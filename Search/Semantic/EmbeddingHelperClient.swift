import Foundation
import Shared

/// Turns text into model vectors. The production implementation is `EmbeddingHelperClient`; tests supply their own.
public protocol TextEmbedding: Sendable {
    /// One L2-normalized vector per text, in order, at the model's full dimension.
    func embed(_ texts: [String]) async throws -> [[Float]]

    /// Frees the model. The next `embed` loads it again.
    func shutdown()
}

/// A screenshot ready for the image encoder: packed RGB, 8 bits a channel, already at its final size
/// (`ImageSizing.targetSize`), so the encoder never rescales it.
public struct EmbeddingImage: Sendable {
    public let width: Int
    public let height: Int
    public let rgb: Data

    public init(width: Int, height: Int, rgb: Data) {
        self.width = width
        self.height = height
        self.rgb = rgb
    }
}

/// Turns screenshots into model vectors, in the same space as `TextEmbedding`'s.
public protocol ImageEmbedding: Sendable {
    /// One L2-normalized vector per image, in order, at the model's full dimension.
    func embed(images: [EmbeddingImage]) async throws -> [[Float]]

    /// Frees the model. The next `embed` loads it again.
    func shutdown()
}

public enum EmbeddingHelperError: Error, CustomStringConvertible {
    case helperNotFound
    case modelNotFound(String)
    case failedToStart(String)
    case exited
    case requestFailed(String)
    /// The helper didn't answer in time. The value says what was being waited for.
    case timedOut(String)

    public var description: String {
        switch self {
        case .helperNotFound: return "RetraceEmbeddingHelper isn't part of this build"
        case .modelNotFound(let path): return "no embedding model at \(path)"
        case .failedToStart(let why): return "the embedding helper didn't start: \(why)"
        case .exited: return "the embedding helper exited"
        case .requestFailed(let why): return "the embedding helper failed: \(why)"
        case .timedOut(let what): return "the embedding helper timed out: \(what)"
        }
    }
}

/// Runs `RetraceEmbeddingHelper` (the embedding model on llama.cpp) as a child process and talks to it over
/// stdin/stdout, one JSON line each way.
///
/// The model lives in that process so its memory goes back to the system when it stops, and a crash in llama.cpp
/// can't take Retrace down. The helper starts on the first request and stops after `idleTimeout` without one.
///
/// Given a `mediaModelPath` the helper also loads the image encoder (about a gigabyte in all, against half
/// for text alone), so screenshots get a client of their own that only runs while there are screenshots to embed.
///
/// A call waits at most `startupTimeout` for a helper to load and `requestTimeout` for its reply. A helper that
/// dies, hangs, or answers with something unusable makes the call throw; it is then killed and the next call
/// starts a fresh one.
public final class EmbeddingHelperClient: TextEmbedding, ImageEmbedding, @unchecked Sendable {
    private let helperURL: URL
    private let modelPath: String
    private let mediaModelPath: String?
    private let idleTimeout: TimeInterval
    private let requestTimeout: TimeInterval
    private let startupTimeout: TimeInterval

    /// How long a helper that was asked to quit has to exit before it is killed.
    private static let quitGrace: TimeInterval = 0.5
    /// How long to wait for a killed helper to be gone. SIGKILL takes a few milliseconds.
    private static let killGrace: TimeInterval = 1

    /// All process state is touched only on this queue. Waits on the helper block it, never a Swift task thread.
    private let queue = DispatchQueue(label: "io.retrace.embedding-helper", qos: .utility)
    private var process: Process?
    /// The helper's stdin. Non-blocking, and a write to a helper that is gone fails with EPIPE instead of
    /// raising SIGPIPE.
    private var input: FileHandle?
    /// The helper's stdout. Non-blocking.
    private var output: FileHandle?
    /// Signalled once the helper has exited and been reaped. Nil when there is nothing left to wait for.
    private var exited: DispatchSemaphore?
    /// What the helper has printed and `readReply` hasn't consumed yet.
    private var buffer = Data()
    /// How much of `buffer` is already known to hold no newline.
    private var scanned = 0
    private var nextID = 1
    private var idleTimer: DispatchSourceTimer?

    public init(
        helperURL: URL,
        modelPath: String,
        mediaModelPath: String? = nil,
        idleTimeout: TimeInterval = 120,
        requestTimeout: TimeInterval = 120,
        startupTimeout: TimeInterval = 90
    ) {
        self.helperURL = helperURL
        self.modelPath = modelPath
        self.mediaModelPath = mediaModelPath
        self.idleTimeout = idleTimeout
        self.requestTimeout = requestTimeout
        self.startupTimeout = startupTimeout
    }

    deinit {
        // Nothing else can reach this object any more, so its state is safe to touch off the queue. Nothing
        // waits here (this can run on any thread): the helper exits when its stdin closes, the kill makes sure
        // of it, and Foundation reaps the child when it does.
        idleTimer?.cancel()
        try? input?.close()
        if let process, process.isRunning { kill(process.processIdentifier, SIGKILL) }
    }

    /// Where the helper is in this build: beside the running executable (SwiftPM and dev builds) or in the app
    /// bundle's helper folders. Nil when the build has none. Only locations inside the build are considered, so
    /// nothing outside it can be substituted for the helper.
    public static func locateHelper() -> URL? {
        let name = "RetraceEmbeddingHelper"
        var candidates: [URL] = []
        if let executable = Bundle.main.executableURL {
            candidates.append(executable.deletingLastPathComponent().appendingPathComponent(name))
        }
        let bundle = Bundle.main.bundleURL
        candidates.append(bundle.appendingPathComponent("Contents/Helpers/\(name)"))
        candidates.append(bundle.appendingPathComponent("Contents/Library/Helpers/\(name)"))
        return candidates.first { FileManager.default.isExecutableFile(atPath: $0.path) }
    }

    public func embed(_ texts: [String]) async throws -> [[Float]] {
        guard !texts.isEmpty else { return [] }
        return try await send(["op": "embed", "inputs": texts], count: texts.count)
    }

    public func embed(images: [EmbeddingImage]) async throws -> [[Float]] {
        guard !images.isEmpty else { return [] }
        guard mediaModelPath != nil else {
            throw EmbeddingHelperError.requestFailed("this client was created without the image encoder")
        }
        let payload = images.map { image in
            ["width": image.width, "height": image.height, "rgb": image.rgb.base64EncodedString()] as [String: Any]
        }
        return try await send(["op": "image", "images": payload], count: images.count)
    }

    private func send(_ body: [String: Any], count: Int) async throws -> [[Float]] {
        // Requests wait their turn on the queue. One whose caller has been cancelled by the time its turn comes
        // (a search that was superseded, or gave up on a slow helper) is dropped there, so it neither starts
        // the helper nor holds up the requests behind it. A request already with the helper runs to its reply
        // or its timeout.
        let cancelled = CancellationFlag()
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                queue.async {
                    guard !cancelled.isSet else {
                        continuation.resume(throwing: CancellationError())
                        return
                    }
                    do {
                        continuation.resume(returning: try self.request(body, count: count))
                    } catch {
                        // A helper in an unknown state is not reused, and gets nothing more written to it.
                        self.stop(graceful: false)
                        continuation.resume(throwing: error)
                    }
                }
            }
        } onCancel: {
            cancelled.set()
        }
    }

    private final class CancellationFlag: @unchecked Sendable {
        private let lock = NSLock()
        private var value = false
        var isSet: Bool { lock.withLock { value } }
        func set() { lock.withLock { value = true } }
    }

    public func shutdown() {
        queue.async { self.stop(graceful: true) }
    }

    // MARK: - On the queue

    private func request(_ body: [String: Any], count: Int) throws -> [[Float]] {
        try startIfNeeded()

        let id = nextID
        nextID += 1
        var message = body
        message["id"] = id
        var line = try JSONSerialization.data(withJSONObject: message)
        line.append(0x0A)

        // A helper that hangs would block this queue for good (searches wait on it): give up on it after
        // `requestTimeout`.
        let deadline = DispatchTime.now() + requestTimeout
        let timeout = EmbeddingHelperError.timedOut("no reply after \(Self.seconds(requestTimeout))")
        try write(line, until: deadline, orThrow: timeout)
        let reply = try readReply(until: deadline, orThrow: timeout, limit: Self.replyLimit(count: count))
        guard (reply["id"] as? Int) == id else {
            throw EmbeddingHelperError.requestFailed("reply out of order")
        }
        guard reply["ok"] as? Bool == true,
              let encoded = reply["vectors"] as? String,
              let data = Data(base64Encoded: encoded),
              let dimension = reply["dim"] as? Int, dimension > 0 else {
            throw EmbeddingHelperError.requestFailed(reply["error"] as? String ?? "no vectors")
        }
        // `dimension` is checked against the bytes first so that a nonsense value can't overflow the product.
        guard dimension <= data.count, data.count == count * dimension * MemoryLayout<Float>.stride else {
            throw EmbeddingHelperError.requestFailed("got \(data.count) bytes for \(count) inputs")
        }

        var floats = [Float](repeating: 0, count: count * dimension)
        _ = floats.withUnsafeMutableBytes { data.copyBytes(to: $0) }
        scheduleIdleStop()
        return (0..<count).map { Array(floats[($0 * dimension)..<(($0 + 1) * dimension)]) }
    }

    private func startIfNeeded() throws {
        if let process, process.isRunning, !helperHungUp() { return }
        stop(graceful: false)

        guard FileManager.default.isExecutableFile(atPath: helperURL.path) else {
            throw EmbeddingHelperError.helperNotFound
        }
        guard FileManager.default.fileExists(atPath: modelPath) else {
            throw EmbeddingHelperError.modelNotFound(modelPath)
        }
        if let mediaModelPath, !FileManager.default.fileExists(atPath: mediaModelPath) {
            throw EmbeddingHelperError.modelNotFound(mediaModelPath)
        }

        let process = Process()
        let stdin = Pipe()
        let stdout = Pipe()
        let input = stdin.fileHandleForWriting
        let output = stdout.fileHandleForReading
        // Without F_SETNOSIGPIPE a write to a helper that has died raises SIGPIPE, which nothing in Retrace
        // handles: the app would die with its helper. With it the write fails with EPIPE. Both ends are
        // non-blocking so that every wait on the helper can have a deadline. (These are this process's ends of
        // the pipes; the helper's own stdin and stdout stay blocking.)
        guard fcntl(input.fileDescriptor, F_SETNOSIGPIPE, 1) != -1,
              Self.makeNonBlocking(input.fileDescriptor),
              Self.makeNonBlocking(output.fileDescriptor) else {
            throw EmbeddingHelperError.failedToStart("couldn't set up its pipes: \(String(cString: strerror(errno)))")
        }
        let exited = DispatchSemaphore(value: 0)
        process.terminationHandler = { _ in exited.signal() }
        process.executableURL = helperURL
        process.arguments = ["--model", modelPath] + (mediaModelPath.map { ["--media", $0] } ?? [])
        process.standardInput = stdin
        process.standardOutput = stdout
        process.standardError = FileHandle.nullDevice
        process.qualityOfService = .utility
        do {
            try process.run()
        } catch {
            throw EmbeddingHelperError.failedToStart(error.localizedDescription)
        }
        self.process = process
        self.input = input
        self.output = output
        self.exited = exited
        buffer.removeAll()
        scanned = 0

        // A helper that stalls while loading the model would block this queue for good (searches wait on it):
        // give up on it after `startupTimeout`.
        let started = Date()
        let ready: [String: Any]
        do {
            ready = try readReply(
                until: .now() + startupTimeout,
                orThrow: .timedOut("not ready after \(Self.seconds(startupTimeout))"),
                limit: Self.replyLimit(count: 0)
            )
        } catch EmbeddingHelperError.exited {
            throw EmbeddingHelperError.failedToStart("it exited before it was ready")
        }
        guard ready["ready"] as? Bool == true else {
            throw EmbeddingHelperError.failedToStart(ready["error"] as? String ?? "no ready line")
        }
        Log.info(
            "[Semantic] Embedding helper ready in \(String(format: "%.2f", Date().timeIntervalSince(started))) s",
            category: .search
        )
    }

    /// Whether the helper has closed its end of the pipe. That shows a helper that died while idle before
    /// `isRunning` does.
    private func helperHungUp() -> Bool {
        guard let output else { return true }
        var descriptor = pollfd(fd: output.fileDescriptor, events: Int16(POLLIN), revents: 0)
        return poll(&descriptor, 1, 0) > 0 && descriptor.revents & Int16(POLLHUP | POLLERR | POLLNVAL) != 0
    }

    /// Sends `data` to the helper. While the helper's stdin is full this also takes in whatever the helper
    /// prints, so the two can't end up each waiting for the other to read.
    private func write(_ data: Data, until deadline: DispatchTime, orThrow timeout: EmbeddingHelperError) throws {
        guard let input else { throw EmbeddingHelperError.exited }
        var sent = 0
        while sent < data.count {
            let (written, code) = data.withUnsafeBytes { bytes -> (Int, Int32) in
                let written = Darwin.write(input.fileDescriptor, bytes.baseAddress! + sent, bytes.count - sent)
                return (written, written < 0 ? errno : 0)
            }
            if written > 0 {
                sent += written
            } else if code == EAGAIN {
                if try wait(forRoomToWrite: true, until: deadline, orThrow: timeout) { try readAvailable() }
            } else if code != EINTR {
                throw EmbeddingHelperError.exited   // EPIPE: the helper is gone
            }
        }
    }

    /// The next line the helper prints, as JSON. `limit` caps how much a helper gone wrong can make this hold.
    private func readReply(
        until deadline: DispatchTime,
        orThrow timeout: EmbeddingHelperError,
        limit: Int
    ) throws -> [String: Any] {
        while true {
            // Blank lines carry nothing: drop a run of them in one go.
            let blank = buffer.prefix { $0 == 0x0A }.count
            if blank > 0 {
                buffer.removeSubrange(buffer.startIndex..<(buffer.startIndex + blank))
                scanned = 0
            }
            if let newline = buffer[(buffer.startIndex + scanned)...].firstIndex(of: 0x0A) {
                let line = Data(buffer[buffer.startIndex..<newline])
                buffer.removeSubrange(buffer.startIndex...newline)
                scanned = 0
                guard let object = (try? JSONSerialization.jsonObject(with: line)) as? [String: Any] else {
                    throw EmbeddingHelperError.requestFailed("unreadable reply")
                }
                return object
            }
            scanned = buffer.count
            guard buffer.count <= limit else {
                throw EmbeddingHelperError.requestFailed("reply longer than \(limit) bytes")
            }
            if try wait(forRoomToWrite: false, until: deadline, orThrow: timeout) { try readAvailable() }
        }
    }

    /// Waits until the helper has printed something or closed its stdout, or, with `forRoomToWrite`, until its
    /// stdin can take more. Returns whether there is output to take in. Throws `timeout` once `deadline` passes.
    private func wait(
        forRoomToWrite: Bool,
        until deadline: DispatchTime,
        orThrow timeout: EmbeddingHelperError
    ) throws -> Bool {
        guard let output else { throw EmbeddingHelperError.exited }
        var descriptors = [pollfd(fd: output.fileDescriptor, events: Int16(POLLIN), revents: 0)]
        if forRoomToWrite {
            guard let input else { throw EmbeddingHelperError.exited }
            descriptors.append(pollfd(fd: input.fileDescriptor, events: Int16(POLLOUT), revents: 0))
        }
        while true {
            let now = DispatchTime.now()
            guard now < deadline else { throw timeout }
            let milliseconds = (deadline.uptimeNanoseconds - now.uptimeNanoseconds) / 1_000_000 + 1
            let ready = poll(&descriptors, nfds_t(descriptors.count), Int32(min(milliseconds, 60_000)))
            if ready > 0 { return descriptors[0].revents != 0 }
            if ready < 0, errno != EINTR { throw EmbeddingHelperError.exited }
        }
    }

    /// Moves what the helper has printed so far into `buffer`. Never blocks. Throws once the helper has closed
    /// its stdout, which is how its death shows.
    private func readAvailable() throws {
        guard let output else { throw EmbeddingHelperError.exited }
        var chunk = [UInt8](repeating: 0, count: 64 * 1024)
        while true {
            let count = read(output.fileDescriptor, &chunk, chunk.count)
            if count > 0 {
                buffer.append(chunk, count: count)
                return
            }
            if count == 0 { throw EmbeddingHelperError.exited }
            if errno == EAGAIN { return }
            if errno != EINTR { throw EmbeddingHelperError.exited }
        }
    }

    private func scheduleIdleStop() {
        idleTimer?.cancel()
        let timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(deadline: .now() + idleTimeout)
        timer.setEventHandler { [weak self] in
            guard let self else { return }
            Log.info("[Semantic] Stopping the idle embedding helper", category: .search)
            self.stop(graceful: true)
        }
        timer.resume()
        idleTimer = timer
    }

    /// Stops the helper and waits, briefly, until it is gone and reaped.
    ///
    /// `graceful` is for a helper that is idle and healthy: it is asked to quit and given `quitGrace` to do so.
    /// After a failed read or write the helper is dead, hung or confused: it is sent nothing more and killed
    /// outright, so the only wait is for the kill to land.
    private func stop(graceful: Bool) {
        idleTimer?.cancel()
        idleTimer = nil
        if let process {
            var askedToQuit = false
            if graceful, process.isRunning {
                let quit = Data(#"{"id":0,"op":"quit"}"#.utf8) + [0x0A]
                askedToQuit = (try? write(quit, until: .now() + Self.quitGrace, orThrow: .exited)) != nil
            }
            try? input?.close()   // end of input makes the helper exit too
            input = nil
            if !(askedToQuit && waitForExit(Self.quitGrace)) {
                // `isRunning` is still true for a child that has died but hasn't been reaped, so its process ID
                // can't have gone to another process yet.
                if process.isRunning { kill(process.processIdentifier, SIGKILL) }
                if !waitForExit(Self.killGrace) {
                    Log.warning(
                        "[Semantic] Embedding helper \(process.processIdentifier) is still there after SIGKILL",
                        category: .search
                    )
                }
            }
        }
        try? output?.close()
        process = nil
        input = nil
        output = nil
        exited = nil
        buffer.removeAll()
        scanned = 0
    }

    /// Whether the helper exited and was reaped within `seconds`.
    private func waitForExit(_ seconds: TimeInterval) -> Bool {
        guard let exited else { return true }
        guard exited.wait(timeout: .now() + seconds) == .success else { return false }
        self.exited = nil
        return true
    }

    private static func makeNonBlocking(_ descriptor: Int32) -> Bool {
        let flags = fcntl(descriptor, F_GETFL)
        return flags != -1 && fcntl(descriptor, F_SETFL, flags | O_NONBLOCK) != -1
    }

    /// The most bytes a reply for `count` inputs may take: far above any real one (a vector is a few kilobytes),
    /// far below what would hurt.
    private static func replyLimit(count: Int) -> Int {
        (1 << 20) + count * (128 << 10)
    }

    private static func seconds(_ interval: TimeInterval) -> String {
        String(format: "%g s", interval)
    }
}
