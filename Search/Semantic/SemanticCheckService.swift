import CoreGraphics
import Foundation
import ImageIO
import Shared
import UniformTypeIdentifiers

/// The result check (prototype): a small local decision model looks at each of the best semantic results and
/// says how sure it is that the screenshot shows what was searched for.
///
/// The model runs in `llama-server` (llama.cpp), started on the first check and stopped after `idleTimeout`
/// without one, so it costs memory only around searches. Each result is one request to the server's decision
/// endpoint (`/v1/systemone`): the screenshot, the app and window title, and one yes/no question. The answer is a
/// probability; nothing is generated.
///
/// A search that asks for a picture (`SemanticCheckSettings.asksForPicture`) gets two looks per result. Shown the
/// full screenshot, the model says yes to a page that only has the words on it; shown one too small to read, it
/// can only go by what is pictured, but mistakes more blurred shapes for the thing. The two are wrong about
/// different frames, so a result has to pass both, and its confidence is the lower of the two.
///
/// Search does not wait for this. Results are shown as the embeddings ranked them, and the answers arrive one
/// at a time (`verdicts`) while the user is already looking; nothing is asked past the deadline, whether the
/// model is still loading or an answer is slow.
///
/// Prototype limits: the server binary and the model files are whatever `SemanticCheckSettings` points at
/// (nothing is bundled or downloaded), requests go one at a time, and a server that cannot start simply means
/// no check.
public actor SemanticCheckService: SemanticResultChecking {
    /// Reads one frame from the video file as it is now. Checks for different searches can overlap, so a source
    /// must not share video decoders between calls: a chunk rewritten to burn in a redaction would otherwise be
    /// read through a decoder another call opened before the rewrite.
    public typealias ImageSource = @Sendable (SemanticCheckCandidate) async throws -> CGImage
    /// A value that changes whenever a frame's pixels change, or nil while they are not final (still being
    /// encoded, processed or redacted, or deleted): the frame must then not be looked at.
    public typealias FrameRevision = @Sendable (Int64) async -> String?

    private let serverURL: URL
    private let modelPath: String
    private let projectorPath: String
    private let image: ImageSource
    private let revision: FrameRevision
    private let fixedPort: Int?
    private let idleTimeout: TimeInterval
    private let startTimeout: TimeInterval
    private let session: URLSession

    private var process: Process?
    private var startTask: Task<Bool, Never>?
    /// The server answered its health check and has not been stopped since.
    private var isReady = false
    /// Where this launch of the server listens, and the key it was started with. Both are new for every launch,
    /// and a server counts as ours only if it accepts this key and refuses another: a screenshot is never sent
    /// to whatever else may be answering on the port.
    private var port = 0
    private var apiKey = ""
    /// Counts starts and stops, so a start that was overtaken by a stop knows to give up.
    private var startGeneration = 0
    private var lastUse = Date.distantPast
    private var idleTask: Task<Void, Never>?

    public init(
        serverURL: URL,
        modelPath: String,
        projectorPath: String,
        port: Int? = nil,
        idleTimeout: TimeInterval = 120,
        startTimeout: TimeInterval = 30,
        image: @escaping ImageSource,
        revision: @escaping FrameRevision = { _ in "" }
    ) {
        self.serverURL = serverURL
        self.modelPath = modelPath
        self.projectorPath = projectorPath
        self.fixedPort = port
        self.idleTimeout = idleTimeout
        self.startTimeout = startTimeout
        self.image = image
        self.revision = revision
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 20
        configuration.waitsForConnectivity = false
        self.session = URLSession(configuration: configuration)
    }

    // MARK: - SemanticResultChecking

    public func confidences(
        for query: String,
        candidates: [SemanticCheckCandidate],
        deadline: Duration
    ) async -> [Int64: Double] {
        await check(for: query, candidates: candidates, deadline: deadline, onAnswer: nil)
    }

    public nonisolated func verdicts(
        for query: String,
        candidates: [SemanticCheckCandidate],
        deadline: Duration
    ) -> AsyncStream<SemanticCheckVerdict> {
        AsyncStream { continuation in
            let work = Task {
                _ = await self.check(for: query, candidates: candidates, deadline: deadline) { verdict in
                    continuation.yield(verdict)
                }
                continuation.finish()
            }
            // Whoever reads the answers went away (a new search, the overlay closed): stop asking.
            continuation.onTermination = { _ in work.cancel() }
        }
    }

    /// Looks at the candidates in order until they are done, the deadline runs out or the task is cancelled, and
    /// reports each answer as it comes.
    private func check(
        for query: String,
        candidates: [SemanticCheckCandidate],
        deadline: Duration,
        onAnswer: (@Sendable (SemanticCheckVerdict) -> Void)?
    ) async -> [Int64: Double] {
        // What was searched for, without `-excluded` words and `app:` style filters: those were applied when
        // the results were fetched, and the embeddings were not shown them either.
        let text = SemanticSearchService.queryText(query)
        guard !text.isEmpty, !candidates.isEmpty else { return [:] }
        let started = ContinuousClock.now
        let end = started + deadline
        guard await ensureRunning(until: end) else { return [:] }
        lastUse = Date()
        defer {
            lastUse = Date()
            scheduleIdleStop()
        }

        var result: [Int64: Double] = [:]
        var reading = Duration.zero, asking = Duration.zero
        // The model is what takes the time, and it does not need the processor that reads frames out of video
        // files. So the next result's frame is read while the model answers for this one: one frame ahead, no
        // more, so that little is read for nothing when the search is abandoned.
        let twoLooks = SemanticCheckSettings.asksForPicture(text)
        var ahead: Task<PreparedFrame?, Never>? = Task { [self] in await prepare(candidates[0], small: twoLooks) }
        defer { ahead?.cancel() }
        for (index, candidate) in candidates.enumerated() {
            let remaining = end - ContinuousClock.now
            guard remaining > .zero, !Task.isCancelled else { break }
            let preparing = ahead
            ahead = index + 1 < candidates.count
                ? Task { [self] in await prepare(candidates[index + 1], small: twoLooks) } : nil
            do {
                guard let prepared = await preparing?.value else { continue }
                reading += prepared.took
                // The frame was read from final pixels, but it has waited for the answer before it since then.
                // It is sent only if it still is what it was: a frame queued for redaction in between stays here.
                guard await revision(candidate.frameID) == prepared.revision else { continue }
                let left = end - ContinuousClock.now
                guard left > .zero else { break }
                let askStarted = ContinuousClock.now
                defer { asking += ContinuousClock.now - askStarted }
                var glance: Double?
                if let small = prepared.small {
                    // The small look first: it takes a sixth of the time, and a result it turns down needs no
                    // second look. It goes without the window title, which is words too.
                    glance = try await ask(query: text, candidate: candidate, jpeg: small, withWindow: false, timeout: left)
                    guard let glance else { continue }
                    if glance < SemanticCheckSettings.cutoff {
                        result[candidate.frameID] = glance
                        onAnswer?(SemanticCheckVerdict(frameID: candidate.frameID, confidence: glance))
                        continue
                    }
                }
                let leftNow = end - ContinuousClock.now
                guard leftNow > .zero else { break }
                if let look = try await ask(query: text, candidate: candidate, jpeg: prepared.jpeg, timeout: leftNow) {
                    let confidence = min(look, glance ?? 1)
                    result[candidate.frameID] = confidence
                    onAnswer?(SemanticCheckVerdict(frameID: candidate.frameID, confidence: confidence))
                }
            } catch is CancellationError {
                break
            } catch {
                // One failed request: that result stays unchecked.
                Log.debug("[SemanticCheck] Frame \(candidate.frameID) not checked: \(error)", category: .search)
            }
        }
        let elapsed = ContinuousClock.now - started
        Log.info(
            "[SemanticCheck] Checked \(result.count) of \(candidates.count) results in \(elapsed)"
                + " (the model \(asking); reading frames \(reading), done while the model worked)",
            category: .search
        )
        return result
    }

    /// A result's screenshot, ready to send, and what its frame was when it was read.
    private struct PreparedFrame: Sendable {
        let jpeg: Data
        /// The same screenshot too small to read, when the search asks for a picture.
        let small: Data?
        let revision: String
        let took: Duration
    }

    /// Reads one result's screenshot, or nil if it may not be looked at or cannot be read. Not on the actor: it
    /// runs while the actor waits for the model.
    ///
    /// The caller picked frames whose pixels were final, but that was before the model loaded and the results
    /// ahead of this one were checked. So the frame is asked about again here, and counts only if nothing changed
    /// while it was being read.
    private nonisolated func prepare(_ candidate: SemanticCheckCandidate, small: Bool) async -> PreparedFrame? {
        let started = ContinuousClock.now
        guard !Task.isCancelled, let before = await revision(candidate.frameID) else { return nil }
        do {
            let frame = try await image(candidate)
            guard await revision(candidate.frameID) == before,
                  let jpeg = Self.jpeg(from: frame, longestSide: SemanticCheckSettings.imageWidth) else { return nil }
            var smallJPEG: Data?
            if small {
                smallJPEG = Self.jpeg(from: frame, longestSide: SemanticCheckSettings.pictureImageWidth)
                guard smallJPEG != nil else { return nil }
            }
            return PreparedFrame(jpeg: jpeg, small: smallJPEG, revision: before, took: ContinuousClock.now - started)
        } catch {
            // A frame that can't be read: that result stays unchecked.
            Log.debug("[SemanticCheck] Frame \(candidate.frameID) not read: \(error)", category: .search)
            return nil
        }
    }

    /// Stops the server now (app shutdown, or the check was switched off).
    public func stop() {
        idleTask?.cancel()
        idleTask = nil
        startGeneration += 1
        startTask?.cancel()
        startTask = nil
        isReady = false
        if let process, process.isRunning { process.terminate() }
        process = nil
    }

    // MARK: - One request

    static func requestBody(
        query: String,
        candidate: SemanticCheckCandidate,
        jpeg: Data,
        withWindow: Bool = true
    ) -> [String: Any] {
        let app = candidate.appName.flatMap { $0.isEmpty ? nil : $0 } ?? "unknown"
        let window = candidate.windowName.flatMap { $0.isEmpty ? nil : $0 } ?? "none"
        return [
            "state": withWindow ? "App: \(app). Window title: \(window)." : "",
            "images": ["data:image/jpeg;base64," + jpeg.base64EncodedString()],
            "questions": [
                "match": [
                    "type": "noul",
                    "instructions": SemanticCheckSettings.question(for: query),
                ] as [String: Any],
            ],
        ]
    }

    /// The probability of "yes" in a decision endpoint reply, or nil if it isn't there.
    static func confidence(inReply data: Data) -> Double? {
        guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let answers = object["answers"] as? [String: Any],
              let match = answers["match"] as? [String: Any],
              let value = match["noul"] as? Double,
              value.isFinite else {
            return nil
        }
        return min(max(value, 0), 1)
    }

    private func ask(
        query: String,
        candidate: SemanticCheckCandidate,
        jpeg: Data,
        withWindow: Bool = true,
        timeout: Duration
    ) async throws -> Double? {
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/systemone")!)
        request.httpMethod = "POST"
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        // The server sends nothing until it has the answer, so this bounds the whole request.
        let seconds = Double(timeout.components.seconds) + Double(timeout.components.attoseconds) / 1e18
        request.timeoutInterval = min(20, max(0.05, seconds))
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(
            withJSONObject: Self.requestBody(query: query, candidate: candidate, jpeg: jpeg, withWindow: withWindow)
        )
        let (data, response) = try await session.data(for: request)
        guard (response as? HTTPURLResponse)?.statusCode == 200 else { return nil }
        return Self.confidence(inReply: data)
    }

    // MARK: - Image

    /// `image` as JPEG, scaled down so its longest side is at most `longestSide`.
    static func jpeg(from image: CGImage, longestSide: Int, quality: Double = 0.8) -> Data? {
        let scale = min(1, Double(longestSide) / Double(max(image.width, image.height, 1)))
        let width = max(1, Int((Double(image.width) * scale).rounded()))
        let height = max(1, Int((Double(image.height) * scale).rounded()))
        var output = image
        if scale < 1 {
            guard let context = CGContext(
                data: nil,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: 0,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
            ) else { return nil }
            context.interpolationQuality = .high
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            guard let scaled = context.makeImage() else { return nil }
            output = scaled
        }
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(data, UTType.jpeg.identifier as CFString, 1, nil) else {
            return nil
        }
        CGImageDestinationAddImage(
            destination,
            output,
            [kCGImageDestinationLossyCompressionQuality: quality] as CFDictionary
        )
        return CGImageDestinationFinalize(destination) ? data as Data : nil
    }

    // MARK: - Server

    /// True once the server is up. Starts it if needed and waits for it, but not past `end`: a model that takes
    /// longer to load than a search may wait goes on loading for the next search, and this one does without.
    private func ensureRunning(until end: ContinuousClock.Instant) async -> Bool {
        if isReady, let process, process.isRunning { return true }
        let attempt: Task<Bool, Never>
        if let startTask {
            attempt = startTask
        } else {
            isReady = false
            startGeneration += 1
            let generation = startGeneration
            attempt = Task { await self.start(generation: generation) }
            startTask = attempt
        }
        while startTask == attempt {
            guard ContinuousClock.now < end, !Task.isCancelled else { return false }
            try? await Task.sleep(for: .milliseconds(25))
        }
        if let process, process.isRunning { return isReady }
        return false
    }

    private func start(generation: Int) async -> Bool {
        if let process, process.isRunning { process.terminate() }
        process = nil
        isReady = false
        guard let launchPort = fixedPort ?? Self.freePort() else {
            Log.warning("[SemanticCheck] No port for the check model", category: .search)
            if generation == startGeneration { startTask = nil }
            return false
        }
        port = launchPort
        apiKey = UUID().uuidString + UUID().uuidString

        // Launched through a shell that also watches this app: if Retrace goes away without stopping the
        // server (a crash, a force quit), the shell stops it. `exec`-ing the server directly would leave a
        // model in memory with nothing to end it.
        let launched = Process()
        launched.executableURL = URL(fileURLWithPath: "/bin/sh")
        launched.arguments = [
            "-c",
            """
            "$0" "$@" & server=$!
            trap 'kill $server 2>/dev/null; exit 0' TERM INT
            while kill -0 \(ProcessInfo.processInfo.processIdentifier) 2>/dev/null && kill -0 $server 2>/dev/null; do sleep 2 & wait $!; done
            kill $server 2>/dev/null
            """,
            serverURL.path,
            "-m", modelPath,
            "--mmproj", projectorPath,
            "-ngl", "99",
            "--host", "127.0.0.1",
            "--port", String(port),
            "-c", "4096", "-b", "4096", "-ub", "4096",
            "-np", "1",
            "--image-max-tokens", String(SemanticCheckSettings.imageTokens),
        ]
        // The key goes in the environment, not on the command line, where any process listing would show it.
        var environment = ProcessInfo.processInfo.environment
        environment["LLAMA_API_KEY"] = apiKey
        launched.environment = environment
        launched.standardInput = FileHandle.nullDevice
        launched.standardOutput = FileHandle.nullDevice
        launched.standardError = FileHandle.nullDevice
        do {
            try launched.run()
        } catch {
            Log.warning("[SemanticCheck] Could not start the check model: \(error)", category: .search)
            if generation == startGeneration { startTask = nil }
            return false
        }
        process = launched

        let began = Date()
        var healthy = false
        while Date().timeIntervalSince(began) < startTimeout {
            if generation != startGeneration || !launched.isRunning { break }
            if await isOurs() {
                healthy = true
                break
            }
            try? await Task.sleep(for: .milliseconds(150))
        }
        guard generation == startGeneration else {
            // Stopped while it was starting; whoever stopped it owns `process` and `startTask` now.
            if launched.isRunning { launched.terminate() }
            return false
        }
        startTask = nil
        if healthy {
            isReady = true
            // The search that asked for this may have given up waiting; the model must still go away when
            // nothing uses it.
            lastUse = Date()
            scheduleIdleStop()
            Log.info(
                "[SemanticCheck] Check model ready in \(String(format: "%.1f", Date().timeIntervalSince(began))) s",
                category: .search
            )
            return true
        }
        Log.warning("[SemanticCheck] The check model did not come up", category: .search)
        if launched.isRunning { launched.terminate() }
        process = nil
        return false
    }

    /// True when the server on `port` is up and is the one this launch started: it lets this launch's key in
    /// and keeps any other out. Something else listening there (another app, another copy of Retrace with its
    /// own model) fails one half or the other.
    private func isOurs() async -> Bool {
        func status(key: String) async -> Int? {
            var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/props")!)
            request.timeoutInterval = 1
            request.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
            guard let (_, response) = try? await session.data(for: request) else { return nil }
            return (response as? HTTPURLResponse)?.statusCode
        }
        guard await status(key: apiKey) == 200 else { return false }
        return await status(key: "not-" + apiKey) == 401
    }

    /// A port nothing is listening on right now.
    static func freePort() -> Int? {
        let descriptor = socket(AF_INET, SOCK_STREAM, 0)
        guard descriptor >= 0 else { return nil }
        defer { close(descriptor) }
        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_addr.s_addr = inet_addr("127.0.0.1")
        address.sin_port = 0
        let bound = withUnsafePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.bind(descriptor, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard bound == 0 else { return nil }
        var length = socklen_t(MemoryLayout<sockaddr_in>.size)
        let named = withUnsafeMutablePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { getsockname(descriptor, $0, &length) }
        }
        guard named == 0 else { return nil }
        return Int(UInt16(bigEndian: address.sin_port))
    }

    private func scheduleIdleStop(after delay: TimeInterval? = nil) {
        idleTask?.cancel()
        let timeout = delay ?? idleTimeout
        idleTask = Task { [weak self] in
            try? await Task.sleep(for: .seconds(timeout))
            guard !Task.isCancelled else { return }
            await self?.stopIfIdle()
        }
    }

    private func stopIfIdle() {
        let idle = Date().timeIntervalSince(lastUse)
        guard idle >= idleTimeout, startTask == nil else {
            // Used, or being started, since this timer was set. Whoever used it normally sets a new timer when it
            // is done, but a search stuck reading a frame never gets that far: look again when the time is up.
            scheduleIdleStop(after: max(idleTimeout - idle, 0.05))
            return
        }
        if let process, process.isRunning { process.terminate() }
        process = nil
        isReady = false
        Log.info("[SemanticCheck] Check model stopped after \(Int(idleTimeout)) s without a search", category: .search)
    }
}
