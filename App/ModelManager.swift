import CryptoKit
import Foundation
import Shared

/// Manages runtime downloads of AI models (whisper.cpp + embeddings)
/// Downloads models on first app launch with user consent
/// Owner: APP integration
public actor ModelManager {

    // MARK: - Model Configuration

    public struct ModelInfo: Sendable {
        let name: String
        let filename: String
        let url: String
        let sizeBytes: Int64
        let purpose: String
        /// Lowercase hex SHA-256 of the file. When set, a download that doesn't match is rejected.
        var sha256: String? = nil

        /// Size in megabytes as Finder and the model's download page show it (1 MB = 1,000,000 bytes).
        var sizeMB: Int {
            ModelManager.megabytes(sizeBytes)
        }
    }

    /// Bytes as whole decimal megabytes (1 MB = 1,000,000 bytes), rounded to the nearest.
    static func megabytes(_ bytes: Int64) -> Int {
        Int((Double(bytes) / 1_000_000).rounded())
    }

    public static let whisperModel = ModelInfo(
        name: "Whisper Small",
        filename: "ggml-small.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin",
        sizeBytes: 465_000_000, // ~465 MB
        purpose: "Speech-to-text transcription"
    )

    /// EmbeddingGemma 2's text model (Google DeepMind, Apache 2.0), ggml-org's 8-bit GGUF at one pinned revision.
    /// The image and audio encoders are a separate file and are not needed for text.
    public static let embeddingModel = ModelInfo(
        name: "EmbeddingGemma 2 (text)",
        filename: SemanticSearchSettings.modelFileName,
        url: "https://huggingface.co/ggml-org/embeddinggemma-2-GGUF/resolve/bfcd298762cc34d0357ece5ebdd31791a3a374d8/embeddinggemma-2-Q8_0.gguf",
        sizeBytes: 309_855_456, // ~310 MB
        purpose: "Semantic search embeddings",
        sha256: "2188ac1deca4b77dffefd603c2776a9d76d9d74ec01841392982ebb840b09135"
    )

    /// EmbeddingGemma 2's image (and audio) encoders, from the same pinned revision. Needed only to embed
    /// screenshots; searching and text embedding use the text model alone.
    public static let imageEncoderModel = ModelInfo(
        name: "EmbeddingGemma 2 (image encoder)",
        filename: SemanticSearchSettings.imageModelFileName,
        url: "https://huggingface.co/ggml-org/embeddinggemma-2-GGUF/resolve/bfcd298762cc34d0357ece5ebdd31791a3a374d8/mmproj-embeddinggemma-2-Q8_0.gguf",
        sizeBytes: 554_821_024, // ~555 MB
        purpose: "Semantic search over screenshots",
        sha256: "c4a8a52691ecef40618438928bdf9e68379b854e24166f292592353db0aab64f"
    )

    public static let allModels = [whisperModel, embeddingModel, imageEncoderModel]

    // MARK: - Paths

    /// Downloads a URL to a temporary file and returns where it is. The caller then owns that file (it moves or
    /// deletes it). When it throws, it leaves nothing behind; a cancelled download throws `CancellationError`.
    typealias Fetch = @Sendable (
        _ url: URL,
        _ progress: @escaping @Sendable (_ bytesWritten: Int64, _ bytesExpected: Int64) -> Void
    ) async throws -> URL

    private let modelsDirectory: URL
    private let fetch: Fetch
    /// The download running for each model file, shared by everyone who asked for that model.
    private var inFlight: [String: InFlightDownload] = [:]
    /// A cancelled download that is still stopping. The next download of the same file waits for it.
    private var stopping: [String: Task<Void, Never>] = [:]

    public init(modelsDirectory: URL? = nil) {
        self.modelsDirectory = Self.resolvedModelsDirectory(modelsDirectory)
        self.fetch = { url, progress in try await ModelDownload.run(from: url, progress: progress) }
    }

    /// `fetch` replaces the network step (tests download from memory).
    init(modelsDirectory: URL?, fetch: @escaping Fetch) {
        self.modelsDirectory = Self.resolvedModelsDirectory(modelsDirectory)
        self.fetch = fetch
    }

    private static func resolvedModelsDirectory(_ modelsDirectory: URL?) -> URL {
        // AppPaths respects a custom storage location
        modelsDirectory ?? URL(fileURLWithPath: AppPaths.expandedStorageRoot).appendingPathComponent("models")
    }

    // MARK: - Model Status

    public struct ModelStatus: Sendable {
        public let model: ModelInfo
        public let isDownloaded: Bool
        public let localPath: URL?
        public let fileSizeBytes: Int64?

        public var isValid: Bool {
            guard let path = localPath,
                  let fileSize = fileSizeBytes,
                  isDownloaded else {
                return false
            }

            // Check if file exists and size is reasonable (within 10% of expected)
            guard FileManager.default.fileExists(atPath: path.path) else {
                return false
            }

            let expectedSize = model.sizeBytes
            let tolerance = expectedSize / 10
            return abs(fileSize - expectedSize) < tolerance
        }
    }

    public func getModelStatus(_ model: ModelInfo) async -> ModelStatus {
        Self.status(of: model, in: modelsDirectory)
    }

    private static func status(of model: ModelInfo, in directory: URL) -> ModelStatus {
        let localPath = directory.appendingPathComponent(model.filename)

        guard FileManager.default.fileExists(atPath: localPath.path) else {
            return ModelStatus(
                model: model,
                isDownloaded: false,
                localPath: nil,
                fileSizeBytes: nil
            )
        }

        let fileSize = try? FileManager.default.attributesOfItem(atPath: localPath.path)[.size] as? Int64

        return ModelStatus(
            model: model,
            isDownloaded: true,
            localPath: localPath,
            fileSizeBytes: fileSize
        )
    }

    public func getAllModelStatuses() async -> [ModelStatus] {
        var statuses: [ModelStatus] = []
        for model in Self.allModels {
            let status = await getModelStatus(model)
            statuses.append(status)
        }
        return statuses
    }

    public func areAllModelsDownloaded() async -> Bool {
        let statuses = await getAllModelStatuses()
        return statuses.allSatisfy { $0.isValid }
    }

    // MARK: - Download

    public enum DownloadError: Error, LocalizedError {
        case downloadFailed(String)
        case invalidFileSize(expected: Int64, actual: Int64)
        case fileSystemError(String)

        public var errorDescription: String? {
            switch self {
            case .downloadFailed(let message):
                return "Download failed: \(message)"
            case .invalidFileSize(let expected, let actual):
                return "Invalid file size: expected \(expected) bytes, got \(actual) bytes"
            case .fileSystemError(let message):
                return "File system error: \(message)"
            }
        }
    }

    public struct DownloadProgress: Sendable {
        public let model: ModelInfo
        public let bytesDownloaded: Int64
        public let totalBytes: Int64
        public let percentage: Double

        public var percentageString: String {
            String(format: "%.1f%%", percentage * 100)
        }

        /// Decimal megabytes (1 MB = 1,000,000 bytes).
        public var downloadedMB: Int {
            ModelManager.megabytes(bytesDownloaded)
        }

        /// Decimal megabytes (1 MB = 1,000,000 bytes).
        public var totalMB: Int {
            ModelManager.megabytes(totalBytes)
        }
    }

    /// Downloads `model` into the models folder, or returns at once if a valid copy is already there.
    ///
    /// Calls for the same model share one download: a second caller waits for the one already running and gets
    /// its result. `progressHandler` is called a few times a second (not once per network chunk), then once with
    /// 100% when the file is in place. It may be called on any thread.
    ///
    /// Cancelling the calling task throws `CancellationError`. The download itself stops once nobody is waiting
    /// for it any more. No partial or rejected file is ever left behind.
    public func downloadModel(
        _ model: ModelInfo,
        progressHandler: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws -> URL {
        try Task.checkCancellation()
        let key = model.filename

        if inFlight[key] == nil {
            let status = Self.status(of: model, in: modelsDirectory)
            if status.isValid, let localPath = status.localPath {
                Log.info("Model \(model.name) already downloaded", category: .app)
                return localPath
            }
        }

        let waiterID = UUID()
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                join(model, waiterID: waiterID, progressHandler: progressHandler, continuation: continuation)
            }
        } onCancel: {
            // Runs after `join`: the actor is busy until this call suspends.
            Task { await self.abandon(waiterID: waiterID, key: key) }
        }
    }

    public func downloadAllModels(
        progressHandler: (@Sendable (ModelInfo, DownloadProgress) -> Void)? = nil
    ) async throws -> [URL] {
        var downloadedURLs: [URL] = []

        for model in Self.allModels {
            let url = try await downloadModel(model) { progress in
                progressHandler?(model, progress)
            }
            downloadedURLs.append(url)
        }

        return downloadedURLs
    }

    /// How many callers are waiting for `model`'s download right now (0 when none is running).
    func waiterCount(for model: ModelInfo) -> Int {
        inFlight[model.filename]?.waiters.count ?? 0
    }

    // MARK: - Shared downloads

    /// One running download and the callers waiting for it. Only touched on the actor.
    private final class InFlightDownload {
        let model: ModelInfo
        let progress = DownloadProgressFanout()
        var waiters: [UUID: CheckedContinuation<URL, Error>] = [:]
        var task: Task<Void, Never>?
        var isCancelled = false

        init(model: ModelInfo) {
            self.model = model
        }
    }

    /// Adds a caller to the model's running download, starting one if there is none.
    private func join(
        _ model: ModelInfo,
        waiterID: UUID,
        progressHandler: (@Sendable (DownloadProgress) -> Void)?,
        continuation: CheckedContinuation<URL, Error>
    ) {
        let download = inFlight[model.filename] ?? start(model)
        download.waiters[waiterID] = continuation
        if let progressHandler {
            download.progress.add(progressHandler, id: waiterID)
        }
    }

    private func start(_ model: ModelInfo) -> InFlightDownload {
        let key = model.filename
        let download = InFlightDownload(model: model)
        let previous = stopping[key]
        let directory = modelsDirectory
        let fetch = fetch
        let progress = download.progress
        download.task = Task {
            // Two downloads of one file never overlap, so neither can disturb the other's files.
            await previous?.value
            let result: Result<URL, Error>
            do {
                result = .success(
                    try await Self.fetchAndInstall(model, into: directory, fetch: fetch, progress: progress)
                )
            } catch {
                result = .failure(error)
            }
            self.finish(download, with: result)
        }
        inFlight[key] = download
        return download
    }

    /// A caller was cancelled. Others keep waiting; when it was the last one, the download is stopped.
    private func abandon(waiterID: UUID, key: String) {
        guard let download = inFlight[key], let continuation = download.waiters[waiterID] else { return }
        download.progress.remove(waiterID)
        if download.waiters.count > 1 {
            download.waiters[waiterID] = nil
            continuation.resume(throwing: CancellationError())
            return
        }
        // Nobody is waiting any more. The last caller hears back from `finish`, once the temporary file is gone;
        // anyone asking from now on gets a fresh download, which waits for this one to stop.
        download.isCancelled = true
        download.task?.cancel()
        inFlight[key] = nil
        stopping[key] = download.task
    }

    private func finish(_ download: InFlightDownload, with result: Result<URL, Error>) {
        let key = download.model.filename
        if inFlight[key] === download {
            inFlight[key] = nil
        }
        if stopping[key] == download.task {
            stopping[key] = nil
        }

        var outcome = result
        if case .failure(let error) = result {
            if download.isCancelled || Self.isCancellation(error) {
                outcome = .failure(CancellationError())
                Log.info("Download of \(download.model.name) cancelled", category: .app)
            } else {
                Log.warning("Download of \(download.model.name) failed: \(error)", category: .app)
            }
        }
        for continuation in download.waiters.values {
            continuation.resume(with: outcome)
        }
        download.waiters.removeAll()
    }

    /// `URLSessionTask.cancel()` surfaces as `URLError(.cancelled)`, not `CancellationError`.
    static func isCancellation(_ error: Error) -> Bool {
        error is CancellationError || (error as? URLError)?.code == .cancelled
    }

    /// The download itself: fetch to a temporary file, check it, move it into place. Runs off the actor (hashing
    /// half a gigabyte takes a second or two, and status checks shouldn't wait for it). Whatever goes wrong,
    /// the temporary file is deleted and the models folder is left as it was.
    private static func fetchAndInstall(
        _ model: ModelInfo,
        into directory: URL,
        fetch: Fetch,
        progress: DownloadProgressFanout
    ) async throws -> URL {
        try Task.checkCancellation()
        let fileManager = FileManager.default
        try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        let destinationURL = directory.appendingPathComponent(model.filename)

        guard let url = URL(string: model.url) else {
            throw DownloadError.downloadFailed("Invalid URL: \(model.url)")
        }

        Log.info("Downloading \(model.name) from \(model.url)", category: .app)

        let downloadedURL: URL
        do {
            downloadedURL = try await fetch(url) { written, expected in
                progress.report(
                    model: model,
                    bytesDownloaded: written,
                    totalBytes: expected > 0 ? expected : model.sizeBytes
                )
            }
        } catch where isCancellation(error) {
            throw CancellationError()
        }

        let fileSize: Int64
        do {
            try Task.checkCancellation()
            fileSize = try verify(downloadedURL, is: model)
            try Task.checkCancellation()
            if fileManager.fileExists(atPath: destinationURL.path) {
                // Only an invalid file can be here (a valid one is returned without downloading)
                Log.warning("Existing model \(model.name) is invalid, replacing it", category: .app)
                try fileManager.removeItem(at: destinationURL)
            }
            try fileManager.moveItem(at: downloadedURL, to: destinationURL)
        } catch {
            try? fileManager.removeItem(at: downloadedURL)
            throw error
        }

        Log.info("Successfully downloaded \(model.name) (\(megabytes(fileSize)) MB)", category: .app)
        progress.finish(model: model, totalBytes: fileSize)
        return destinationURL
    }

    /// Checks a downloaded file before it is moved into place, and returns its size.
    private static func verify(_ fileURL: URL, is model: ModelInfo) throws -> Int64 {
        let attributes = try FileManager.default.attributesOfItem(atPath: fileURL.path)
        guard let fileSize = attributes[.size] as? Int64 else {
            throw DownloadError.fileSystemError("Could not determine file size")
        }

        // Within 10% of expected (models without a pinned checksum may vary slightly)
        let expectedSize = model.sizeBytes
        guard abs(fileSize - expectedSize) < expectedSize / 10 else {
            throw DownloadError.invalidFileSize(expected: expectedSize, actual: fileSize)
        }

        // A pinned checksum is the only thing that says these are the bytes that were tested
        if let expectedHash = model.sha256 {
            let actualHash = try ModelDownload.sha256(of: fileURL)
            guard actualHash == expectedHash else {
                throw DownloadError.downloadFailed("Checksum mismatch for \(model.name)")
            }
        }
        return fileSize
    }

    // MARK: - Model Paths

    public func getWhisperModelPath() async -> URL? {
        let status = await getModelStatus(Self.whisperModel)
        return status.isValid ? status.localPath : nil
    }

    public func getEmbeddingModelPath() async -> URL? {
        let status = await getModelStatus(Self.embeddingModel)
        return status.isValid ? status.localPath : nil
    }

    // MARK: - Cleanup

    public func deleteModel(_ model: ModelInfo) async throws {
        let modelPath = modelsDirectory.appendingPathComponent(model.filename)

        guard FileManager.default.fileExists(atPath: modelPath.path) else {
            return
        }

        try FileManager.default.removeItem(at: modelPath)
        Log.info("Deleted model: \(model.name)", category: .app)
    }

    public func deleteAllModels() async throws {
        for model in Self.allModels {
            try await deleteModel(model)
        }
    }
}

// MARK: - Download progress

/// Passes one download's progress on to everyone waiting for it, a few times a second at most.
/// The network reports progress for every chunk (hundreds of times a second on a fast connection); nobody
/// watching a progress bar needs more than this, and each report costs the listener a hop to its own thread.
final class DownloadProgressFanout: @unchecked Sendable {
    typealias Handler = @Sendable (ModelManager.DownloadProgress) -> Void

    private let minimumInterval: Duration
    private let lock = NSLock()
    private var handlers: [UUID: Handler] = [:]
    private var latest: ModelManager.DownloadProgress?
    private var lastReport: ContinuousClock.Instant?

    init(minimumInterval: Duration = .milliseconds(100)) {
        self.minimumInterval = minimumInterval
    }

    /// Someone who joins a download already under way is told at once how far it is.
    func add(_ handler: @escaping Handler, id: UUID) {
        lock.lock()
        handlers[id] = handler
        let latest = latest
        lock.unlock()
        if let latest {
            handler(latest)
        }
    }

    func remove(_ id: UUID) {
        lock.lock()
        handlers[id] = nil
        lock.unlock()
    }

    func report(model: ModelManager.ModelInfo, bytesDownloaded: Int64, totalBytes: Int64) {
        let update = ModelManager.DownloadProgress(
            model: model,
            bytesDownloaded: bytesDownloaded,
            totalBytes: totalBytes,
            percentage: totalBytes > 0 ? min(1, Double(bytesDownloaded) / Double(totalBytes)) : 0
        )
        let now = ContinuousClock.now
        lock.lock()
        latest = update
        if let lastReport, now - lastReport < minimumInterval {
            lock.unlock()
            return
        }
        lastReport = now
        let handlers = Array(handlers.values)
        lock.unlock()
        handlers.forEach { $0(update) }
    }

    /// The file is in place: everyone hears 100%, whenever the last report was.
    func finish(model: ModelManager.ModelInfo, totalBytes: Int64) {
        let update = ModelManager.DownloadProgress(
            model: model,
            bytesDownloaded: totalBytes,
            totalBytes: totalBytes,
            percentage: 1.0
        )
        lock.lock()
        latest = update
        let handlers = Array(handlers.values)
        lock.unlock()
        handlers.forEach { $0(update) }
    }
}

// MARK: - Download with progress

/// One file download that reports progress, as an async call.
enum ModelDownload {
    /// Downloads `url` to a temporary file and returns where it is. The caller moves or deletes it.
    /// Cancelling the calling task stops the transfer and throws `CancellationError`; nothing is left behind.
    static func run(
        from url: URL,
        progress: @escaping @Sendable (_ bytesWritten: Int64, _ bytesExpected: Int64) -> Void
    ) async throws -> URL {
        try Task.checkCancellation()
        let delegate = Delegate(progress: progress)
        let session = URLSession(configuration: .default, delegate: delegate, delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }
        let task = session.downloadTask(with: url)
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                delegate.start(task, continuation: continuation)
            }
        } onCancel: {
            // May run before `start` (the calling task was cancelled already) or at any time after it
            delegate.cancel()
        }
    }

    /// Lowercase hex SHA-256 of a file, read in pieces.
    static func sha256(of url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while let chunk = try handle.read(upToCount: 8 << 20), !chunk.isEmpty {
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    /// The session calls this on its delegate queue while the caller's task may be cancelled from any thread:
    /// everything mutable is behind `lock`, and the continuation is resumed exactly once.
    private final class Delegate: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
        private let progress: @Sendable (Int64, Int64) -> Void
        private let lock = NSLock()
        private var continuation: CheckedContinuation<URL, Error>?
        private var task: URLSessionDownloadTask?
        private var result: Result<URL, Error>?
        private var isCancelled = false

        init(progress: @escaping @Sendable (Int64, Int64) -> Void) {
            self.progress = progress
        }

        /// Starts the transfer, unless the caller was cancelled before it got this far.
        func start(_ task: URLSessionDownloadTask, continuation: CheckedContinuation<URL, Error>) {
            lock.lock()
            if isCancelled || Task.isCancelled {
                isCancelled = true
                lock.unlock()
                task.cancel()
                continuation.resume(throwing: CancellationError())
                return
            }
            self.continuation = continuation
            self.task = task
            task.resume()
            lock.unlock()
        }

        func cancel() {
            lock.lock()
            isCancelled = true
            let continuation = continuation
            let task = task
            let kept = try? result?.get()
            self.continuation = nil
            self.result = nil
            lock.unlock()

            task?.cancel()
            if let kept {
                try? FileManager.default.removeItem(at: kept)
            }
            continuation?.resume(throwing: CancellationError())
        }

        func urlSession(
            _ session: URLSession,
            downloadTask: URLSessionDownloadTask,
            didWriteData bytesWritten: Int64,
            totalBytesWritten: Int64,
            totalBytesExpectedToWrite: Int64
        ) {
            progress(totalBytesWritten, totalBytesExpectedToWrite)
        }

        func urlSession(
            _ session: URLSession,
            downloadTask: URLSessionDownloadTask,
            didFinishDownloadingTo location: URL
        ) {
            lock.lock()
            defer { lock.unlock() }
            guard !isCancelled else { return }

            if let response = downloadTask.response as? HTTPURLResponse,
               !(200...299).contains(response.statusCode) {
                result = .failure(ModelManager.DownloadError.downloadFailed("HTTP \(response.statusCode)"))
                return
            }
            // The file at `location` is removed when this method returns: keep it under another name.
            let kept = FileManager.default.temporaryDirectory
                .appendingPathComponent("retrace-model-\(UUID().uuidString).download")
            do {
                try FileManager.default.moveItem(at: location, to: kept)
                result = .success(kept)
            } catch {
                result = .failure(error)
            }
        }

        func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
            lock.lock()
            let continuation = continuation
            let result = result
            self.continuation = nil
            self.result = nil
            lock.unlock()

            let kept = try? result?.get()
            let outcome: Result<URL, Error>
            if let error {
                outcome = .failure(ModelManager.isCancellation(error) ? CancellationError() : error)
            } else {
                outcome = result ?? .failure(ModelManager.DownloadError.downloadFailed("No file received"))
            }
            if let kept, continuation == nil || error != nil {
                // Nobody will take the file: the caller was cancelled, or the transfer failed after all
                try? FileManager.default.removeItem(at: kept)
            }
            continuation?.resume(with: outcome)
        }
    }
}
