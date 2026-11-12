import CryptoKit
import XCTest
@testable import App

/// `ModelManager.downloadModel` against real files in a temporary folder: what ends up on disk, who gets an
/// answer, and what is left behind when a download is rejected or cancelled. Only the network is stood in for
/// (and two tests run the real `URLSession` step against a local file).
final class ModelManagerDownloadTests: XCTestCase {
    private var root: URL!
    /// The manager's models folder. Not created here: the manager makes it when it first needs it.
    private var modelsDirectory: URL!
    /// Where the stand-in network writes its temporary files, so a leftover is easy to see.
    private var downloadsDirectory: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("ModelManagerDownloadTests_\(UUID().uuidString)", isDirectory: true)
        modelsDirectory = root.appendingPathComponent("models", isDirectory: true)
        downloadsDirectory = root.appendingPathComponent("downloads", isDirectory: true)
        try FileManager.default.createDirectory(at: downloadsDirectory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    // MARK: - One download per model

    func testTwoCallersAtOnceShareOneDownloadAndBothGetTheFile() async throws {
        let payload = Self.payload(seed: 1)
        let model = Self.model(for: payload)
        let network = StubNetwork(payload: payload, directory: downloadsDirectory, isHeld: true)
        let manager = ModelManager(modelsDirectory: modelsDirectory, fetch: network.fetch)
        let firstProgress = ProgressLog()
        let secondProgress = ProgressLog()

        let first = Task { try await manager.downloadModel(model) { firstProgress.append($0.percentage) } }
        try await waitUntil { await manager.waiterCount(for: model) == 1 }
        let second = Task { try await manager.downloadModel(model) { secondProgress.append($0.percentage) } }
        try await waitUntil { await manager.waiterCount(for: model) == 2 }
        network.release()

        let firstURL = try await first.value
        let secondURL = try await second.value

        XCTAssertEqual(network.fetchCount, 1, "the second caller must wait for the first download, not start another")
        XCTAssertEqual(firstURL, secondURL)
        XCTAssertEqual(try Data(contentsOf: firstURL), payload)
        XCTAssertEqual(try contents(of: modelsDirectory), [model.filename])
        XCTAssertEqual(try contents(of: downloadsDirectory), [], "the temporary file must be moved, not copied")
        XCTAssertEqual(firstProgress.values.last, 1.0)
        XCTAssertEqual(secondProgress.values.last, 1.0)
        let waiting = await manager.waiterCount(for: model)
        XCTAssertEqual(waiting, 0)
    }

    func testAValidFileAlreadyInPlaceIsNotDownloadedAgain() async throws {
        let payload = Self.payload(seed: 2)
        let model = Self.model(for: payload)
        try FileManager.default.createDirectory(at: modelsDirectory, withIntermediateDirectories: true)
        try payload.write(to: modelsDirectory.appendingPathComponent(model.filename))
        let network = StubNetwork(payload: payload, directory: downloadsDirectory)
        let manager = ModelManager(modelsDirectory: modelsDirectory, fetch: network.fetch)

        let url = try await manager.downloadModel(model)

        XCTAssertEqual(network.fetchCount, 0)
        XCTAssertEqual(url.lastPathComponent, model.filename)
        XCTAssertEqual(try Data(contentsOf: url), payload)
    }

    func testAnInvalidFileAlreadyInPlaceIsReplaced() async throws {
        let payload = Self.payload(seed: 3)
        let model = Self.model(for: payload)
        try FileManager.default.createDirectory(at: modelsDirectory, withIntermediateDirectories: true)
        let destination = modelsDirectory.appendingPathComponent(model.filename)
        try Data("cut short".utf8).write(to: destination)
        let network = StubNetwork(payload: payload, directory: downloadsDirectory)
        let manager = ModelManager(modelsDirectory: modelsDirectory, fetch: network.fetch)

        let url = try await manager.downloadModel(model)

        XCTAssertEqual(network.fetchCount, 1)
        XCTAssertEqual(try Data(contentsOf: url), payload)
        XCTAssertEqual(try contents(of: downloadsDirectory), [])
    }

    // MARK: - Rejected downloads

    func testAChecksumMismatchThrowsAndLeavesNoFileAnywhere() async throws {
        let expected = Self.payload(seed: 4)
        let tampered = Self.payload(seed: 5)
        XCTAssertEqual(expected.count, tampered.count)
        let model = Self.model(for: expected)
        let network = StubNetwork(payload: tampered, directory: downloadsDirectory)
        let manager = ModelManager(modelsDirectory: modelsDirectory, fetch: network.fetch)

        do {
            _ = try await manager.downloadModel(model)
            XCTFail("a file with the wrong checksum must be rejected")
        } catch let error as ModelManager.DownloadError {
            guard case .downloadFailed(let message) = error else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertTrue(message.contains("Checksum mismatch"), message)
        }

        XCTAssertEqual(try contents(of: modelsDirectory), [])
        XCTAssertEqual(try contents(of: downloadsDirectory), [])
        let status = await manager.getModelStatus(model)
        XCTAssertFalse(status.isDownloaded)
    }

    func testAFileOfTheWrongSizeThrowsAndLeavesNoFileAnywhere() async throws {
        let expected = Self.payload(seed: 6)
        let model = Self.model(for: expected)
        let network = StubNetwork(payload: expected.prefix(expected.count / 2), directory: downloadsDirectory)
        let manager = ModelManager(modelsDirectory: modelsDirectory, fetch: network.fetch)

        do {
            _ = try await manager.downloadModel(model)
            XCTFail("half a file must be rejected")
        } catch let error as ModelManager.DownloadError {
            guard case .invalidFileSize(let wanted, let actual) = error else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertEqual(wanted, Int64(expected.count))
            XCTAssertEqual(actual, Int64(expected.count / 2))
        }

        XCTAssertEqual(try contents(of: modelsDirectory), [])
        XCTAssertEqual(try contents(of: downloadsDirectory), [])
    }

    // MARK: - Cancelling

    func testCancellingThrowsCancellationLeavesNothingAndALaterCallDownloadsAgain() async throws {
        let payload = Self.payload(seed: 7)
        let model = Self.model(for: payload)
        let network = StubNetwork(payload: payload, directory: downloadsDirectory, isHeld: true)
        let manager = ModelManager(modelsDirectory: modelsDirectory, fetch: network.fetch)

        let download = Task { try await manager.downloadModel(model) }
        try await waitUntil { network.fetchCount == 1 }
        download.cancel()

        do {
            _ = try await download.value
            XCTFail("a cancelled download must not return a file")
        } catch {
            XCTAssertTrue(error is CancellationError, "cancelling is not a failure: \(error)")
        }
        XCTAssertEqual(try contents(of: modelsDirectory), [])
        XCTAssertEqual(try contents(of: downloadsDirectory), [])

        network.release()
        let url = try await manager.downloadModel(model)

        XCTAssertEqual(network.fetchCount, 2)
        XCTAssertEqual(try Data(contentsOf: url), payload)
        XCTAssertEqual(try contents(of: downloadsDirectory), [])
    }

    func testAFileThatArrivesAfterCancellingIsDeleted() async throws {
        // The transfer finished just as the caller gave up: the network step still hands over its file
        let payload = Self.payload(seed: 8)
        let model = Self.model(for: payload)
        let network = StubNetwork(
            payload: payload,
            directory: downloadsDirectory,
            isHeld: true,
            stopsWhenCancelled: false
        )
        let manager = ModelManager(modelsDirectory: modelsDirectory, fetch: network.fetch)

        let download = Task { try await manager.downloadModel(model) }
        try await waitUntil { network.fetchCount == 1 }
        download.cancel()
        // Once the manager has seen the cancellation, the transfer "finishes" anyway
        try await waitUntil { await manager.waiterCount(for: model) == 0 }
        network.release()

        do {
            _ = try await download.value
            XCTFail("a cancelled download must not return a file")
        } catch {
            XCTAssertTrue(error is CancellationError, "\(error)")
        }
        XCTAssertEqual(try contents(of: modelsDirectory), [])
        XCTAssertEqual(try contents(of: downloadsDirectory), [])
    }

    func testOneOfTwoCallersCancellingDoesNotStopTheDownloadForTheOther() async throws {
        let payload = Self.payload(seed: 9)
        let model = Self.model(for: payload)
        let network = StubNetwork(payload: payload, directory: downloadsDirectory, isHeld: true)
        let manager = ModelManager(modelsDirectory: modelsDirectory, fetch: network.fetch)

        let staying = Task { try await manager.downloadModel(model) }
        try await waitUntil { await manager.waiterCount(for: model) == 1 }
        let leaving = Task { try await manager.downloadModel(model) }
        try await waitUntil { await manager.waiterCount(for: model) == 2 }

        leaving.cancel()
        do {
            _ = try await leaving.value
            XCTFail("the cancelled caller must not get a file")
        } catch {
            XCTAssertTrue(error is CancellationError, "\(error)")
        }
        let waiting = await manager.waiterCount(for: model)
        XCTAssertEqual(waiting, 1)

        network.release()
        let url = try await staying.value

        XCTAssertEqual(network.fetchCount, 1)
        XCTAssertEqual(try Data(contentsOf: url), payload)
    }

    func testACancelledTransferIsReportedAsCancellationNotAsANetworkError() async throws {
        // URLSessionTask.cancel() ends a transfer with URLError(.cancelled)
        let payload = Self.payload(seed: 10)
        let model = Self.model(for: payload)
        let network = StubNetwork(payload: payload, directory: downloadsDirectory, failure: URLError(.cancelled))
        let manager = ModelManager(modelsDirectory: modelsDirectory, fetch: network.fetch)

        do {
            _ = try await manager.downloadModel(model)
            XCTFail("expected the download to stop")
        } catch {
            XCTAssertTrue(error is CancellationError, "\(error)")
        }
    }

    func testANetworkFailureIsPassedOnAndTheNextCallTriesAgain() async throws {
        let payload = Self.payload(seed: 11)
        let model = Self.model(for: payload)
        let offline = StubNetwork(
            payload: payload,
            directory: downloadsDirectory,
            failure: URLError(.notConnectedToInternet)
        )
        let manager = ModelManager(modelsDirectory: modelsDirectory, fetch: offline.fetch)

        for attempt in 1...2 {
            do {
                _ = try await manager.downloadModel(model)
                XCTFail("expected the download to fail")
            } catch {
                XCTAssertEqual((error as? URLError)?.code, .notConnectedToInternet)
            }
            XCTAssertEqual(offline.fetchCount, attempt)
        }
    }

    // MARK: - Progress

    func testProgressIsPassedOnAFewTimesNotOncePerChunkAndEndsAtOneHundredPercent() async throws {
        let payload = Self.payload(seed: 12)
        let model = Self.model(for: payload)
        let network = StubNetwork(payload: payload, directory: downloadsDirectory, progressReports: 5_000)
        let manager = ModelManager(modelsDirectory: modelsDirectory, fetch: network.fetch)
        let progress = ProgressLog()

        _ = try await manager.downloadModel(model) { progress.append($0.percentage) }

        let values = progress.values
        XCTAssertEqual(values.last, 1.0)
        XCTAssertEqual(values, values.sorted(), "progress must not go backwards")
        XCTAssertGreaterThan(values.count, 1)
        // 5,000 reports arrive within a few milliseconds; at most one per 100 ms is passed on, plus the last
        XCTAssertLessThan(values.count, 50)
    }

    // MARK: - The real network step, against a local file

    func testTheURLSessionStepDownloadsAFileAndHandsOverACopy() async throws {
        let payload = Self.payload(seed: 13)
        let source = root.appendingPathComponent("served.bin")
        try payload.write(to: source)

        let downloaded = try await ModelDownload.run(from: source) { _, _ in }
        defer { try? FileManager.default.removeItem(at: downloaded) }

        XCTAssertNotEqual(downloaded.path, source.path)
        XCTAssertEqual(try Data(contentsOf: downloaded), payload)
        XCTAssertEqual(try ModelDownload.sha256(of: downloaded), Self.sha256(payload))
    }

    func testADownloadThroughTheDefaultNetworkStepIsVerifiedAndInstalled() async throws {
        let payload = Self.payload(seed: 14)
        let source = root.appendingPathComponent("served-model.bin")
        try payload.write(to: source)
        let model = ModelManager.ModelInfo(
            name: "Local file",
            filename: "local-model.bin",
            url: source.absoluteString,
            sizeBytes: Int64(payload.count),
            purpose: "Test",
            sha256: Self.sha256(payload)
        )
        let manager = ModelManager(modelsDirectory: modelsDirectory)

        let url = try await manager.downloadModel(model)

        XCTAssertEqual(try Data(contentsOf: url), payload)
        let status = await manager.getModelStatus(model)
        XCTAssertTrue(status.isValid)
    }

    func testTheURLSessionStepDoesNotStartForACallerAlreadyCancelled() async throws {
        let payload = Self.payload(seed: 15)
        let source = root.appendingPathComponent("served-late.bin")
        try payload.write(to: source)
        let progress = ProgressLog()

        let download = Task { () -> URL in
            // Cancelled before the transfer is asked for
            withUnsafeCurrentTask { $0?.cancel() }
            return try await ModelDownload.run(from: source) { written, _ in progress.append(Double(written)) }
        }

        do {
            let url = try await download.value
            try? FileManager.default.removeItem(at: url)
            XCTFail("a cancelled caller must not download anything")
        } catch {
            XCTAssertTrue(error is CancellationError, "\(error)")
        }
        XCTAssertEqual(progress.values, [])
    }

    // MARK: - Sizes shown to the user

    func testModelSizesAreDecimalMegabytesLikeFinder() {
        // 309,855,456 and 554,821,024 bytes: Finder shows 309.9 MB and 554.8 MB
        XCTAssertEqual(ModelManager.embeddingModel.sizeMB, 310)
        XCTAssertEqual(ModelManager.imageEncoderModel.sizeMB, 555)
    }

    // MARK: - Helpers

    /// 64 kB of bytes that differ from seed to seed.
    private static func payload(seed: UInt8) -> Data {
        Data((0..<64_000).map { UInt8(truncatingIfNeeded: $0 &* 31 &+ Int(seed) &* 7) })
    }

    private static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// A model whose pinned size and checksum are those of `payload`.
    private static func model(for payload: Data) -> ModelManager.ModelInfo {
        ModelManager.ModelInfo(
            name: "Test model",
            filename: "test-model.gguf",
            url: "https://models.invalid/test-model.gguf",
            sizeBytes: Int64(payload.count),
            purpose: "Test",
            sha256: sha256(payload)
        )
    }

    /// File names in a folder; empty when the folder doesn't exist.
    private func contents(of directory: URL) throws -> [String] {
        guard FileManager.default.fileExists(atPath: directory.path) else { return [] }
        return try FileManager.default.contentsOfDirectory(atPath: directory.path).sorted()
    }

    private func waitUntil(
        timeout: Duration = .seconds(10),
        _ condition: @escaping () async -> Bool,
        file: StaticString = #filePath,
        line: UInt = #line
    ) async throws {
        let deadline = ContinuousClock.now + timeout
        while !(await condition()) {
            guard ContinuousClock.now < deadline else {
                XCTFail("timed out waiting", file: file, line: line)
                throw CancellationError()
            }
            try await Task.sleep(for: .milliseconds(2))
        }
    }
}

// MARK: - Stand-ins

/// Stands in for the network: writes its payload to a temporary file and returns it, as the `URLSession` step
/// does. While held, a transfer has started but not finished, which is when callers join or cancel.
private final class StubNetwork: @unchecked Sendable {
    private let payload: Data
    private let directory: URL
    private let progressReports: Int
    private let failure: Error?
    private let stopsWhenCancelled: Bool
    private let lock = NSLock()
    private var isHeld: Bool
    private var started = 0

    init(
        payload: Data,
        directory: URL,
        isHeld: Bool = false,
        stopsWhenCancelled: Bool = true,
        progressReports: Int = 4,
        failure: Error? = nil
    ) {
        self.payload = payload
        self.directory = directory
        self.isHeld = isHeld
        self.stopsWhenCancelled = stopsWhenCancelled
        self.progressReports = progressReports
        self.failure = failure
    }

    var fetchCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return started
    }

    /// Lets held transfers finish.
    func release() {
        lock.lock()
        isHeld = false
        lock.unlock()
    }

    var fetch: ModelManager.Fetch {
        { [self] _, progress in
            noteStarted()
            while isStillHeld {
                if stopsWhenCancelled {
                    try Task.checkCancellation()
                }
                try? await Task.sleep(for: .milliseconds(2))
            }
            if stopsWhenCancelled {
                try Task.checkCancellation()
            }
            if let failure {
                throw failure
            }

            let total = Int64(payload.count)
            for report in 1...progressReports {
                progress(total * Int64(report) / Int64(progressReports), total)
            }
            let file = directory.appendingPathComponent("\(UUID().uuidString).download")
            try payload.write(to: file)
            return file
        }
    }

    private func noteStarted() {
        lock.lock()
        started += 1
        lock.unlock()
    }

    private var isStillHeld: Bool {
        lock.lock()
        defer { lock.unlock() }
        return isHeld
    }
}

private final class ProgressLog: @unchecked Sendable {
    private let lock = NSLock()
    private var recorded: [Double] = []

    func append(_ value: Double) {
        lock.lock()
        recorded.append(value)
        lock.unlock()
    }

    var values: [Double] {
        lock.lock()
        defer { lock.unlock() }
        return recorded
    }
}
