import XCTest
import CoreGraphics
import Foundation
import ImageIO
import Shared
@testable import Search

/// `SemanticCheckService` (prototype) against a real child process: a script that takes `llama-server`'s command
/// line and answers the two endpoints the service uses (`/props`, `/v1/systemone`) without a model, asking for the
/// key it was started with (`LLAMA_API_KEY`) as the real server does.
///
/// The script takes its orders from the text in the model file (`ok`, `never-ready`, `ready-after-2`, `slow`,
/// `very-slow`, `garbage`, `error`, and `open` for a server that lets any key in)
/// and answers with the confidence written in the window title ("sure 0.91"), so a test decides what the "model"
/// says. It appends its process ID to a `pids` file beside the model, its arguments to `args` and every request to
/// `requests`, which is how the tests know what was started, what it was asked, and that nothing is left running.
///
/// `testLiveModelSeparatesRightFromWrong` runs the real thing and is skipped unless its environment is set.
final class SemanticCheckServiceTests: XCTestCase {
    private var folder: URL!
    private var serverURL: URL!
    private var modelURL: URL!
    private var services: [SemanticCheckService] = []

    override func setUp() async throws {
        guard FileManager.default.isExecutableFile(atPath: "/usr/bin/python3") else {
            throw XCTSkip("the stand-in server needs /usr/bin/python3")
        }
        folder = FileManager.default.temporaryDirectory
            .appendingPathComponent("retrace-check-server-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        serverURL = folder.appendingPathComponent("fake-llama-server")
        modelURL = folder.appendingPathComponent("model")
        try Self.fakeServer.write(to: serverURL, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: serverURL.path)
    }

    override func tearDown() async throws {
        for service in services { await service.stop() }
        services.removeAll()
        if folder != nil {
            for pid in startedProcesses() where Self.isRunning(pid) { kill(pid, SIGKILL) }
            try? FileManager.default.removeItem(at: folder)
        }
    }

    // MARK: - Asking

    func testEachResultIsAskedAboutOnceAndGetsTheModelsConfidence() async throws {
        let service = try makeService(mode: "ok")

        let answers = await service.confidences(
            for: "  someone playing the piano ",
            candidates: [candidate(1, window: "sure 0.91"), candidate(2, window: "sure 0.08")],
            deadline: .seconds(10)
        )

        XCTAssertEqual(answers, [1: 0.91, 2: 0.08])
        let requests = try requests()
        XCTAssertEqual(requests.count, 2)
        XCTAssertEqual(requests.map { $0["path"] as? String }, ["/v1/systemone", "/v1/systemone"])
        XCTAssertEqual(requests.first?["state"] as? String, "App: browser. Window title: sure 0.91.")
        let question = try XCTUnwrap(requests.first?["question"] as? [String: Any])
        XCTAssertEqual(question["type"] as? String, "noul")
        XCTAssertEqual(question["instructions"] as? String, "Does this screenshot show: someone playing the piano?")
        XCTAssertEqual(requests.first?["image"] as? String, "data:image/jpeg;base64,")
        XCTAssertEqual(startedProcesses().count, 1, "one server for the whole search")
    }

    func testTheServerGetsOneSlotAndTheImageBudget() async throws {
        let service = try makeService(mode: "ok")
        _ = await service.confidences(for: "anything", candidates: [candidate(1)], deadline: .seconds(10))

        let arguments = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(contentsOf: folder.appendingPathComponent("args"))) as? [String]
        )
        XCTAssertEqual(value(after: "-m", in: arguments), modelURL.path)
        XCTAssertEqual(value(after: "--mmproj", in: arguments), "/projector")
        XCTAssertEqual(value(after: "--host", in: arguments), "127.0.0.1", "never reachable from another machine")
        XCTAssertEqual(value(after: "-np", in: arguments), "1")
        XCTAssertEqual(value(after: "--image-max-tokens", in: arguments), String(SemanticCheckSettings.imageTokens))
    }

    func testAnEmptySearchOrNoResultsStartsNothing() async throws {
        let service = try makeService(mode: "ok")

        let noText = await service.confidences(for: "   ", candidates: [candidate(1)], deadline: .seconds(5))
        let noResults = await service.confidences(for: "piano", candidates: [], deadline: .seconds(5))

        XCTAssertEqual(noText, [:])
        XCTAssertEqual(noResults, [:])
        XCTAssertTrue(startedProcesses().isEmpty)
    }

    func testAFrameThatCannotBeReadLeavesOnlyThatResultUnchecked() async throws {
        let service = try makeService(mode: "ok") { candidate in
            if candidate.frameID == 2 { throw CocoaError(.fileReadNoSuchFile) }
            return Self.image(width: 64, height: 40)
        }

        let answers = await service.confidences(
            for: "piano",
            candidates: [candidate(1, window: "sure 0.70"), candidate(2, window: "sure 0.90"), candidate(3, window: "sure 0.20")],
            deadline: .seconds(10)
        )

        XCTAssertEqual(answers, [1: 0.70, 3: 0.20])
    }

    func testRepliesThatAreNotAnAnswerLeaveTheResultsUnchecked() async throws {
        for mode in ["garbage", "error"] {
            let service = try makeService(mode: mode)
            let answers = await service.confidences(for: "piano", candidates: [candidate(1)], deadline: .seconds(10))
            XCTAssertEqual(answers, [:], mode)
            await service.stop()
        }
    }

    func testItStopsAskingWhenTheTimeIsUp() async throws {
        let service = try makeService(mode: "slow")   // 0.6 s an answer
        let candidates = (1...6).map { candidate(Int64($0), window: "sure 0.80") }

        let answers = await service.confidences(for: "piano", candidates: candidates, deadline: .seconds(2))

        XCTAssertGreaterThanOrEqual(answers.count, 1)
        XCTAssertLessThan(answers.count, 6, "the last results were not waited for")
        XCTAssertTrue(answers.keys.allSatisfy { $0 <= Int64(answers.count) }, "asked in order, best first")
    }

    func testAnswersAreHandedOverOneAtATimeAsTheyCome() async throws {
        let service = try makeService(mode: "slow")   // 0.6 s an answer
        let candidates = [candidate(1, window: "sure 0.70"), candidate(2, window: "sure 0.10"), candidate(3, window: "sure 0.90")]
        var answers: [SemanticCheckVerdict] = [], arrivals: [Date] = []

        for await verdict in service.verdicts(for: "piano", candidates: candidates, deadline: .seconds(15)) {
            answers.append(verdict)
            arrivals.append(Date())
        }

        XCTAssertEqual(answers, [
            SemanticCheckVerdict(frameID: 1, confidence: 0.70),
            SemanticCheckVerdict(frameID: 2, confidence: 0.10),
            SemanticCheckVerdict(frameID: 3, confidence: 0.90),
        ])
        let first = try XCTUnwrap(arrivals.first), last = try XCTUnwrap(arrivals.last)
        XCTAssertGreaterThan(last.timeIntervalSince(first), 1.0, "the first answer was there long before the last")
    }

    func testTheNextFrameIsReadWhileTheModelAnswers() async throws {
        try "slow".write(to: modelURL, atomically: true, encoding: .utf8)   // 0.6 s an answer
        let service = SemanticCheckService(
            serverURL: serverURL, modelPath: modelURL.path, projectorPath: "/projector",
            image: { _ in
                try await Task.sleep(for: .milliseconds(400))   // a slow frame read
                return Self.image(width: 64, height: 40)
            }
        )
        services.append(service)
        let candidates = (1...4).map { candidate(Int64($0), window: "sure 0.80") }
        var arrivals: [Date] = []

        for await _ in service.verdicts(for: "piano", candidates: candidates, deadline: .seconds(20)) {
            arrivals.append(Date())
        }

        XCTAssertEqual(arrivals.count, 4)
        // One after the other, each answer would come a second after the one before (0.4 s to read, 0.6 s to
        // answer): 3 s from the first to the last. With the read done during the answer before it, 1.8 s.
        let first = try XCTUnwrap(arrivals.first), last = try XCTUnwrap(arrivals.last)
        XCTAssertLessThan(last.timeIntervalSince(first), 2.5)
    }

    func testAFrameThatChangesAfterItWasReadAheadIsNotSent() async throws {
        final class Revisions: @unchecked Sendable {
            private let lock = NSLock()
            private var asked: [Int64: Int] = [:]
            /// Frame 2 is fine before and after it is read (the first two looks), and has been queued for
            /// redaction by the time it would be sent (the third).
            func next(for id: Int64) -> String {
                lock.withLock {
                    asked[id, default: 0] += 1
                    return id == 2 && asked[id]! > 2 ? "being redacted" : "final"
                }
            }
        }
        let revisions = Revisions()
        try "ok".write(to: modelURL, atomically: true, encoding: .utf8)
        let service = SemanticCheckService(
            serverURL: serverURL, modelPath: modelURL.path, projectorPath: "/projector",
            image: { _ in Self.image(width: 64, height: 40) },
            revision: { revisions.next(for: $0) }
        )
        services.append(service)

        let answers = await service.confidences(
            for: "piano",
            candidates: [candidate(1, window: "sure 0.70"), candidate(2, window: "sure 0.90"), candidate(3, window: "sure 0.20")],
            deadline: .seconds(10)
        )

        XCTAssertEqual(answers, [1: 0.70, 3: 0.20])
        XCTAssertEqual(try requests().count, 2, "the changed frame never left the app")
    }

    func testOnlyOneFrameIsReadAheadOfTheModel() async throws {
        final class Reads: @unchecked Sendable {
            private let lock = NSLock()
            private var ids: [Int64] = []
            func note(_ id: Int64) { lock.withLock { ids.append(id) } }
            var all: [Int64] { lock.withLock { ids } }
        }
        let reads = Reads()
        try "slow".write(to: modelURL, atomically: true, encoding: .utf8)   // 0.6 s an answer
        let service = SemanticCheckService(
            serverURL: serverURL, modelPath: modelURL.path, projectorPath: "/projector",
            image: { candidate in
                reads.note(candidate.frameID)
                return Self.image(width: 64, height: 40)
            }
        )
        services.append(service)
        let candidates = (1...8).map { candidate(Int64($0), window: "sure 0.80") }

        for await _ in service.verdicts(for: "piano", candidates: candidates, deadline: .seconds(20)) {
            break   // a new search after the first answer
        }
        try await Task.sleep(for: .seconds(2))

        XCTAssertLessThanOrEqual(reads.all.count, 4, "read: \(reads.all)")
        XCTAssertEqual(Set(reads.all.prefix(2)), [1, 2], "the first two straight away")
    }

    func testWhenNobodyIsListeningAnyMoreTheRestIsNotAsked() async throws {
        let service = try makeService(mode: "slow")   // 0.6 s an answer
        let candidates = (1...6).map { candidate(Int64($0), window: "sure 0.80") }

        for await _ in service.verdicts(for: "piano", candidates: candidates, deadline: .seconds(15)) {
            break   // a new search: the reader leaves after the first answer
        }
        try await Task.sleep(for: .seconds(2))

        XCTAssertLessThan(try requests().count, 4, "the check stopped instead of going through all six")
    }

    func testFiltersAndExcludedWordsAreNotPartOfTheQuestion() async throws {
        let service = try makeService(mode: "ok")

        _ = await service.confidences(
            for: "someone playing the piano -guitar app:Chrome",
            candidates: [candidate(1)],
            deadline: .seconds(10)
        )
        let onlyFilters = await service.confidences(for: "app:Chrome -guitar", candidates: [candidate(2)], deadline: .seconds(10))

        let question = try XCTUnwrap(try requests().first?["question"] as? [String: Any])
        XCTAssertEqual(question["instructions"] as? String, "Does this screenshot show: someone playing the piano?")
        XCTAssertEqual(onlyFilters, [:], "nothing to ask about")
        XCTAssertEqual(try requests().count, 1)
    }

    func testAnAnswerThatTakesTooLongIsNotWaitedFor() async throws {
        let service = try makeService(mode: "very-slow")   // 5 s an answer
        let started = Date()

        let answers = await service.confidences(
            for: "piano",
            candidates: [candidate(1), candidate(2), candidate(3)],
            deadline: .seconds(1)
        )

        XCTAssertEqual(answers, [:])
        XCTAssertLessThan(Date().timeIntervalSince(started), 2.5, "back at the deadline, not after the answer")
        XCTAssertEqual(try requests().count, 1, "the others were not started")
    }

    func testAFrameWhosePixelsAreNotFinalIsNeitherReadNorSent() async throws {
        final class Reads: @unchecked Sendable {
            private let lock = NSLock()
            private var ids: [Int64] = []
            func note(_ id: Int64) { lock.withLock { ids.append(id) } }
            var all: [Int64] { lock.withLock { ids } }
        }
        let reads = Reads()
        try "ok".write(to: modelURL, atomically: true, encoding: .utf8)
        let service = SemanticCheckService(
            serverURL: serverURL, modelPath: modelURL.path, projectorPath: "/projector",
            image: { candidate in
                reads.note(candidate.frameID)
                return Self.image(width: 64, height: 40)
            },
            revision: { frameID in frameID == 2 ? nil : "final" }   // the second is being redacted
        )
        services.append(service)

        let answers = await service.confidences(
            for: "piano",
            candidates: [candidate(1, window: "sure 0.70"), candidate(2, window: "sure 0.90"), candidate(3, window: "sure 0.20")],
            deadline: .seconds(10)
        )

        XCTAssertEqual(answers, [1: 0.70, 3: 0.20])
        XCTAssertEqual(reads.all, [1, 3])
        XCTAssertEqual(try requests().count, 2)
    }

    func testAFrameThatChangesWhileItIsReadIsNotSent() async throws {
        final class Revisions: @unchecked Sendable {
            private let lock = NSLock()
            private var asked: [Int64: Int] = [:]
            /// Frame 2 is rewritten between the look before the read and the look after it.
            func next(for id: Int64) -> String {
                lock.withLock {
                    asked[id, default: 0] += 1
                    return id == 2 ? "revision \(asked[id]!)" : "final"
                }
            }
        }
        let revisions = Revisions()
        try "ok".write(to: modelURL, atomically: true, encoding: .utf8)
        let service = SemanticCheckService(
            serverURL: serverURL, modelPath: modelURL.path, projectorPath: "/projector",
            image: { _ in Self.image(width: 64, height: 40) },
            revision: { revisions.next(for: $0) }
        )
        services.append(service)

        let answers = await service.confidences(
            for: "piano",
            candidates: [candidate(1, window: "sure 0.70"), candidate(2, window: "sure 0.90")],
            deadline: .seconds(10)
        )

        XCTAssertEqual(answers, [1: 0.70])
        XCTAssertEqual(try requests().count, 1, "the changed frame never left the app")
    }

    func testAServerThatLetsAnyKeyInIsNotTrustedWithScreenshots() async throws {
        // Something else on the port (another app, another Retrace with its own model) answers everything.
        let service = try makeService(mode: "open", startTimeout: 1.5)

        let answers = await service.confidences(for: "piano", candidates: [candidate(1)], deadline: .seconds(5))

        XCTAssertEqual(answers, [:])
        XCTAssertFalse(FileManager.default.fileExists(atPath: folder.appendingPathComponent("requests").path))
    }

    func testEachLaunchHasItsOwnKeyAndItIsNotOnTheCommandLine() async throws {
        let service = try makeService(mode: "ok", idleTimeout: 0.5)
        _ = await service.confidences(for: "piano", candidates: [candidate(1)], deadline: .seconds(10))
        try await assertNothingIsLeftRunning()
        _ = await service.confidences(for: "piano", candidates: [candidate(1)], deadline: .seconds(10))

        let keys = try String(contentsOf: folder.appendingPathComponent("keys"), encoding: .utf8).split(separator: "\n")
        XCTAssertEqual(keys.count, 2)
        XCTAssertNotEqual(keys[0], keys[1])
        XCTAssertTrue(keys.allSatisfy { $0.count >= 32 })
        let arguments = try String(contentsOf: folder.appendingPathComponent("args"), encoding: .utf8)
        XCTAssertFalse(arguments.contains(String(keys[1])))
    }

    // MARK: - The server's life

    func testASearchDoesNotWaitPastItsDeadlineForTheModelToLoadButTheNextOneGetsIt() async throws {
        let service = try makeService(mode: "ready-after-2")   // "loads" for two seconds
        let started = Date()

        let first = await service.confidences(for: "piano", candidates: [candidate(1, window: "sure 0.70")], deadline: .milliseconds(500))

        XCTAssertEqual(first, [:])
        XCTAssertLessThan(Date().timeIntervalSince(started), 1.5, "back at the deadline, not when the model is up")

        let second = await service.confidences(for: "piano", candidates: [candidate(1, window: "sure 0.70")], deadline: .seconds(10))

        XCTAssertEqual(second, [1: 0.70])
        XCTAssertEqual(startedProcesses().count, 1, "the start went on in the background; it was not begun again")
    }

    func testAModelThatFinishesLoadingAfterTheSearchGaveUpStillGoesAwayWhenIdle() async throws {
        let service = try makeService(mode: "ready-after-2", idleTimeout: 0.5)
        _ = await service.confidences(for: "piano", candidates: [candidate(1)], deadline: .milliseconds(300))

        try await Task.sleep(for: .milliseconds(1200))
        XCTAssertTrue(startedProcesses().contains(where: Self.isRunning), "still loading, for the next search")

        try await assertNothingIsLeftRunning()   // up at 2 s, unused for half a second, stopped
        XCTAssertEqual(startedProcesses().count, 1)
    }

    func testASearchStuckReadingAFrameDoesNotKeepTheModelLoadedForever() async throws {
        final class Hang: @unchecked Sendable {
            private let lock = NSLock()
            private var seconds = 0.0
            var duration: Double {
                get { lock.withLock { seconds } }
                set { lock.withLock { seconds = newValue } }
            }
        }
        let hang = Hang()
        try "ok".write(to: modelURL, atomically: true, encoding: .utf8)
        let service = SemanticCheckService(
            serverURL: serverURL, modelPath: modelURL.path, projectorPath: "/projector", idleTimeout: 0.6,
            image: { _ in
                let seconds = hang.duration
                if seconds > 0 {
                    // Not interruptible, like a video decoder that does not come back.
                    await withCheckedContinuation { continuation in
                        DispatchQueue.global().asyncAfter(deadline: .now() + seconds) { continuation.resume() }
                    }
                }
                return Self.image(width: 64, height: 40)
            }
        )
        services.append(service)
        _ = await service.confidences(for: "piano", candidates: [candidate(1)], deadline: .seconds(10))   // timer set
        try await Task.sleep(for: .milliseconds(300))
        hang.duration = 5
        let stuck = Task { await service.confidences(for: "piano", candidates: [candidate(2)], deadline: .seconds(1)) }

        // The first timer fires while the stuck search counts as recent use. The server still has to go once
        // that use is 0.6 s old, long before the frame read comes back.
        let deadline = Date().addingTimeInterval(3.5)
        while Date() < deadline, startedProcesses().contains(where: Self.isRunning) {
            try await Task.sleep(for: .milliseconds(100))
        }
        XCTAssertFalse(startedProcesses().contains(where: Self.isRunning), "stopped while the read was still stuck")
        _ = await stuck.value
    }

    func testStoppingWhileTheModelLoadsEndsTheServer() async throws {
        let service = try makeService(mode: "ready-after-2")
        _ = await service.confidences(for: "piano", candidates: [candidate(1)], deadline: .milliseconds(300))
        // The stand-in notes its process id as it starts, which on a busy machine can take longer than the search waited.
        let noted = Date().addingTimeInterval(1.5)
        while startedProcesses().isEmpty, Date() < noted { try await Task.sleep(for: .milliseconds(20)) }
        XCTAssertEqual(startedProcesses().count, 1)

        await service.stop()

        try await assertNothingIsLeftRunning()
        let after = await service.confidences(for: "piano", candidates: [candidate(1, window: "sure 0.70")], deadline: .seconds(10))
        XCTAssertEqual(after, [1: 0.70], "and it starts again for the next search")
        XCTAssertEqual(startedProcesses().count, 2)
    }

    func testAServerThatIsNotThereMeansNoAnswers() async throws {
        let service = try makeService(mode: "ok", server: folder.appendingPathComponent("missing"))

        let answers = await service.confidences(for: "piano", candidates: [candidate(1)], deadline: .seconds(5))

        XCTAssertEqual(answers, [:])
    }

    func testAServerThatNeverGetsReadyIsStoppedAndMeansNoAnswers() async throws {
        let service = try makeService(mode: "never-ready", startTimeout: 1.5)

        let answers = await service.confidences(for: "piano", candidates: [candidate(1)], deadline: .seconds(5))

        XCTAssertEqual(answers, [:])
        XCTAssertEqual(startedProcesses().count, 1)
        try await assertNothingIsLeftRunning()
    }

    func testStoppingEndsTheServer() async throws {
        let service = try makeService(mode: "ok")
        _ = await service.confidences(for: "piano", candidates: [candidate(1)], deadline: .seconds(10))
        let pid = try XCTUnwrap(startedProcesses().first)
        XCTAssertTrue(Self.isRunning(pid))

        await service.stop()

        try await assertNothingIsLeftRunning()
    }

    func testTheServerGoesAwayWhenSearchesStopAndComesBackForTheNextOne() async throws {
        let service = try makeService(mode: "ok", idleTimeout: 0.5)
        let first = await service.confidences(for: "piano", candidates: [candidate(1, window: "sure 0.60")], deadline: .seconds(10))
        XCTAssertEqual(first, [1: 0.60])

        try await assertNothingIsLeftRunning()   // half a second without a search

        let second = await service.confidences(for: "piano", candidates: [candidate(2, window: "sure 0.30")], deadline: .seconds(10))
        XCTAssertEqual(second, [2: 0.30])
        XCTAssertEqual(startedProcesses().count, 2, "started again for the second search")
    }

    func testSearchesCloseTogetherShareOneServer() async throws {
        let service = try makeService(mode: "ok")

        async let first = service.confidences(for: "piano", candidates: [candidate(1, window: "sure 0.60")], deadline: .seconds(10))
        async let second = service.confidences(for: "guitar", candidates: [candidate(2, window: "sure 0.30")], deadline: .seconds(10))
        let answers = await [first, second]

        XCTAssertEqual(answers, [[1: 0.60], [2: 0.30]])
        XCTAssertEqual(startedProcesses().count, 1)
    }

    // MARK: - Pieces

    func testConfidenceIsReadOnlyFromAWellFormedReply() {
        func confidence(_ json: String) -> Double? { SemanticCheckService.confidence(inReply: Data(json.utf8)) }

        XCTAssertEqual(confidence(#"{"answers":{"match":{"noul":0.42}}}"#), 0.42)
        XCTAssertEqual(confidence(#"{"answers":{"match":{"noul":1}}}"#), 1)
        XCTAssertEqual(confidence(#"{"answers":{"match":{"noul":1.7}}}"#), 1, "clamped")
        XCTAssertEqual(confidence(#"{"answers":{"match":{"noul":-0.2}}}"#), 0, "clamped")
        XCTAssertNil(confidence(#"{"answers":{"match":{"noul":"yes"}}}"#))
        XCTAssertNil(confidence(#"{"answers":{"other":{"noul":0.9}}}"#))
        XCTAssertNil(confidence(#"{"error":"model not loaded"}"#))
        XCTAssertNil(confidence("not json"))
    }

    func testAResultWithoutAppOrTitleStillMakesAQuestion() throws {
        let body = SemanticCheckService.requestBody(
            query: "a whiteboard",
            candidate: SemanticCheckCandidate(frameID: 1, appName: "", windowName: nil, videoPath: nil, frameIndex: 0, videoFrameRate: nil),
            jpeg: Data([0xFF, 0xD8])
        )

        XCTAssertEqual(body["state"] as? String, "App: unknown. Window title: none.")
        XCTAssertEqual((body["images"] as? [String])?.count, 1)
    }

    func testLargeScreenshotsAreScaledDownAndSmallOnesAreLeftAlone() throws {
        let large = try XCTUnwrap(SemanticCheckService.jpeg(from: Self.image(width: 3024, height: 1964), longestSide: 1536))
        let small = try XCTUnwrap(SemanticCheckService.jpeg(from: Self.image(width: 800, height: 600), longestSide: 1536))

        XCTAssertEqual(try Self.size(ofJPEG: large).width, 1536)
        XCTAssertEqual(try Self.size(ofJPEG: large).height, 998)
        XCTAssertEqual(try Self.size(ofJPEG: small).width, 800)
        XCTAssertEqual(try Self.size(ofJPEG: small).height, 600)
    }

    // MARK: - The real model

    /// The real server and model on real screenshots. Set
    ///   RETRACE_CHECK_LIVE_SERVER     llama-server (a build that knows the model's decision format)
    ///   RETRACE_CHECK_LIVE_MODEL, RETRACE_CHECK_LIVE_PROJECTOR
    ///   RETRACE_CHECK_LIVE_CASES      a JSON file: [{"query": ..., "right": [image paths], "wrong": [image paths],
    ///                                 "apps": {image path: app name}, "titles": {image path: window title}}]
    /// Every right screenshot has to be kept at the cut-off and to score above every wrong one for the same search.
    func testLiveModelSeparatesRightFromWrong() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let server = environment["RETRACE_CHECK_LIVE_SERVER"],
              let model = environment["RETRACE_CHECK_LIVE_MODEL"],
              let projector = environment["RETRACE_CHECK_LIVE_PROJECTOR"],
              let casesPath = environment["RETRACE_CHECK_LIVE_CASES"] else {
            throw XCTSkip("set RETRACE_CHECK_LIVE_SERVER, _MODEL, _PROJECTOR and _CASES to run")
        }
        let cases = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: casesPath))) as? [[String: Any]]
        )
        final class Paths: @unchecked Sendable {
            private let lock = NSLock()
            private var stored: [Int64: String] = [:]
            func set(_ path: String, for id: Int64) { lock.withLock { stored[id] = path } }
            func path(for id: Int64) -> String? { lock.withLock { stored[id] } }
        }
        let paths = Paths()
        let service = SemanticCheckService(
            serverURL: URL(fileURLWithPath: server),
            modelPath: model,
            projectorPath: projector,
            startTimeout: 60
        ) { candidate in
            guard let path = paths.path(for: candidate.frameID),
                  let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
                  let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
                throw CocoaError(.fileReadCorruptFile)
            }
            return image
        }
        services.append(service)

        var nextID: Int64 = 0
        for (number, entry) in cases.enumerated() {
            let query = try XCTUnwrap(entry["query"] as? String)
            let titles = entry["titles"] as? [String: String] ?? [:]
            let apps = entry["apps"] as? [String: String] ?? [:]
            func candidates(_ key: String) -> [SemanticCheckCandidate] {
                (entry[key] as? [String] ?? []).map { path in
                    nextID += 1
                    paths.set(path, for: nextID)
                    return SemanticCheckCandidate(
                        frameID: nextID, appName: apps[path], windowName: titles[path],
                        videoPath: nil, frameIndex: 0, videoFrameRate: nil
                    )
                }
            }
            let right = candidates("right"), wrong = candidates("wrong")
            let started = ContinuousClock.now
            let answers = await service.confidences(for: query, candidates: right + wrong, deadline: .seconds(120))
            let elapsed = ContinuousClock.now - started
            let rightScores = right.compactMap { answers[$0.frameID] }, wrongScores = wrong.compactMap { answers[$0.frameID] }
            print("[live check] search \(number + 1): \(right.count + wrong.count) screenshots in \(elapsed);"
                + " right \(rightScores.map { String(format: "%.2f", $0) }), wrong \(wrongScores.map { String(format: "%.2f", $0) })")
            XCTAssertEqual(answers.count, right.count + wrong.count, "an answer for every screenshot (search \(number + 1))")
            let lowestRight = try XCTUnwrap(rightScores.min()), highestWrong = wrongScores.max() ?? 0
            XCTAssertGreaterThanOrEqual(lowestRight, SemanticCheckSettings.defaultCutoff, "a right result is kept (search \(number + 1))")
            XCTAssertGreaterThan(lowestRight, highestWrong, "right above wrong (search \(number + 1))")
        }
        await service.stop()
    }

    // MARK: - Helpers

    private func makeService(
        mode: String,
        server: URL? = nil,
        idleTimeout: TimeInterval = 120,
        startTimeout: TimeInterval = 10,
        image: SemanticCheckService.ImageSource? = nil
    ) throws -> SemanticCheckService {
        try mode.write(to: modelURL, atomically: true, encoding: .utf8)
        let service = SemanticCheckService(
            serverURL: server ?? serverURL,
            modelPath: modelURL.path,
            projectorPath: "/projector",
            port: try Self.freePort(),
            idleTimeout: idleTimeout,
            startTimeout: startTimeout,
            image: image ?? { _ in Self.image(width: 64, height: 40) }
        )
        services.append(service)
        return service
    }

    private func candidate(_ id: Int64, window: String = "sure 0.50") -> SemanticCheckCandidate {
        SemanticCheckCandidate(
            frameID: id, appName: "browser", windowName: window, videoPath: "chunks/test", frameIndex: Int(id), videoFrameRate: 2
        )
    }

    private func value(after flag: String, in arguments: [String]) -> String? {
        arguments.firstIndex(of: flag).flatMap { $0 + 1 < arguments.count ? arguments[$0 + 1] : nil }
    }

    private func startedProcesses() -> [pid_t] {
        guard let text = try? String(contentsOf: folder.appendingPathComponent("pids"), encoding: .utf8) else { return [] }
        return text.split(separator: "\n").compactMap { pid_t($0) }
    }

    private func requests() throws -> [[String: Any]] {
        try String(contentsOf: folder.appendingPathComponent("requests"), encoding: .utf8)
            .split(separator: "\n")
            .compactMap { try JSONSerialization.jsonObject(with: Data($0.utf8)) as? [String: Any] }
    }

    private static func isRunning(_ pid: pid_t) -> Bool { kill(pid, 0) == 0 }

    private func assertNothingIsLeftRunning(file: StaticString = #filePath, line: UInt = #line) async throws {
        let deadline = Date().addingTimeInterval(6)
        while Date() < deadline {
            if !startedProcesses().contains(where: Self.isRunning) { return }
            try await Task.sleep(for: .milliseconds(100))
        }
        XCTFail("a server is still running: \(startedProcesses().filter(Self.isRunning))", file: file, line: line)
    }

    private static func image(width: Int, height: Int) -> CGImage {
        let context = CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        )!
        context.setFillColor(CGColor(red: 0.2, green: 0.4, blue: 0.6, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        return context.makeImage()!
    }

    private static func size(ofJPEG data: Data) throws -> (width: Int, height: Int) {
        let source = try XCTUnwrap(CGImageSourceCreateWithData(data as CFData, nil))
        let image = try XCTUnwrap(CGImageSourceCreateImageAtIndex(source, 0, nil))
        return (image.width, image.height)
    }

    /// A port nothing is listening on right now.
    private static func freePort() throws -> Int {
        let descriptor = socket(AF_INET, SOCK_STREAM, 0)
        guard descriptor >= 0 else { throw POSIXError(.EMFILE) }
        defer { close(descriptor) }
        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_addr.s_addr = inet_addr("127.0.0.1")
        address.sin_port = 0
        let bound = withUnsafePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { Darwin.bind(descriptor, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) }
        }
        guard bound == 0 else { throw POSIXError(.EADDRINUSE) }
        var length = socklen_t(MemoryLayout<sockaddr_in>.size)
        let named = withUnsafeMutablePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { getsockname(descriptor, $0, &length) }
        }
        guard named == 0 else { throw POSIXError(.EINVAL) }
        return Int(UInt16(bigEndian: address.sin_port))
    }

    private static let fakeServer = #"""
    #!/usr/bin/python3
    import json, os, re, sys, time
    from http.server import BaseHTTPRequestHandler, HTTPServer

    args = sys.argv[1:]
    model = args[args.index("-m") + 1]
    port = int(args[args.index("--port") + 1])
    folder = os.path.dirname(model)
    mode = open(model).read().strip()
    key = os.environ.get("LLAMA_API_KEY", "")
    began = time.time()
    with open(os.path.join(folder, "keys"), "a") as f: f.write(key + "\n")
    with open(os.path.join(folder, "pids"), "a") as f: f.write(f"{os.getpid()}\n")
    with open(os.path.join(folder, "args"), "w") as f: json.dump(args, f)


    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a): pass

        def reply(self, status, body):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def allowed(self):
            return mode == "open" or self.headers.get("Authorization") == "Bearer " + key

        def do_GET(self):
            loading = mode == "never-ready" or (mode == "ready-after-2" and time.time() - began < 2)
            if self.path != "/props": self.reply(404, {})
            elif loading: self.reply(503, {"error": "loading"})
            elif not self.allowed(): self.reply(401, {"error": "invalid key"})
            else: self.reply(200, {"status": "ok"})

        def do_POST(self):
            if not self.allowed(): return self.reply(401, {"error": "invalid key"})
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with open(os.path.join(folder, "requests"), "a") as f:
                f.write(json.dumps({"path": self.path, "state": body.get("state"), "question": body["questions"]["match"],
                                    "image": body["images"][0][:23]}) + "\n")
            if mode == "slow": time.sleep(0.6)
            if mode == "very-slow": time.sleep(5)
            if mode == "garbage": return self.reply(200, b"not json")
            if mode == "error": return self.reply(500, {})
            sure = re.search(r"sure (\d\.\d+)", body.get("state", ""))
            self.reply(200, {"answers": {"match": {"noul": float(sure.group(1)) if sure else 0.5}}})


    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
    """#
}
