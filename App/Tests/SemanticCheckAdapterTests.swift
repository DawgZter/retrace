import XCTest
import CoreGraphics
import Foundation
import Shared
import Database
import Search
import Storage
@testable import App

/// The result check (prototype) as the app runs it: `DataAdapter.search` over a real SQLite database with fixed
/// semantic matches, then `semanticCheckEvents` for what it returned, with a checker that returns the confidences
/// it was given.
final class SemanticCheckAdapterTests: XCTestCase {
    private struct StubImageExtractor: ImageExtractor {
        struct NotImplemented: Error {}

        func extractFrame(videoPath: String, frameIndex: Int, frameRate: Double?) async throws -> Data {
            throw NotImplemented()
        }

        func extractFrameCGImage(videoPath: String, frameIndex: Int, frameRate: Double?) async throws -> CGImage {
            throw NotImplemented()
        }
    }

    private final class FixedSemanticSearch: SemanticSearchProtocol, @unchecked Sendable {
        private let lock = NSLock()
        private var fixedMatches: [SemanticMatch] = []

        var matches: [SemanticMatch] {
            get { lock.withLock { fixedMatches } }
            set { lock.withLock { fixedMatches = newValue } }
        }

        func matches(for query: String, limit: Int) async -> [SemanticMatch] {
            lock.withLock { Array(fixedMatches.prefix(limit)) }
        }
    }

    /// Answers with the confidences it was given and records what it was asked.
    private final class FixedChecker: SemanticResultChecking, @unchecked Sendable {
        private let lock = NSLock()
        private var fixed: [Int64: Double] = [:]
        private var asked: [[SemanticCheckCandidate]] = []
        private var askedQueries: [String] = []
        private var answerDelay: Duration = .zero

        /// How long it takes to answer, whatever deadline it is given (a frame that hangs while being read).
        var delay: Duration {
            get { lock.withLock { answerDelay } }
            set { lock.withLock { answerDelay = newValue } }
        }

        var confidences: [Int64: Double] {
            get { lock.withLock { fixed } }
            set { lock.withLock { fixed = newValue } }
        }
        var requests: [[SemanticCheckCandidate]] { lock.withLock { asked } }
        var queries: [String] { lock.withLock { askedQueries } }

        func confidences(
            for query: String,
            candidates: [SemanticCheckCandidate],
            deadline: Duration
        ) async -> [Int64: Double] {
            let wait = delay
            if wait > .zero { try? await Task.sleep(for: wait) }
            return lock.withLock {
                asked.append(candidates)
                askedQueries.append(query)
                return fixed.filter { entry in candidates.contains { $0.frameID == entry.key } }
            }
        }
    }

    private var database: DatabaseManager!
    private var adapter: DataAdapter!
    private var semantic: FixedSemanticSearch!
    private var checker: FixedChecker!
    private var videoID: Int64 = 0
    private var frameCount = 0
    private let start = Date(timeIntervalSince1970: 1_790_000_000)
    private let enabled = OSAllocatedUnfairLockBox(true)

    /// A switch the test can flip while the adapter reads it from another context.
    private final class OSAllocatedUnfairLockBox: @unchecked Sendable {
        private let lock = NSLock()
        private var stored: Bool
        init(_ value: Bool) { stored = value }
        var value: Bool {
            get { lock.withLock { stored } }
            set { lock.withLock { stored = newValue } }
        }
    }

    override func setUp() async throws {
        database = DatabaseManager(databasePath: "file:semantic_check_\(UUID().uuidString)?mode=memory&cache=private")
        try await database.initialize()
        guard let pointer = await database.getConnection() else {
            throw XCTSkip("no database connection")
        }
        let connection = SQLiteConnection(db: pointer)
        adapter = DataAdapter(
            retraceConnection: connection,
            retraceReadConnectionPool: SQLiteReadConnectionPool(label: "test_semantic_check", sharedConnection: connection),
            retraceConfig: .retrace(storageRoot: "/tmp"),
            retraceImageExtractor: StubImageExtractor(),
            database: database
        )
        try await adapter.initialize()
        semantic = FixedSemanticSearch()
        checker = FixedChecker()
        await adapter.setSemanticSearch(semantic)
        let enabled = self.enabled
        await adapter.setSemanticChecker(checker, isEnabled: { enabled.value }, cutoff: { 0.4 })

        videoID = try await database.insertVideoSegment(
            VideoSegment(
                id: VideoSegmentID(value: 0),
                startTime: start,
                endTime: start.addingTimeInterval(300),
                frameCount: 150,
                fileSizeBytes: 1024,
                relativePath: "chunks/test",
                width: 1920,
                height: 1080,
                source: .native
            )
        )
        try await database.markVideoFinalized(id: videoID, frameCount: 150, fileSize: 1024)
        frameCount = 0
    }

    override func tearDown() async throws {
        await adapter.shutdown()
        try await database.close()
        adapter = nil
        database = nil
    }

    /// A frame in its own app segment, with indexed OCR text, an hour after the previous one.
    private func insertFrame(text: String, app: String = "com.test.browser", window: String? = nil) async throws -> Int64 {
        frameCount += 1
        let timestamp = start.addingTimeInterval(Double(frameCount) * 3600)
        let segmentID = try await database.insertSegment(
            bundleID: app,
            startDate: timestamp,
            endDate: timestamp.addingTimeInterval(60),
            windowName: window ?? "Window \(frameCount)",
            browserUrl: nil,
            type: 0
        )
        let frameID = try await database.insertFrame(
            FrameReference(
                id: FrameID(value: 0),
                timestamp: timestamp,
                segmentID: AppSegmentID(value: segmentID),
                videoID: VideoSegmentID(value: videoID),
                frameIndexInSegment: frameCount,
                metadata: .empty,
                source: .native
            )
        )
        _ = try await database.indexFrameText(
            mainText: text,
            chromeText: nil,
            windowTitle: window,
            segmentId: segmentID,
            frameId: frameID
        )
        try await database.updateFrameProcessingStatus(frameID: frameID, status: 2)   // OCR finished: pixels final
        return frameID
    }

    /// One search, then the check's events for what it returned.
    private func searchAndCheck(
        _ text: String,
        mode: SearchMode = .relevant,
        limit: Int = 50
    ) async throws -> (results: [Int64], events: [SemanticCheckEvent]) {
        let results = try await adapter.search(query: SearchQuery(text: text, limit: limit, mode: mode))
        var events: [SemanticCheckEvent] = []
        for await event in adapter.semanticCheckEvents(for: results) { events.append(event) }
        return (results.results.map(\.id.value), events)
    }

    /// Strong matches (cosine above the weak-match cap), best first in the order given.
    private func setMatches(_ frameIDs: [Int64]) {
        semantic.matches = frameIDs.enumerated().map { offset, frameID in
            SemanticMatch(frameID: frameID, score: 9 - Double(offset) * 0.1, cosine: 0.8)
        }
    }

    func testSearchReturnsTheEmbeddingsOrderAndTheCheckThenSaysWhichResultsToDrop() async throws {
        let homePage = try await insertFrame(text: "Recommended for you trending subscriptions", window: "YouTube")
        let pianoCover = try await insertFrame(text: "Moonlight Sonata piano cover", window: "Moonlight Sonata - piano cover - YouTube")
        let concert = try await insertFrame(text: "Live at the hall grand piano", window: "Recital - YouTube")
        setMatches([homePage, pianoCover, concert])
        checker.confidences = [homePage: 0.08, pianoCover: 0.62, concert: 0.91]

        let results = try await adapter.search(
            query: SearchQuery(text: "someone performing music on a keyboard instrument", limit: 50, mode: .relevant)
        )

        XCTAssertEqual(results.results.map(\.id.value), [homePage, pianoCover, concert], "search itself changes nothing")
        XCTAssertTrue(checker.requests.isEmpty, "and does not ask the model")

        var events: [SemanticCheckEvent] = []
        for await event in adapter.semanticCheckEvents(for: results) { events.append(event) }

        XCTAssertEqual(events, [
            .started(asked: 3),
            .decided(frameID: homePage, confidence: 0.08, isKept: false),
            .decided(frameID: pianoCover, confidence: 0.62, isKept: true),
            .decided(frameID: concert, confidence: 0.91, isKept: true),
        ])
        XCTAssertEqual(checker.queries, ["someone performing music on a keyboard instrument"])
        let asked = try XCTUnwrap(checker.requests.first)
        XCTAssertEqual(asked.map(\.frameID), [homePage, pianoCover, concert], "asked in the embeddings' order")
        XCTAssertEqual(asked.first?.windowName, "YouTube")
        XCTAssertEqual(asked.first?.appName, "browser", "the last part of the bundle id")
        XCTAssertEqual(asked.first?.videoPath, "chunks/test")
    }

    func testSearchDoesNotWaitForTheCheck() async throws {
        let first = try await insertFrame(text: "Quarterly planning notes")
        setMatches([first])
        checker.confidences = [first: 0.01]
        checker.delay = .seconds(5)
        let started = Date()

        let results = try await adapter.search(query: SearchQuery(text: "meeting preparation", limit: 50, mode: .relevant))

        XCTAssertLessThan(Date().timeIntervalSince(started), 2)
        XCTAssertEqual(results.results.map(\.id.value), [first])
    }

    func testACheckWithNoAnswersDecidesNothing() async throws {
        let first = try await insertFrame(text: "Quarterly planning notes")
        let second = try await insertFrame(text: "Budget review agenda")
        setMatches([first, second])
        checker.confidences = [:]   // the model could not be started

        let (results, events) = try await searchAndCheck("meeting preparation")

        XCTAssertEqual(results, [first, second])
        XCTAssertEqual(events, [.started(asked: 2)])
    }

    func testSwitchedOffTheModelIsNotAsked() async throws {
        let first = try await insertFrame(text: "Quarterly planning notes")
        setMatches([first])
        checker.confidences = [first: 0.01]
        enabled.value = false

        let (results, events) = try await searchAndCheck("meeting preparation")

        XCTAssertEqual(results, [first])
        XCTAssertEqual(events, [])
        XCTAssertTrue(checker.requests.isEmpty)
    }

    func testFramesThatContainTheWordsAreNotSentToTheModel() async throws {
        let withWords = try await insertFrame(text: "Invoice 4471 payment overdue reminder")
        let byMeaning = try await insertFrame(text: "Outstanding balance on your account")
        setMatches([withWords, byMeaning])
        checker.confidences = [withWords: 0.01, byMeaning: 0.9]

        let (results, events) = try await searchAndCheck("invoice")

        XCTAssertEqual(Set(results), [withWords, byMeaning])
        XCTAssertEqual(checker.requests.first?.map(\.frameID), [byMeaning])
        XCTAssertEqual(events, [.started(asked: 1), .decided(frameID: byMeaning, confidence: 0.9, isKept: true)])
    }

    func testWhenTheSearchAsksForAPictureAFrameThatContainsTheWordsIsTheModelsToRejectToo() async throws {
        let withWords = try await insertFrame(text: "Photo of a leather jacket: a buying guide and care tips")
        let byMeaning = try await insertFrame(text: "Black biker coat hanging on a rail")
        setMatches([withWords, byMeaning])
        checker.confidences = [withWords: 0.05, byMeaning: 0.9]

        let (results, events) = try await searchAndCheck("photo of a leather jacket")

        XCTAssertEqual(Set(results), [withWords, byMeaning])
        XCTAssertEqual(Set(checker.requests.first?.map(\.frameID) ?? []), [withWords, byMeaning])
        XCTAssertTrue(events.contains(.decided(frameID: withWords, confidence: 0.05, isKept: false)), "\(events)")
        XCTAssertTrue(events.contains(.decided(frameID: byMeaning, confidence: 0.9, isKept: true)), "\(events)")
    }

    func testAFrameWithTheWordsBelowThePageIsNotTheModelsToReject() async throws {
        let first = try await insertFrame(text: "Refund issued for your laptop order")
        let belowThePage = try await insertFrame(text: "Refund policy link in the page footer")
        let byMeaning = try await insertFrame(text: "We have returned your money to the original card")
        let unrelated = try await insertFrame(text: "Garden tomato seedlings")
        setMatches([belowThePage, unrelated, byMeaning])
        checker.confidences = [belowThePage: 0.01, unrelated: 0.02, byMeaning: 0.9]

        // A page of one keyword result: the other frame with the word comes back through meaning.
        let (results, events) = try await searchAndCheck("refund", limit: 1)

        XCTAssertEqual(Set(results), [first, belowThePage, byMeaning, unrelated])
        XCTAssertEqual(checker.requests.first?.map(\.frameID), [unrelated, byMeaning], "the frame with the word is not asked about")
        XCTAssertEqual(events, [
            .started(asked: 2),
            .decided(frameID: unrelated, confidence: 0.02, isKept: false),
            .decided(frameID: byMeaning, confidence: 0.9, isKept: true),
        ])
    }

    func testACheckThatHangsIsGivenUpOnAfterItsTimeLimit() async throws {
        let first = try await insertFrame(text: "Quarterly planning notes")
        setMatches([first])
        checker.confidences = [first: 0.01]
        checker.delay = .seconds(8)
        let enabled = self.enabled
        await adapter.setSemanticChecker(checker, isEnabled: { enabled.value }, cutoff: { 0.4 }, deadline: .milliseconds(300))
        let started = Date()

        let (results, events) = try await searchAndCheck("meeting preparation")

        XCTAssertLessThan(Date().timeIntervalSince(started), 4, "the deadline and its grace, not the eight seconds")
        XCTAssertEqual(results, [first])
        XCTAssertEqual(events, [.started(asked: 1)], "nothing was decided, so the result stays")
    }

    func testAScreenshotStillBeingRedactedIsNotShownToTheModel() async throws {
        let final = try await insertFrame(text: "Sprint retrospective notes")
        let beingRedacted = try await insertFrame(text: "Team offsite agenda")
        let unrelated = try await insertFrame(text: "Garden tomato seedlings")
        try await database.updateFrameProcessingStatus(frameID: beingRedacted, status: 5, rewritePurpose: "redaction")
        setMatches([beingRedacted, final, unrelated])
        checker.confidences = [beingRedacted: 0.01, final: 0.9, unrelated: 0.02]

        let (results, events) = try await searchAndCheck("meeting preparation")

        XCTAssertEqual(results, [beingRedacted, final, unrelated])
        XCTAssertEqual(checker.requests.first?.map(\.frameID), [final, unrelated], "its pixels are not final yet")
        XCTAssertEqual(events, [
            .started(asked: 2),
            .decided(frameID: final, confidence: 0.9, isKept: true),
            .decided(frameID: unrelated, confidence: 0.02, isKept: false),
        ])
    }

    func testWithNoFinalScreenshotAmongTheResultsTheModelIsNotAsked() async throws {
        let first = try await insertFrame(text: "Quarterly planning notes")
        try await database.updateFrameProcessingStatus(frameID: first, status: 8, rewritePurpose: "redaction")
        setMatches([first])
        checker.confidences = [first: 0.01]

        let (results, events) = try await searchAndCheck("meeting preparation")

        XCTAssertEqual(results, [first])
        XCTAssertEqual(events, [])
        XCTAssertTrue(checker.requests.isEmpty)
    }

    func testEveryResultSearchReturnsIsLookedAt() async throws {
        var frames: [Int64] = []
        for index in 0..<34 {
            frames.append(try await insertFrame(text: "Note number \(index) about gardening"))
        }
        setMatches(frames)
        checker.confidences = Dictionary(uniqueKeysWithValues: frames.map { ($0, 0.9) })

        let (results, events) = try await searchAndCheck("growing vegetables at home")

        // Search returns at most thirty results found by meaning alone, and the check looks at all of them.
        XCTAssertEqual(SemanticCheckSettings.depth, 30)
        XCTAssertEqual(results, Array(frames.prefix(30)))
        XCTAssertEqual(checker.requests.first?.map(\.frameID), Array(frames.prefix(30)))
        XCTAssertEqual(events.first, .started(asked: 30))
        XCTAssertEqual(events.count, 31)
    }

    func testTheCutoffDecidesWhatIsKept() async throws {
        let first = try await insertFrame(text: "Sprint retrospective notes")
        let second = try await insertFrame(text: "Team offsite agenda")
        setMatches([first, second])
        checker.confidences = [first: 0.4, second: 0.39]

        let (_, events) = try await searchAndCheck("meeting preparation")

        XCTAssertEqual(events, [
            .started(asked: 2),
            .decided(frameID: first, confidence: 0.4, isKept: true),
            .decided(frameID: second, confidence: 0.39, isKept: false),
        ])
    }
}

/// The result check through the real search path: a copy of a real database, its real video files, the real
/// embedding helper and the real decision model. Skipped unless its environment is set:
///
///   RETRACE_SEMANTIC_LIVE_DB        a copy of retrace.db that already holds embeddings (it is opened read-write)
///   RETRACE_SEMANTIC_LIVE_MODEL     embeddinggemma-2-Q8_0.gguf
///   RETRACE_EMBEDDING_HELPER        the built RetraceEmbeddingHelper
///   RETRACE_SEMANTIC_LIVE_STORAGE   the storage root whose chunks/ the database's videos are in (only read)
///   RETRACE_CHECK_LIVE_SERVER, RETRACE_CHECK_LIVE_MODEL, RETRACE_CHECK_LIVE_PROJECTOR
///   RETRACE_SEMANTIC_LIVE_QUERIES   searches, separated by |
final class SemanticCheckLiveTests: XCTestCase {
    private struct StubImageExtractor: ImageExtractor {
        struct NotImplemented: Error {}
        func extractFrame(videoPath: String, frameIndex: Int, frameRate: Double?) async throws -> Data { throw NotImplemented() }
        func extractFrameCGImage(videoPath: String, frameIndex: Int, frameRate: Double?) async throws -> CGImage { throw NotImplemented() }
    }

    func testSearchACopyOfARealDatabaseWithAndWithoutTheCheck() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let databasePath = environment["RETRACE_SEMANTIC_LIVE_DB"],
              let modelPath = environment["RETRACE_SEMANTIC_LIVE_MODEL"],
              let helperPath = environment["RETRACE_EMBEDDING_HELPER"],
              let storageRoot = environment["RETRACE_SEMANTIC_LIVE_STORAGE"],
              let server = environment["RETRACE_CHECK_LIVE_SERVER"],
              let checkModel = environment["RETRACE_CHECK_LIVE_MODEL"],
              let projector = environment["RETRACE_CHECK_LIVE_PROJECTOR"] else {
            throw XCTSkip("set the RETRACE_SEMANTIC_LIVE_* and RETRACE_CHECK_LIVE_* variables to run")
        }
        let queries = (environment["RETRACE_SEMANTIC_LIVE_QUERIES"] ?? "")
            .split(separator: "|").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
        try XCTSkipIf(queries.isEmpty, "set RETRACE_SEMANTIC_LIVE_QUERIES")

        let database = DatabaseManager(databasePath: databasePath)
        try await database.initialize()
        let maybePointer = await database.getConnection()
        let connection = SQLiteConnection(db: try XCTUnwrap(maybePointer))
        let adapter = DataAdapter(
            retraceConnection: connection,
            retraceReadConnectionPool: SQLiteReadConnectionPool(label: "live_semantic_check", sharedConnection: connection),
            retraceConfig: .retrace(storageRoot: storageRoot),
            retraceImageExtractor: StubImageExtractor(),
            database: database
        )
        try await adapter.initialize()
        let embedder = EmbeddingHelperClient(helperURL: URL(fileURLWithPath: helperPath), modelPath: modelPath)
        let service = SemanticSearchService(store: database, embedder: embedder, isEnabled: { true })
        await adapter.setSemanticSearch(service)
        let checker = SemanticCheckService.readingRetraceStorage(
            storageRoot: storageRoot,
            database: database,
            serverURL: URL(fileURLWithPath: server),
            modelPath: checkModel,
            projectorPath: projector
        )
        await adapter.setSemanticChecker(checker, isEnabled: { true }, cutoff: { SemanticCheckSettings.defaultCutoff })

        // The first search starts the embedding helper and loads the vectors; it is not part of what is measured.
        for _ in 0..<20 {
            let warm = try await adapter.search(query: SearchQuery(text: queries[0], limit: 50, mode: .relevant))
            if warm.results.contains(where: { $0.semanticScore != nil }) { break }
            try await Task.sleep(for: .seconds(1))
        }

        var asked = 0, answered = 0, rejected = 0
        for query in queries {
            let started = Date()
            let results = try await adapter.search(query: SearchQuery(text: query, limit: 50, mode: .relevant))
            let searchSeconds = Date().timeIntervalSince(started)
            var turnedDown = Set<Int64>(), confidence: [Int64: Double] = [:], firstAnswerSeconds: Double?
            for await event in adapter.semanticCheckEvents(for: results) {
                switch event {
                case .started(let count):
                    asked += count
                case .decided(let frameID, let value, let isKept):
                    if firstAnswerSeconds == nil { firstAnswerSeconds = Date().timeIntervalSince(started) }
                    answered += 1
                    confidence[frameID] = value
                    if !isKept { turnedDown.insert(frameID); rejected += 1 }
                }
            }
            let before = results.results.filter(\.isSemanticOnly)
            let after = before.filter { !turnedDown.contains($0.id.value) }
            print("[live check] \"\(query)\": results in \(String(format: "%.1f", searchSeconds)) s, by meaning alone \(before.count) -> \(after.count)"
                + ", first answer after \(firstAnswerSeconds.map { String(format: "%.1f", $0) } ?? "-") s"
                + String(format: ", all answers after %.1f s", Date().timeIntervalSince(started)))
            for result in after.prefix(6) {
                print("[live check]     kept     \(confidence[result.id.value].map { String(format: "%.2f", $0) } ?? " -- ")  \(result.metadata.windowName?.prefix(70) ?? "")")
            }
            for result in before.prefix(30) where turnedDown.contains(result.id.value) {
                print("[live check]     left out \(confidence[result.id.value].map { String(format: "%.2f", $0) } ?? " -- ")  \(result.metadata.windowName?.prefix(70) ?? "")")
            }
            // The check only ever speaks about results found by meaning alone.
            let semanticOnlyIDs = Set(before.map(\.id.value))
            XCTAssertTrue(Set(confidence.keys).isSubset(of: semanticOnlyIDs), "keyword results are untouched (\(query))")
        }
        await checker.stop()
        embedder.shutdown()
        XCTAssertGreaterThan(asked, 0, "the model was asked about real screenshots")
        XCTAssertGreaterThan(answered, 0, "and answered (all of them, unless the deadline ran out)")
        print("[live check] total: asked \(asked), answered \(answered), rejected \(rejected)")
    }
}
