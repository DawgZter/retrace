import XCTest
import CoreGraphics
import Foundation
import Shared
import Database
import Search
import Storage
@testable import App

/// How semantic matches reach search results: through `DataAdapter.search`, over a real SQLite database, with a
/// provider that returns fixed matches in place of the embedding model.
final class SemanticSearchAdapterTests: XCTestCase {
    private struct StubImageExtractor: ImageExtractor {
        struct NotImplemented: Error {}

        func extractFrame(videoPath: String, frameIndex: Int, frameRate: Double?) async throws -> Data {
            throw NotImplemented()
        }

        func extractFrameCGImage(videoPath: String, frameIndex: Int, frameRate: Double?) async throws -> CGImage {
            throw NotImplemented()
        }
    }

    /// Returns the matches it was given (after `delay`, if set) and counts how often it was asked.
    private final class FixedSemanticSearch: SemanticSearchProtocol, @unchecked Sendable {
        private let lock = NSLock()
        private var fixedMatches: [SemanticMatch] = []
        private var answerDelay: Duration?
        private var calls = 0
        private var limits: [Int] = []

        var matches: [SemanticMatch] {
            get { lock.withLock { fixedMatches } }
            set { lock.withLock { fixedMatches = newValue } }
        }
        var delay: Duration? {
            get { lock.withLock { answerDelay } }
            set { lock.withLock { answerDelay = newValue } }
        }
        var callCount: Int { lock.withLock { calls } }
        var requestedLimits: [Int] { lock.withLock { limits } }

        func matches(for query: String, limit: Int) async -> [SemanticMatch] {
            let (delay, matches) = lock.withLock { () -> (Duration?, [SemanticMatch]) in
                calls += 1
                limits.append(limit)
                return (answerDelay, Array(fixedMatches.prefix(limit)))
            }
            // Like the real service, this can't be interrupted: it answers when it answers.
            if let delay { try? await Task.sleep(for: delay) }
            return matches
        }
    }

    private var database: DatabaseManager!
    private var adapter: DataAdapter!
    private var semantic: FixedSemanticSearch!
    private var videoID: Int64 = 0
    private var frameCount = 0
    private let start = Date(timeIntervalSince1970: 1_790_000_000)

    override func setUp() async throws {
        database = DatabaseManager(databasePath: "file:semantic_adapter_\(UUID().uuidString)?mode=memory&cache=private")
        try await database.initialize()
        guard let pointer = await database.getConnection() else {
            throw XCTSkip("no database connection")
        }
        let connection = SQLiteConnection(db: pointer)
        adapter = DataAdapter(
            retraceConnection: connection,
            retraceReadConnectionPool: SQLiteReadConnectionPool(label: "test_semantic_search", sharedConnection: connection),
            retraceConfig: .retrace(storageRoot: "/tmp"),
            retraceImageExtractor: StubImageExtractor(),
            database: database
        )
        try await adapter.initialize()
        semantic = FixedSemanticSearch()
        await adapter.setSemanticSearch(semantic)

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
        return frameID
    }

    private func search(
        _ text: String,
        mode: SearchMode,
        filters: SearchFilters = .none,
        limit: Int = 50,
        cursor: SearchPageCursor? = nil
    ) async throws -> SearchResults {
        try await adapter.search(
            query: SearchQuery(text: text, filters: filters, limit: limit, cursor: cursor, mode: mode)
        )
    }

    func testAllModeFallsBackToMeaningWhenNoFrameHasTheWords() async throws {
        let pricing = try await insertFrame(text: "Plans start at three thousand dollars a month")
        let unrelated = try await insertFrame(text: "Garden tomato seedlings watering calendar")
        semantic.matches = [SemanticMatch(frameID: pricing, score: 4.2, cosine: 0.61)]

        let results = try await search("how much does it cost", mode: .all)

        XCTAssertEqual(results.results.map(\.id.value), [pricing])
        let result = try XCTUnwrap(results.results.first)
        XCTAssertTrue(result.isSemanticOnly)
        XCTAssertEqual(result.semanticScore, 4.2)
        XCTAssertEqual(result.metadata.appBundleID, "com.test.browser")
        XCTAssertEqual(result.videoID.value, videoID)
        XCTAssertNil(results.nextCursor)
        XCTAssertFalse(results.results.contains { $0.id.value == unrelated })
    }

    func testAllModeListsMeaningMatchesAfterTheKeywordMatches() async throws {
        let older = try await insertFrame(text: "Invoice 4471 payment overdue reminder")
        let related = try await insertFrame(text: "Outstanding balance on your account")
        let newer = try await insertFrame(text: "Invoice 4480 sent to the customer")
        semantic.matches = [
            SemanticMatch(frameID: related, score: 5, cosine: 0.7),
            SemanticMatch(frameID: older, score: 4, cosine: 0.6),
        ]

        let results = try await search("invoice", mode: .all)

        // Keyword matches keep their order (newest first); what only meaning found comes after them.
        XCTAssertEqual(results.results.map(\.id.value), [newer, older, related])
        XCTAssertEqual(results.results.map(\.isSemanticOnly), [false, false, true])
    }

    func testMeaningMatchesWaitUntilKeywordMatchesFitOnOnePage() {
        let query = SearchQuery(text: "invoice", limit: 50, mode: .all)
        let complete = SearchResults(query: query, results: [], searchTimeMs: 0, nextCursor: nil)
        let hasMorePages = SearchResults(
            query: query,
            results: [],
            searchTimeMs: 0,
            nextCursor: SearchPageCursor(native: SearchSourceCursor(timestamp: start, frameID: 1))
        )
        XCTAssertTrue(DataAdapter.shouldAddSemanticMatches(to: complete, for: query))
        XCTAssertFalse(
            DataAdapter.shouldAddSemanticMatches(to: hasMorePages, for: query),
            "a chronological list that continues on the next page isn't interrupted"
        )

        let relevant = SearchQuery(text: "invoice", limit: 50, mode: .relevant)
        XCTAssertTrue(DataAdapter.shouldAddSemanticMatches(to: hasMorePages, for: relevant))
        let secondPage = SearchQuery(text: "invoice", limit: 50, cursor: hasMorePages.nextCursor, mode: .relevant)
        XCTAssertFalse(DataAdapter.shouldAddSemanticMatches(to: complete, for: secondPage))
        let blank = SearchQuery(text: "  ", limit: 50, mode: .relevant)
        XCTAssertFalse(DataAdapter.shouldAddSemanticMatches(to: complete, for: blank))
    }

    func testWithRewindHistoryMeaningMatchesFollowRetracesOwnKeywordMatches() {
        // With Rewind connected, a newest-first page of Retrace results always carries a cursor: the next page
        // looks in Rewind. Retrace's matches are all shown once that cursor has no Retrace position left.
        let native = SearchResult(
            id: FrameID(value: 1),
            timestamp: start,
            snippet: "",
            matchedText: "invoice",
            relevanceScore: 0.5,
            metadata: .empty,
            segmentID: AppSegmentID(value: 1),
            frameIndex: 0,
            source: .native
        )
        let rewindNext = SearchPageCursor(native: nil, rewind: SearchSourceCursor(timestamp: start, frameID: 9))

        let newestFirst = SearchQuery(text: "invoice", limit: 50, mode: .all, sortOrder: .newestFirst)
        XCTAssertTrue(DataAdapter.shouldAddSemanticMatches(
            to: SearchResults(query: newestFirst, results: [native], searchTimeMs: 0, nextCursor: rewindNext),
            for: newestFirst
        ))
        XCTAssertTrue(DataAdapter.shouldAddSemanticMatches(
            to: SearchResults(query: newestFirst, results: [native], searchTimeMs: 0, nextCursor: SearchPageCursor()),
            for: newestFirst
        ))

        // Oldest first, Rewind's matches come before Retrace's: the first page is not where Retrace's end.
        let oldestFirst = SearchQuery(text: "invoice", limit: 50, mode: .all, sortOrder: .oldestFirst)
        XCTAssertFalse(DataAdapter.shouldAddSemanticMatches(
            to: SearchResults(query: oldestFirst, results: [native], searchTimeMs: 0, nextCursor: rewindNext),
            for: oldestFirst
        ))
    }

    func testSemanticMatchesRespectAppFilterAndExcludedWords() async throws {
        let inBrowser = try await insertFrame(text: "Plans start at three thousand dollars a month", app: "com.test.browser")
        let inChat = try await insertFrame(text: "They quoted three thousand dollars a month in the draft", app: "com.test.chat")
        semantic.matches = [
            SemanticMatch(frameID: inBrowser, score: 4, cosine: 0.6),
            SemanticMatch(frameID: inChat, score: 3.5, cosine: 0.55),
        ]

        let everything = try await search("how much does it cost", mode: .all)
        XCTAssertEqual(everything.results.map(\.id.value), [inBrowser, inChat])

        let onlyChat = try await search(
            "how much does it cost",
            mode: .all,
            filters: SearchFilters(appBundleIDs: ["com.test.chat"])
        )
        XCTAssertEqual(onlyChat.results.map(\.id.value), [inChat])

        let withoutDraft = try await search("how much does it cost -draft", mode: .all)
        XCTAssertEqual(withoutDraft.results.map(\.id.value), [inBrowser])

        // Exclusion means what it means for keyword search: whole words. "-raft" is not in "draft".
        let withoutRaft = try await search("how much does it cost -raft", mode: .all)
        XCTAssertEqual(withoutRaft.results.map(\.id.value), [inBrowser, inChat])

        let beforeSecondFrame = try await search(
            "how much does it cost",
            mode: .all,
            filters: SearchFilters(endDate: start.addingTimeInterval(1.5 * 3600))
        )
        XCTAssertEqual(beforeSecondFrame.results.map(\.id.value), [inBrowser])
    }

    func testAFilteredSearchLooksPastTheBestMatchesOutsideTheFilter() async throws {
        // 250 strong matches in the browser, then one weaker match in the chat app.
        var matches: [SemanticMatch] = []
        for index in 0..<250 {
            let frame = try await insertFrame(text: "Plans start at three thousand dollars a month, copy \(index)", app: "com.test.browser")
            matches.append(SemanticMatch(frameID: frame, score: 9 - Double(index) * 0.01, cosine: 0.7))
        }
        let inChat = try await insertFrame(text: "They quoted three thousand dollars a month", app: "com.test.chat")
        matches.append(SemanticMatch(frameID: inChat, score: 2.5, cosine: 0.5))
        semantic.matches = matches

        let onlyChat = try await search(
            "how much does it cost",
            mode: .all,
            filters: SearchFilters(appBundleIDs: ["com.test.chat"])
        )

        XCTAssertEqual(onlyChat.results.map(\.id.value), [inChat])
        XCTAssertEqual(semantic.requestedLimits.last, DataAdapter.semanticFilteredCandidateLimit)

        _ = try await search("how much does it cost", mode: .all)
        XCTAssertEqual(semantic.requestedLimits.last, DataAdapter.semanticCandidateLimit)
    }

    func testAFilteredSearchGoesAsFarDownTheMatchesAsItTakes() async throws {
        // Thousands of better matches that the filter leaves out (here: frames the adapter gets no row for),
        // then the one match in the chat app.
        var matches: [SemanticMatch] = (0..<5_000).map { index in
            SemanticMatch(frameID: 90_000_000 + Int64(index), score: 9 - Double(index) * 0.001, cosine: 0.75)
        }
        let inChat = try await insertFrame(text: "They quoted three thousand dollars a month", app: "com.test.chat")
        matches.append(SemanticMatch(frameID: inChat, score: 2.5, cosine: 0.72))
        semantic.matches = matches

        let onlyChat = try await search(
            "how much does it cost",
            mode: .relevant,
            filters: SearchFilters(appBundleIDs: ["com.test.chat"])
        )

        XCTAssertEqual(onlyChat.results.map(\.id.value), [inChat], "a match inside the filter is found however many rank above it outside")
    }

    func testALookupPutsFramesWithTheWordFirst() async throws {
        let keywordAndMeaning = try await insertFrame(text: "Refund issued for your laptop order")
        let keywordOnly = try await insertFrame(text: "Refund policy link in the page footer")
        let meaningOnly = try await insertFrame(text: "We have returned your money to the original card")
        semantic.matches = [
            SemanticMatch(frameID: meaningOnly, score: 3.4, cosine: 0.6),
            SemanticMatch(frameID: keywordAndMeaning, score: 2.6, cosine: 0.5),
        ]

        let results = try await search("refund", mode: .relevant)

        // One word is a lookup: the frame with the word and the meaning, then the word alone, then meaning alone.
        XCTAssertEqual(results.results.map(\.id.value), [keywordAndMeaning, keywordOnly, meaningOnly])
        XCTAssertEqual(results.results.map(\.isSemanticOnly), [false, false, true])
        XCTAssertEqual(results.results.first?.semanticScore, 2.6)
    }

    func testALookupKeepsEveryKeywordResultAheadOfMeaningOnlyOnesOnALongPage() {
        // A page of 150 keyword results, as Relevance order fetches them, and one strong meaning-only match.
        let keyword = (1...150).map { frame in
            SearchResult(
                id: FrameID(value: Int64(frame)),
                timestamp: start.addingTimeInterval(Double(frame)),
                snippet: "",
                matchedText: "refund",
                relevanceScore: 0.5,
                metadata: .empty,
                segmentID: AppSegmentID(value: Int64(frame)),
                frameIndex: 0,
                source: .native
            )
        }
        let meaningOnly = row(frame: 999, segment: 999, window: "Money returned", minutes: 1)
            .withSemanticScore(9, relevanceScore: 0.9)

        let lookup = DataAdapter.mergeSemanticResults(
            keywordResults: keyword,
            matches: [SemanticMatch(frameID: 999, score: 9, cosine: 0.8)],
            semanticOnly: [meaningOnly],
            mode: .relevant,
            isLookup: true
        )
        XCTAssertEqual(lookup.last?.id.value, 999, "after all 150, not in among the later ones")
        XCTAssertEqual(lookup.prefix(150).map(\.id.value), Array(1...150).map(Int64.init))

        // A description is different: the best meaning-only match sits near the top.
        let description = DataAdapter.mergeSemanticResults(
            keywordResults: keyword,
            matches: [SemanticMatch(frameID: 999, score: 9, cosine: 0.8)],
            semanticOnly: [meaningOnly],
            mode: .relevant,
            isLookup: false
        )
        XCTAssertEqual(description.firstIndex { $0.id.value == 999 }, 1)
    }

    func testADescriptionRanksBothFirstThenInterleavesWordsAndMeaning() async throws {
        let both = try await insertFrame(text: "Your money was returned to the card you paid with")
        let wordsOnly = try await insertFrame(text: "Card games: money is returned to the bank when a player folds")
        let bestMeaning = try await insertFrame(text: "Refund issued for your laptop order")
        let weakerMeaning = try await insertFrame(text: "Chargeback dispute opened by the customer")
        semantic.matches = [
            SemanticMatch(frameID: bestMeaning, score: 5.0, cosine: 0.7),
            SemanticMatch(frameID: both, score: 3.0, cosine: 0.6),
            SemanticMatch(frameID: weakerMeaning, score: 2.5, cosine: 0.5),
        ]

        let results = try await search("money returned to card", mode: .relevant)
        let order = results.results.map(\.id.value)

        XCTAssertEqual(order.count, 4)
        XCTAssertEqual(order.first, both, "a frame with the words whose meaning also stands out leads")
        XCTAssertEqual(Set(order[1...2]), [wordsOnly, bestMeaning], "then the best of each kind")
        XCTAssertEqual(order.last, weakerMeaning)
    }

    func testAFrameWithTheWordsBelowThePageIsNotLabelledAsMeaningOnly() async throws {
        let first = try await insertFrame(text: "Refund issued for your laptop order")
        let second = try await insertFrame(text: "Refund policy link in the page footer")
        let meaningOnly = try await insertFrame(text: "We have returned your money to the original card")
        semantic.matches = [first, second, meaningOnly].enumerated().map { offset, frame in
            SemanticMatch(frameID: frame, score: 4 - Double(offset) * 0.5, cosine: 0.6)
        }

        // A page of one: the other frame with the word comes back through meaning.
        let results = try await search("refund", mode: .relevant, limit: 1)

        XCTAssertEqual(Set(results.results.map(\.id.value)), [first, second, meaningOnly])
        for result in results.results where result.id.value != meaningOnly {
            XCTAssertFalse(result.isSemanticOnly, "frame \(result.id.value) contains the word")
        }
        XCTAssertEqual(results.results.first { $0.id.value == meaningOnly }?.isSemanticOnly, true)
    }

    func testKeywordResultsDoNotWaitForASlowSemanticSearch() async throws {
        let keyword = try await insertFrame(text: "Refund issued for your laptop order")
        let meaningOnly = try await insertFrame(text: "We have returned your money to the original card")
        semantic.matches = [SemanticMatch(frameID: meaningOnly, score: 3.4, cosine: 0.6)]
        semantic.delay = .seconds(5)   // a model that is still starting

        let started = Date()
        let results = try await search("refund", mode: .relevant)
        let waited = Date().timeIntervalSince(started)

        XCTAssertEqual(results.results.map(\.id.value), [keyword])
        XCTAssertLessThan(waited, 1.5, "keyword results are returned at the deadline, not when the model is ready")
        XCTAssertEqual(semantic.callCount, 1)
    }

    func testLaterPagesAreKeywordOnly() async throws {
        // More frames with the word than one page by date holds (pages are read 150 rows at a time).
        var withTheWord: [Int64] = []
        for index in 0..<160 {
            withTheWord.append(try await insertFrame(text: "Refund issued for order number \(index)"))
        }
        let meaningOnly = try await insertFrame(text: "We have returned your money to the original card")
        semantic.matches = [SemanticMatch(frameID: meaningOnly, score: 3.4, cosine: 0.6)]

        // By date: the list continues on the next page, so meaning isn't asked for.
        let firstPage = try await search("refund", mode: .all)
        XCTAssertEqual(firstPage.results.count, 150)
        XCTAssertFalse(firstPage.results.contains { $0.id.value == meaningOnly })
        let cursor = try XCTUnwrap(firstPage.nextCursor)
        XCTAssertEqual(semantic.callCount, 0)

        // The page reached through the cursor is the last one, and still keyword-only.
        let secondPage = try await search("refund", mode: .all, cursor: cursor)
        XCTAssertEqual(secondPage.results.count, 10)
        XCTAssertEqual(
            Set(firstPage.results.map(\.id.value)).union(secondPage.results.map(\.id.value)),
            Set(withTheWord)
        )
        XCTAssertEqual(semantic.callCount, 0, "a page reached through a cursor never asks for meaning")

        // In relevance order the first page does ask; a page after it does not.
        let relevantFirst = try await search("refund", mode: .relevant)
        XCTAssertTrue(relevantFirst.results.contains { $0.id.value == meaningOnly })
        XCTAssertEqual(semantic.callCount, 1)
        if let relevantCursor = relevantFirst.nextCursor {
            let relevantSecond = try await search("refund", mode: .relevant, cursor: relevantCursor)
            XCTAssertFalse(relevantSecond.results.contains { $0.id.value == meaningOnly })
            XCTAssertEqual(semantic.callCount, 1)
        }
    }

    func testAMatchForADeletedFrameIsDropped() async throws {
        let kept = try await insertFrame(text: "Plans start at three thousand dollars a month")
        let deleted = try await insertFrame(text: "Pricing tiers for enterprise customers")
        try await database.deleteFrame(id: FrameID(value: deleted))
        // The in-memory index can still hold a vector for a frame deleted since launch.
        semantic.matches = [
            SemanticMatch(frameID: deleted, score: 5, cosine: 0.7),
            SemanticMatch(frameID: kept, score: 4, cosine: 0.6),
        ]

        let results = try await search("how much does it cost", mode: .all)

        XCTAssertEqual(results.results.map(\.id.value), [kept])
    }

    func testAMatchForAFrameDeletedFromTheTimelineIsDropped() async throws {
        let kept = try await insertFrame(text: "Plans start at three thousand dollars a month")
        let deleted = try await insertFrame(text: "Pricing tiers for enterprise customers")
        // Deleted from the timeline: the frame's row is still there until its video has been rewritten.
        let outcome = try await database.deleteOrScheduleNativeFramesForDeletion(frameIDs: [deleted])
        XCTAssertEqual(outcome.scheduledJobs.map(\.frameID), [deleted])
        semantic.matches = [
            SemanticMatch(frameID: deleted, score: 5, cosine: 0.7),
            SemanticMatch(frameID: kept, score: 4, cosine: 0.6),
        ]

        let results = try await search("how much does it cost", mode: .all)

        XCTAssertEqual(results.results.map(\.id.value), [kept])
    }

    // MARK: - Thinning

    private func row(
        frame: Int64,
        segment: Int64,
        app: String = "com.test.browser",
        window: String,
        minutes: Double
    ) -> SearchResult {
        SearchResult(
            id: FrameID(value: frame),
            timestamp: start.addingTimeInterval(minutes * 60),
            snippet: "",
            matchedText: "",
            relevanceScore: 0,
            metadata: FrameMetadata(appBundleID: app, appName: nil, windowName: window, browserURL: nil, displayID: 0),
            segmentID: AppSegmentID(value: segment),
            frameIndex: 0
        )
    }

    func testOneMomentIsShownOnce() {
        let rows: [Int64: SearchResult] = [
            1: row(frame: 1, segment: 10, window: "Pricing", minutes: 0),
            2: row(frame: 2, segment: 10, window: "Pricing", minutes: 0.1),     // same segment as 1
            3: row(frame: 3, segment: 11, window: "Pricing", minutes: 2),       // same window, two minutes later
            4: row(frame: 4, segment: 12, window: "Pricing", minutes: 90),      // same window, much later
            5: row(frame: 5, segment: 13, window: "Checkout", minutes: 1),      // another window
        ]
        let matches = (1...5).map { SemanticMatch(frameID: Int64($0), score: 6 - Double($0), cosine: 0.5) }

        let shown = DataAdapter.semanticOnlyResults(keywordResults: [], matches: matches, semanticRows: rows)

        XCTAssertEqual(shown.map(\.id.value), [1, 4, 5])
    }

    func testOnlyAFewWeakMatchesAreShownHoweverManyStandOut() {
        // Ten frames stand out for the query; two of them are similar enough to be taken at their word.
        var rows: [Int64: SearchResult] = [:]
        var matches: [SemanticMatch] = []
        for frame in 1...10 {
            rows[Int64(frame)] = row(frame: Int64(frame), segment: Int64(100 + frame), window: "Window \(frame)", minutes: Double(frame) * 60)
            let strong = frame == 2 || frame == 7
            matches.append(SemanticMatch(frameID: Int64(frame), score: 6 - Double(frame) * 0.3, cosine: strong ? 0.74 : 0.62))
        }

        let shown = DataAdapter.semanticOnlyResults(keywordResults: [], matches: matches, semanticRows: rows)

        // The three best weak ones and both strong ones, still best first.
        XCTAssertEqual(shown.map(\.id.value), [1, 2, 3, 4, 7])

        // A query about nothing in the history: everything is weak, so three results, not thirty.
        let nothingStrong = matches.map { SemanticMatch(frameID: $0.frameID, score: $0.score, cosine: 0.62) }
        let weakOnly = DataAdapter.semanticOnlyResults(keywordResults: [], matches: nothingStrong, semanticRows: rows)
        XCTAssertEqual(weakOnly.map(\.id.value), [1, 2, 3])
    }

    func testSegmentsKeywordSearchAlreadyReturnedAreNotRepeated() {
        let keyword = SearchResult(
            id: FrameID(value: 1),
            timestamp: start,
            snippet: "",
            matchedText: "pricing",
            relevanceScore: 0.5,
            metadata: FrameMetadata(appBundleID: "com.test.browser", appName: nil, windowName: "Pricing", browserURL: nil, displayID: 0),
            segmentID: AppSegmentID(value: 10),
            frameIndex: 0
        )
        let rows: [Int64: SearchResult] = [
            2: row(frame: 2, segment: 10, window: "Pricing", minutes: 0.1),
            3: row(frame: 3, segment: 20, window: "Plans", minutes: 30),
        ]
        let matches = [
            SemanticMatch(frameID: 2, score: 5, cosine: 0.7),
            SemanticMatch(frameID: 3, score: 3, cosine: 0.5),
        ]

        let shown = DataAdapter.semanticOnlyResults(keywordResults: [keyword], matches: matches, semanticRows: rows)

        XCTAssertEqual(shown.map(\.id.value), [3], "frame 2 is the moment keyword search already found")
    }

    func testLookupAndDescriptionQueriesAreToldApart() {
        XCTAssertTrue(DataAdapter.isLookupQuery("stripe"))
        XCTAssertTrue(DataAdapter.isLookupQuery("stripe invoice"))
        XCTAssertTrue(DataAdapter.isLookupQuery("the page that said \"payment overdue\""))
        XCTAssertTrue(DataAdapter.isLookupQuery("stripe invoice -draft"), "an excluded word describes nothing")
        XCTAssertFalse(DataAdapter.isLookupQuery("checking whether a domain is available"))
    }
}

/// The whole path with the real model, against a copy of a real Retrace database. Skipped unless pointed at one:
///
///   RETRACE_SEMANTIC_LIVE_DB        a COPY of retrace.db (this test migrates it and writes vectors into it)
///   RETRACE_SEMANTIC_LIVE_MODEL     embeddinggemma-2-Q8_0.gguf
///   RETRACE_EMBEDDING_HELPER        the built RetraceEmbeddingHelper
///   RETRACE_SEMANTIC_LIVE_QUERIES   queries separated by "|"
///   RETRACE_SEMANTIC_LIVE_QUERY_FILE  or: a text file with one query a line
///   RETRACE_SEMANTIC_LIVE_RESULTS   optional: where to write, as JSON, the frame ids each query returned with
///                                   semantic search off and on (to score against your own answer key)
///   RETRACE_SEMANTIC_LIVE_MAX_PASSES  optional cap on indexing batches (default: until up to date)
///
/// Besides the top results of each query it prints the median and 95th-percentile time of a search with
/// semantic search off and on (Relevance order, three warm rounds over the queries).
///
/// To embed screenshots as well:
///
///   RETRACE_SEMANTIC_LIVE_MEDIA     mmproj-embeddinggemma-2-Q8_0.gguf
///   RETRACE_SEMANTIC_LIVE_STORAGE   the storage root whose chunks/ the database's videos are in (only read)
final class SemanticSearchLiveTests: XCTestCase {
    private struct StubImageExtractor: ImageExtractor {
        struct NotImplemented: Error {}

        func extractFrame(videoPath: String, frameIndex: Int, frameRate: Double?) async throws -> Data {
            throw NotImplemented()
        }

        func extractFrameCGImage(videoPath: String, frameIndex: Int, frameRate: Double?) async throws -> CGImage {
            throw NotImplemented()
        }
    }

    func testIndexAndSearchACopyOfARealDatabase() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let databasePath = environment["RETRACE_SEMANTIC_LIVE_DB"],
              let modelPath = environment["RETRACE_SEMANTIC_LIVE_MODEL"],
              let helperPath = environment["RETRACE_EMBEDDING_HELPER"] else {
            throw XCTSkip("set RETRACE_SEMANTIC_LIVE_DB, RETRACE_SEMANTIC_LIVE_MODEL and RETRACE_EMBEDDING_HELPER to run")
        }
        var queries = (environment["RETRACE_SEMANTIC_LIVE_QUERIES"] ?? "")
            .split(separator: "|").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
        if let queryFile = environment["RETRACE_SEMANTIC_LIVE_QUERY_FILE"] {
            queries += try String(contentsOfFile: queryFile, encoding: .utf8)
                .split(whereSeparator: \.isNewline).map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
        }
        let maxPasses = environment["RETRACE_SEMANTIC_LIVE_MAX_PASSES"].flatMap(Int.init) ?? Int.max

        let database = DatabaseManager(databasePath: databasePath)
        try await database.initialize()
        let maybePointer = await database.getConnection()
        let pointer = try XCTUnwrap(maybePointer)
        let connection = SQLiteConnection(db: pointer)
        let adapter = DataAdapter(
            retraceConnection: connection,
            retraceReadConnectionPool: SQLiteReadConnectionPool(label: "live_semantic_search", sharedConnection: connection),
            retraceConfig: .retrace(storageRoot: URL(fileURLWithPath: databasePath).deletingLastPathComponent().path),
            retraceImageExtractor: StubImageExtractor(),
            database: database
        )
        try await adapter.initialize()

        let helperURL = URL(fileURLWithPath: helperPath)
        let embedder = EmbeddingHelperClient(helperURL: helperURL, modelPath: modelPath)
        var imageEmbedder: EmbeddingHelperClient?
        var imageProvider: StorageFrameImageProvider?
        if let mediaPath = environment["RETRACE_SEMANTIC_LIVE_MEDIA"],
           let storageRoot = environment["RETRACE_SEMANTIC_LIVE_STORAGE"] {
            imageEmbedder = EmbeddingHelperClient(helperURL: helperURL, modelPath: modelPath, mediaModelPath: mediaPath)
            imageProvider = StorageFrameImageProvider(extractor: HEVCStorageExtractor(storageRoot: storageRoot))
        }
        let withImages = imageEmbedder != nil
        let service = SemanticSearchService(
            store: database,
            embedder: embedder,
            imageEmbedder: imageEmbedder,
            imageProvider: imageProvider,
            isEnabled: { true }
        )
        let indexer = SemanticIndexer(
            store: database,
            service: service,
            isEnabled: { true },
            isImagesEnabled: { withImages },
            isReady: { _ in nil },
            isOnPower: { true }
        )
        await adapter.setSemanticSearch(service)

        let indexingStarted = Date()
        let before = await indexer.status()
        var passes = 0
        var state = SemanticIndexStatus.State.indexing
        var textFinishedAfter: TimeInterval?
        while passes < maxPasses {
            _ = await indexer.runPass()
            passes += 1
            state = await indexer.status().state
            if passes % 250 == 0 {
                let progress = await indexer.status()
                if textFinishedAfter == nil, progress.embeddedFrames >= progress.indexedFrames {
                    textFinishedAfter = Date().timeIntervalSince(indexingStarted)
                }
                print("[live] pass \(passes): \(progress.embeddedFrames) by text, \(progress.embeddedImages) by screenshot, of \(progress.indexedFrames) frames, \(Int(Date().timeIntervalSince(indexingStarted))) s")
            }
            if state != .indexing { break }
        }
        let after = await indexer.status()
        let seconds = Date().timeIntervalSince(indexingStarted)
        let textNow = after.embeddedFrames - before.embeddedFrames
        let imagesNow = after.embeddedImages - before.embeddedImages
        print("[live] state \(state.rawValue)\(after.detail.map { " (\($0))" } ?? ""): \(after.embeddedFrames) by text and \(after.embeddedImages) by screenshot, of \(after.indexedFrames) frames")
        print("[live] this run: \(textNow) texts and \(imagesNow) screenshots in \(String(format: "%.1f", seconds)) s"
            + (textFinishedAfter.map { String(format: " (text done within %.0f s)", $0) } ?? ""))
        XCTAssertNotEqual(state, .failed)
        XCTAssertGreaterThan(after.embeddedFrames, 0)
        if withImages {
            XCTAssertGreaterThan(after.embeddedImages, 0)
        }

        for query in queries {
            for mode in [SearchMode.all, .relevant] {
                let started = Date()
                let results = try await adapter.search(query: SearchQuery(text: query, limit: 50, mode: mode))
                let milliseconds = Int(Date().timeIntervalSince(started) * 1000)
                let semanticOnly = results.results.filter(\.isSemanticOnly).count
                print("[live] \"\(query)\" (\(mode.rawValue)): \(results.results.count) results, \(semanticOnly) by meaning alone, \(milliseconds) ms")
                for result in results.results.prefix(5) {
                    let score = result.semanticScore.map { String(format: "%.1f", $0) } ?? "-"
                    let kind = result.isSemanticOnly ? "meaning" : "keyword"
                    print("[live]     \(kind) z=\(score)  \(result.metadata.appBundleID ?? "?")  \(result.metadata.windowName?.prefix(70) ?? "")")
                }
            }
        }

        // The same searches with semantic search off and on: what each returns, and what it costs.
        if !queries.isEmpty {
            func percentiles(_ values: [Double]) -> String {
                let sorted = values.sorted()
                return String(
                    format: "%.0f ms median, %.0f ms p95",
                    sorted[sorted.count / 2],
                    sorted[Int(Double(sorted.count - 1) * 0.95)]
                )
            }
            var frames: [String: [String: [Int64]]] = [:]
            for (label, provider) in [("keyword", nil), ("semantic", service)] as [(String, SemanticSearchService?)] {
                await adapter.setSemanticSearch(provider)
                for mode in [SearchMode.relevant, .all] {
                    var times: [Double] = []
                    for round in 0..<3 {
                        for query in queries {
                            let started = Date()
                            let results = try await adapter.search(query: SearchQuery(text: query, limit: 50, mode: mode))
                            times.append(Date().timeIntervalSince(started) * 1000)
                            if round == 0 {
                                frames[query, default: [:]]["\(label)_\(mode.rawValue)"] = results.results.map(\.id.value)
                            }
                        }
                    }
                    // The first round also pays for cold caches; report the two warm rounds.
                    let warm = Array(times.dropFirst(queries.count))
                    print("[live] \(label), \(mode.rawValue) order, \(queries.count) queries: \(percentiles(warm)) a search")
                }
            }
            if let resultsPath = environment["RETRACE_SEMANTIC_LIVE_RESULTS"] {
                let data = try JSONSerialization.data(withJSONObject: frames, options: [.prettyPrinted, .sortedKeys])
                try data.write(to: URL(fileURLWithPath: resultsPath))
                print("[live] wrote the frame ids each query returned to \(resultsPath)")
            }
        }

        embedder.shutdown()
        imageEmbedder?.shutdown()
        await adapter.shutdown()
        try await database.close()
    }
}
