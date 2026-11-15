import XCTest
import Shared
import App
@testable import Retrace

/// The result check (prototype) as the search overlay shows it: results appear as search returned them, and
/// a result found by meaning alone is taken out when the check turns it down.
@MainActor
final class SearchResultCheckTests: XCTestCase {
    /// Hands out one stream of check events per search, and lets a test send the events when it wants to.
    private final class Feed: @unchecked Sendable {
        private let lock = NSLock()
        private var continuations: [AsyncStream<SemanticCheckEvent>.Continuation] = []
        private var ended: [Bool] = []
        private var asked: [[Int64]] = []

        /// The results found by meaning alone that each check was asked about.
        var askedAbout: [[Int64]] { lock.withLock { asked } }

        func stream(for results: SearchResults) -> AsyncStream<SemanticCheckEvent> {
            AsyncStream { continuation in
                let index = lock.withLock { () -> Int in
                    continuations.append(continuation)
                    ended.append(false)
                    asked.append(results.results.filter(\.isSemanticOnly).map(\.id.value))
                    return continuations.count - 1
                }
                continuation.onTermination = { [weak self] _ in
                    self?.lock.withLock { self?.ended[index] = true }
                }
            }
        }

        var count: Int { lock.withLock { continuations.count } }
        func hasEnded(_ index: Int) -> Bool { lock.withLock { ended[index] } }
        func send(_ event: SemanticCheckEvent, to index: Int = 0) { _ = lock.withLock { continuations[index] }.yield(event) }
        func finish(_ index: Int = 0) { lock.withLock { continuations[index] }.finish() }
    }

    private func result(_ frameID: Int64, byMeaningAlone: Bool) -> SearchResult {
        let result = SearchResult(
            id: FrameID(value: frameID),
            timestamp: Date(timeIntervalSince1970: Double(1_700_000_000 + frameID)),
            snippet: "",
            matchedText: byMeaningAlone ? "" : "refund",
            relevanceScore: 0.5,
            metadata: .empty,
            segmentID: AppSegmentID(value: frameID),
            frameIndex: 0,
            source: .native
        )
        return byMeaningAlone ? result.withSemanticScore(5 - Double(frameID) * 0.1, relevanceScore: 0.5) : result
    }

    /// Frame 1 was found by its words; the others by meaning alone.
    private func page(_ frameIDs: [Int64] = [1, 2, 3, 4]) -> SearchResults {
        SearchResults(
            query: SearchQuery(text: "refund", limit: 50, mode: .relevant),
            results: frameIDs.map { result($0, byMeaningAlone: $0 != 1) },
            searchTimeMs: 1
        )
    }

    private func makeViewModel(_ feed: Feed, results: SearchResults) -> SearchViewModel {
        let viewModel = SearchViewModel(
            coordinator: AppCoordinator(),
            executeSearch: { _ in results },
            resultCheckEvents: { results in feed.stream(for: results) }
        )
        viewModel.searchMode = .relevant
        viewModel.searchQuery = "refund"
        viewModel.resumeResultCheckIfNeeded()   // the overlay is on screen
        return viewModel
    }

    private func waitUntil(_ condition: @escaping @MainActor () -> Bool) async throws {
        let clock = ContinuousClock()
        let deadline = clock.now.advanced(by: .seconds(2))
        while clock.now < deadline {
            if condition() { return }
            try await Task.sleep(for: .milliseconds(10), clock: .continuous)
        }
        XCTFail("Timed out waiting for condition")
    }

    private func shown(_ viewModel: SearchViewModel) -> [Int64] {
        viewModel.results?.results.map(\.id.value) ?? []
    }

    func testResultsAreShownAtOnceAndTurnedDownOnesGoAsTheAnswersArrive() async throws {
        let feed = Feed()
        let viewModel = makeViewModel(feed, results: page())

        await viewModel.performSearch(query: "refund", trigger: "test")

        XCTAssertEqual(shown(viewModel), [1, 2, 3, 4], "everything search returned, before any answer")
        XCTAssertFalse(viewModel.isSearching)
        try await waitUntil { feed.count == 1 }

        feed.send(.started(asked: 3))
        try await waitUntil { viewModel.isCheckingResults }

        feed.send(.decided(frameID: 3, confidence: 0.1, isKept: false))
        try await waitUntil { self.shown(viewModel) == [1, 2, 4] }

        feed.send(.decided(frameID: 2, confidence: 0.5, isKept: true))
        feed.send(.decided(frameID: 4, confidence: 0.99, isKept: true))
        feed.finish()
        try await waitUntil { !viewModel.isCheckingResults }

        XCTAssertEqual(shown(viewModel), [1, 2, 4], "what stays keeps its place: the surer result does not jump ahead")
    }

    func testAResultFoundByItsWordsIsNeverTakenOut() async throws {
        let feed = Feed()
        let viewModel = makeViewModel(feed, results: page())
        await viewModel.performSearch(query: "refund", trigger: "test")
        try await waitUntil { feed.count == 1 }

        feed.send(.started(asked: 3))
        feed.send(.decided(frameID: 1, confidence: 0.0, isKept: false))
        feed.send(.decided(frameID: 2, confidence: 0.0, isKept: false))
        try await waitUntil { self.shown(viewModel) == [1, 3, 4] }
    }

    func testTheResultTheUserHasOpenStays() async throws {
        let feed = Feed()
        let results = page()
        let viewModel = makeViewModel(feed, results: results)
        await viewModel.performSearch(query: "refund", trigger: "test")
        try await waitUntil { feed.count == 1 }
        viewModel.selectedResult = results.results[2]   // frame 3

        feed.send(.started(asked: 3))
        feed.send(.decided(frameID: 3, confidence: 0.0, isKept: false))
        feed.send(.decided(frameID: 4, confidence: 0.0, isKept: false))
        try await waitUntil { self.shown(viewModel) == [1, 2, 3] }
    }

    func testANewSearchStopsTheCheckOfTheOldOne() async throws {
        let feed = Feed()
        let viewModel = makeViewModel(feed, results: page())
        await viewModel.performSearch(query: "refund", trigger: "test")
        try await waitUntil { feed.count == 1 }
        feed.send(.started(asked: 3))
        try await waitUntil { viewModel.isCheckingResults }

        await viewModel.performSearch(query: "refund policy", trigger: "test")
        try await waitUntil { feed.count == 2 }

        XCTAssertTrue(feed.hasEnded(0), "nobody is listening to the first search's check any more")
        XCTAssertFalse(viewModel.isCheckingResults, "the new search's check has not said it started")
        feed.send(.decided(frameID: 2, confidence: 0.0, isKept: false), to: 0)
        feed.send(.started(asked: 3), to: 1)
        feed.send(.decided(frameID: 4, confidence: 0.0, isKept: false), to: 1)
        try await waitUntil { self.shown(viewModel) == [1, 2, 3] }
    }

    func testClosingTheOverlayStopsTheCheck() async throws {
        let feed = Feed()
        let viewModel = makeViewModel(feed, results: page())
        await viewModel.performSearch(query: "refund", trigger: "test")
        try await waitUntil { feed.count == 1 }
        feed.send(.started(asked: 3))
        try await waitUntil { viewModel.isCheckingResults }

        viewModel.cancelSearch()

        XCTAssertFalse(viewModel.isCheckingResults)
        try await waitUntil { feed.hasEnded(0) }
        XCTAssertEqual(shown(viewModel), [1, 2, 3, 4])
    }

    func testReopeningTheOverlayFinishesACheckThatClosingItCutShort() async throws {
        let feed = Feed()
        let viewModel = makeViewModel(feed, results: page())
        await viewModel.performSearch(query: "refund", trigger: "test")
        try await waitUntil { feed.count == 1 }
        feed.send(.started(asked: 3))
        feed.send(.decided(frameID: 2, confidence: 0.9, isKept: true))
        feed.send(.decided(frameID: 3, confidence: 0.1, isKept: false))
        try await waitUntil { self.shown(viewModel) == [1, 2, 4] }

        viewModel.cancelSearch()            // opening a result closes the overlay
        try await waitUntil { feed.hasEnded(0) }
        viewModel.resumeResultCheckIfNeeded()   // and Cmd+K brings it back with the same results
        try await waitUntil { feed.count == 2 }

        XCTAssertEqual(feed.askedAbout[1], [4], "only what the first check had not got to")
        feed.send(.started(asked: 1), to: 1)
        feed.send(.decided(frameID: 4, confidence: 0.0, isKept: false), to: 1)
        feed.finish(1)
        try await waitUntil { self.shown(viewModel) == [1, 2] && !viewModel.isCheckingResults }

        viewModel.resumeResultCheckIfNeeded()
        try await Task.sleep(for: .milliseconds(100))
        XCTAssertEqual(feed.count, 2, "a check that finished is not run again")
    }

    func testHidingTheOverlayStopsTheCheckAndShowingItAgainFinishesIt() async throws {
        let feed = Feed()
        let viewModel = makeViewModel(feed, results: page())
        await viewModel.performSearch(query: "refund", trigger: "test")
        try await waitUntil { feed.count == 1 }
        feed.send(.started(asked: 3))
        feed.send(.decided(frameID: 2, confidence: 0.9, isKept: true))
        try await waitUntil { viewModel.isCheckingResults }

        viewModel.pauseResultCheck()            // another dialog took the overlay's place
        try await waitUntil { feed.hasEnded(0) }
        XCTAssertFalse(viewModel.isCheckingResults)
        XCTAssertEqual(shown(viewModel), [1, 2, 3, 4], "nothing is lost by pausing")

        viewModel.resumeResultCheckIfNeeded()
        try await waitUntil { feed.count == 2 }
        XCTAssertEqual(feed.askedAbout[1], [3, 4])
    }

    func testClearingTheResultsEndsTheirCheck() async throws {
        let feed = Feed()
        let viewModel = makeViewModel(feed, results: page())
        await viewModel.performSearch(query: "refund", trigger: "test")
        try await waitUntil { feed.count == 1 }
        feed.send(.started(asked: 3))
        try await waitUntil { viewModel.isCheckingResults }

        viewModel.results = nil   // the search field was cleared

        XCTAssertFalse(viewModel.isCheckingResults, "no spinner beside an empty field")
        try await waitUntil { feed.hasEnded(0) }
        viewModel.resumeResultCheckIfNeeded()
        try await Task.sleep(for: .milliseconds(100))
        XCTAssertEqual(feed.count, 1, "and nothing left to pick up")
    }

    func testARestoredListIsCheckedAgainFromTheStart() async throws {
        let feed = Feed()
        let saved = page()
        let viewModel = makeViewModel(feed, results: saved)
        await viewModel.performSearch(query: "refund", trigger: "test")
        try await waitUntil { feed.count == 1 }
        feed.send(.started(asked: 3))
        feed.send(.decided(frameID: 2, confidence: 0.9, isKept: true))
        feed.send(.decided(frameID: 3, confidence: 0.1, isKept: false))
        try await waitUntil { self.shown(viewModel) == [1, 2, 4] }

        // The list as it was saved before those answers comes back (the timeline was refreshed).
        viewModel.results = saved
        viewModel.resetResultCheck(forRestored: saved, checkWasComplete: false)
        try await waitUntil { feed.count == 2 }

        XCTAssertTrue(feed.hasEnded(0))
        XCTAssertEqual(feed.askedAbout[1], [2, 3, 4], "the result that had been turned down is asked about again")
        feed.send(.started(asked: 3), to: 1)
        feed.send(.decided(frameID: 3, confidence: 0.1, isKept: false), to: 1)
        try await waitUntil { self.shown(viewModel) == [1, 2, 4] }
    }

    func testARestoredListWhoseCheckHadFinishedIsLeftAlone() async throws {
        let feed = Feed()
        let saved = page()
        let viewModel = makeViewModel(feed, results: saved)
        await viewModel.performSearch(query: "refund", trigger: "test")
        try await waitUntil { feed.count == 1 }
        feed.send(.started(asked: 3))
        try await waitUntil { viewModel.isCheckingResults }

        viewModel.results = saved
        viewModel.resetResultCheck(forRestored: saved, checkWasComplete: true)

        XCTAssertFalse(viewModel.isCheckingResults, "the old list's check is over; no spinner is left behind")
        try await waitUntil { feed.hasEnded(0) }
        viewModel.resumeResultCheckIfNeeded()
        try await Task.sleep(for: .milliseconds(100))
        XCTAssertEqual(feed.count, 1)
    }

    func testResultsThatArriveWhileTheOverlayIsHiddenAreCheckedWhenItComesBack() async throws {
        let feed = Feed()
        let viewModel = makeViewModel(feed, results: page())
        viewModel.pauseResultCheck()            // another dialog took the overlay's place while the search ran

        await viewModel.performSearch(query: "refund", trigger: "test")
        try await Task.sleep(for: .milliseconds(100))

        XCTAssertEqual(shown(viewModel), [1, 2, 3, 4])
        XCTAssertEqual(feed.count, 0, "nothing is checked for a list nobody can see")
        XCTAssertFalse(viewModel.isCheckingResults)

        viewModel.resumeResultCheckIfNeeded()
        try await waitUntil { feed.count == 1 }
        XCTAssertEqual(feed.askedAbout[0], [2, 3, 4])
    }

    func testAViewModelWhoseOverlayWasNeverShownChecksNothing() async throws {
        let feed = Feed()
        let results = page()
        let viewModel = SearchViewModel(
            coordinator: AppCoordinator(),
            executeSearch: { _ in results },
            resultCheckEvents: { results in feed.stream(for: results) }
        )
        viewModel.searchMode = .relevant
        viewModel.searchQuery = "refund"

        await viewModel.performSearch(query: "refund", trigger: "test")
        try await Task.sleep(for: .milliseconds(100))

        XCTAssertEqual(feed.count, 0)
    }

    func testClosingBeforeTheCheckHasSaidAWordStillGetsItFinishedLater() async throws {
        let feed = Feed()
        let viewModel = makeViewModel(feed, results: page())
        await viewModel.performSearch(query: "refund", trigger: "test")
        try await waitUntil { feed.count == 1 }

        viewModel.cancelSearch()                // closed before `.started` arrived
        viewModel.resumeResultCheckIfNeeded()
        try await waitUntil { feed.count == 2 }

        XCTAssertEqual(feed.askedAbout[1], [2, 3, 4])
    }

    func testAnOpenResultFromAnotherSourceDoesNotProtectTheSameFrameNumber() async throws {
        let feed = Feed()
        let viewModel = makeViewModel(feed, results: page())
        await viewModel.performSearch(query: "refund", trigger: "test")
        try await waitUntil { feed.count == 1 }
        // The user has a Rewind result open that happens to have frame number 3.
        viewModel.selectedResult = SearchResult(
            id: FrameID(value: 3),
            timestamp: Date(timeIntervalSince1970: 1_600_000_000),
            snippet: "",
            matchedText: "refund",
            relevanceScore: 0.5,
            metadata: .empty,
            segmentID: AppSegmentID(value: 3),
            frameIndex: 0,
            source: .rewind
        )

        feed.send(.started(asked: 3))
        feed.send(.decided(frameID: 3, confidence: 0.0, isKept: false))
        try await waitUntil { self.shown(viewModel) == [1, 2, 4] }
    }

    func testAPageLoadedWhileTheCheckRunsDoesNotBringBackWhatItTookOut() async throws {
        /// The first page at once; the next page only when the test lets it through.
        actor Pages {
            private var release: CheckedContinuation<Void, Never>?
            private var waiting = false
            let first: SearchResults, second: SearchResults
            init(first: SearchResults, second: SearchResults) { self.first = first; self.second = second }
            var isWaiting: Bool { waiting }
            func next(for query: SearchQuery) async -> SearchResults {
                guard query.cursor != nil else { return first }
                waiting = true
                await withCheckedContinuation { release = $0 }
                return second
            }
            func letThrough() { release?.resume(); release = nil }
        }
        let cursor = SearchPageCursor(native: SearchSourceCursor(timestamp: Date(timeIntervalSince1970: 1_700_000_000), frameID: 4))
        let firstPage = SearchResults(query: page().query, results: page().results, searchTimeMs: 1, nextCursor: cursor)
        let secondPage = SearchResults(query: page().query, results: [result(9, byMeaningAlone: false)], searchTimeMs: 1)
        let pages = Pages(first: firstPage, second: secondPage)
        let feed = Feed()
        let viewModel = SearchViewModel(
            coordinator: AppCoordinator(),
            executeSearch: { query in await pages.next(for: query) },
            resultCheckEvents: { results in feed.stream(for: results) }
        )
        viewModel.searchMode = .relevant
        viewModel.searchQuery = "refund"
        viewModel.resumeResultCheckIfNeeded()   // the overlay is on screen
        await viewModel.performSearch(query: "refund", trigger: "test")
        try await waitUntil { feed.count == 1 }

        viewModel.loadMore()
        for _ in 0..<200 where await !pages.isWaiting { try await Task.sleep(for: .milliseconds(10)) }
        let pageIsBeingFetched = await pages.isWaiting
        XCTAssertTrue(pageIsBeingFetched)
        feed.send(.started(asked: 3))
        feed.send(.decided(frameID: 3, confidence: 0.1, isKept: false))
        try await waitUntil { self.shown(viewModel) == [1, 2, 4] }
        await pages.letThrough()

        try await waitUntil { self.shown(viewModel).contains(9) }
        XCTAssertEqual(shown(viewModel), [1, 2, 4, 9], "the page is added to the list as it is now")
    }

    func testTheKeyboardHighlightFollowsItsResultWhenOneAboveItIsTakenOut() {
        // Highlight on "c" (third); "a" is taken out above it.
        XCTAssertEqual(
            SearchViewModel.keyboardSelectionIndex(previousIndex: 2, previousKey: "c", resultKeys: ["b", "c", "d"]), 1
        )
        // Taken out below it: nothing moves.
        XCTAssertEqual(
            SearchViewModel.keyboardSelectionIndex(previousIndex: 1, previousKey: "b", resultKeys: ["a", "b", "d"]), 1
        )
        // Its own result is taken out: the one that took its place is highlighted...
        XCTAssertEqual(
            SearchViewModel.keyboardSelectionIndex(previousIndex: 1, previousKey: "b", resultKeys: ["a", "c", "d"]), 1
        )
        // ...or the last one, if it was at the end.
        XCTAssertEqual(
            SearchViewModel.keyboardSelectionIndex(previousIndex: 2, previousKey: "c", resultKeys: ["a", "b"]), 1
        )
        // A page added below changes nothing; no highlight yet means the first; no results means none.
        XCTAssertEqual(
            SearchViewModel.keyboardSelectionIndex(previousIndex: 1, previousKey: "b", resultKeys: ["a", "b", "c", "d", "e"]), 1
        )
        XCTAssertEqual(SearchViewModel.keyboardSelectionIndex(previousIndex: nil, previousKey: nil, resultKeys: ["a", "b"]), 0)
        XCTAssertNil(SearchViewModel.keyboardSelectionIndex(previousIndex: 1, previousKey: "b", resultKeys: []))
    }

    func testWithNothingFoundByMeaningAloneThereIsNoCheck() async throws {
        let feed = Feed()
        let viewModel = makeViewModel(feed, results: page([1]))

        await viewModel.performSearch(query: "refund", trigger: "test")
        try await Task.sleep(for: .milliseconds(100))

        XCTAssertEqual(feed.count, 0)
        XCTAssertFalse(viewModel.isCheckingResults)
    }

    func testACheckWithNothingToDoLeavesTheResultsAlone() async throws {
        let feed = Feed()
        let viewModel = makeViewModel(feed, results: page())
        await viewModel.performSearch(query: "refund", trigger: "test")
        try await waitUntil { feed.count == 1 }

        feed.finish()   // switched off, or no model: the stream ends without a word
        try await Task.sleep(for: .milliseconds(100))

        XCTAssertFalse(viewModel.isCheckingResults)
        XCTAssertEqual(shown(viewModel), [1, 2, 3, 4])
    }
}
