import XCTest
import Shared
import App
@testable import Retrace

/// Loading more results when a later keyword page repeats frames the first page already showed (as semantic
/// search results do when a frame also contains the words).
@MainActor
final class SearchLoadMoreDeduplicationTests: XCTestCase {
    /// Answers each search with the next page it was given.
    private actor Pages {
        private var pages: [SearchResults]
        private(set) var cursors: [SearchPageCursor?] = []

        init(_ pages: [SearchResults]) {
            self.pages = pages
        }

        func next(for query: SearchQuery) -> SearchResults {
            cursors.append(query.cursor)
            return pages.isEmpty ? SearchResults(query: query, results: [], searchTimeMs: 0) : pages.removeFirst()
        }
    }

    private func cursor(_ frameID: Int64) -> SearchPageCursor {
        SearchPageCursor(native: SearchSourceCursor(timestamp: Date(timeIntervalSince1970: 1_700_000_000), frameID: frameID))
    }

    private func page(_ frameIDs: [Int64], next: SearchPageCursor?) -> SearchResults {
        let query = SearchQuery(text: "refund", limit: 50, mode: .relevant)
        let results = frameIDs.map { frameID in
            SearchResult(
                id: FrameID(value: frameID),
                timestamp: Date(timeIntervalSince1970: Double(1_700_000_000 + frameID)),
                snippet: "",
                matchedText: "refund",
                relevanceScore: 0.5,
                metadata: .empty,
                segmentID: AppSegmentID(value: frameID),
                frameIndex: 0,
                source: .native
            )
        }
        return SearchResults(query: query, results: results, searchTimeMs: 1, nextCursor: next)
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

    private func makeViewModel(_ pages: Pages) -> SearchViewModel {
        let viewModel = SearchViewModel(coordinator: AppCoordinator(), executeSearch: { query in await pages.next(for: query) })
        viewModel.searchMode = .relevant
        viewModel.searchQuery = "refund"
        return viewModel
    }

    func testAFrameAlreadyShownIsNotRepeatedByALaterPage() async throws {
        let pages = Pages([
            page([1, 2, 3], next: cursor(3)),
            page([3, 4], next: nil),   // 3 was on the first page (found by meaning there, by keyword here)
        ])
        let viewModel = makeViewModel(pages)
        await viewModel.performSearch(query: "refund", trigger: "test")

        viewModel.loadMore()
        try await waitUntil { viewModel.results?.results.count == 4 }

        XCTAssertEqual(viewModel.results?.results.map(\.id.value), [1, 2, 3, 4])
        XCTAssertFalse(viewModel.canLoadMore)
    }

    func testAPageOfFramesAlreadyShownDoesNotStopScrolling() async throws {
        let pages = Pages([
            page([1, 2, 3], next: cursor(3)),
            page([2, 3], next: cursor(5)),   // nothing new on this page, and more to come
            page([6, 7], next: nil),
        ])
        let viewModel = makeViewModel(pages)
        await viewModel.performSearch(query: "refund", trigger: "test")

        // One trigger, as the last cards scroll into view. With nothing appended they would never appear again.
        viewModel.loadMore()
        try await waitUntil { viewModel.results?.results.count == 5 }

        XCTAssertEqual(viewModel.results?.results.map(\.id.value), [1, 2, 3, 6, 7])
        let cursors = await pages.cursors
        XCTAssertEqual(cursors, [nil, cursor(3), cursor(5)], "the page after the repeated one was fetched in the same load")
        XCTAssertFalse(viewModel.isLoadingMore)
    }

    func testManyPagesOfFramesAlreadyShownInARowDoNotStopScrolling() async throws {
        // Thirty frames on the first page (as many as meaning can add), then six keyword pages that repeat them.
        let shown = Array(Int64(1)...30)
        var later: [SearchResults] = []
        for index in 0..<6 {
            later.append(page(Array(shown[(index * 5)..<(index * 5 + 5)]), next: cursor(Int64(100 + index))))
        }
        let pages = Pages([page(shown, next: cursor(30))] + later + [page([40, 41], next: nil)])
        let viewModel = makeViewModel(pages)
        await viewModel.performSearch(query: "refund", trigger: "test")

        viewModel.loadMore()
        try await waitUntil { viewModel.results?.results.count == 32 }

        XCTAssertEqual(viewModel.results?.results.suffix(2).map(\.id.value), [40, 41])
        let cursors = await pages.cursors
        XCTAssertEqual(cursors.count, 8, "all six repeated pages were skipped in one load")
        XCTAssertFalse(viewModel.isLoadingMore)
    }

    func testARepeatedPageWhoseCursorDoesNotMoveIsNotFetchedAgainAndAgain() async throws {
        let pages = Pages([
            page([1, 2, 3], next: cursor(3)),
            page([2, 3], next: cursor(3)),   // nothing new, and the same cursor as before
            page([2, 3], next: cursor(3)),
            page([2, 3], next: cursor(3)),
        ])
        let viewModel = makeViewModel(pages)
        await viewModel.performSearch(query: "refund", trigger: "test")

        viewModel.loadMore()
        try await Task.sleep(for: .milliseconds(300), clock: .continuous)
        try await waitUntil { !viewModel.isLoadingMore }

        XCTAssertEqual(viewModel.results?.results.map(\.id.value), [1, 2, 3])
        let cursors = await pages.cursors
        XCTAssertEqual(cursors, [nil, cursor(3)], "one fetch, not a loop")
    }
}
