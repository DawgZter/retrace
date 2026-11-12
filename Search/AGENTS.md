# SEARCH Agent Instructions

You are responsible for the **Search** module of Retrace. Your job is to implement search functionality including query parsing, full-text search via SQLite FTS5, result ranking, and semantic search.

**Status**: ✅ Full-text search with FTS5 fully implemented. Query parser supports filters (app:, date:, -exclude). Semantic search over OCR text and, optionally, screenshots is implemented in `Search/Semantic/` and is off by default (see "Semantic Search" below).

## Your Directory

```
Search/
├── SearchManager.swift            # Main SearchProtocol implementation
├── IngestionManager.swift         # Search index ingestion
├── QueryParser/
│   ├── QueryParser.swift          # QueryParserProtocol implementation
│   └── QueryTokenizer.swift       # Shared query tokenization + shell-option classification
├── Ranking/
│   └── ResultRanker.swift         # Rank and sort results
├── Semantic/                      # Semantic search over OCR text and screenshots (off by default)
│   ├── EmbeddingHelperClient.swift  # Runs RetraceEmbeddingHelper as a child process (one JSON line each way);
│   │                                # starts on the first request, stops when idle; locates the helper in the build
│   ├── VectorIndex.swift            # One kind's vectors in memory as float16 rows, scored by brute force;
│   │                                # a match must stand out from all rows (z-score gate). Also vector truncation
│   │                                # and the float16 storage format (EmbeddingVector)
│   ├── ImageSizing.swift            # Screenshot size for a soft-token budget; CGImage → packed RGB at that size
│   ├── SemanticSearchService.swift  # SemanticSearchProtocol: query and document prompts, text and screenshot
│   │                                # embedding batches, loading stored vectors, matches across both kinds
│   └── SemanticIndexer.swift        # Background loop: new text, new screenshots, then history, newest first;
│                                    # pauses in Low Power Mode and under thermal pressure, history and screenshots
│                                    # only on AC power; screenshots only once a frame's pixels are final, retried
│                                    # until then
├── VectorSearchTODO/              # Earlier Nomic-based prototype, superseded by Semantic/ (excluded from build)
│   ├── Embedding/
│   ├── VectorStore/
│   ├── Embedding.swift
│   └── HybridSearchManager.swift
└── Tests/
    ├── SemanticSearchTests.swift  # Text and screenshot indexing, service, VectorIndex, ImageSizing (real SQLite)
    ├── EmbeddingHelperClientTests.swift # The client against a stand-in helper that dies, hangs or misbehaves
    ├── QueryParserTests.swift
    └── TestLogger.swift
```

## Protocols You Must Implement

### 1. `SearchProtocol` (from `Shared/Protocols/SearchProtocol.swift`)
- Full-text search via FTS5
- Indexing text content
- Search statistics

### 2. `QueryParserProtocol` (from `Shared/Protocols/SearchProtocol.swift`)
- Parse query syntax
- Extract filters (app:, date:, -exclude)

### 3. `SemanticSearchProtocol` (from `Shared/Protocols/SemanticSearchProtocol.swift`)
- `matches(for:limit:)`: frames whose meaning stands out for a query, best first
- Never throws and returns nothing when semantic search is off or the model is missing, so keyword search keeps working
- Implemented by `SemanticSearchService`

The same file defines `EmbeddingStoreProtocol` (the `frame_embedding` table, implemented by the DATABASE module), `FrameImageProviding` (reads a frame's screenshot, implemented in `App/ServiceContainer.swift`), `SemanticSearchSettings`, `SemanticMatch` and `SemanticIndexStatus`.

## Key Implementation Details

### 1. Query Parser

Support rich query syntax:
- `error message` - Basic keyword search
- `"exact phrase"` - Exact phrase matching
- `-excluded` - Exclude term
- `app:Chrome` - Filter by app
- `after:2024-01-01` - Date filters

```swift
struct QueryParser: QueryParserProtocol {
    func parse(rawQuery: String) throws -> ParsedQuery {
        var searchTerms: [String] = []
        var phrases: [String] = []
        var excludedTerms: [String] = []
        var appFilter: String? = nil
        var startDate: Date? = nil
        var endDate: Date? = nil

        // Tokenize preserving quotes
        let tokens = tokenize(rawQuery)

        for token in tokens {
            if token.hasPrefix("\"") && token.hasSuffix("\"") {
                // Exact phrase
                let phrase = String(token.dropFirst().dropLast())
                phrases.append(phrase)
            } else if token.hasPrefix("-") {
                // Excluded term
                excludedTerms.append(String(token.dropFirst()))
            } else if token.lowercased().hasPrefix("app:") {
                // App filter
                appFilter = String(token.dropFirst(4))
            } else if token.lowercased().hasPrefix("after:") {
                // Start date
                let dateStr = String(token.dropFirst(6))
                startDate = parseDate(dateStr)
            } else if token.lowercased().hasPrefix("before:") {
                // End date
                let dateStr = String(token.dropFirst(7))
                endDate = parseDate(dateStr)
            } else {
                // Regular search term
                searchTerms.append(token)
            }
        }

        return ParsedQuery(
            searchTerms: searchTerms,
            phrases: phrases,
            excludedTerms: excludedTerms,
            appFilter: appFilter,
            dateRange: (startDate, endDate)
        )
    }

    private func tokenize(_ query: String) -> [String] {
        var tokens: [String] = []
        var current = ""
        var inQuotes = false

        for char in query {
            if char == "\"" {
                inQuotes.toggle()
                current.append(char)
            } else if char == " " && !inQuotes {
                if !current.isEmpty {
                    tokens.append(current)
                    current = ""
                }
            } else {
                current.append(char)
            }
        }

        if !current.isEmpty {
            tokens.append(current)
        }

        return tokens
    }

    private func parseDate(_ str: String) -> Date? {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd"
        return formatter.date(from: str)
    }
}
```

### 2. FTS Query Builder

Convert parsed query to SQLite FTS5 syntax:

```swift
extension ParsedQuery {
    func toFTSQuery() -> String {
        var parts: [String] = []

        // Regular terms (with prefix matching)
        for term in searchTerms {
            parts.append("\(term)*")  // Prefix match
        }

        // Exact phrases
        for phrase in phrases {
            parts.append("\"\(phrase)\"")
        }

        // Excluded terms
        for term in excludedTerms {
            parts.append("NOT \(term)")
        }

        return parts.joined(separator: " ")
    }
}
```

### 3. Search Manager

```swift
public actor SearchManager: SearchProtocol {
    private let database: any FTSProtocol
    private let queryParser: QueryParser
    private let resultRanker: ResultRanker
    private var config: SearchConfig

    public func search(query: SearchQuery) async throws -> SearchResults {
        let startTime = Date()

        // Parse query
        let parsed = try queryParser.parse(rawQuery: query.text)

        // Build FTS query
        let ftsQuery = parsed.toFTSQuery()

        // Build filters
        var filters = query.filters
        if let appFilter = parsed.appFilter {
            filters = SearchFilters(
                startDate: filters.startDate ?? parsed.dateRange.start,
                endDate: filters.endDate ?? parsed.dateRange.end,
                appBundleIDs: [appFilter],
                excludedAppBundleIDs: filters.excludedAppBundleIDs
            )
        }

        // Execute FTS search
        let ftsMatches = try await database.search(
            query: ftsQuery,
            filters: filters,
            limit: query.limit,
            offset: query.offset
        )

        // Get total count for pagination
        let totalCount = try await database.getMatchCount(query: ftsQuery, filters: filters)

        // Convert to SearchResults
        let results = ftsMatches.map { match in
            SearchResult(
                id: match.frameID,
                timestamp: match.timestamp,
                snippet: match.snippet,
                matchedText: extractMatchedText(from: match.snippet),
                relevanceScore: normalizeRank(match.rank),
                metadata: FrameMetadata(
                    appName: match.appName,
                    windowTitle: match.windowTitle
                ),
                segmentID: SegmentID(), // Would need to join with frames table
                frameIndex: 0
            )
        }

        // Rank results
        let rankedResults = resultRanker.rank(results, forQuery: query.text)

        let searchTimeMs = Int(Date().timeIntervalSince(startTime) * 1000)

        return SearchResults(
            query: query,
            results: rankedResults,
            totalCount: totalCount,
            searchTimeMs: searchTimeMs
        )
    }

    public func search(text: String, limit: Int) async throws -> SearchResults {
        return try await search(query: SearchQuery(text: text, limit: limit))
    }

    private func normalizeRank(_ bm25Rank: Double) -> Double {
        // BM25 returns negative values (more negative = better match)
        // Normalize to 0-1 range
        return 1.0 / (1.0 + abs(bm25Rank))
    }

    private func extractMatchedText(from snippet: String) -> String {
        // Extract text between <mark> tags
        let pattern = "<mark>(.*?)</mark>"
        guard let regex = try? NSRegularExpression(pattern: pattern),
              let match = regex.firstMatch(in: snippet, range: NSRange(snippet.startIndex..., in: snippet)),
              let range = Range(match.range(at: 1), in: snippet) else {
            return snippet
        }
        return String(snippet[range])
    }
}
```

### 4. Result Ranker

Apply additional ranking on top of FTS:

```swift
struct ResultRanker {
    func rank(_ results: [SearchResult], forQuery query: String) -> [SearchResult] {
        let queryTerms = Set(query.lowercased().split(separator: " ").map(String.init))

        return results.sorted { a, b in
            let scoreA = computeScore(a, queryTerms: queryTerms)
            let scoreB = computeScore(b, queryTerms: queryTerms)
            return scoreA > scoreB
        }
    }

    private func computeScore(_ result: SearchResult, queryTerms: Set<String>) -> Double {
        var score = result.relevanceScore

        // Boost for recency
        let ageInDays = Date().timeIntervalSince(result.timestamp) / 86400
        let recencyBoost = max(0, 1.0 - (ageInDays / 30.0)) * 0.2
        score += recencyBoost

        // Boost if query appears in window title
        if let title = result.metadata.windowTitle?.lowercased() {
            let titleMatches = queryTerms.filter { title.contains($0) }.count
            score += Double(titleMatches) * 0.1
        }

        return score
    }
}
```

### 5. Autocomplete Suggestions

```swift
extension SearchManager {
    public func getSuggestions(prefix: String, limit: Int) async throws -> [String] {
        // Query FTS for terms starting with prefix
        let sql = """
            SELECT DISTINCT term FROM documents_fts_vocab
            WHERE term LIKE ? || '%'
            ORDER BY doc_count DESC
            LIMIT ?
        """
        // Note: This requires creating an FTS vocab table:
        // CREATE VIRTUAL TABLE documents_fts_vocab USING fts5vocab(documents_fts, instance);

        // Execute and return suggestions
        return []  // Implement with actual query
    }
}
```

### 6. Semantic Search

Semantic search finds frames by meaning. It is optional and off by default: the user turns it on in Settings → Advanced → Semantic Search, which downloads the model. The code is in `Search/Semantic/`.

**Model and helper**

- The model is EmbeddingGemma 2 (8-bit GGUF), run on llama.cpp by `RetraceEmbeddingHelper` (`Sources/RetraceEmbeddingHelper/`), a separate process. The Search module does not link llama.cpp.
- `EmbeddingHelperClient` starts the helper on the first request, sends one JSON line per request on stdin and reads one JSON line back, and stops the helper when it has had no request for its idle timeout (120 seconds by default). A helper that dies, hangs past its timeout, or replies with something unusable makes the call throw; it is killed and the next call starts a fresh one. Writes to a dead helper fail with an error (the pipe is set not to raise SIGPIPE).
- `EmbeddingHelperClient.locateHelper()` looks beside the running executable and in the app bundle's `Contents/Helpers` and `Contents/Library/Helpers`. In a build without the helper it returns nil; Settings then disables the switches and says to run `scripts/build_llama_embedding.sh`.
- Text and screenshots use separate clients (created in `App/ServiceContainer.swift`). The client for screenshots also loads the image encoder and has a 60-second idle timeout, so it only runs while there are screenshots to embed.

**Vectors**

- The model returns 768 values. `EmbeddingVector.truncated` keeps the first 256 (`SemanticSearchSettings.dimension`) and normalizes them again. Vectors are stored as float16.
- There are two kinds (`EmbeddingKind`): `text` (OCR text with the window title) and `image` (the screenshot). Both are in one vector space, so one query vector is compared with both.
- Vectors are stored in the `frame_embedding` table through `EmbeddingStoreProtocol`, one row per frame and kind, tagged with `SemanticSearchSettings.modelIdentifier(for:)`. Vectors made by another model identifier are deleted and the frames are embedded again.
- Every frame the indexer has looked at gets a row with a state (`FrameEmbeddingRecord.State`): a vector; "nothing to embed" (fewer than 24 characters of text, a frame the user deleted, a failed redaction), which is not looked at again; or "try again later" (a screenshot that isn't final or readable yet, or a frame that changed while it was being embedded).
- The vectors are held in memory only around searches: they are read when the search box opens and dropped after 10 minutes without a search (`SemanticSearchService.indexIdleTimeout`). A vector is 512 bytes, so 100,000 frames of one kind are about 50 MB while loaded.

**Indexing** (`SemanticIndexer`)

- Runs in the background after OCR has indexed a frame's text, never in the capture or OCR path.
- Walks FTS documents by id, separately for each kind. Order of work: new text, new screenshots, text history, screenshot history. History is walked newest first.
- A screenshot is embedded only when the frame's pixels are final (encoded, OCR finished, any redaction applied). `ImageSizing` scales it for a budget of 280 soft tokens.
- A screenshot that isn't final yet, or whose video can't be read right now, is marked "try again later" and the walk moves on. Those frames are looked at again between batches, and given up on 24 hours after they were first tried (`retryTimeout`).
- Nothing is embedded from a frame the user has deleted, even while it is still waiting to be removed from its video.
- Pauses in Low Power Mode and under serious thermal pressure. History and all screenshot work run only on AC power, with a pause after each batch of history.
- The models are held only while there is work: new frames are embedded in small batches (once 16 are waiting or the oldest was captured a minute ago), and whenever a pass finds nothing to do the indexer stops both models, unless a search was made in the last two minutes. While Retrace records, the helper therefore runs for a second or two about once a minute instead of staying in memory.
- The store writes a record only if the frame still has the document and text it was made from; a screenshot vector also needs the frame not to have been processed or rewritten since its pixels were read. A record that is refused is replaced by a "try again later" marker, so the frame is embedded again from what it holds now.

**Query** (`SemanticSearchService.matches`)

- Strips keyword-search syntax from the query (`queryText`, using `QueryTokenizer`, so excluded words and phrases are the same ones keyword search excludes), embeds it with the query prompt, and scores every stored vector in memory (`VectorIndex`, one per kind).
- A frame's score is how many standard deviations its cosine similarity is above the mean of all frames of that kind. Only frames at or above the gate (2.0) are matches. A frame is a match when either of its vectors clears the gate; a frame with both kinds is ranked by a weighted mean of the two (`SemanticSearchService.textWeight`, 0.7 text). The gate puts every query on one scale; about 2% of frames clear it for any query, so it does not decide whether anything truly matches. That is what `SemanticSearchSettings.strongMatchCosine` (0.69) is for: `DataAdapter` shows at most three matches below it, so a query about something that was never on screen gets a few results, not a page.
- Matches whose vector is no longer in the database are dropped.
- `App/DataAdapter.swift` applies the query's filters to the matches and adds them to the first page of keyword results. Later pages are keyword-only. In Relevance order the two are combined by reciprocal rank fusion; in date order the semantic matches follow the keyword matches. Keyword results wait at most 400 ms for the semantic side (3 s when there are no keyword results).
- `Log.recordLatency` metrics: `search.semantic.match_ms` (embed the query and scan), `search.semantic.index_load_ms`, and in the App module `search.semantic.added_ms` (what the semantic side added once keyword results were ready).

**Superseded code**: `Search/VectorSearchTODO/` is an earlier Nomic-based prototype. It is not used by `Search/Semantic/` and is still excluded from the build in `Package.swift`.

### 7. Indexing Pipeline

This is the FTS path. Embeddings are not made here: `SemanticIndexer` embeds a frame later, in the background, and the database deletes a frame's vectors when the frame or its FTS document is deleted.

```swift
extension SearchManager {
    public func index(text: ExtractedText) async throws {
        // Create indexed document
        let document = IndexedDocument(
            id: 0,  // Will be assigned by DB
            frameID: text.frameID,
            timestamp: text.timestamp,
            content: text.fullText,
            appName: text.metadata.appName,
            windowTitle: text.metadata.windowTitle,
            browserURL: text.metadata.browserURL
        )

        // Insert into database (which updates FTS)
        let docID = try await database.insertDocument(document)
    }

    public func removeFromIndex(frameID: FrameID) async throws {
        // Remove from database
        if let doc = try await database.getDocument(frameID: frameID) {
            try await database.deleteDocument(id: doc.id)
        }
    }
}
```

## Error Handling

Use errors from `Shared/Models/Errors.swift`:
```swift
throw SearchError.invalidQuery(reason: "Empty query")
throw SearchError.indexNotReady
```

The embedding helper's failures are `EmbeddingHelperError` (`Search/Semantic/EmbeddingHelperClient.swift`). `SemanticSearchService.matches` catches them, logs a warning and returns no matches.

## Testing Strategy

1. Test query parsing with various syntax
2. Test FTS query generation
3. Test result ranking
4. Test autocomplete suggestions
5. Test semantic search against a real SQLite database with a stand-in embedding model (`Tests/SemanticSearchTests.swift`)
6. Test indexing and removal
7. Test edge cases (empty queries, special characters)

## Dependencies

- **Input from**: PROCESSING module (ExtractedText to index)
- **Output to**: UI (SearchResults)
- **Uses**: DATABASE module (FTSProtocol for queries, EmbeddingStoreProtocol for semantic search vectors)
- **Uses**: `RetraceEmbeddingHelper` (a separate executable, run as a child process) for semantic search
- **Uses types**: `SearchQuery`, `SearchResults`, `SearchResult`, `ExtractedText`, `ParsedQuery`, `SemanticMatch`, `FrameEmbeddingSource`, `FrameEmbeddingRecord`

## DO NOT

- Modify any files outside `Search/`
- Import from other module directories (only `Shared/`)
- Implement database operations (use DATABASE's FTSProtocol)
- Handle OCR or text extraction (that's PROCESSING's job)

## Performance Targets

- Search: <100ms for typical queries
- Autocomplete: <50ms
- Indexing: <10ms per document
- Semantic search: one scan of all stored vectors per query, linear in the number of frames (see `VectorIndexBenchmarks` in `Tests/SemanticSearchTests.swift` to measure it on your machine, and `FrameEmbeddingLoadBenchmarks` in `Database/Tests` for the time to read the vectors into memory)

## Getting Started

1. Create `Search/QueryParser/QueryParser.swift`
2. Create `Search/Ranking/ResultRanker.swift`
3. Create `Search/SearchManager.swift` conforming to `SearchProtocol`
4. Write tests for query parsing

Start with the query parser and FTS integration, then add ranking. Semantic search is separate from this path: its code is in `Search/Semantic/` (see "Semantic Search" above).
