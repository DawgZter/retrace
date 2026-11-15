import CoreGraphics
import Foundation
import Shared

/// Search by meaning over frames' OCR text and, optionally, their screenshots.
///
/// While search is in use it holds every stored vector in memory (one `VectorIndex` per kind), embeds a query
/// with the same model that embedded the frames, and returns the frames whose meaning stands out. Text and image
/// vectors share one space, so a single query vector is compared with both, and a frame that has both is ranked
/// by a weighted mean of how far each stands out. The vectors are read from the
/// database when the search box opens and dropped again after `indexIdleTimeout` without a search, so they cost
/// memory only around searches (512 bytes a vector: about 50 MB per 100,000 frames of one kind).
///
/// Vectors are written by `SemanticIndexer` through `embed(sources:kind:)`, and `didStore(_:)` keeps the
/// in-memory indexes current.
///
/// Model: EmbeddingGemma 2, run by `RetraceEmbeddingHelper`. Prompts follow the model card: queries as
/// `task: search result | query: …`, documents as `title: <window title> | text: <OCR text>`; images take no prompt.
/// Owner: SEARCH agent
public actor SemanticSearchService: SemanticSearchProtocol {
    /// Fewer characters of OCR text than this say too little to embed.
    static let minimumTextLength = 24
    /// OCR text past this is cut before embedding (the model reads about 2,000 tokens of one input).
    static let maximumTextLength = 8000
    /// Texts sent to the helper per request.
    static let textBatchSize = 16
    /// Screenshots sent to the helper per request.
    static let imageBatchSize = 4
    /// How far a frame must stand out (standard deviations above the mean of all frames) to count as a match.
    /// This puts every query on one scale; it does not decide whether anything truly matches. About 2% of
    /// frames clear it for any query, so callers must also cap how many they take.
    static let defaultGate = 2.0
    /// The share of a frame's standing that comes from its text when it also has a screenshot vector.
    /// Measured over 26,000 frames. On 32 searches for an activity, where many frames are right, an even split
    /// put more right frames in the first ten than 0.7 did (68% against 64%). On 70 searches with one right
    /// frame it put that frame first as often (39 times each). Leaning further towards the screenshot added a
    /// point or two to the first and began to cost the second (34 at 0.2).
    static let textWeight = 0.5
    /// The text model is kept running this long after a search, for the next one.
    public static let searchWarmInterval: TimeInterval = 2 * 60
    /// The vectors are dropped from memory this long after the last search.
    public static let indexIdleTimeout: TimeInterval = 10 * 60

    private let store: any EmbeddingStoreProtocol
    private let textEmbedder: any TextEmbedding
    private let imageEmbedder: (any ImageEmbedding)?
    private let imageProvider: (any FrameImageProviding)?
    private let model: @Sendable (EmbeddingKind) -> String
    private let dimension: Int
    private let imageTokenBudget: Int
    private let isEnabled: @Sendable () -> Bool

    private var indexes: [EmbeddingKind: VectorIndex] = [:]
    private var indexLoaded = false
    private var indexLoadTask: Task<Void, Error>?
    /// Goes up every time the vectors are dropped, so a read that was under way knows its result is unwanted.
    private var indexGeneration = 0
    private var lastSearchAt: Date?
    /// Records stored while the index was being read from the database; applied once the read finishes.
    private var storedDuringLoad: [FrameEmbeddingRecord] = []

    public init(
        store: any EmbeddingStoreProtocol,
        embedder: any TextEmbedding,
        imageEmbedder: (any ImageEmbedding)? = nil,
        imageProvider: (any FrameImageProviding)? = nil,
        model: @escaping @Sendable (EmbeddingKind) -> String = { SemanticSearchSettings.modelIdentifier(for: $0) },
        dimension: Int = SemanticSearchSettings.dimension,
        imageTokenBudget: Int = SemanticSearchSettings.imageTokenBudget,
        isEnabled: @escaping @Sendable () -> Bool = { SemanticSearchSettings.isEnabled }
    ) {
        self.store = store
        self.textEmbedder = embedder
        self.imageEmbedder = imageEmbedder
        self.imageProvider = imageProvider
        self.model = model
        self.dimension = dimension
        self.imageTokenBudget = imageTokenBudget
        self.isEnabled = isEnabled
        for kind in EmbeddingKind.allCases {
            indexes[kind] = VectorIndex(dimension: dimension)
        }
    }

    /// Whether this service can embed screenshots at all (it was given an image encoder and a way to read frames).
    public nonisolated var canEmbedImages: Bool {
        imageEmbedder != nil && imageProvider != nil
    }

    // MARK: - SemanticSearchProtocol

    public func matches(for query: String, limit: Int) async -> [SemanticMatch] {
        await matches(for: query, limit: limit, gate: Self.defaultGate)
    }

    func matches(for query: String, limit: Int, gate: Double) async -> [SemanticMatch] {
        guard isEnabled() else { return [] }
        let text = Self.queryText(query)
        guard !text.isEmpty, limit > 0 else { return [] }

        let started = Date()
        lastSearchAt = started
        do {
            try await loadIndexIfNeeded()
            guard indexes.values.contains(where: { $0.count > 0 }) else { return [] }
            guard let vector = try await textEmbedder.embed([Self.queryPrompt(text)]).first,
                  let queryVector = EmbeddingVector.truncated(vector, to: dimension) else {
                return []
            }
            // Other calls run on this actor whenever this one waits (another search, the indexer, an unload),
            // and any of them may drop or reload the vectors. So the vectors are made sure of again after the
            // query has been embedded, and a ranking is only kept if the same set of vectors was in place from
            // its first step to its last.
            var matches: [SemanticMatch] = []
            var rereadBecauseStale = false
            for _ in 0..<4 {
                // The feature may have been switched off while this search was waiting: then the vectors are
                // not read again and nothing is returned.
                guard isEnabled() else { return [] }
                try await loadIndexIfNeeded()
                let generation = indexGeneration
                let ranked = try await rankedMatches(for: queryVector, limit: limit, gate: gate)
                matches = ranked.matches
                if generation != indexGeneration {
                    continue   // the vectors were dropped under this ranking: rank again on the current ones
                }
                if ranked.staleVectors > 0, !rereadBecauseStale {
                    // What is in memory has fallen behind the database (frames were deleted or redacted since
                    // it was read). The best matches may all have been among the stale ones, with good ones
                    // left below the cutoff: read the vectors again and rank once more.
                    rereadBecauseStale = true
                    dropIndex()
                    continue
                }
                break
            }
            let elapsedMs = Date().timeIntervalSince(started) * 1000
            Log.recordLatency(
                "search.semantic.match_ms",
                valueMs: elapsedMs,
                category: .search,
                summaryEvery: 10,
                warningThresholdMs: 250
            )
            Log.debug("[Semantic] \(matches.count) matches over \(vectorCounts) in \(Int(elapsedMs)) ms", category: .search)
            return isEnabled() ? Array(matches) : []
        } catch is CancellationError {
            return []   // the caller stopped waiting, or the vectors were dropped: nothing went wrong
        } catch {
            Log.warning("[Semantic] Search unavailable: \(error)", category: .search)
            return []
        }
    }

    /// Scores every vector for `queryVector` and returns the frames that stand out, best first, together with
    /// how many of the best turned out to have no vector in the database any more (those are left out).
    private func rankedMatches(
        for queryVector: [Float],
        limit: Int,
        gate: Double
    ) async throws -> (matches: [SemanticMatch], staleVectors: Int) {
        // Each kind is put on its own scale (image vectors sit at a different level than text vectors for
        // the same query). Everything from here to the fused list happens without suspending, so the scans
        // describe the indexes as they are.
        var scans: [EmbeddingKind: VectorIndex.Scan] = [:]
        var candidates = Set<Int64>()
        let kinds = EmbeddingKind.allCases.filter { (indexes[$0]?.count ?? 0) > 0 }
        for kind in kinds {
            guard let index = indexes[kind] else { continue }
            let scan = index.scan(for: queryVector)
            scans[kind] = scan
            // A frame is a match when either of its vectors stands out, so turning screenshots on never
            // hides a frame its text alone would have found. Where it ranks is decided below.
            if kinds.count == 1 {
                // One kind: its own order is the final order, so its best `limit` are all that can be returned.
                candidates.formUnion(index.matches(in: scan, limit: limit, gate: gate, kind: kind).map(\.frameID))
            } else {
                // Two kinds: a frame's place depends on both scores, so every frame that stands out in either
                // is ranked before the list is cut. Cutting each kind's list first would lose a frame that
                // is just below the cut in both and, combined, above everything kept.
                candidates.formUnion(index.frames(standingOutIn: scan, gate: gate))
            }
        }
        var fused: [(match: SemanticMatch, scored: [EmbeddingKind: UInt32])] = []
        for frameID in candidates {
            let text = scans[.text].flatMap { indexes[.text]?.standing(of: frameID, in: $0) }
            let image = scans[.image].flatMap { indexes[.image]?.standing(of: frameID, in: $0) }
            let score: Double
            switch (text, image) {
            case let (text?, image?): score = Self.textWeight * text.score + (1 - Self.textWeight) * image.score
            case let (text?, nil): score = text.score
            case let (nil, image?): score = image.score
            case (nil, nil): continue
            }
            let imageLeads = (image?.score ?? -.infinity) > (text?.score ?? -.infinity)
            fused.append((
                SemanticMatch(
                    frameID: frameID,
                    score: score,
                    cosine: max(text?.cosine ?? -1, image?.cosine ?? -1),
                    kind: imageLeads ? .image : .text
                ),
                [EmbeddingKind.text: text?.version, .image: image?.version].compactMapValues { $0 }
            ))
        }
        fused.sort {
            $0.match.score > $1.match.score || ($0.match.score == $1.match.score && $0.match.frameID > $1.match.frameID)
        }
        let best = Array(fused.prefix(limit))
        let matches = try await stillStored(best)
        return (matches, best.count - matches.count)
    }

    /// Drops matches scored on a vector that is no longer the frame's vector, and forgets vectors the database
    /// no longer has.
    ///
    /// The database deletes a frame's vectors the moment its text is redacted, reprocessed or deleted (triggers);
    /// the copies in memory are only as fresh as the last write this process made. Two things are checked for
    /// each match, so that a vector made from removed text never produces a result: the database still has a
    /// vector for the frame, and the vector in memory is still the one that was scored (the indexer may have
    /// written a new one, from new text, while the database was being asked).
    private func stillStored(
        _ fused: [(match: SemanticMatch, scored: [EmbeddingKind: UInt32])]
    ) async throws -> [SemanticMatch] {
        var stale = Set<Int64>()
        for kind in EmbeddingKind.allCases {
            let frameIDs = fused.filter { $0.scored[kind] != nil }.map(\.match.frameID)
            guard !frameIDs.isEmpty else { continue }
            let stored = try await store.framesWithEmbedding(kind: kind, model: model(kind), among: frameIDs)
            for frameID in frameIDs where !stored.contains(frameID) {
                indexes[kind]?.remove(frameID: frameID)
                stale.insert(frameID)
            }
        }
        // Checked last, after every wait: nothing can change the indexes between here and the return.
        for (match, scored) in fused {
            for (kind, version) in scored where indexes[kind]?.version(of: match.frameID) != version {
                stale.insert(match.frameID)
            }
        }
        return fused.map(\.match).filter { !stale.contains($0.frameID) }
    }

    /// Loads the vectors and the model ahead of a search (the search box just opened), so the first query
    /// doesn't pay for either.
    public func prepareForSearch() async {
        guard isEnabled() else { return }
        lastSearchAt = Date()
        do {
            try await loadIndexIfNeeded()
            guard indexes.values.contains(where: { $0.count > 0 }) else { return }
            _ = try await textEmbedder.embed([Self.queryPrompt("warm up")])
        } catch {
            Log.warning("[Semantic] Couldn't prepare for search: \(error)", category: .search)
        }
    }

    // MARK: - Indexing

    /// Embeds frames for `kind`. Every source gets a record: a vector, a marker that there is nothing to embed
    /// (too little text, a frame the user deleted, a failed redaction), or for a screenshot that isn't final or
    /// couldn't be read, a marker to try again later. Call `didStore(_:)` once the records are written.
    public func embed(sources: [FrameEmbeddingSource], kind: EmbeddingKind) async throws -> [FrameEmbeddingRecord] {
        switch kind {
        case .text: return try await embedText(sources)
        case .image: return try await embedImages(sources)
        }
    }

    private func embedText(_ sources: [FrameEmbeddingSource]) async throws -> [FrameEmbeddingRecord] {
        var prompts: [String] = []
        var promptIndex: [String: Int] = [:]
        var promptOfSource: [Int?] = []
        for source in sources {
            guard !source.isMarkedForDeletion,
                  let prompt = Self.documentPrompt(windowName: source.windowName, text: source.text) else {
                promptOfSource.append(nil)
                continue
            }
            // Consecutive frames often carry identical text: embed each distinct prompt once.
            if let existing = promptIndex[prompt] {
                promptOfSource.append(existing)
            } else {
                promptIndex[prompt] = prompts.count
                promptOfSource.append(prompts.count)
                prompts.append(prompt)
            }
        }

        var vectors: [Data?] = []
        vectors.reserveCapacity(prompts.count)
        var start = 0
        while start < prompts.count {
            try Task.checkCancellation()   // a stop is noticed between requests, not only after the batch
            let end = min(start + Self.textBatchSize, prompts.count)
            let batch = try await textEmbedder.embed(Array(prompts[start..<end]))
            guard batch.count == end - start else {
                throw EmbeddingHelperError.requestFailed("got \(batch.count) vectors for \(end - start) texts")
            }
            vectors.append(contentsOf: batch.map(storedVector))
            start = end
        }

        return sources.enumerated().map { offset, source in
            let vector = promptOfSource[offset].flatMap { vectors[$0] } ?? Data()
            return FrameEmbeddingRecord(frameID: source.frameID, kind: .text, docID: source.docID, vector: vector)
        }
    }

    private func embedImages(_ sources: [FrameEmbeddingSource]) async throws -> [FrameEmbeddingRecord] {
        guard let imageEmbedder, let imageProvider else {
            throw EmbeddingHelperError.requestFailed("this build can't embed screenshots")
        }

        // Read and size each screenshot. Nothing cached from an earlier batch is reused: see
        // `FrameImageProviding.discardCachedFrames`.
        enum Outcome {
            case image(Int)
            case nothingToEmbed
            case retryLater
        }
        imageProvider.discardCachedFrames()
        // ...and nothing is left cached when this batch ends, however it ends: there may be no next batch
        // (indexing finished, or was switched off), and open video decoders are memory.
        defer { imageProvider.discardCachedFrames() }
        var images: [EmbeddingImage] = []
        var outcomes: [Outcome] = []
        for source in sources {
            if source.isMarkedForDeletion || source.imageState == .unavailable {
                outcomes.append(.nothingToEmbed)
                continue
            }
            guard source.imageState == .ready else {
                outcomes.append(.retryLater)   // still being encoded, processed or redacted
                continue
            }
            do {
                let frame = try await imageProvider.frameImage(for: source)
                guard let image = ImageSizing.embeddingImage(from: frame, budget: imageTokenBudget) else {
                    outcomes.append(.nothingToEmbed)
                    continue
                }
                outcomes.append(.image(images.count))
                images.append(image)
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                // Often temporary: the video is being finalized or rewritten.
                Log.debug("[Semantic] No screenshot for frame \(source.frameID) yet: \(error)", category: .search)
                outcomes.append(.retryLater)
            }
        }

        var vectors: [Data?] = []
        vectors.reserveCapacity(images.count)
        var start = 0
        while start < images.count {
            try Task.checkCancellation()
            let end = min(start + Self.imageBatchSize, images.count)
            let batch = try await imageEmbedder.embed(images: Array(images[start..<end]))
            guard batch.count == end - start else {
                throw EmbeddingHelperError.requestFailed("got \(batch.count) vectors for \(end - start) images")
            }
            vectors.append(contentsOf: batch.map(storedVector))
            start = end
        }

        return zip(sources, outcomes).map { source, outcome in
            switch outcome {
            case .image(let index):
                return FrameEmbeddingRecord(
                    frameID: source.frameID,
                    kind: .image,
                    docID: source.docID,
                    vector: vectors[index] ?? Data()
                )
            case .nothingToEmbed:
                return FrameEmbeddingRecord(frameID: source.frameID, kind: .image, docID: source.docID, state: .nothingToEmbed)
            case .retryLater:
                return FrameEmbeddingRecord(frameID: source.frameID, kind: .image, docID: source.docID, state: .retryLater)
            }
        }
    }

    private func storedVector(_ vector: [Float]) -> Data? {
        EmbeddingVector.truncated(vector, to: dimension).map(EmbeddingVector.storageData)
    }

    /// Call just before records are written. The frames' vectors in memory are forgotten, so a search that has
    /// already scored one of them notices when it checks that the vector is still current (see `stillStored`).
    public func willStore(_ records: [FrameEmbeddingRecord]) {
        if indexLoaded {
            for record in records {
                indexes[record.kind]?.remove(frameID: record.frameID)
            }
        } else if indexLoadTask != nil {
            // The vectors are being read right now and the old ones for these frames may already have been
            // read: have them dropped when the read finishes (what `didStore` reports comes after, in order).
            storedDuringLoad.append(contentsOf: records.map {
                FrameEmbeddingRecord(frameID: $0.frameID, kind: $0.kind, docID: $0.docID, state: .nothingToEmbed)
            })
        }
    }

    /// Call with the records the store wrote, so searches see them without reloading. A marker means the frame
    /// has no vector now, so any it had is forgotten.
    public func didStore(_ records: [FrameEmbeddingRecord]) {
        if indexLoaded {
            apply(records)
        } else if indexLoadTask != nil {
            // The index is being read right now, possibly already past these frames: apply them when it's done.
            storedDuringLoad.append(contentsOf: records)
        }
        // Not loaded and not loading: the first search reads everything from the database.
    }

    private func apply(_ records: [FrameEmbeddingRecord]) {
        for record in records {
            if record.state == .vector, !record.vector.isEmpty {
                indexes[record.kind]?.upsert(frameID: record.frameID, vector: record.vector)
            } else {
                indexes[record.kind]?.remove(frameID: record.frameID)
            }
        }
    }

    /// Frees the vectors and stops the models (semantic search was turned off, or the index was removed).
    public func unload() {
        dropIndex()
        textEmbedder.shutdown()
        imageEmbedder?.shutdown()
    }

    /// Frees the vectors once nobody has searched for `indexIdleTimeout`. The next search reads them again.
    /// The indexer calls this between batches.
    public func releaseIndexIfIdle(now: Date = Date(), after timeout: TimeInterval = indexIdleTimeout) {
        guard indexLoaded, indexLoadTask == nil else { return }
        if let lastSearchAt, now.timeIntervalSince(lastSearchAt) < timeout { return }
        dropIndex()
        Log.debug("[Semantic] Released the in-memory vectors after \(Int(timeout)) s without a search", category: .search)
    }

    private func dropIndex() {
        guard indexLoaded || indexLoadTask != nil else { return }
        indexGeneration += 1
        indexLoadTask?.cancel()
        indexLoadTask = nil
        for index in indexes.values { index.removeAll() }
        indexLoaded = false
        storedDuringLoad.removeAll()
    }

    /// Stops the image encoder (about 1 GB) without touching the text model or the vectors.
    public func releaseImageEmbedder() {
        imageEmbedder?.shutdown()
    }

    /// Stops both models, unless someone searched within `searchWarmInterval` (the text model then stays for the
    /// next query). The indexer calls this whenever it has nothing to embed, so the models' memory (0.5 GB for
    /// text, 1 GB with the image encoder) is held only while there is work, not for as long as Retrace records.
    public func releaseModelsUnlessSearching(now: Date = Date()) {
        imageEmbedder?.shutdown()
        if let lastSearchAt, now.timeIntervalSince(lastSearchAt) < Self.searchWarmInterval { return }
        textEmbedder.shutdown()
    }

    func loadedVectorCount(kind: EmbeddingKind) -> Int {
        indexes[kind]?.count ?? 0
    }

    private var vectorCounts: String {
        EmbeddingKind.allCases
            .map { "\(indexes[$0]?.count ?? 0) \($0 == .text ? "text" : "image") vectors" }
            .joined(separator: " and ")
    }

    // MARK: - Index loading

    /// Reads every stored vector, on the first search after launch or after the vectors were released.
    private func loadIndexIfNeeded() async throws {
        if indexLoaded { return }
        if let indexLoadTask {
            try await indexLoadTask.value
            return
        }
        let generation = indexGeneration
        let task = Task { try await self.loadIndex(generation: generation) }
        indexLoadTask = task
        defer {
            // Only tidy up after this read: the vectors may have been dropped, and another read started, since.
            if indexGeneration == generation {
                indexLoadTask = nil
                if !indexLoaded { storedDuringLoad.removeAll() }   // a failed read starts over next time
            }
        }
        try await task.value
    }

    /// Reads into fresh indexes and swaps them in at the end, unless the vectors were dropped meanwhile.
    private func loadIndex(generation: Int) async throws {
        let started = Date()
        var loaded: [EmbeddingKind: VectorIndex] = [:]
        for kind in EmbeddingKind.allCases {
            let index = VectorIndex(dimension: dimension)
            // The count includes the few frames with nothing to embed: close enough to size the storage once.
            index.reserve(rows: try await store.frameEmbeddingCount(kind: kind, model: model(kind)))
            var lastFrameID: Int64 = -1
            while true {
                let page = try await store.frameEmbeddings(
                    kind: kind,
                    model: model(kind),
                    afterFrameID: lastFrameID,
                    limit: 5000
                )
                guard generation == indexGeneration else { throw CancellationError() }
                guard let last = page.last else { break }
                for record in page {
                    index.upsert(frameID: record.frameID, vector: record.vector)
                }
                lastFrameID = last.frameID
            }
            loaded[kind] = index
        }
        indexes = loaded
        // Vectors written while the pages above were being read (history is embedded oldest-last, so new
        // rows often land behind the read position).
        apply(storedDuringLoad)
        storedDuringLoad.removeAll()
        indexLoaded = true
        let elapsedMs = Date().timeIntervalSince(started) * 1000
        Log.recordLatency("search.semantic.index_load_ms", valueMs: elapsedMs, category: .search, summaryEvery: 1)
        Log.info("[Semantic] Loaded \(vectorCounts) in \(Int(elapsedMs)) ms", category: .search)
    }

    // MARK: - Prompts

    static func queryPrompt(_ text: String) -> String {
        "task: search result | query: \(text)"
    }

    /// Nil when there is too little text to say anything.
    static func documentPrompt(windowName: String?, text: String) -> String? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.count >= minimumTextLength else { return nil }
        let title = windowName?.trimmingCharacters(in: .whitespacesAndNewlines)
        return "title: \((title?.isEmpty ?? true) ? "none" : title!) | text: \(trimmed.prefix(maximumTextLength))"
    }

    /// The words of a search box query that describe what to find: terms and quoted phrases, without `-excluded`
    /// words or phrases and without `app:` style filters. Uses the keyword search's own tokenizer, so both
    /// searches agree on what was excluded.
    static func queryText(_ query: String) -> String {
        QueryTokenizer().tokenize(query)
            .compactMap { token -> String? in
                switch token.kind {
                case .excludedTerm, .excludedPhrase:
                    return nil
                case .phrase, .ignoredShellOption:
                    return token.text
                case .term:
                    let lowered = token.text.lowercased()
                    if ["app:", "after:", "before:"].contains(where: { lowered.hasPrefix($0) && lowered.count > $0.count }) {
                        return nil
                    }
                    return token.text
                }
            }
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .joined(separator: " ")
    }
}
