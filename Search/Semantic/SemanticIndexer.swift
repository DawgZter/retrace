import Foundation
import Shared

/// Keeps the embedding index in step with the FTS index, in the background.
///
/// Frames are embedded after OCR has indexed their text, never in the capture or OCR path. The indexer walks FTS
/// documents by id (they are numbered in the order they were indexed), separately for each kind: first anything
/// newer than the newest embedded document, then history, newest first. Both resume points are read from the
/// database each pass, so quitting mid-way, deleting frames or reprocessing OCR needs no bookkeeping.
///
/// Order of work: new text, new screenshots, text history, screenshot history. Text is cheap (tens of
/// milliseconds a frame) and makes a frame searchable; screenshots cost about ten times as much.
///
/// A screenshot is embedded only when its pixels are final (encoded, OCR finished, redaction applied). A frame
/// that isn't final yet, or whose video can't be read right now, is marked "try again later" and the walk moves
/// on; those frames are looked at again between batches until they can be embedded or `retryTimeout` passes.
///
/// It pauses in Low Power Mode and under thermal pressure. History, and all screenshot work, run only on AC power.
///
/// The models are held only while there is work. New frames are embedded in small batches (about once a minute
/// while the Mac is in use), and whenever a pass finds nothing to do the indexer stops the models, unless a
/// search was made in the last two minutes.
/// Owner: SEARCH agent
public actor SemanticIndexer {
    public struct Configuration: Sendable {
        /// Frames read from the database per text pass.
        public var batchSize = 64
        /// Frames read from the database per screenshot pass.
        public var imageBatchSize = 8
        /// Wait between passes when there is nothing to do, or while paused.
        public var idleInterval: Duration = .seconds(5)
        /// Pause after each batch of history, so the GPU is free part of the time while a backlog is worked
        /// through. New frames are embedded without the pause.
        public var backfillInterval: Duration = .milliseconds(500)
        /// Wait after a failure before trying again.
        public var retryInterval: Duration = .seconds(60)
        /// A frame that still can't be embedded this long after it was first tried is given up on.
        /// A video chunk is only final once it is full, which on a quiet day can take hours.
        public var retryTimeout: TimeInterval = 24 * 60 * 60
        /// ...and only once this loop has itself seen it waiting for this long. The first limit is by the clock;
        /// without the second, frames marked before the Mac slept for a weekend would be given up on at the
        /// first look afterwards, minutes before their video is finished.
        public var retryGrace: TimeInterval = 10 * 60
        /// A frame that was tried without success (its screenshot couldn't be read, or it changed while it was
        /// being embedded) is not tried again sooner than this.
        public var retryBackoff: TimeInterval = 5 * 60
        /// Frames waiting to be retried that are looked at per pass (looking is one query; only the ones that
        /// have become ready are embedded, `imageBatchSize` at a time). With more waiting than this, each pass
        /// looks at the next page, so every waiting frame gets its turn.
        public var retryScanLimit = 256
        /// New frames are embedded in batches, so that the model runs for a moment and exits instead of being
        /// kept in memory by one frame every few seconds: a batch starts once this many frames are waiting...
        public var newFrameBatchMinimum = 16
        /// ...or once the oldest of them was captured this long ago.
        public var newFrameMaximumWait: TimeInterval = 60

        public init() {}
    }

    private let store: any EmbeddingStoreProtocol
    private let service: SemanticSearchService
    private let model: @Sendable (EmbeddingKind) -> String
    private let configuration: Configuration
    private let isEnabled: @Sendable () -> Bool
    private let isImagesEnabled: @Sendable () -> Bool
    private let isReady: @Sendable (EmbeddingKind) -> SemanticIndexStatus.State?
    private let isOnPower: @Sendable () -> Bool

    private var task: Task<Void, Never>?
    private var state: SemanticIndexStatus.State = .off
    private var detail: String?
    private var prunedOtherModels: Set<EmbeddingKind> = []
    /// A frame waiting to be tried again for one kind.
    private struct Waiting: Hashable {
        let kind: EmbeddingKind
        let frameID: Int64
    }

    /// When each waiting frame was last tried without success (this launch only).
    private var lastFailedTry: [Waiting: Date] = [:]
    /// When this loop first saw each waiting frame (this launch only).
    private var firstSeenWaiting: [Waiting: Date] = [:]
    /// Where the walk through each kind's waiting frames continues (none: from the newest).
    private var retryCursor: [EmbeddingKind: Int64] = [:]

    /// - Parameters:
    ///   - isReady: nil when `kind`'s model and the helper are in place, else the state to report.
    ///   - isOnPower: whether the Mac is on AC power (history and screenshots wait for it).
    public init(
        store: any EmbeddingStoreProtocol,
        service: SemanticSearchService,
        model: @escaping @Sendable (EmbeddingKind) -> String = { SemanticSearchSettings.modelIdentifier(for: $0) },
        configuration: Configuration = Configuration(),
        isEnabled: @escaping @Sendable () -> Bool = { SemanticSearchSettings.isEnabled },
        isImagesEnabled: @escaping @Sendable () -> Bool = { SemanticSearchSettings.isImagesEnabled },
        isReady: @escaping @Sendable (EmbeddingKind) -> SemanticIndexStatus.State?,
        isOnPower: @escaping @Sendable () -> Bool = { !PowerStateMonitor.shared.isOnBattery }
    ) {
        self.store = store
        self.service = service
        self.model = model
        self.configuration = configuration
        self.isEnabled = isEnabled
        self.isImagesEnabled = isImagesEnabled
        self.isReady = isReady
        self.isOnPower = isOnPower
    }

    // MARK: - Lifecycle

    public func start() {
        guard task == nil else { return }
        task = Task(priority: .utility) { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                let wait = await self.runPass()
                try? await Task.sleep(for: wait)
            }
        }
    }

    public func stop() {
        task?.cancel()
        task = nil
    }

    /// Stops the loop and waits for the batch in flight to finish or be abandoned, so nothing is written after
    /// this returns. Call `start()` to resume.
    public func stopAndWait() async {
        let running = task
        task = nil
        running?.cancel()
        await running?.value
    }

    public func status() async -> SemanticIndexStatus {
        let text = (try? await store.frameEmbeddingCount(kind: .text, model: model(.text))) ?? 0
        let images = (try? await store.frameEmbeddingCount(kind: .image, model: model(.image))) ?? 0
        let indexed = (try? await store.indexedDocumentCount()) ?? 0
        return SemanticIndexStatus(
            state: state,
            embeddedFrames: text,
            embeddedImages: images,
            indexedFrames: indexed,
            imagesEnabled: imagesWanted,
            detail: detail
        )
    }

    private var imagesWanted: Bool {
        isImagesEnabled() && service.canEmbedImages
    }

    // MARK: - One pass

    private enum Step {
        /// Records were written. `backfill`: they were history rather than new frames.
        case embedded(backfill: Bool)
        /// There was a batch, but nothing could be written (its frames changed while they were being embedded).
        case stalled
        case nothing
    }

    /// Does one batch of work and returns how long to wait before the next. `start()` calls this in a loop;
    /// it is public so a caller (a test, a one-off import) can drive indexing itself.
    @discardableResult
    public func runPass() async -> Duration {
        guard isEnabled() else {
            // Every pass while off, not only the first: the search box may have loaded the vectors and started
            // the model between two passes, without this loop ever seeing the feature on.
            await service.unload()
            set(.off)
            return configuration.idleInterval
        }
        await service.releaseIndexIfIdle()
        if let blocked = isReady(.text) {
            set(blocked)
            return configuration.idleInterval
        }
        if ProcessInfo.processInfo.isLowPowerModeEnabled
            || ProcessInfo.processInfo.thermalState.rawValue >= ProcessInfo.ThermalState.serious.rawValue {
            await service.releaseModelsUnlessSearching()
            set(.paused, detail: "Paused to save power")
            return configuration.idleInterval
        }

        do {
            let outcome = try await nextStep()
            switch outcome.step {
            case .embedded(let backfill):
                set(.indexing)
                return backfill ? configuration.backfillInterval : .zero
            case .stalled:
                // Don't spin on a batch that keeps changing under us: come back to it shortly.
                set(.indexing)
                return configuration.idleInterval
            case .nothing:
                await service.releaseModelsUnlessSearching()
                if let waiting = outcome.waitingDetail {
                    set(.paused, detail: waiting)
                } else {
                    set(.upToDate)
                }
                return configuration.idleInterval
            }
        } catch is CancellationError {
            return .zero
        } catch {
            Log.warning("[Semantic] Indexing failed, will retry: \(error)", category: .search)
            set(.failed, detail: "\(error)")
            return configuration.retryInterval
        }
    }

    /// The most useful batch that can be done right now, and why anything is being held back.
    private func nextStep() async throws -> (step: Step, waitingDetail: String?) {
        let onPower = isOnPower()
        let images = imagesWanted && isReady(.image) == nil
        var waitingDetail: String?
        if imagesWanted, let blocked = isReady(.image), blocked == .modelMissing {
            waitingDetail = "The image encoder isn't downloaded yet"
        }

        // 1. New text, then text that was changed in place and is waiting to be embedded again.
        var step = try await embed(kind: .text, backfill: false)
        if case .nothing = step {} else { return (step, nil) }
        if try await retryWaiting(kind: .text) { return (.embedded(backfill: false), nil) }
        // 2. New screenshots, then the ones that weren't ready earlier.
        if images {
            if onPower {
                step = try await embed(kind: .image, backfill: false)
                if case .nothing = step {} else { return (step, nil) }
                if try await retryWaiting(kind: .image) { return (.embedded(backfill: false), nil) }
            } else if try await hasWork(kind: .image) {
                waitingDetail = "Screenshots are indexed while on power"
            }
        }
        // 3. Text history, then 4. screenshot history.
        if onPower {
            step = try await embed(kind: .text, backfill: true)
            if case .nothing = step {} else { return (step, nil) }
            if images {
                step = try await embed(kind: .image, backfill: true)
                if case .nothing = step {} else { return (step, nil) }
            }
        } else if try await hasWork(kind: .text) {
            waitingDetail = "Older history is indexed while on power"
        }
        return (.nothing, waitingDetail)
    }

    /// Embeds one batch of `kind`: frames newer than its newest embedded document, or (`backfill`) older than
    /// its oldest. With nothing embedded yet, either direction starts from the most recent frames, which are the
    /// ones searched most.
    private func embed(kind: EmbeddingKind, backfill: Bool) async throws -> Step {
        let model = model(kind)
        if !prunedOtherModels.contains(kind) {
            // Vectors from another model or dimension can't be compared with this one's.
            let removed = try await store.deleteFrameEmbeddings(kind: kind, exceptModel: model)
            if removed > 0 {
                Log.info("[Semantic] Removed \(removed) vectors made by another model", category: .search)
            }
            prunedOtherModels.insert(kind)
        }

        let limit = kind == .image ? configuration.imageBatchSize : configuration.batchSize
        let sources: [FrameEmbeddingSource]
        var hasEmbeddedBefore = false
        if let bounds = try await store.embeddingDocIDBounds(kind: kind, model: model) {
            hasEmbeddedBefore = true
            sources = backfill
                ? try await store.embeddingSources(beforeDocID: bounds.min, limit: limit)
                : try await store.embeddingSources(afterDocID: bounds.max, limit: limit)
        } else {
            guard let latest = try await store.latestIndexedDocID() else { return .nothing }
            sources = try await store.embeddingSources(beforeDocID: latest + 1, limit: limit)
        }
        guard let oldest = sources.first else { return .nothing }

        // A few new frames wait for company, or for a minute, whichever comes first (see `newFrameBatchMinimum`).
        if !backfill, hasEmbeddedBefore,
           sources.count < min(configuration.newFrameBatchMinimum, limit),
           Date().timeIntervalSince(oldest.createdAt) < configuration.newFrameMaximumWait {
            return .nothing
        }

        let stored = try await embedAndStore(sources, kind: kind, model: model)
        return stored.isEmpty ? .stalled : .embedded(backfill: backfill)
    }

    /// Looks again at frames marked "try again later": screenshots that weren't final or readable, and frames
    /// whose text was changed in place. Returns whether anything changed.
    private func retryWaiting(kind: EmbeddingKind) async throws -> Bool {
        let model = model(kind)
        let waiting = try await store.embeddingSources(
            retryingKind: kind,
            model: model,
            beforeFrameID: retryCursor[kind],
            limit: configuration.retryScanLimit
        )
        guard !waiting.isEmpty else {
            if retryCursor[kind] == nil {
                // Nothing of this kind is waiting at all: what was remembered about its frames is obsolete.
                // (Only this kind's: the other kind's frames may still be waiting.)
                lastFailedTry = lastFailedTry.filter { $0.key.kind != kind }
                firstSeenWaiting = firstSeenWaiting.filter { $0.key.kind != kind }
            }
            retryCursor[kind] = nil
            return false
        }

        let now = Date()
        for source in waiting {
            let key = Waiting(kind: kind, frameID: source.frameID)
            if firstSeenWaiting[key] == nil { firstSeenWaiting[key] = now }
        }
        // Worth another try: a frame that wasn't tried without success a moment ago and, for a screenshot, is
        // no longer pending.
        let allDue = waiting.filter { source in
            if kind == .image, !source.isMarkedForDeletion, source.imageState == .pending { return false }
            guard let failed = lastFailedTry[Waiting(kind: kind, frameID: source.frameID)] else { return true }
            return now.timeIntervalSince(failed) >= configuration.retryBackoff
        }
        let due = Array(allDue.prefix(kind == .image ? configuration.imageBatchSize : configuration.batchSize))
        // Waited long enough: stop trying. Never a frame that is ready and only waiting for room in a batch.
        let readyFrames = Set(allDue.map(\.frameID))
        let expired = waiting.filter { source in
            guard !readyFrames.contains(source.frameID),
                  let since = source.retryingSince,
                  let firstSeen = firstSeenWaiting[Waiting(kind: kind, frameID: source.frameID)] else {
                return false
            }
            return now.timeIntervalSince(since) > configuration.retryTimeout
                && now.timeIntervalSince(firstSeen) >= configuration.retryGrace
        }
        // Stay on this page while it has more frames ready than one batch takes; otherwise move on to the
        // next page, and back to the newest after the last.
        if allDue.count <= due.count {
            retryCursor[kind] = waiting.count < configuration.retryScanLimit ? nil : waiting.last?.frameID
        }

        var changed = false
        if !due.isEmpty {
            changed = try await !embedAndStore(due, kind: kind, model: model).isEmpty
        }
        if !expired.isEmpty {
            let records = expired.map {
                FrameEmbeddingRecord(frameID: $0.frameID, kind: kind, docID: $0.docID, state: .nothingToEmbed)
            }
            await service.willStore(records)
            let stored = try await store.storeFrameEmbeddings(records, from: expired, model: model)
            await service.didStore(stored)
            forgetWaiting(stored, kind: kind)
            changed = changed || !stored.isEmpty
        }
        return changed
    }

    /// Embeds `sources` and writes what came of it. Returns the records written.
    private func embedAndStore(
        _ sources: [FrameEmbeddingSource],
        kind: EmbeddingKind,
        model: String
    ) async throws -> [FrameEmbeddingRecord] {
        var records = try await service.embed(sources: sources, kind: kind)
        try Task.checkCancellation()

        // A frame found not ready for the first time gets a "try again later" row. One that is already waiting
        // keeps its row as it is and, once it has waited long enough, is given up on.
        let now = Date()
        let sourceOfFrame = Dictionary(sources.map { ($0.frameID, $0) }, uniquingKeysWith: { first, _ in first })
        records = records.compactMap { record in
            let key = Waiting(kind: kind, frameID: record.frameID)
            guard record.state == .retryLater, let source = sourceOfFrame[record.frameID] else {
                lastFailedTry[key] = nil
                return record
            }
            if source.imageState == .ready {
                lastFailedTry[key] = now   // its video was read without success just now
            }
            guard let since = source.retryingSince else { return record }
            guard now.timeIntervalSince(since) > configuration.retryTimeout,
                  now.timeIntervalSince(firstSeenWaiting[key] ?? now) >= configuration.retryGrace else {
                return nil
            }
            lastFailedTry[key] = nil
            return FrameEmbeddingRecord(
                frameID: record.frameID,
                kind: record.kind,
                docID: record.docID,
                state: .nothingToEmbed
            )
        }

        // Before anything is written, the vectors these frames have in memory are forgotten. A search that
        // has already scored one of them then finds it gone when it checks, instead of trusting a score made
        // from the old text because the database has a (new) vector for the frame.
        await service.willStore(records)
        // The store writes only records whose frame still has the text (and pixels) they were made from.
        var stored = try await store.storeFrameEmbeddings(records, from: sources, model: model)

        // A record is refused when its frame changed while it was being embedded: its text was edited in
        // place, or its video was processed or rewritten. The frames around it were written, so the walk has
        // moved past it: it is marked to be tried again, or it would never be looked at. (If the frame is gone,
        // or has a new document, this marker is refused too; a new document is ahead of the walk anyway.)
        let written = Set(stored.map(\.frameID))
        let refused = records.filter { $0.state != .retryLater && !written.contains($0.frameID) }
        if !refused.isEmpty {
            for record in refused {
                lastFailedTry[Waiting(kind: kind, frameID: record.frameID)] = now
            }
            stored += try await store.storeFrameEmbeddings(
                refused.map { FrameEmbeddingRecord(frameID: $0.frameID, kind: kind, docID: $0.docID, state: .retryLater) },
                from: sources,
                model: model
            )
        }
        await service.didStore(stored)
        forgetWaiting(stored, kind: kind)
        return stored
    }

    /// Drops what was remembered about frames that are no longer waiting.
    private func forgetWaiting(_ stored: [FrameEmbeddingRecord], kind: EmbeddingKind) {
        for record in stored where record.state != .retryLater {
            let key = Waiting(kind: kind, frameID: record.frameID)
            lastFailedTry[key] = nil
            firstSeenWaiting[key] = nil
        }
    }

    /// Whether `kind` has frames left to embed in either direction.
    private func hasWork(kind: EmbeddingKind) async throws -> Bool {
        guard let bounds = try await store.embeddingDocIDBounds(kind: kind, model: model(kind)) else {
            return try await store.latestIndexedDocID() != nil
        }
        if try await !store.embeddingSources(afterDocID: bounds.max, limit: 1).isEmpty { return true }
        return try await !store.embeddingSources(beforeDocID: bounds.min, limit: 1).isEmpty
    }

    private func set(_ state: SemanticIndexStatus.State, detail: String? = nil) {
        self.state = state
        self.detail = detail
    }
}
