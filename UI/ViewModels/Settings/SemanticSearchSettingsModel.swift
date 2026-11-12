import Foundation
import Shared
import App

/// What the Semantic Search settings need from the app. `AppCoordinator` is the real one.
protocol SemanticSearchSettingsBackend: AnyObject, Sendable {
    var isSemanticSearchEnabled: Bool { get }
    func setSemanticSearchEnabled(_ enabled: Bool)
    var isSemanticImagesEnabled: Bool { get }
    func setSemanticImagesEnabled(_ enabled: Bool)
    /// Whether this build can run the embedding model. May touch the disk, so it is async.
    func semanticSearchAvailabilityInBuild() async -> Bool
    func isSemanticModelDownloaded(_ model: AppCoordinator.SemanticModel) async -> Bool
    func semanticModelSizeMB(_ model: AppCoordinator.SemanticModel) -> Int
    /// Throws `CancellationError` when the calling task is cancelled.
    func downloadSemanticModel(
        _ model: AppCoordinator.SemanticModel,
        progress: @escaping @Sendable (Double) -> Void
    ) async throws
    /// Counts frames in the database: not something to call more often than needed.
    func semanticSearchStatus() async -> SemanticIndexStatus
    func removeSemanticIndex() async throws
}

extension AppCoordinator: SemanticSearchSettingsBackend {}

/// State behind the Semantic Search card: the two switches (search by meaning, and screenshots too), their
/// one-time model downloads, and indexing progress.
///
/// There is one of these per `AppCoordinator`, and it outlives the card. The card comes and goes with the
/// Settings tab, and can be on screen twice (the Advanced tab and settings search); a download and the
/// "turn on when it has finished" that goes with it must not depend on either staying open.
@MainActor
final class SemanticSearchSettingsModel: ObservableObject {
    typealias Model = AppCoordinator.SemanticModel

    // MARK: - Shared instance

    private static var instances: [ObjectIdentifier: SemanticSearchSettingsModel] = [:]

    /// The model every Semantic Search card of `coordinator` observes.
    static func shared(for coordinator: AppCoordinator) -> SemanticSearchSettingsModel {
        let key = ObjectIdentifier(coordinator)
        if let existing = instances[key] {
            return existing
        }
        // The model keeps its coordinator alive, so the identifier can't be reused for another one
        let model = SemanticSearchSettingsModel(backend: coordinator)
        instances[key] = model
        return model
    }

    // MARK: - State

    /// The first switch as shown: the stored setting, or what was asked for while its model downloads.
    @Published private(set) var isEnabled: Bool
    /// The second switch as shown, likewise.
    @Published private(set) var isImagesEnabled: Bool
    /// Whether this build has the embedding helper. Nil until that has been looked up.
    @Published private(set) var isAvailableInBuild: Bool?
    @Published private(set) var downloading: Model?
    @Published private(set) var downloadProgress: Double = 0
    @Published private(set) var status: SemanticIndexStatus?
    @Published private(set) var errorMessage: String?
    @Published private(set) var isRemovingIndex = false

    let textModelSizeMB: Int
    let imageModelSizeMB: Int

    private let backend: any SemanticSearchSettingsBackend

    /// Downloads whatever the switches still need and then turns the settings on. Nil when nothing is pending.
    private var enableTask: Task<Void, Never>?
    /// Goes up whenever pending work is started or dropped: work from an older generation changes nothing.
    private var enableGeneration = 0

    private var visibleCards = 0
    private var pollTask: Task<Void, Never>?
    private var refreshGeneration = 0
    /// Status reads still to be made at the fast pace after something changed (the indexer takes a few
    /// seconds to notice a switch).
    private var settlingReads = 0

    init(backend: any SemanticSearchSettingsBackend) {
        self.backend = backend
        self.isEnabled = backend.isSemanticSearchEnabled
        self.isImagesEnabled = backend.isSemanticImagesEnabled
        self.textModelSizeMB = backend.semanticModelSizeMB(.text)
        self.imageModelSizeMB = backend.semanticModelSizeMB(.imageEncoder)
    }

    // MARK: - Switches

    var canChangeSwitches: Bool {
        isAvailableInBuild == true
    }

    /// Turning semantic search on downloads the text model first if it isn't there yet.
    /// Turning it off stops a download that was going to turn something on.
    func setEnabled(_ enabled: Bool) {
        errorMessage = nil
        if enabled {
            guard isAvailableInBuild != false else { return }
            isEnabled = true
            startPendingWork()
        } else {
            cancelPendingWork()
            isEnabled = false
            // "Screenshots too" goes back to its stored position: anything that was only pending is dropped
            isImagesEnabled = backend.isSemanticImagesEnabled
            if backend.isSemanticSearchEnabled {
                backend.setSemanticSearchEnabled(false)
            }
        }
        stateDidChange()
    }

    /// Turning screenshots on downloads the image encoder first if it isn't there yet.
    func setImagesEnabled(_ enabled: Bool) {
        errorMessage = nil
        if enabled {
            guard isAvailableInBuild != false, isEnabled else { return }
            isImagesEnabled = true
            startPendingWork()
        } else {
            isImagesEnabled = false
            if downloading == .imageEncoder {
                cancelPendingWork()
            }
            if backend.isSemanticImagesEnabled {
                backend.setSemanticImagesEnabled(false)
            }
        }
        stateDidChange()
    }

    /// Reset to Defaults: both off. A download still running is stopped and turns nothing on; the models
    /// already downloaded and the index are kept. A setting that was already off is left alone, so nothing is
    /// recorded for it.
    func resetToDefaults() {
        cancelPendingWork()
        errorMessage = nil
        isEnabled = false
        isImagesEnabled = false
        if backend.isSemanticImagesEnabled {
            backend.setSemanticImagesEnabled(false)
        }
        if backend.isSemanticSearchEnabled {
            backend.setSemanticSearchEnabled(false)
        }
        stateDidChange()
    }

    // MARK: - Download, then turn on

    private func startPendingWork() {
        // Work already running picks up a switch turned on meanwhile when it gets to it
        guard enableTask == nil else { return }
        enableGeneration += 1
        let generation = enableGeneration
        enableTask = Task { [weak self] in
            await self?.runPendingWork(generation)
        }
    }

    private func cancelPendingWork() {
        enableGeneration += 1
        enableTask?.cancel()
        enableTask = nil
        downloading = nil
        downloadProgress = 0
    }

    /// The next switch that is on while its setting isn't yet: the text model first, since screenshots are
    /// only indexed with semantic search on.
    private func nextModelToEnable() -> Model? {
        guard isEnabled else { return nil }
        if !backend.isSemanticSearchEnabled {
            return .text
        }
        if isImagesEnabled, !backend.isSemanticImagesEnabled {
            return .imageEncoder
        }
        return nil
    }

    private func isWanted(_ model: Model) -> Bool {
        switch model {
        case .text: return isEnabled
        case .imageEncoder: return isEnabled && isImagesEnabled
        }
    }

    private func isStoredOn(_ model: Model) -> Bool {
        switch model {
        case .text: return backend.isSemanticSearchEnabled
        case .imageEncoder: return backend.isSemanticImagesEnabled
        }
    }

    private func runPendingWork(_ generation: Int) async {
        let available = await resolveAvailability()
        guard generation == enableGeneration else { return }
        defer {
            if generation == enableGeneration {
                enableTask = nil
                downloading = nil
                stateDidChange()
            }
        }
        guard available else {
            isEnabled = backend.isSemanticSearchEnabled
            isImagesEnabled = backend.isSemanticImagesEnabled
            return
        }

        while let model = nextModelToEnable() {
            do {
                let isDownloaded = await backend.isSemanticModelDownloaded(model)
                guard generation == enableGeneration else { return }
                if !isDownloaded {
                    guard isWanted(model) else { continue }
                    downloading = model
                    downloadProgress = 0
                    stateDidChange()
                    try await backend.downloadSemanticModel(model) { [weak self] fraction in
                        Task { @MainActor in
                            self?.noteDownloadProgress(fraction, of: model, generation: generation)
                        }
                    }
                    guard generation == enableGeneration else { return }
                    downloading = nil
                }
                // The switch may have been turned off meanwhile: the model is kept, nothing is turned on
                guard isWanted(model) else { continue }
                switch model {
                case .text: backend.setSemanticSearchEnabled(true)
                case .imageEncoder: backend.setSemanticImagesEnabled(true)
                }
                // A setting that didn't take would be tried again for ever
                guard isStoredOn(model) else { return }
            } catch {
                guard generation == enableGeneration else { return }
                downloading = nil
                switch model {
                case .text:
                    isEnabled = false
                    isImagesEnabled = backend.isSemanticImagesEnabled
                case .imageEncoder:
                    isImagesEnabled = false
                }
                // Stopping a download is something the user did, not something that went wrong
                if !(error is CancellationError) {
                    errorMessage = "The model download didn't finish. \(error.localizedDescription)"
                }
            }
        }
    }

    private func noteDownloadProgress(_ fraction: Double, of model: Model, generation: Int) {
        guard generation == enableGeneration, downloading == model else { return }
        // Reports hop here one by one and may arrive out of order
        downloadProgress = max(downloadProgress, min(1, fraction))
    }

    private func resolveAvailability() async -> Bool {
        if let isAvailableInBuild {
            return isAvailableInBuild
        }
        let available = await backend.semanticSearchAvailabilityInBuild()
        isAvailableInBuild = available
        return available
    }

    // MARK: - Status

    /// A card came on screen. Status is kept current only while at least one is showing.
    func cardDidAppear() {
        visibleCards += 1
        if visibleCards == 1 {
            restartPolling()
        }
    }

    func cardDidDisappear() {
        visibleCards = max(0, visibleCards - 1)
        if visibleCards == 0 {
            pollTask?.cancel()
            pollTask = nil
        }
    }

    /// How long to wait before reading status again: often while a download runs or frames are being indexed
    /// (the numbers on screen are moving), seldom otherwise.
    var pollInterval: Duration {
        if downloading != nil || settlingReads > 0 {
            return .seconds(2)
        }
        if isEnabled, status?.state == .indexing {
            return .seconds(2)
        }
        return .seconds(15)
    }

    /// Reading status counts frames in the database. With semantic search off and nothing indexed, nothing
    /// there can change, so once that is known the counts aren't read again until something is switched.
    var needsStatusRead: Bool {
        guard let status else { return true }
        if settlingReads > 0 || backend.isSemanticSearchEnabled {
            return true
        }
        return status.embeddedFrames + status.embeddedImages > 0
    }

    /// A switch, a download or the index changed: read status again now and for the next few seconds.
    private func stateDidChange() {
        settlingReads = 4
        if visibleCards > 0 {
            restartPolling()
        }
    }

    private func restartPolling() {
        pollTask?.cancel()
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                let interval: Duration
                do {
                    guard let self else { return }
                    await self.refresh()
                    interval = self.pollInterval
                }
                do {
                    try await Task.sleep(for: interval)
                } catch {
                    return
                }
            }
        }
    }

    /// Reads what the card shows that can change behind its back: whether the build has the helper (once),
    /// indexing progress, and the stored switch positions.
    func refresh() async {
        refreshGeneration += 1
        let generation = refreshGeneration
        _ = await resolveAvailability()

        let readsStatus = needsStatusRead
        settlingReads = max(0, settlingReads - 1)
        let newStatus = readsStatus ? await backend.semanticSearchStatus() : nil
        // A newer refresh has started since: its answer is the one to show
        guard generation == refreshGeneration else { return }

        if let newStatus, newStatus != status {
            status = newStatus
        }
        // The switches follow the stored settings, except while a download is on its way to turning one on
        if enableTask == nil {
            let storedEnabled = backend.isSemanticSearchEnabled
            let storedImagesEnabled = backend.isSemanticImagesEnabled
            if isEnabled != storedEnabled {
                isEnabled = storedEnabled
            }
            if isImagesEnabled != storedImagesEnabled {
                isImagesEnabled = storedImagesEnabled
            }
        }
    }

    // MARK: - Index

    var hasIndex: Bool {
        ((status?.embeddedFrames ?? 0) + (status?.embeddedImages ?? 0)) > 0
    }

    /// What "Remove Index" is about to do, for its confirmation.
    var removeIndexConfirmation: String {
        let frames = status?.embeddedFrames ?? 0
        let images = status?.embeddedImages ?? 0
        var text = "This deletes the stored vectors for \(frames.formatted()) \(frames == 1 ? "frame" : "frames")"
        if images > 0 {
            text += " and \(images.formatted()) \(images == 1 ? "screenshot" : "screenshots")"
        }
        text += ". Your recordings and keyword search are not affected. "
        text += isEnabled
            ? "Semantic search is on, so indexing will start again from the beginning."
            : "Indexing will start again if you turn semantic search back on."
        return text
    }

    func removeIndex() {
        guard !isRemovingIndex else { return }
        errorMessage = nil
        isRemovingIndex = true
        Task { [weak self, backend] in
            var failure: String?
            do {
                try await backend.removeSemanticIndex()
            } catch {
                failure = "Couldn't remove the index: \(error.localizedDescription)"
            }
            guard let self else { return }
            self.isRemovingIndex = false
            self.errorMessage = failure
            // The counts on screen are no longer true, whether or not a card is showing right now
            self.status = nil
            self.stateDidChange()
        }
    }

    // MARK: - Text

    /// One line under the switches saying what is happening, or nil when there is nothing to say.
    var statusText: String? {
        guard let isAvailableInBuild else { return nil }
        guard isAvailableInBuild else { return "Not available in this build." }
        if let downloading {
            let size = downloading == .text ? textModelSizeMB : imageModelSizeMB
            let name = downloading == .text ? "model" : "image encoder"
            return "Downloading the \(name): \(Int(downloadProgress * 100))% of \(size) MB"
        }
        guard let status else { return nil }
        guard isEnabled else {
            return status.embeddedFrames > 0
                ? "Off. \(status.embeddedFrames.formatted()) frames are indexed and kept."
                : nil
        }
        let progress = Self.progressText(status)
        switch status.state {
        case .off:
            // The indexer looks at the switch every few seconds
            return "Starting…"
        case .modelMissing:
            return "The model file is missing. Turn the switch off and on again to download it."
        case .helperMissing:
            return "Not available in this build."
        case .indexing:
            return "Indexing: \(progress)"
        case .upToDate:
            return "Up to date: \(progress)"
        case .paused:
            return "\(status.detail ?? "Paused"). \(progress)"
        case .failed:
            return "Indexing stopped and will retry: \(status.detail ?? "unknown error")"
        }
    }

    /// "12,340 of 24,944 frames by text, 1,200 by screenshot"
    static func progressText(_ status: SemanticIndexStatus) -> String {
        let total = status.indexedFrames.formatted()
        var text = "\(status.embeddedFrames.formatted()) of \(total) frames by text"
        if status.imagesEnabled || status.embeddedImages > 0 {
            text += ", \(status.embeddedImages.formatted()) by screenshot"
        }
        return text
    }
}
