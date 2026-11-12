import XCTest
import App
import Shared
@testable import Retrace

/// The Semantic Search settings model, driven the way the card and Reset to Defaults drive it, against a
/// stand-in for the app whose downloads finish, fail or get cancelled when the test says so.
@MainActor
final class SemanticSearchSettingsModelTests: XCTestCase {
    private typealias Model = AppCoordinator.SemanticModel

    // MARK: - A download outlives the card

    func testADownloadStartedFromACardThatIsGoneStillTurnsTheSettingOn() async throws {
        let backend = StubBackend()
        let model = SemanticSearchSettingsModel(backend: backend)
        model.cardDidAppear()

        model.setEnabled(true)
        try await waitUntil { model.downloading == .text }
        XCTAssertTrue(model.isEnabled, "the switch shows what was asked for while the model downloads")
        XCTAssertFalse(backend.isSemanticSearchEnabled, "nothing is turned on before the model is there")

        // The Settings tab is switched: the card is gone
        model.cardDidDisappear()
        backend.reportProgress(0.5, of: .text)
        try await waitUntil { model.downloadProgress == 0.5 }
        backend.finishDownload(of: .text)
        try await waitUntil { backend.isSemanticSearchEnabled }

        try await waitUntil { model.downloading == nil }
        XCTAssertTrue(model.isEnabled)
        XCTAssertNil(model.errorMessage)
        XCTAssertEqual(backend.recordedToggles, ["on"])
    }

    func testEveryCardOfACoordinatorObservesTheSameModel() {
        let coordinator = AppCoordinator()
        let other = AppCoordinator()

        let advancedTabCard = SemanticSearchSettingsModel.shared(for: coordinator)
        let settingsSearchCard = SemanticSearchSettingsModel.shared(for: coordinator)

        XCTAssertTrue(advancedTabCard === settingsSearchCard)
        XCTAssertFalse(advancedTabCard === SemanticSearchSettingsModel.shared(for: other))
    }

    func testFlippingTheSwitchAgainWhileDownloadingDoesNotStartASecondDownload() async throws {
        let backend = StubBackend()
        let model = SemanticSearchSettingsModel(backend: backend)

        model.setEnabled(true)
        try await waitUntil { model.downloading == .text }
        // The same switch in the other card
        model.setEnabled(true)
        backend.finishDownload(of: .text)
        try await waitUntil { backend.isSemanticSearchEnabled }

        XCTAssertEqual(backend.downloadsStarted, [.text])
        XCTAssertEqual(backend.recordedToggles, ["on"])
    }

    func testAModelAlreadyOnDiskTurnsTheSettingOnWithoutADownload() async throws {
        let backend = StubBackend(downloaded: [.text])
        let model = SemanticSearchSettingsModel(backend: backend)

        model.setEnabled(true)
        try await waitUntil { backend.isSemanticSearchEnabled }

        XCTAssertEqual(backend.downloadsStarted, [])
        XCTAssertNil(model.downloading)
    }

    func testScreenshotsSwitchedOnDuringTheTextDownloadWaitsItsTurn() async throws {
        let backend = StubBackend()
        let model = SemanticSearchSettingsModel(backend: backend)

        model.setEnabled(true)
        try await waitUntil { model.downloading == .text }
        model.setImagesEnabled(true)
        XCTAssertTrue(model.isImagesEnabled)
        XCTAssertNil(model.errorMessage)

        backend.finishDownload(of: .text)
        try await waitUntil { model.downloading == .imageEncoder }
        XCTAssertTrue(backend.isSemanticSearchEnabled)
        XCTAssertFalse(backend.isSemanticImagesEnabled)

        backend.finishDownload(of: .imageEncoder)
        try await waitUntil { backend.isSemanticImagesEnabled }

        XCTAssertEqual(backend.downloadsStarted, [.text, .imageEncoder])
        XCTAssertEqual(backend.recordedToggles, ["on", "images_on"])
    }

    // MARK: - Stopping

    func testSwitchingOffWhileDownloadingStopsTheDownloadAndIsNotAnError() async throws {
        let backend = StubBackend()
        let model = SemanticSearchSettingsModel(backend: backend)

        model.setEnabled(true)
        try await waitUntil { model.downloading == .text }
        model.setEnabled(false)

        XCTAssertFalse(model.isEnabled)
        XCTAssertNil(model.downloading)
        try await waitUntil { backend.downloadsCancelled == [.text] }
        try await settle()

        XCTAssertNil(model.errorMessage, "stopping a download is not a failure")
        XCTAssertFalse(model.isEnabled)
        XCTAssertFalse(backend.isSemanticSearchEnabled)
        XCTAssertEqual(backend.recordedToggles, [], "it was never on, so nothing was turned off")
    }

    func testSwitchingBackOnRightAfterStoppingDownloadsAgain() async throws {
        let backend = StubBackend()
        let model = SemanticSearchSettingsModel(backend: backend)

        model.setEnabled(true)
        try await waitUntil { model.downloading == .text }
        model.setEnabled(false)
        model.setEnabled(true)
        try await waitUntil { backend.downloadsStarted == [.text, .text] && model.downloading == .text }
        backend.finishDownload(of: .text)
        try await waitUntil { backend.isSemanticSearchEnabled }

        XCTAssertTrue(model.isEnabled)
        XCTAssertNil(model.errorMessage)
        XCTAssertEqual(backend.downloadsCancelled, [.text])
    }

    func testResetWhileDownloadingTurnsNothingOnEvenIfTheDownloadFinishes() async throws {
        // A download that can't be stopped any more and completes after the reset
        let backend = StubBackend(downloadsStopWhenCancelled: false)
        let model = SemanticSearchSettingsModel(backend: backend)

        model.setEnabled(true)
        try await waitUntil { model.downloading == .text }
        model.setImagesEnabled(true)
        model.resetToDefaults()
        backend.finishDownload(of: .text)
        try await waitUntil { backend.downloadsRunning == 0 }
        try await settle()

        XCTAssertFalse(model.isEnabled)
        XCTAssertFalse(model.isImagesEnabled)
        XCTAssertNil(model.downloading)
        XCTAssertNil(model.errorMessage)
        XCTAssertFalse(backend.isSemanticSearchEnabled)
        XCTAssertFalse(backend.isSemanticImagesEnabled)
        XCTAssertEqual(backend.downloadsStarted, [.text], "the image encoder that was waiting is not fetched")
        XCTAssertEqual(backend.recordedToggles, [])
    }

    func testResetTurnsOffOnlyWhatWasOn() {
        let neverOn = StubBackend()
        SemanticSearchSettingsModel(backend: neverOn).resetToDefaults()
        XCTAssertEqual(neverOn.recordedToggles, [])

        let textOnly = StubBackend(enabled: true)
        let textOnlyModel = SemanticSearchSettingsModel(backend: textOnly)
        textOnlyModel.resetToDefaults()
        XCTAssertEqual(textOnly.recordedToggles, ["off"])
        XCTAssertFalse(textOnlyModel.isEnabled)

        let both = StubBackend(enabled: true, imagesEnabled: true)
        let bothModel = SemanticSearchSettingsModel(backend: both)
        bothModel.resetToDefaults()
        XCTAssertEqual(both.recordedToggles, ["images_off", "off"])
        XCTAssertFalse(bothModel.isImagesEnabled)
    }

    func testAFailedDownloadPutsTheSwitchBackAndSaysWhy() async throws {
        let backend = StubBackend()
        let model = SemanticSearchSettingsModel(backend: backend)

        model.setEnabled(true)
        try await waitUntil { model.downloading == .text }
        backend.failDownload(of: .text, with: URLError(.notConnectedToInternet))
        try await waitUntil { model.errorMessage != nil }

        XCTAssertFalse(model.isEnabled)
        XCTAssertNil(model.downloading)
        XCTAssertFalse(backend.isSemanticSearchEnabled)
        XCTAssertEqual(backend.recordedToggles, [])
    }

    // MARK: - A build without the helper

    func testWithoutTheHelperNothingCanBeTurnedOnAndTheCardSaysSoPlainly() async throws {
        let backend = StubBackend(hasHelper: false)
        let model = SemanticSearchSettingsModel(backend: backend)
        XCTAssertFalse(model.canChangeSwitches, "the switches wait until the build has been checked")

        await model.refresh()

        XCTAssertFalse(model.canChangeSwitches)
        XCTAssertEqual(model.statusText, "Not available in this build.")

        model.setEnabled(true)
        try await settle()
        XCTAssertFalse(model.isEnabled)
        XCTAssertEqual(backend.downloadsStarted, [])
    }

    // MARK: - How often status is read

    func testFramesAreNotCountedAgainWhileOffWithNothingIndexed() async {
        let backend = StubBackend()
        let model = SemanticSearchSettingsModel(backend: backend)

        await model.refresh()
        XCTAssertEqual(backend.statusReads, 1, "the first look has to count")
        await model.refresh()
        await model.refresh()

        XCTAssertEqual(backend.statusReads, 1)
        XCTAssertEqual(model.pollInterval, .seconds(15))
    }

    func testAnIndexKeptWhileOffIsStillCountedButSeldom() async {
        let backend = StubBackend()
        backend.status = SemanticIndexStatus(state: .off, embeddedFrames: 1_200, indexedFrames: 5_000)
        let model = SemanticSearchSettingsModel(backend: backend)

        await model.refresh()
        await model.refresh()

        XCTAssertEqual(backend.statusReads, 2)
        XCTAssertEqual(model.pollInterval, .seconds(15))
        XCTAssertEqual(model.statusText, "Off. \(1_200.formatted()) frames are indexed and kept.")
    }

    func testStatusIsReadOftenOnlyWhileIndexingOrDownloading() async throws {
        let backend = StubBackend(enabled: true, downloaded: [.text])
        backend.status = SemanticIndexStatus(state: .indexing, embeddedFrames: 10, indexedFrames: 5_000)
        let model = SemanticSearchSettingsModel(backend: backend)

        await model.refresh()
        XCTAssertEqual(model.pollInterval, .seconds(2))

        backend.status = SemanticIndexStatus(state: .upToDate, embeddedFrames: 5_000, indexedFrames: 5_000)
        await model.refresh()
        XCTAssertEqual(model.pollInterval, .seconds(15))

        model.setImagesEnabled(true)
        try await waitUntil { model.downloading == .imageEncoder }
        XCTAssertEqual(model.pollInterval, .seconds(2))
        model.setImagesEnabled(false)
    }

    // MARK: - Removing the index

    func testTheRemovalConfirmationSaysWhatGoesAndThatIndexingStartsOver() async {
        let backend = StubBackend(enabled: true, downloaded: [.text])
        backend.status = SemanticIndexStatus(
            state: .upToDate,
            embeddedFrames: 12_340,
            embeddedImages: 1_200,
            indexedFrames: 12_340,
            imagesEnabled: true
        )
        let model = SemanticSearchSettingsModel(backend: backend)
        await model.refresh()

        let whileOn = model.removeIndexConfirmation
        XCTAssertTrue(whileOn.contains("\(12_340.formatted()) frames"), whileOn)
        XCTAssertTrue(whileOn.contains("\(1_200.formatted()) screenshots"), whileOn)
        XCTAssertTrue(whileOn.contains("indexing will start again from the beginning"), whileOn)

        model.setEnabled(false)
        let whileOff = model.removeIndexConfirmation
        XCTAssertTrue(whileOff.contains("\(12_340.formatted()) frames"), whileOff)
        XCTAssertTrue(whileOff.contains("if you turn semantic search back on"), whileOff)
    }

    func testRemovingTheIndexReadsTheCountsAgain() async throws {
        let backend = StubBackend()
        backend.status = SemanticIndexStatus(state: .off, embeddedFrames: 800, indexedFrames: 5_000)
        let model = SemanticSearchSettingsModel(backend: backend)
        model.cardDidAppear()
        try await waitUntil { model.hasIndex }

        model.removeIndex()
        model.removeIndex()
        try await waitUntil { !model.isRemovingIndex && model.status?.embeddedFrames == 0 }

        XCTAssertEqual(backend.indexRemovals, 1, "a second click while removing does nothing")
        XCTAssertFalse(model.hasIndex)
        XCTAssertNil(model.errorMessage)
        model.cardDidDisappear()
    }

    // MARK: - Helpers

    private func waitUntil(
        timeout: Duration = .seconds(10),
        _ condition: @MainActor () -> Bool,
        file: StaticString = #filePath,
        line: UInt = #line
    ) async throws {
        let deadline = ContinuousClock.now + timeout
        while !condition() {
            guard ContinuousClock.now < deadline else {
                XCTFail("timed out waiting", file: file, line: line)
                throw CancellationError()
            }
            try await Task.sleep(for: .milliseconds(2))
        }
    }

    /// Gives work that was already under way time to finish, before asserting that something did not happen.
    private func settle() async throws {
        try await Task.sleep(for: .milliseconds(100))
    }
}

// MARK: - Stand-in for the app

/// Settings held in memory, model files as a set, and downloads that end when the test says so.
/// `recordedToggles` is what the real setters would record as `semantic_search_toggled` events.
private final class StubBackend: SemanticSearchSettingsBackend, @unchecked Sendable {
    typealias Model = AppCoordinator.SemanticModel

    private let lock = NSLock()
    private let hasHelper: Bool
    private let downloadsStopWhenCancelled: Bool
    private var enabled: Bool
    private var imagesEnabled: Bool
    private var downloaded: Set<Model>
    private var toggles: [String] = []
    private var started: [Model] = []
    private var cancelled: [Model] = []
    private var running = 0
    private var statusReadCount = 0
    private var removals = 0
    private var currentStatus = SemanticIndexStatus(state: .off, embeddedFrames: 0, indexedFrames: 0)
    private var pending: [Model: PendingDownload] = [:]
    private var cancelledBeforeStarting: Set<UUID> = []

    private struct PendingDownload {
        let id: UUID
        let continuation: CheckedContinuation<Void, Error>
        let progress: @Sendable (Double) -> Void
    }

    init(
        enabled: Bool = false,
        imagesEnabled: Bool = false,
        downloaded: Set<Model> = [],
        hasHelper: Bool = true,
        downloadsStopWhenCancelled: Bool = true
    ) {
        self.enabled = enabled
        self.imagesEnabled = imagesEnabled
        self.downloaded = downloaded
        self.hasHelper = hasHelper
        self.downloadsStopWhenCancelled = downloadsStopWhenCancelled
    }

    private func locked<T>(_ body: () -> T) -> T {
        lock.lock()
        defer { lock.unlock() }
        return body()
    }

    // What the test looks at

    var recordedToggles: [String] { locked { toggles } }
    var downloadsStarted: [Model] { locked { started } }
    var downloadsCancelled: [Model] { locked { cancelled } }
    var downloadsRunning: Int { locked { running } }
    var statusReads: Int { locked { statusReadCount } }
    var indexRemovals: Int { locked { removals } }

    var status: SemanticIndexStatus {
        get { locked { currentStatus } }
        set { locked { currentStatus = newValue } }
    }

    // What the test does

    func reportProgress(_ fraction: Double, of model: Model) {
        locked { pending[model]?.progress }?(fraction)
    }

    func finishDownload(of model: Model) {
        let download = locked { () -> PendingDownload? in
            downloaded.insert(model)
            return pending.removeValue(forKey: model)
        }
        download?.continuation.resume()
    }

    func failDownload(of model: Model, with error: Error) {
        locked { pending.removeValue(forKey: model) }?.continuation.resume(throwing: error)
    }

    // SemanticSearchSettingsBackend

    var isSemanticSearchEnabled: Bool { locked { enabled } }

    func setSemanticSearchEnabled(_ enabled: Bool) {
        locked {
            self.enabled = enabled
            toggles.append(enabled ? "on" : "off")
        }
    }

    var isSemanticImagesEnabled: Bool { locked { imagesEnabled } }

    func setSemanticImagesEnabled(_ enabled: Bool) {
        locked {
            imagesEnabled = enabled
            toggles.append(enabled ? "images_on" : "images_off")
        }
    }

    func semanticSearchAvailabilityInBuild() async -> Bool {
        hasHelper
    }

    func isSemanticModelDownloaded(_ model: Model) async -> Bool {
        locked { downloaded.contains(model) }
    }

    func semanticModelSizeMB(_ model: Model) -> Int {
        model == .text ? 310 : 555
    }

    func downloadSemanticModel(_ model: Model, progress: @escaping @Sendable (Double) -> Void) async throws {
        let id = UUID()
        locked {
            started.append(model)
            running += 1
        }
        defer { locked { running -= 1 } }

        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                let alreadyCancelled = locked { () -> Bool in
                    if cancelledBeforeStarting.contains(id) {
                        return true
                    }
                    pending[model] = PendingDownload(id: id, continuation: continuation, progress: progress)
                    return false
                }
                if alreadyCancelled {
                    continuation.resume(throwing: CancellationError())
                }
            }
        } onCancel: {
            guard downloadsStopWhenCancelled else { return }
            let download = locked { () -> PendingDownload? in
                cancelled.append(model)
                guard pending[model]?.id == id else {
                    cancelledBeforeStarting.insert(id)
                    return nil
                }
                return pending.removeValue(forKey: model)
            }
            download?.continuation.resume(throwing: CancellationError())
        }
    }

    func semanticSearchStatus() async -> SemanticIndexStatus {
        locked {
            statusReadCount += 1
            return currentStatus
        }
    }

    func removeSemanticIndex() async throws {
        locked {
            removals += 1
            currentStatus = SemanticIndexStatus(
                state: currentStatus.state,
                embeddedFrames: 0,
                indexedFrames: currentStatus.indexedFrames
            )
        }
    }
}
