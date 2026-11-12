import XCTest
import CoreGraphics
import Foundation
import Shared
import Database
import SQLCipher
@testable import Search

/// Semantic indexing and search against a real SQLite database, with a stand-in for the embedding model.
///
/// `WordEmbedder` gives every word a fixed random direction and a text the normalized sum of its words, so texts
/// that share words are close and everything else is roughly orthogonal. That is enough to exercise what this
/// code owns: which frames get embedded and in what order, what is stored, and what a query returns.
final class SemanticSearchTests: XCTestCase {
    private let model = "test-words/64"
    private let dimension = 64
    private var database: DatabaseManager!
    /// The database, with switches to hold a read or refuse a write at a chosen moment.
    private var store: InterruptibleStore!
    private var embedder: WordEmbedder!
    private var service: SemanticSearchService!
    private var videoID: Int64 = 0
    private var segmentID: Int64 = 0
    private var frameCount = 0

    override func setUp() async throws {
        database = DatabaseManager(databasePath: "file:memdb_\(UUID().uuidString)?mode=memory&cache=private")
        try await database.initialize()
        store = InterruptibleStore(base: database)
        videoID = try await database.insertVideoSegment(
            VideoSegment(
                id: VideoSegmentID(value: 0),
                startTime: Date(),
                endTime: Date().addingTimeInterval(300),
                frameCount: 150,
                fileSizeBytes: 1024,
                relativePath: "chunks/test",
                width: 1920,
                height: 1080,
                source: .native
            )
        )
        segmentID = try await database.insertSegment(
            bundleID: "com.test.app",
            startDate: Date(),
            endDate: Date().addingTimeInterval(300),
            windowName: "Notes",
            browserUrl: nil,
            type: 0
        )
        embedder = WordEmbedder()
        service = SemanticSearchService(
            store: store,
            embedder: embedder,
            model: { [model] _ in model },
            dimension: dimension,
            isEnabled: { true }
        )
        frameCount = 0
    }

    override func tearDown() async throws {
        try await database.close()
        database = nil
    }

    private func makeIndexer(
        enabled: Bool = true,
        ready: SemanticIndexStatus.State? = nil,
        canBackfill: Bool = true,
        batchSize: Int = 4,
        newFrameBatchMinimum: Int = 1,
        newFrameMaximumWait: TimeInterval = 0,
        retryBackoff: TimeInterval = 300
    ) -> SemanticIndexer {
        var configuration = SemanticIndexer.Configuration()
        configuration.batchSize = batchSize
        configuration.retryBackoff = retryBackoff
        // By default a new frame is embedded on the next pass, as most tests expect.
        configuration.newFrameBatchMinimum = newFrameBatchMinimum
        configuration.newFrameMaximumWait = newFrameMaximumWait
        return SemanticIndexer(
            store: store,
            service: service,
            model: { [model] _ in model },
            configuration: configuration,
            isEnabled: { enabled },
            isImagesEnabled: { false },
            isReady: { _ in ready },
            isOnPower: { canBackfill }
        )
    }

    @discardableResult
    private func insertFrame(text: String, capturedAt: Date? = nil) async throws -> Int64 {
        frameCount += 1
        let frameID = try await database.insertFrame(
            FrameReference(
                id: FrameID(value: 0),
                timestamp: capturedAt ?? Date().addingTimeInterval(Double(frameCount)),
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
            windowTitle: "Notes",
            segmentId: segmentID,
            frameId: frameID
        )
        return frameID
    }

    /// Twelve frames on unrelated subjects, so one of them can stand out.
    private func insertUnrelatedFrames() async throws {
        let subjects = [
            "quarterly revenue forecast spreadsheet columns",
            "kitchen renovation contractor estimate tiles",
            "marathon training schedule intervals tempo",
            "database migration rollback procedure checklist",
            "birthday party guest list balloons cake",
            "mortgage refinance interest rate comparison",
            "guitar chord progression practice session",
            "customer support ticket escalation policy",
            "garden tomato seedlings watering calendar",
            "conference keynote speaker biography draft",
            "bicycle brake cable replacement instructions",
            "podcast episode editing waveform timeline",
        ]
        for subject in subjects {
            try await insertFrame(text: subject)
        }
    }

    private func runUntilSettled(_ indexer: SemanticIndexer, maxPasses: Int = 50) async -> SemanticIndexStatus.State {
        for _ in 0..<maxPasses {
            _ = await indexer.runPass()
            let state = await indexer.status().state
            if state != .indexing { return state }
        }
        return await indexer.status().state
    }

    func testIndexerEmbedsHistoryAndSearchFindsTheFrameByItsWords() async throws {
        try await insertUnrelatedFrames()
        let flight = try await insertFrame(text: "airline flight booking confirmation Lisbon departure gate")
        let tooShort = try await insertFrame(text: "ok")

        let indexer = makeIndexer()
        let state = await runUntilSettled(indexer)
        XCTAssertEqual(state, .upToDate)

        let status = await indexer.status()
        XCTAssertEqual(status.indexedFrames, 14)
        XCTAssertEqual(status.embeddedFrames, 14, "every frame is accounted for, including the one with no text to embed")

        let stored = try await database.frameEmbeddings(kind: .text, model: model, afterFrameID: -1, limit: 100)
        XCTAssertEqual(stored.count, 13)
        XCTAssertFalse(stored.contains { $0.frameID == tooShort })
        XCTAssertTrue(stored.allSatisfy { $0.vector.count == dimension * 2 }, "vectors are stored as float16")

        let matches = await service.matches(for: "flight booking", limit: 5)
        XCTAssertEqual(matches.first?.frameID, flight)
        XCTAssertEqual(matches.count, 1, "nothing else stands out for this query")
    }

    func testNewestFramesAreEmbeddedFirst() async throws {
        try await insertUnrelatedFrames()
        let newest = try await insertFrame(text: "airline flight booking confirmation Lisbon departure gate")

        let indexer = makeIndexer(batchSize: 4)
        _ = await indexer.runPass()

        let stored = try await database.frameEmbeddings(kind: .text, model: model, afterFrameID: -1, limit: 100)
        XCTAssertEqual(stored.count, 4)
        XCTAssertTrue(stored.contains { $0.frameID == newest })
    }

    func testFrameCapturedAfterIndexingIsSearchableWithoutReloading() async throws {
        try await insertUnrelatedFrames()
        let indexer = makeIndexer()
        _ = await runUntilSettled(indexer)

        let none = await service.matches(for: "flight booking", limit: 5)
        XCTAssertTrue(none.isEmpty)

        let flight = try await insertFrame(text: "airline flight booking confirmation Lisbon departure gate")
        _ = await runUntilSettled(indexer)

        let matches = await service.matches(for: "flight booking", limit: 5)
        XCTAssertEqual(matches.first?.frameID, flight)
    }

    func testAFewNewFramesWaitForABatchOrAMinuteSoTheModelIsNotKeptRunning() async throws {
        try await insertUnrelatedFrames()
        let indexer = makeIndexer(batchSize: 64, newFrameBatchMinimum: 3, newFrameMaximumWait: 60)
        _ = await runUntilSettled(indexer)
        let embeddedHistory = embedder.embeddedTextCount
        XCTAssertGreaterThan(embedder.shutdownCount, 0, "with nothing left to embed, the model is stopped")

        // Two frames have just been captured: not enough for a batch, and not waiting long.
        let first = try await insertFrame(text: "airline flight booking confirmation Lisbon departure gate", capturedAt: Date())
        try await insertFrame(text: "quarterly budget review meeting agenda items", capturedAt: Date())
        var state = await runUntilSettled(indexer)
        XCTAssertEqual(state, .upToDate)
        XCTAssertEqual(embedder.embeddedTextCount, embeddedHistory, "the model was not started for two frames")

        // A third arrives: that is a batch.
        try await insertFrame(text: "hiking trail map with elevation profile", capturedAt: Date())
        state = await runUntilSettled(indexer)
        XCTAssertEqual(embedder.embeddedTextCount, embeddedHistory + 3)
        var stored = try await database.frameEmbeddings(kind: .text, model: model, afterFrameID: -1, limit: 100)
        XCTAssertTrue(stored.contains { $0.frameID == first })

        // One frame alone is embedded once it has waited a minute.
        let lone = try await insertFrame(text: "dentist appointment reminder for Thursday morning", capturedAt: Date().addingTimeInterval(-61))
        _ = await runUntilSettled(indexer)
        stored = try await database.frameEmbeddings(kind: .text, model: model, afterFrameID: -1, limit: 100)
        XCTAssertTrue(stored.contains { $0.frameID == lone })
    }

    func testTheModelStaysForTheNextQueryRightAfterASearch() async throws {
        try await insertUnrelatedFrames()
        let indexer = makeIndexer()
        _ = await runUntilSettled(indexer)
        let stopsBefore = embedder.shutdownCount

        _ = await service.matches(for: "revenue forecast", limit: 5)
        _ = await indexer.runPass()   // nothing to embed

        XCTAssertEqual(embedder.shutdownCount, stopsBefore, "someone is searching: the text model is kept")
    }

    func testHistoryWaitsForPowerButNewFramesDoNot() async throws {
        try await insertUnrelatedFrames()
        let indexer = makeIndexer(canBackfill: false, batchSize: 4)

        // The first pass always takes the newest frames; after that only history is left.
        let state = await runUntilSettled(indexer)
        XCTAssertEqual(state, .paused)
        let partial = try await database.frameEmbeddingCount(kind: .text, model: model)
        XCTAssertEqual(partial, 4)

        let fresh = try await insertFrame(text: "airline flight booking confirmation Lisbon departure gate")
        _ = await runUntilSettled(indexer)
        let stored = try await database.frameEmbeddings(kind: .text, model: model, afterFrameID: -1, limit: 100)
        XCTAssertEqual(stored.count, 5)
        XCTAssertTrue(stored.contains { $0.frameID == fresh })
    }

    func testIdenticalTextIsEmbeddedOnce() async throws {
        let first = try await insertFrame(text: "airline flight booking confirmation Lisbon departure gate")
        let second = try await insertFrame(text: "airline flight booking confirmation Lisbon departure gate")

        let indexer = makeIndexer(batchSize: 8)
        _ = await runUntilSettled(indexer)

        XCTAssertEqual(embedder.embeddedTextCount, 1)
        let stored = try await database.frameEmbeddings(kind: .text, model: model, afterFrameID: -1, limit: 100)
        XCTAssertEqual(Set(stored.map(\.frameID)), [first, second])
        XCTAssertEqual(stored.first?.vector, stored.last?.vector)
    }

    func testNothingIsEmbeddedOrReturnedWhileTurnedOffOrModelMissing() async throws {
        try await insertUnrelatedFrames()

        let off = makeIndexer(enabled: false)
        _ = await off.runPass()
        var status = await off.status()
        XCTAssertEqual(status.state, .off)
        XCTAssertEqual(status.embeddedFrames, 0)

        let missing = makeIndexer(ready: .modelMissing)
        _ = await missing.runPass()
        status = await missing.status()
        XCTAssertEqual(status.state, .modelMissing)
        XCTAssertEqual(status.embeddedFrames, 0)

        let disabledService = SemanticSearchService(
            store: database,
            embedder: embedder,
            model: { [model] _ in model },
            dimension: dimension,
            isEnabled: { false }
        )
        let matches = await disabledService.matches(for: "revenue forecast", limit: 5)
        XCTAssertTrue(matches.isEmpty)
        XCTAssertEqual(embedder.embeddedTextCount, 0)
    }

    func testRedactedTextStopsMatchingEvenWhileItsVectorIsStillInMemory() async throws {
        try await insertUnrelatedFrames()
        let flight = try await insertFrame(text: "airline flight booking confirmation Lisbon departure gate")
        let indexer = makeIndexer()
        _ = await runUntilSettled(indexer)
        let before = await service.matches(for: "flight booking", limit: 5)
        XCTAssertEqual(before.first?.frameID, flight)

        // A redaction rewrites the frame's text. The database drops its vector at once; the copy in memory is
        // still there, and the indexer hasn't run again.
        _ = try await database.indexFrameText(
            mainText: "ok",
            chromeText: nil,
            windowTitle: "Notes",
            segmentId: segmentID,
            frameId: flight
        )

        let after = await service.matches(for: "flight booking", limit: 5)
        XCTAssertFalse(after.contains { $0.frameID == flight })
        let loaded = await service.loadedVectorCount(kind: .text)
        XCTAssertEqual(loaded, 12, "the stale vector is forgotten, not just hidden")

        // Once the indexer has seen the new text (too short to embed), nothing brings the old vector back.
        _ = await runUntilSettled(indexer)
        let later = await service.matches(for: "flight booking", limit: 5)
        XCTAssertFalse(later.contains { $0.frameID == flight })
    }

    func testASearchAfterTheBestMatchWasDeletedStillFindsTheNextBest() async throws {
        try await insertUnrelatedFrames()
        let monday = try await insertFrame(text: "airline flight booking confirmation Lisbon departure gate Monday")
        let tuesday = try await insertFrame(text: "airline flight booking confirmation Lisbon departure gate Tuesday")
        _ = await runUntilSettled(makeIndexer())
        // Only the best match is asked for; the other stands out too, just below the cutoff.
        let before = await service.matches(for: "flight booking Lisbon departure", limit: 1)
        let best = try XCTUnwrap(before.first?.frameID)
        let next = best == monday ? tuesday : monday

        // Retention (or a delete) removes exactly the frame the vectors in memory rank best.
        try await database.deleteFrame(id: FrameID(value: best))

        let after = await service.matches(for: "flight booking Lisbon departure", limit: 1)
        XCTAssertEqual(after.map(\.frameID), [next], "the match below the cutoff is found in the same search")
        let loaded = await service.loadedVectorCount(kind: .text)
        XCTAssertEqual(loaded, 13, "and the deleted frame's vector is gone from memory")
    }

    func testTextRedactedAndIndexedAgainDuringASearchDoesNotProduceAMatch() async throws {
        try await insertUnrelatedFrames()
        let flight = try await insertFrame(text: "airline flight booking confirmation Lisbon departure gate")
        let indexer = makeIndexer()
        _ = await runUntilSettled(indexer)

        // A search has scored the frame's vector and is about to ask the database whether it is still stored.
        let gate = Gate()
        await store.holdStoredChecks(at: gate)
        let search = Task { await self.service.matches(for: "flight booking", limit: 5) }
        await gate.waitUntilReached()

        // Meanwhile the frame's text is redacted and the indexer embeds what is left. The database now has a
        // vector for the frame again: a different one.
        _ = try await database.indexFrameText(
            mainText: "garden tomato seedlings watering calendar for the allotment",
            chromeText: nil,
            windowTitle: "Notes",
            segmentId: segmentID,
            frameId: flight
        )
        _ = await runUntilSettled(indexer)
        let present = try await database.framesWithEmbedding(kind: .text, model: model, among: [flight])
        XCTAssertEqual(present, [flight])

        gate.open()
        await store.holdStoredChecks(at: nil)
        let matches = await search.value
        XCTAssertFalse(matches.contains { $0.frameID == flight }, "the score was made from text that is gone")
    }

    func testASearchDoesNotTrustAVectorTheIndexerHasWrittenButNotYetAnnounced() async throws {
        try await insertUnrelatedFrames()
        let flight = try await insertFrame(text: "airline flight booking confirmation Lisbon departure gate")
        let indexer = makeIndexer()
        _ = await runUntilSettled(indexer)

        // A search has scored the frame's vector and is about to ask the database whether it is still stored.
        let searchGate = Gate()
        await store.holdStoredChecks(at: searchGate)
        let search = Task { await self.service.matches(for: "flight booking", limit: 5) }
        await searchGate.waitUntilReached()

        // The frame's text is redacted, and the indexer gets as far as committing the new vector to the
        // database, but not as far as telling the search service.
        _ = try await database.indexFrameText(
            mainText: "garden tomato seedlings watering calendar for the allotment",
            chromeText: nil,
            windowTitle: "Notes",
            segmentId: segmentID,
            frameId: flight
        )
        let writeGate = Gate()
        await store.holdAfterWrites(at: writeGate)
        let indexing = Task { await indexer.runPass() }
        await writeGate.waitUntilReached()

        // The search resumes exactly there: the database has a vector for the frame, a different one.
        searchGate.open()
        await store.holdStoredChecks(at: nil)
        let matches = await search.value
        XCTAssertFalse(matches.contains { $0.frameID == flight }, "the score was made from text that is gone")

        writeGate.open()
        await store.holdAfterWrites(at: nil)
        _ = await indexing.value
    }

    func testTextChangedInPlaceStopsMatchingAtOnceAndIsEmbeddedAgain() async throws {
        try await insertUnrelatedFrames()
        let flight = try await insertFrame(text: "airline flight booking confirmation Lisbon departure gate")
        let indexer = makeIndexer()
        _ = await runUntilSettled(indexer)
        let before = await service.matches(for: "flight booking", limit: 5)
        XCTAssertEqual(before.first?.frameID, flight)

        // The document's text is rewritten in place: no document is deleted, so no trigger fires.
        let maybeDocument = try await database.getDocument(frameID: FrameID(value: flight))
        let document = try XCTUnwrap(maybeDocument)
        try await database.updateDocument(id: document.id, content: "dentist appointment reminder Thursday morning checkup")

        let stale = await service.matches(for: "flight booking", limit: 5)
        XCTAssertFalse(stale.contains { $0.frameID == flight }, "the old text's vector is not used, before any re-indexing")

        _ = await runUntilSettled(indexer)
        let old = await service.matches(for: "flight booking", limit: 5)
        XCTAssertFalse(old.contains { $0.frameID == flight })
        let new = await service.matches(for: "dentist appointment", limit: 5)
        XCTAssertEqual(new.first?.frameID, flight, "the new text is searchable by meaning")
        let status = await indexer.status()
        XCTAssertEqual(status.embeddedFrames, 13)
    }

    func testTextChangedInPlaceWhileItIsBeingEmbeddedIsEmbeddedAgainNotSkipped() async throws {
        try await insertUnrelatedFrames()
        let flight = try await insertFrame(text: "airline flight booking confirmation Lisbon departure gate")
        let indexer = makeIndexer(batchSize: 16, retryBackoff: 0)

        // The first batch, with the frame in it, is in the model...
        let gate = Gate()
        embedder.hold = gate
        let pass = Task { await indexer.runPass() }
        await gate.waitUntilReached()
        // ...when the frame's text is rewritten in place. It has no vector yet, so there is nothing to requeue.
        let maybeDocument = try await database.getDocument(frameID: FrameID(value: flight))
        let document = try XCTUnwrap(maybeDocument)
        try await database.updateDocument(id: document.id, content: "dentist appointment reminder Thursday morning checkup")
        embedder.hold = nil
        gate.open()
        _ = await pass.value

        // The vector made from the old text is refused. The twelve frames around it are written, so the walk
        // has moved past it: it must be waiting to be tried again, not forgotten.
        let present = try await database.framesWithEmbedding(kind: .text, model: model, among: [flight])
        XCTAssertTrue(present.isEmpty)
        let written = try await database.frameEmbeddingCount(kind: .text, model: model)
        XCTAssertEqual(written, 12)
        let waiting = try await database.embeddingSources(retryingKind: .text, model: model, beforeFrameID: nil, limit: 10)
        XCTAssertEqual(waiting.map(\.frameID), [flight])

        _ = await runUntilSettled(indexer)
        let old = await service.matches(for: "flight booking", limit: 5)
        XCTAssertFalse(old.contains { $0.frameID == flight })
        let new = await service.matches(for: "dentist appointment", limit: 5)
        XCTAssertEqual(new.first?.frameID, flight, "the new text is searchable by meaning")
        let status = await indexer.status()
        XCTAssertEqual(status.embeddedFrames, 13)
    }

    func testAFrameWithALeftoverOlderDocumentIsIndexedOnceFromItsCurrentText() async throws {
        try await insertUnrelatedFrames()
        let frame = try await insertFrame(text: "airline flight booking confirmation Lisbon departure gate")
        // A database from before V15 can hold two documents for one frame; the one with the higher id is the
        // current one. Here they are the two newest documents, where both walks of the indexer meet.
        let maybeDocument = try await database.getDocument(frameID: FrameID(value: frame))
        let currentDocID = try XCTUnwrap(maybeDocument).id + 1
        guard let db = await database.getConnection() else { return XCTFail("no connection") }
        for sql in [
            "INSERT INTO searchRanking(rowid, text, otherText, title) VALUES (\(currentDocID), 'dentist appointment reminder Thursday morning checkup', NULL, 'Notes');",
            "INSERT INTO doc_segment (docid, segmentId, frameId) VALUES (\(currentDocID), \(segmentID), \(frame));",
        ] {
            XCTAssertEqual(sqlite3_exec(db, sql, nil, nil, nil), SQLITE_OK)
        }

        // One frame a batch, so the two documents can never be in the same one.
        let indexer = makeIndexer(batchSize: 1)
        let state = await runUntilSettled(indexer)
        XCTAssertEqual(state, .upToDate, "the indexer settles instead of embedding the two documents in turn")
        let embedded = embedder.embeddedTextCount
        _ = await runUntilSettled(indexer)
        XCTAssertEqual(embedder.embeddedTextCount, embedded, "and stays settled")

        let status = await indexer.status()
        XCTAssertEqual(status.embeddedFrames, 13)
        XCTAssertEqual(status.indexedFrames, 13)
        let old = await service.matches(for: "flight booking", limit: 5)
        XCTAssertFalse(old.contains { $0.frameID == frame }, "the leftover document's text is not searchable")
        let current = await service.matches(for: "dentist appointment", limit: 5)
        XCTAssertEqual(current.first?.frameID, frame)
    }

    func testVectorsLoadedBySearchAreFreedWhenTheFeatureIsOffEvenIfIndexingNeverSawItOn() async throws {
        try await insertUnrelatedFrames()
        _ = await runUntilSettled(makeIndexer())
        // The search box loads the vectors and starts the model...
        await service.prepareForSearch()
        var loaded = await service.loadedVectorCount(kind: .text)
        XCTAssertEqual(loaded, 12)

        // ...and the feature is switched off before an indexer that has only ever seen it off runs again.
        let neverSawItOn = makeIndexer(enabled: false)
        _ = await neverSawItOn.runPass()

        loaded = await service.loadedVectorCount(kind: .text)
        XCTAssertEqual(loaded, 0)
        let state = await neverSawItOn.status().state
        XCTAssertEqual(state, .off)
    }

    func testAVectorReplacedWhileTheVectorsAreBeingReadDoesNotSurviveTheRead() async throws {
        try await insertUnrelatedFrames()
        let flight = try await insertFrame(text: "airline flight booking confirmation Lisbon departure gate")
        let indexer = makeIndexer()
        _ = await runUntilSettled(indexer)   // no search yet: the vectors are not in memory

        // The first search starts reading the vectors and has read the frame's (old) one.
        let readGate = Gate()
        await store.holdVectorReads(at: readGate)
        let search = Task { await self.service.matches(for: "flight booking", limit: 5) }
        await readGate.waitUntilReached()

        // The frame's text is redacted, and the indexer commits the new vector but has not yet announced it.
        _ = try await database.indexFrameText(
            mainText: "garden tomato seedlings watering calendar for the allotment",
            chromeText: nil,
            windowTitle: "Notes",
            segmentId: segmentID,
            frameId: flight
        )
        let writeGate = Gate()
        await store.holdAfterWrites(at: writeGate)
        let indexing = Task { await indexer.runPass() }
        await writeGate.waitUntilReached()

        // The read finishes with the old vector in hand, and the search ranks.
        readGate.open()
        await store.holdVectorReads(at: nil)
        let matches = await search.value
        XCTAssertFalse(matches.contains { $0.frameID == flight }, "the vector read before the redaction was dropped")

        writeGate.open()
        await store.holdAfterWrites(at: nil)
        _ = await indexing.value
        let later = await service.matches(for: "flight booking", limit: 5)
        XCTAssertFalse(later.contains { $0.frameID == flight })
    }

    func testTwoSearchesAtOnceBothGetTheirMatchesWhenAVectorHasGoneStale() async throws {
        try await insertUnrelatedFrames()
        let monday = try await insertFrame(text: "airline flight booking confirmation Lisbon departure gate Monday")
        let tuesday = try await insertFrame(text: "airline flight booking confirmation Lisbon departure gate Tuesday")
        _ = await runUntilSettled(makeIndexer())
        let before = await service.matches(for: "flight booking Lisbon departure", limit: 1)
        let best = try XCTUnwrap(before.first?.frameID)
        let next = best == monday ? tuesday : monday
        try await database.deleteFrame(id: FrameID(value: best))   // its vector is still in memory

        // Two searches have both scored the stale vector and are about to check the database.
        let gate = Gate()
        await store.holdStoredChecks(at: gate)
        let first = Task { await self.service.matches(for: "flight booking Lisbon departure", limit: 1) }
        let second = Task { await self.service.matches(for: "flight booking Lisbon gate", limit: 1) }
        await gate.waitUntilReached()
        try await Task.sleep(for: .milliseconds(50))   // let the other one arrive at the gate too
        gate.open()
        await store.holdStoredChecks(at: nil)

        let firstMatches = await first.value
        let secondMatches = await second.value
        XCTAssertEqual(firstMatches.map(\.frameID), [next], "one search re-reading the vectors must not empty the other")
        XCTAssertEqual(secondMatches.map(\.frameID), [next])
    }

    func testASearchUnderWayWhenTheFeatureIsSwitchedOffReturnsNothingAndReloadsNothing() async throws {
        try await insertUnrelatedFrames()
        try await insertFrame(text: "airline flight booking confirmation Lisbon departure gate")
        _ = await runUntilSettled(makeIndexer())
        let switchedOn = Flag()   // set = switched off
        let switchable = SemanticSearchService(
            store: store,
            embedder: embedder,
            model: { [model] _ in model },
            dimension: dimension,
            isEnabled: { !switchedOn.isSet }
        )
        let found = await switchable.matches(for: "flight booking", limit: 5)
        XCTAssertEqual(found.count, 1)

        // A search is waiting for its query to be embedded when the feature is switched off and unloaded.
        let gate = Gate()
        embedder.hold = gate
        let search = Task { await switchable.matches(for: "flight booking", limit: 5) }
        await gate.waitUntilReached()
        switchedOn.set()
        await switchable.unload()
        embedder.hold = nil
        gate.open()

        let matches = await search.value
        XCTAssertTrue(matches.isEmpty, "the feature is off: no matches")
        let loaded = await switchable.loadedVectorCount(kind: .text)
        XCTAssertEqual(loaded, 0, "and the vectors were not read back into memory")
    }

    func testVectorsStoredWhileTheIndexIsBeingLoadedAreNotLost() async throws {
        try await insertUnrelatedFrames()
        let indexer = makeIndexer(batchSize: 4)
        _ = await indexer.runPass()   // the newest four frames are stored; the index isn't loaded yet

        // The first search reads those four from the database and is held there, with the read position
        // already past the older frames.
        let gate = Gate()
        await store.holdVectorReads(at: gate)
        let firstSearch = Task { await self.service.matches(for: "revenue forecast", limit: 5) }
        await gate.waitUntilReached()

        // Meanwhile the indexer writes the eight older frames.
        _ = await runUntilSettled(indexer)
        let stored = try await database.frameEmbeddings(kind: .text, model: model, afterFrameID: -1, limit: 100)
        XCTAssertEqual(stored.count, 12)

        gate.open()
        _ = await firstSearch.value
        let loaded = await service.loadedVectorCount(kind: .text)
        XCTAssertEqual(loaded, 12, "the eight written behind the read position are in memory too")
    }

    func testTurningSearchOffWhileTheIndexIsBeingLoadedLeavesNothingInMemory() async throws {
        try await insertUnrelatedFrames()
        let flight = try await insertFrame(text: "airline flight booking confirmation Lisbon departure gate")
        _ = await runUntilSettled(makeIndexer())

        let gate = Gate()
        await store.holdVectorReads(at: gate)
        let search = Task { await self.service.matches(for: "flight booking", limit: 5) }
        await gate.waitUntilReached()

        // Turned off (or the index removed) while the vectors are still being read.
        await service.unload()
        gate.open()
        let interrupted = await search.value
        XCTAssertTrue(interrupted.isEmpty)
        var loaded = await service.loadedVectorCount(kind: .text)
        XCTAssertEqual(loaded, 0, "the read that was under way must not bring the vectors back")

        // The next search starts a read of its own and works.
        await store.holdVectorReads(at: nil)
        let matches = await service.matches(for: "flight booking", limit: 5)
        XCTAssertEqual(matches.first?.frameID, flight)
        loaded = await service.loadedVectorCount(kind: .text)
        XCTAssertEqual(loaded, 13)
    }

    func testVectorsLeaveMemoryAfterSearchHasBeenIdleAndComeBackOnTheNextSearch() async throws {
        try await insertUnrelatedFrames()
        let flight = try await insertFrame(text: "airline flight booking confirmation Lisbon departure gate")
        _ = await runUntilSettled(makeIndexer())
        _ = await service.matches(for: "flight booking", limit: 5)
        var loaded = await service.loadedVectorCount(kind: .text)
        XCTAssertEqual(loaded, 13)

        await service.releaseIndexIfIdle(now: Date().addingTimeInterval(60), after: 600)
        loaded = await service.loadedVectorCount(kind: .text)
        XCTAssertEqual(loaded, 13, "still in use a minute after a search")

        await service.releaseIndexIfIdle(now: Date().addingTimeInterval(601), after: 600)
        loaded = await service.loadedVectorCount(kind: .text)
        XCTAssertEqual(loaded, 0)

        let matches = await service.matches(for: "flight booking", limit: 5)
        XCTAssertEqual(matches.first?.frameID, flight)
    }

    func testStoppingTheIndexerWaitsForTheBatchInFlightAndWritesNothingAfterwards() async throws {
        try await insertUnrelatedFrames()
        let indexer = makeIndexer(batchSize: 4)

        // The first batch is in the model when the stop is asked for.
        let gate = Gate()
        embedder.hold = gate
        await indexer.start()
        await gate.waitUntilReached()

        let stopped = Flag()
        let stop = Task {
            await indexer.stopAndWait()
            stopped.set()
        }
        try await Task.sleep(for: .milliseconds(100))
        XCTAssertFalse(stopped.isSet, "stopAndWait returns only once the batch in flight has ended")

        gate.open()
        await stop.value
        let written = try await database.frameEmbeddingCount(kind: .text, model: model)
        XCTAssertEqual(written, 0, "the batch that was in flight is abandoned, not written")
        try await Task.sleep(for: .milliseconds(100))
        let later = try await database.frameEmbeddingCount(kind: .text, model: model)
        XCTAssertEqual(later, 0, "and nothing is written after stopAndWait has returned")
    }

    func testABatchThatCannotBeWrittenIsNotRetriedInATightLoop() async throws {
        try await insertUnrelatedFrames()
        var configuration = SemanticIndexer.Configuration()
        configuration.batchSize = 4
        configuration.newFrameMaximumWait = 0
        let indexer = SemanticIndexer(
            store: store,
            service: service,
            model: { [model] _ in model },
            configuration: configuration,
            isEnabled: { true },
            isImagesEnabled: { false },
            isReady: { _ in nil },
            isOnPower: { true }
        )

        // Every frame of the batch changed while it was being embedded: the store writes none of them.
        await store.refuseWrites(true)
        let wait = await indexer.runPass()
        XCTAssertEqual(wait, configuration.idleInterval, "the same batch is next, so the indexer waits before it")
        let state = await indexer.status().state
        XCTAssertEqual(state, .indexing)

        await store.refuseWrites(false)
        let next = await indexer.runPass()
        XCTAssertEqual(next, .zero)
    }

    func testAFrameTheUserDeletedIsNotEmbedded() async throws {
        try await insertUnrelatedFrames()
        let flight = try await insertFrame(text: "airline flight booking confirmation Lisbon departure gate")
        // Deleted from the timeline before the indexer got to it: the frame and its text are still in the
        // database, waiting for the video to be rewritten.
        _ = try await database.deleteOrScheduleNativeFramesForDeletion(frameIDs: [flight])

        let indexer = makeIndexer()
        let state = await runUntilSettled(indexer)

        XCTAssertEqual(state, .upToDate)
        let status = await indexer.status()
        XCTAssertEqual(status.embeddedFrames, 13, "it is accounted for")
        let stored = try await database.frameEmbeddings(kind: .text, model: model, afterFrameID: -1, limit: 100)
        XCTAssertEqual(stored.count, 12)
        XCTAssertFalse(stored.contains { $0.frameID == flight })
        let matches = await service.matches(for: "flight booking", limit: 5)
        XCTAssertFalse(matches.contains { $0.frameID == flight })
    }

    func testDeletingAFrameAfterItWasEmbeddedStopsItMatching() async throws {
        try await insertUnrelatedFrames()
        let flight = try await insertFrame(text: "airline flight booking confirmation Lisbon departure gate")
        let indexer = makeIndexer()
        _ = await runUntilSettled(indexer)
        let before = await service.matches(for: "flight booking", limit: 5)
        XCTAssertEqual(before.first?.frameID, flight)

        _ = try await database.deleteOrScheduleNativeFramesForDeletion(frameIDs: [flight])

        // Its vector is still in memory; the search checks the database before returning it.
        let after = await service.matches(for: "flight booking", limit: 5)
        XCTAssertFalse(after.contains { $0.frameID == flight })
        // And the indexer does not embed it again while it waits to be removed from its video.
        _ = await runUntilSettled(indexer)
        let stored = try await database.frameEmbeddings(kind: .text, model: model, afterFrameID: -1, limit: 100)
        XCTAssertFalse(stored.contains { $0.frameID == flight })
    }

    func testEmbedderFailureLeavesFramesForTheNextPass() async throws {
        try await insertUnrelatedFrames()
        let indexer = makeIndexer(batchSize: 4)

        embedder.failNextCalls = 1
        _ = await indexer.runPass()
        var status = await indexer.status()
        XCTAssertEqual(status.state, .failed)
        XCTAssertEqual(status.embeddedFrames, 0)

        let state = await runUntilSettled(indexer)
        XCTAssertEqual(state, .upToDate)
        status = await indexer.status()
        XCTAssertEqual(status.embeddedFrames, 12)
    }

    // MARK: - VectorIndex

    func testIndexReturnsOnlyRowsThatStandOutBestFirst() {
        let index = VectorIndex(dimension: 4)
        func unit(_ values: [Float]) -> [Float] { EmbeddingVector.truncated(values, to: 4)! }
        // Twenty rows pointing away from the query, two pointing at it to different degrees.
        for frame in 0..<20 {
            index.upsert(frameID: Int64(frame), vector: EmbeddingVector.storageData(unit([0, 1, Float(frame) * 0.01, 0])))
        }
        index.upsert(frameID: 100, vector: EmbeddingVector.storageData(unit([1, 0.1, 0, 0])))
        index.upsert(frameID: 101, vector: EmbeddingVector.storageData(unit([1, 0.6, 0, 0])))

        let matches = index.matches(for: unit([1, 0, 0, 0]), limit: 10, gate: 2.0)
        XCTAssertEqual(matches.map(\.frameID), [100, 101])
        XCTAssertGreaterThan(matches[0].cosine, 0.99)

        // A replaced vector is scored as the new one; data of the wrong size is ignored.
        index.upsert(frameID: 100, vector: EmbeddingVector.storageData(unit([0, 1, 0, 0])))
        index.upsert(frameID: 102, vector: Data(repeating: 0, count: 3))
        XCTAssertEqual(index.count, 22)
        let afterReplace = index.matches(for: unit([1, 0, 0, 0]), limit: 10, gate: 2.0)
        XCTAssertEqual(afterReplace.map(\.frameID), [101])

        // Removing a row moves the last one into its place; the rest still score as before.
        index.remove(frameID: 3)
        index.remove(frameID: 12345)
        XCTAssertEqual(index.count, 21)
        XCTAssertFalse(index.frameIDs.contains(3))
        let afterRemove = index.matches(for: unit([1, 0, 0, 0]), limit: 10, gate: 2.0)
        XCTAssertEqual(afterRemove.map(\.frameID), [101])
        index.remove(frameID: 101)
        XCTAssertTrue(index.matches(for: unit([1, 0, 0, 0]), limit: 10, gate: 2.0).isEmpty)
    }

    func testScoresOfALargeIndexMatchAPlainDotProduct() {
        // More rows than one conversion block (2,048), with a ragged last block.
        let (rows, dimension) = (5000, 8)
        let index = VectorIndex(dimension: dimension)
        var vectors: [[Float]] = []
        var state: UInt64 = 42
        func next() -> Float {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float(Int64(bitPattern: state >> 11) % 2001 - 1000) / 1000
        }
        for row in 0..<rows {
            let vector = EmbeddingVector.truncated((0..<dimension).map { _ in next() }, to: dimension)!
            vectors.append(vector)
            index.upsert(frameID: Int64(row), vector: EmbeddingVector.storageData(vector))
        }
        let query = EmbeddingVector.truncated((0..<dimension).map { _ in next() }, to: dimension)!

        let scores = index.scores(for: query)

        XCTAssertEqual(scores.count, rows)
        for row in [0, 1, 2047, 2048, 2049, 4095, 4096, 4999] {
            let expected = zip(vectors[row], query).reduce(Float(0)) { $0 + $1.0 * $1.1 }
            XCTAssertEqual(scores[row], expected, accuracy: 2e-3, "row \(row) (float16 storage costs about 1e-3)")
        }
        let best = scores.indices.max { scores[$0] < scores[$1] }!
        let expectedBest = vectors.indices.max { a, b in
            zip(vectors[a], query).reduce(Float(0)) { $0 + $1.0 * $1.1 } < zip(vectors[b], query).reduce(Float(0)) { $0 + $1.0 * $1.1 }
        }!
        XCTAssertEqual(index.matches(for: query, limit: 1, gate: 0).first?.frameID, Int64(best))
        XCTAssertEqual(best, expectedBest)
    }

    func testTruncationKeepsLeadingValuesAtUnitLength() throws {
        let truncated = try XCTUnwrap(EmbeddingVector.truncated([3, 4, 12, 99], to: 2))
        XCTAssertEqual(truncated[0], 0.6, accuracy: 1e-6)
        XCTAssertEqual(truncated[1], 0.8, accuracy: 1e-6)
        XCTAssertNil(EmbeddingVector.truncated([0, 0, 1], to: 2), "a zero vector has no direction")
        XCTAssertNil(EmbeddingVector.truncated([1], to: 2))
    }

    // MARK: - Prompts

    func testQueryTextDropsKeywordSearchSyntax() {
        XCTAssertEqual(SemanticSearchService.queryText("pricing page -draft app:Chrome \"expert calls\""), "pricing page expert calls")
        XCTAssertEqual(SemanticSearchService.queryText("meeting at 10:30 re: budget"), "meeting at 10:30 re: budget")
        XCTAssertEqual(SemanticSearchService.queryText("  -only -excluded  "), "")
        // An excluded phrase is excluded whole, exactly as keyword search reads it.
        XCTAssertEqual(SemanticSearchService.queryText("pricing page -\"internal draft\""), "pricing page")
        XCTAssertEqual(SemanticSearchService.queryText("budget after:2026-01-01 before:2026-02-01"), "budget")
    }

    func testDocumentPromptCarriesWindowTitleAndSkipsNearEmptyText() {
        XCTAssertEqual(
            SemanticSearchService.documentPrompt(windowName: " Inbox ", text: "  Your refund has been processed today.  "),
            "title: Inbox | text: Your refund has been processed today."
        )
        XCTAssertEqual(
            SemanticSearchService.documentPrompt(windowName: nil, text: "Your refund has been processed today."),
            "title: none | text: Your refund has been processed today."
        )
        XCTAssertNil(SemanticSearchService.documentPrompt(windowName: "Inbox", text: "ok   "))
    }
}

// MARK: - Benchmark

/// How one scan of the in-memory index scales, on this machine. Skipped unless asked for:
///
///   RETRACE_SEMANTIC_BENCH_ROWS=1000000 swift test --filter VectorIndexBenchmarks
///
/// (several row counts may be given, separated by commas). Prints, per row count: the memory the index holds, the
/// time to build it from stored vectors, and the median and 95th-percentile time of one query (score every row,
/// gate, sort, take 200), which is everything a search does after the query has been embedded. The test build is
/// not optimized, so an app build is somewhat faster than what this prints.
final class VectorIndexBenchmarks: XCTestCase {
    func testScanTimeAndMemoryAtScale() throws {
        guard let setting = ProcessInfo.processInfo.environment["RETRACE_SEMANTIC_BENCH_ROWS"] else {
            throw XCTSkip("set RETRACE_SEMANTIC_BENCH_ROWS (for example 100000,1000000) to run")
        }
        let dimension = SemanticSearchSettings.dimension
        var state: UInt64 = 0x9E3779B97F4A7C15
        func randomUnitVector() -> [Float] {
            var values = [Float](repeating: 0, count: dimension)
            for index in 0..<dimension {
                state = state &* 6364136223846793005 &+ 1442695040888963407
                values[index] = Float(Int64(bitPattern: state >> 11) % 2001 - 1000) / 1000
            }
            return EmbeddingVector.truncated(values, to: dimension)!
        }
        // A few hundred distinct stored vectors, reused: what is timed is the scan, not random number generation.
        let stored = (0..<512).map { _ in EmbeddingVector.storageData(randomUnitVector()) }
        let queries = (0..<40).map { _ in randomUnitVector() }

        for rows in setting.split(separator: ",").compactMap({ Int($0.trimmingCharacters(in: .whitespaces)) }) {
            let before = Self.memoryFootprint()
            let buildStarted = Date()
            let index = VectorIndex(dimension: dimension)
            index.reserve(rows: rows)   // as the service does when it loads stored vectors
            for row in 0..<rows {
                index.upsert(frameID: Int64(row), vector: stored[row % stored.count])
            }
            let buildSeconds = Date().timeIntervalSince(buildStarted)
            let heldMB = Double(Self.memoryFootprint() - before) / 1_000_000

            var milliseconds: [Double] = []
            var matched = 0
            for query in queries {
                let started = Date()
                matched += index.matches(for: query, limit: 200, gate: 2.0).count
                milliseconds.append(Date().timeIntervalSince(started) * 1000)
            }
            milliseconds.sort()
            let median = milliseconds[milliseconds.count / 2]
            let p95 = milliseconds[Int(Double(milliseconds.count - 1) * 0.95)]
            print(String(
                format: "[bench] %d rows × %d: index holds %.0f MB, built in %.2f s, one query %.1f ms median, %.1f ms p95 (%d matches a query)",
                rows, dimension, heldMB, buildSeconds, median, p95, matched / queries.count
            ))
            XCTAssertEqual(index.count, rows)
        }
    }

    /// The process's physical memory footprint in bytes (what Activity Monitor shows as Memory).
    private static func memoryFootprint() -> Int64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return result == KERN_SUCCESS ? Int64(info.phys_footprint) : 0
    }
}

// MARK: - Stand-in embedding model

/// Bag-of-words vectors: every word has a fixed pseudo-random direction; a text is the normalized sum of its words.
/// Prompt scaffolding (`task:`, `title:`, `|` …) is ignored so a query and a document meet on their content words.
private final class WordEmbedder: TextEmbedding, @unchecked Sendable {
    private let lock = NSLock()
    private var texts = 0
    private var failures = 0
    private let dimension = 96

    private var holdGate: Gate?
    private var shutdowns = 0

    var embeddedTextCount: Int { lock.withLock { texts } }
    var shutdownCount: Int { lock.withLock { shutdowns } }
    var failNextCalls: Int {
        get { lock.withLock { failures } }
        set { lock.withLock { failures = newValue } }
    }
    /// When set, every call waits at this gate before answering (a batch that is "in the model").
    var hold: Gate? {
        get { lock.withLock { holdGate } }
        set { lock.withLock { holdGate = newValue } }
    }

    struct Failed: Error {}

    func embed(_ texts: [String]) async throws -> [[Float]] {
        let gate: Gate? = try lock.withLock {
            if failures > 0 {
                failures -= 1
                throw Failed()
            }
            self.texts += texts.count
            return holdGate
        }
        await gate?.pass()
        return texts.map(vector)
    }

    func shutdown() { lock.withLock { shutdowns += 1 } }

    func vector(_ text: String) -> [Float] {
        let scaffolding: Set<String> = ["task", "search", "result", "query", "title", "text", "none", "notes"]
        var sum = [Float](repeating: 0, count: dimension)
        let words = text.lowercased().split { !$0.isLetter && !$0.isNumber }.map(String.init)
        for word in words where !scaffolding.contains(word) {
            var state: UInt64 = 0xcbf29ce484222325
            for byte in word.utf8 { state = (state ^ UInt64(byte)) &* 0x100000001b3 }
            for index in 0..<dimension {
                state = state &* 6364136223846793005 &+ 1442695040888963407
                sum[index] += Float(Int64(bitPattern: state >> 11) % 2001 - 1000) / 1000
            }
        }
        let norm = sum.reduce(0) { $0 + $1 * $1 }.squareRoot()
        return norm > 0 ? sum.map { $0 / norm } : sum.enumerated().map { $0.offset == 0 ? 1 : 0 }
    }
}

// MARK: - Forcing an interleaving

/// A point a task waits at until the test lets it through, so the test decides what happens in between.
private final class Gate: @unchecked Sendable {
    private let lock = NSLock()
    private var reached = false
    private var isOpen = false
    private var reachedWaiters: [CheckedContinuation<Void, Never>] = []
    private var passWaiters: [CheckedContinuation<Void, Never>] = []

    /// Called by the code under test: reports arrival and waits until the gate is open.
    func pass() async {
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            lock.lock()
            reached = true
            let arrived = reachedWaiters
            reachedWaiters = []
            if isOpen {
                lock.unlock()
                continuation.resume()
            } else {
                passWaiters.append(continuation)
                lock.unlock()
            }
            arrived.forEach { $0.resume() }
        }
    }

    /// Called by the test: returns once a task is waiting at the gate.
    func waitUntilReached() async {
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            lock.lock()
            if reached {
                lock.unlock()
                continuation.resume()
            } else {
                reachedWaiters.append(continuation)
                lock.unlock()
            }
        }
    }

    func open() {
        lock.lock()
        isOpen = true
        let waiting = passWaiters
        passWaiters = []
        lock.unlock()
        waiting.forEach { $0.resume() }
    }
}

private final class Flag: @unchecked Sendable {
    private let lock = NSLock()
    private var value = false
    var isSet: Bool { lock.withLock { value } }
    func set() { lock.withLock { value = true } }
}

/// The real database behind two switches: hold a read of stored vectors after it has happened (what was read is
/// now a moment old), and refuse every write (as the store does when a frame changed while it was embedded).
private actor InterruptibleStore: EmbeddingStoreProtocol {
    private let base: DatabaseManager
    private var vectorReadGate: Gate?
    private var storedCheckGate: Gate?
    private var afterWriteGate: Gate?
    private var refusesWrites = false

    init(base: DatabaseManager) {
        self.base = base
    }

    func holdVectorReads(at gate: Gate?) { vectorReadGate = gate }
    /// Holds the question "are these vectors still stored?" before it is put to the database.
    func holdStoredChecks(at gate: Gate?) { storedCheckGate = gate }
    /// Holds a write after it has been committed, before its caller hears about it.
    func holdAfterWrites(at gate: Gate?) { afterWriteGate = gate }
    func refuseWrites(_ refuse: Bool) { refusesWrites = refuse }

    func embeddingSources(afterDocID docID: Int64, limit: Int) async throws -> [FrameEmbeddingSource] {
        try await base.embeddingSources(afterDocID: docID, limit: limit)
    }

    func embeddingSources(beforeDocID docID: Int64, limit: Int) async throws -> [FrameEmbeddingSource] {
        try await base.embeddingSources(beforeDocID: docID, limit: limit)
    }

    func embeddingSources(
        retryingKind kind: EmbeddingKind,
        model: String,
        beforeFrameID: Int64?,
        limit: Int
    ) async throws -> [FrameEmbeddingSource] {
        try await base.embeddingSources(retryingKind: kind, model: model, beforeFrameID: beforeFrameID, limit: limit)
    }

    func embeddingDocIDBounds(kind: EmbeddingKind, model: String) async throws -> (min: Int64, max: Int64)? {
        try await base.embeddingDocIDBounds(kind: kind, model: model)
    }

    func latestIndexedDocID() async throws -> Int64? {
        try await base.latestIndexedDocID()
    }

    func storeFrameEmbeddings(
        _ records: [FrameEmbeddingRecord],
        from sources: [FrameEmbeddingSource],
        model: String
    ) async throws -> [FrameEmbeddingRecord] {
        if refusesWrites { return [] }
        let stored = try await base.storeFrameEmbeddings(records, from: sources, model: model)
        if let gate = afterWriteGate, !stored.isEmpty {
            await gate.pass()
        }
        return stored
    }

    func framesWithEmbedding(kind: EmbeddingKind, model: String, among frameIDs: [Int64]) async throws -> Set<Int64> {
        if let gate = storedCheckGate {
            await gate.pass()
        }
        return try await base.framesWithEmbedding(kind: kind, model: model, among: frameIDs)
    }

    func frameEmbeddings(
        kind: EmbeddingKind,
        model: String,
        afterFrameID frameID: Int64,
        limit: Int
    ) async throws -> [FrameEmbeddingRecord] {
        let page = try await base.frameEmbeddings(kind: kind, model: model, afterFrameID: frameID, limit: limit)
        if let gate = vectorReadGate, !page.isEmpty {
            await gate.pass()
        }
        return page
    }

    func frameEmbeddingCount(kind: EmbeddingKind, model: String) async throws -> Int {
        try await base.frameEmbeddingCount(kind: kind, model: model)
    }

    func indexedDocumentCount() async throws -> Int {
        try await base.indexedDocumentCount()
    }

    func deleteFrameEmbeddings(kind: EmbeddingKind, exceptModel model: String) async throws -> Int {
        try await base.deleteFrameEmbeddings(kind: kind, exceptModel: model)
    }

    func deleteAllFrameEmbeddings(kind: EmbeddingKind?) async throws {
        try await base.deleteAllFrameEmbeddings(kind: kind)
    }
}

// MARK: - Screenshots

/// Screenshot embedding against a real SQLite database. The stand-in image encoder reads a "topic" out of each
/// test image's color and returns the word vector of that topic's text, so an image lands where its description
/// would: enough to check which screenshots are embedded, when, and that a text query finds them.
final class SemanticImageIndexingTests: XCTestCase {
    private let dimension = 64
    private var database: DatabaseManager!
    private var words: WordEmbedder!
    private var images: TopicImageEmbedder!
    private var provider: TopicImageProvider!
    private var service: SemanticSearchService!
    private var videoID: Int64 = 0
    private var segmentID: Int64 = 0
    private var frameCount = 0

    private static let topics = [
        "quarterly revenue forecast spreadsheet columns",
        "airline flight booking confirmation Lisbon departure gate",
        "marathon training schedule intervals tempo",
    ]

    override func setUp() async throws {
        database = DatabaseManager(databasePath: "file:memdb_\(UUID().uuidString)?mode=memory&cache=private")
        try await database.initialize()
        videoID = try await database.insertVideoSegment(
            VideoSegment(
                id: VideoSegmentID(value: 0),
                startTime: Date(),
                endTime: Date().addingTimeInterval(300),
                frameCount: 150,
                fileSizeBytes: 1024,
                relativePath: "chunks/test",
                width: 1920,
                height: 1080,
                source: .native
            )
        )
        try await database.markVideoFinalized(id: videoID, frameCount: 150, fileSize: 1024)
        segmentID = try await database.insertSegment(
            bundleID: "com.test.app",
            startDate: Date(),
            endDate: Date().addingTimeInterval(300),
            windowName: "Notes",
            browserUrl: nil,
            type: 0
        )
        words = WordEmbedder()
        images = TopicImageEmbedder(words: words, topics: Self.topics)
        provider = TopicImageProvider()
        service = SemanticSearchService(
            store: database,
            embedder: words,
            imageEmbedder: images,
            imageProvider: provider,
            model: { $0 == .text ? "test-words/64" : "test-images/64" },
            dimension: dimension,
            isEnabled: { true }
        )
        frameCount = 0
    }

    override func tearDown() async throws {
        try await database.close()
        database = nil
    }

    private func makeIndexer(
        onPower: Bool = true,
        retryBackoff: TimeInterval = 300,
        retryTimeout: TimeInterval = 24 * 3600,
        retryScanLimit: Int = 256,
        retryGrace: TimeInterval = 0
    ) -> SemanticIndexer {
        var configuration = SemanticIndexer.Configuration()
        configuration.batchSize = 4
        configuration.imageBatchSize = 2
        configuration.newFrameMaximumWait = 0
        configuration.retryGrace = retryGrace
        configuration.retryScanLimit = retryScanLimit
        configuration.retryBackoff = retryBackoff
        configuration.retryTimeout = retryTimeout
        return SemanticIndexer(
            store: database,
            service: service,
            model: { $0 == .text ? "test-words/64" : "test-images/64" },
            configuration: configuration,
            isEnabled: { true },
            isImagesEnabled: { true },
            isReady: { _ in nil },
            isOnPower: { onPower }
        )
    }

    /// A frame with OCR text whose screenshot shows `topic`. `status` 2 is "OCR finished", the state in which a
    /// screenshot's pixels are final; a freshly captured frame is 4 ("not yet readable").
    @discardableResult
    private func insertFrame(text: String, topic: Int, status: Int = 2) async throws -> Int64 {
        frameCount += 1
        let frameID = try await database.insertFrame(
            FrameReference(
                id: FrameID(value: 0),
                timestamp: Date().addingTimeInterval(Double(frameCount) - 100),
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
            windowTitle: "Notes",
            segmentId: segmentID,
            frameId: frameID
        )
        try await database.updateFrameProcessingStatus(frameID: frameID, status: status)
        provider.setTopic(topic, forFrame: frameID)
        return frameID
    }

    private func insertBackground(count: Int = 12) async throws {
        for index in 0..<count {
            try await insertFrame(text: "garden tomato seedlings watering calendar entry number \(index)", topic: index % 2 == 0 ? 0 : 2)
        }
    }

    private func settle(_ indexer: SemanticIndexer, maxPasses: Int = 80) async -> SemanticIndexStatus {
        for _ in 0..<maxPasses {
            await indexer.runPass()
            let status = await indexer.status()
            if status.state != .indexing { return status }
        }
        return await indexer.status()
    }

    private func imageVectors() async throws -> [FrameEmbeddingRecord] {
        try await database.frameEmbeddings(kind: .image, model: "test-images/64", afterFrameID: -1, limit: 100)
    }

    func testAScreenshotIsFoundByWhatItShowsWhenItsTextSaysNothing() async throws {
        try await insertBackground()
        // No readable text on screen (a game, a video): only the screenshot can match.
        let flight = try await insertFrame(text: "ok", topic: 1)

        let status = await settle(makeIndexer())
        XCTAssertEqual(status.state, .upToDate)
        XCTAssertEqual(status.embeddedFrames, 13)
        XCTAssertEqual(status.embeddedImages, 13)
        XCTAssertTrue(status.imagesEnabled)

        let matches = await service.matches(for: "flight booking", limit: 5)
        XCTAssertEqual(matches.first?.frameID, flight)
        XCTAssertEqual(matches.first?.kind, .image)

        // 13 screenshots in batches of two: cached video state is dropped before each batch, so a chunk rewritten
        // for a redaction is never read through a decoder opened before the rewrite, and again after it, so no
        // decoder stays open once there is nothing left to index.
        XCTAssertEqual(provider.discardCount, 14)
        XCTAssertFalse(provider.holdsCachedFrames)

        // Every screenshot reached the encoder already at its final size: sides in multiples of 48 pixels.
        XCTAssertFalse(images.sizes.isEmpty)
        XCTAssertTrue(images.sizes.allSatisfy { $0.width % 48 == 0 && $0.height % 48 == 0 })
    }

    func testAFrameThatMatchesByTextAndByScreenshotRanksAboveOneThatMatchesByTextAlone() async throws {
        try await insertBackground()
        // The same words on both; only one of them also shows the thing.
        let wordsOnly = try await insertFrame(text: "airline flight booking confirmation Lisbon departure gate", topic: 0)
        let wordsAndPicture = try await insertFrame(text: "airline flight booking confirmation Lisbon departure gate", topic: 1)
        _ = await settle(makeIndexer())

        let matches = await service.matches(for: "flight booking", limit: 5)

        XCTAssertEqual(matches.map(\.frameID), [wordsAndPicture, wordsOnly])
        let both = try XCTUnwrap(matches.first)
        let textOnly = try XCTUnwrap(matches.last)
        XCTAssertGreaterThan(both.score, textOnly.score, "its screenshot adds to its standing")
    }

    func testAFrameSecondInBothTextAndScreenshotIsNotLostBehindTheFirstOfEach() async throws {
        try await insertBackground(count: 24)
        // Best by text, with a screenshot of something else.
        let textOnly = try await insertFrame(text: "airline flight booking confirmation Lisbon departure gate", topic: 0)
        // Second by text (the words are there among many others) and a screenshot of the flight booking.
        let both = try await insertFrame(
            text: "weekly reminders: renew passport, water the plants, check the flight booking, call the dentist, pay rent",
            topic: 1
        )
        // Best by screenshot (it ties with the frame above and is newer), with text about something else.
        let imageOnly = try await insertFrame(text: "garden tomato seedlings watering calendar entry number 99", topic: 1)
        _ = await settle(makeIndexer())

        let all = await service.matches(for: "flight booking", limit: 10)
        
        XCTAssertEqual(all.first?.frameID, both)

        // Asked for one match: each kind's own best is a different frame, and neither is the best overall.
        let best = await service.matches(for: "flight booking", limit: 1)
        XCTAssertEqual(best.map(\.frameID), [both])
    }

    func testAScreenshotIsEmbeddedOnlyOnceItsPixelsAreFinalAndNewerFramesDoNotWaitForIt() async throws {
        // Still being encoded (its video chunk is not full yet), followed by more frames than fit one batch.
        let pending = try await insertFrame(text: "airline flight booking confirmation Lisbon departure gate", topic: 1, status: 4)
        var ready: [Int64] = []
        for index in 0..<5 {
            ready.append(try await insertFrame(text: "marathon training schedule intervals tempo, day \(index)", topic: 2))
        }

        let indexer = makeIndexer()
        var status = await settle(indexer)
        XCTAssertEqual(status.embeddedFrames, 6, "text doesn't wait for the screenshot")
        var stored = try await imageVectors()
        XCTAssertEqual(Set(stored.map(\.frameID)), Set(ready), "the walk moves past the frame that isn't ready")
        XCTAssertEqual(status.embeddedImages, 5, "the waiting frame is not counted as done")
        XCTAssertFalse(provider.requestedFrames.contains(pending), "its pixels were not read")
        var waiting = try await database.embeddingSources(retryingKind: .image, model: "test-images/64", beforeFrameID: nil, limit: 10)
        XCTAssertEqual(waiting.map(\.frameID), [pending])

        // OCR finishes (and any redaction is applied): now the screenshot is embedded, however long that took.
        try await database.updateFrameProcessingStatus(frameID: pending, status: 2)
        status = await settle(indexer)
        stored = try await imageVectors()
        XCTAssertEqual(Set(stored.map(\.frameID)), Set(ready + [pending]))
        XCTAssertEqual(status.state, .upToDate)
        XCTAssertEqual(status.embeddedImages, 6)
        waiting = try await database.embeddingSources(retryingKind: .image, model: "test-images/64", beforeFrameID: nil, limit: 10)
        XCTAssertTrue(waiting.isEmpty)
    }

    func testAFrameWhoseRedactionFailedIsNotEmbeddedUntilTheRedactionSucceeds() async throws {
        let failed = try await insertFrame(text: "airline flight booking confirmation Lisbon departure gate", topic: 1, status: 8)
        let fine = try await insertFrame(text: "marathon training schedule intervals tempo", topic: 2)

        let indexer = makeIndexer()
        var status = await settle(indexer)

        XCTAssertEqual(status.state, .upToDate)
        var stored = try await imageVectors()
        XCTAssertEqual(stored.map(\.frameID), [fine])
        XCTAssertFalse(provider.requestedFrames.contains(failed), "its pixels, which still show what was to be redacted, were never read")
        XCTAssertEqual(status.embeddedImages, 1, "it is waiting, not written off: the rewrite is retried")

        // The processing queue retries the rewrite and it succeeds. Nothing else about the frame changes.
        try await database.updateFrameProcessingStatus(frameID: failed, status: 7, rewritePurpose: "redaction")
        status = await settle(indexer)
        stored = try await imageVectors()
        XCTAssertEqual(Set(stored.map(\.frameID)), [failed, fine])
        XCTAssertEqual(status.embeddedImages, 2)
    }

    func testARedactionThatNeverSucceedsIsGivenUpOnWithoutItsPixelsBeingRead() async throws {
        let failed = try await insertFrame(text: "airline flight booking confirmation Lisbon departure gate", topic: 1, status: 8)

        let status = await settle(makeIndexer(retryBackoff: 0, retryTimeout: 0))

        XCTAssertEqual(status.embeddedImages, 1, "accounted for as having nothing to embed")
        let stored = try await imageVectors()
        XCTAssertTrue(stored.isEmpty)
        XCTAssertFalse(provider.requestedFrames.contains(failed))
    }

    func testAScreenshotThatCannotBeReadDoesNotStopTheRestAndIsReadAgainLater() async throws {
        let unreadable = try await insertFrame(text: "airline flight booking confirmation Lisbon departure gate", topic: 1)
        let fine = try await insertFrame(text: "marathon training schedule intervals tempo", topic: 2)
        provider.failFrames = [unreadable]

        // Not read again until the back-off has passed.
        var status = await settle(makeIndexer())
        XCTAssertEqual(status.state, .upToDate)
        var stored = try await imageVectors()
        XCTAssertEqual(stored.map(\.frameID), [fine])
        XCTAssertEqual(provider.requestedFrames.filter { $0 == unreadable }.count, 1, "it isn't read again straight away")

        // The video becomes readable (it was being finalized or rewritten): the next try embeds it.
        provider.failFrames = []
        status = await settle(makeIndexer(retryBackoff: 0))
        stored = try await imageVectors()
        XCTAssertEqual(Set(stored.map(\.frameID)), [unreadable, fine])
        XCTAssertEqual(status.embeddedImages, 2)
    }

    func testAScreenshotWhoseFrameChangesWhileItIsEmbeddedIsTriedAgainNotSkipped() async throws {
        let changing = try await insertFrame(text: "airline flight booking confirmation Lisbon departure gate", topic: 1)
        let neighbour = try await insertFrame(text: "marathon training schedule intervals tempo", topic: 2)
        // A redaction is queued for the frame at the moment its pixels are being read.
        let database = self.database!
        provider.onRead = { frameID in
            guard frameID == changing else { return }
            try? await database.updateFrameProcessingStatus(frameID: changing, status: 5, rewritePurpose: "redaction")
        }

        let indexer = makeIndexer(retryBackoff: 0)
        var status = await settle(indexer)

        // Its vector (made from the pixels before the redaction) is refused, its neighbour's is written, and
        // the walk has moved past both. It must be waiting to be tried again, not forgotten.
        var stored = try await imageVectors()
        XCTAssertEqual(stored.map(\.frameID), [neighbour])
        let waiting = try await database.embeddingSources(retryingKind: .image, model: "test-images/64", beforeFrameID: nil, limit: 10)
        XCTAssertEqual(waiting.map(\.frameID), [changing])
        XCTAssertEqual(status.embeddedImages, 1)

        // The redaction is burned into the video: the screenshot is embedded from the new pixels.
        provider.onRead = nil
        try await database.updateFrameProcessingStatus(frameID: changing, status: 7, rewritePurpose: "redaction")
        status = await settle(indexer)
        stored = try await imageVectors()
        XCTAssertEqual(Set(stored.map(\.frameID)), [changing, neighbour])
        XCTAssertEqual(status.embeddedImages, 2)
    }

    func testAReadyScreenshotIsReachedBehindMoreWaitingFramesThanOneScanTakes() async throws {
        let older = try await insertFrame(text: "airline flight booking confirmation Lisbon departure gate", topic: 1, status: 4)
        for index in 0..<3 {
            try await insertFrame(text: "marathon training schedule intervals tempo, day \(index)", topic: 2, status: 4)
        }
        // Four frames wait; a scan looks at two.
        let indexer = makeIndexer(retryScanLimit: 2)
        _ = await settle(indexer)
        var stored = try await imageVectors()
        XCTAssertTrue(stored.isEmpty)

        // The oldest becomes ready while the three newer ones are still pending.
        try await database.updateFrameProcessingStatus(frameID: older, status: 2)
        for _ in 0..<4 { await indexer.runPass() }

        stored = try await imageVectors()
        XCTAssertEqual(stored.map(\.frameID), [older], "the scan moves on to the next page instead of repeating the first")
    }

    func testScreenshotsThatBecameReadyAfterTheTimeoutAreAllEmbeddedNotGivenUpOn() async throws {
        // Three frames whose video was still open when they were first looked at.
        var frames: [Int64] = []
        for index in 0..<3 {
            frames.append(try await insertFrame(text: "marathon training schedule intervals tempo, day \(index)", topic: 2, status: 4))
        }
        let indexer = makeIndexer(retryBackoff: 0, retryTimeout: 0.05)
        _ = await settle(indexer)
        var stored = try await imageVectors()
        XCTAssertTrue(stored.isEmpty)

        // The Mac was asleep past the timeout. Now all three are ready at once: more than one batch of two.
        try await Task.sleep(for: .milliseconds(120))
        for frame in frames {
            try await database.updateFrameProcessingStatus(frameID: frame, status: 2)
        }
        let status = await settle(indexer)

        stored = try await imageVectors()
        XCTAssertEqual(Set(stored.map(\.frameID)), Set(frames), "the one that didn't fit the first batch is embedded in the next, not written off")
        XCTAssertEqual(status.embeddedImages, 3)
    }

    func testAScreenshotThatStaysUnreadableIsGivenUpOnAfterTheTimeout() async throws {
        let unreadable = try await insertFrame(text: "airline flight booking confirmation Lisbon departure gate", topic: 1)
        let fine = try await insertFrame(text: "marathon training schedule intervals tempo", topic: 2)
        provider.failFrames = [unreadable]

        let status = await settle(makeIndexer(retryBackoff: 0, retryTimeout: 0))

        XCTAssertEqual(status.state, .upToDate)
        XCTAssertEqual(status.embeddedImages, 2, "both are accounted for: one vector, one with nothing to embed")
        let stored = try await imageVectors()
        XCTAssertEqual(stored.map(\.frameID), [fine])
        let waiting = try await database.embeddingSources(retryingKind: .image, model: "test-images/64", beforeFrameID: nil, limit: 10)
        XCTAssertTrue(waiting.isEmpty)
        let reads = provider.requestedFrames.filter { $0 == unreadable }.count
        _ = await settle(makeIndexer(retryBackoff: 0, retryTimeout: 0))
        XCTAssertEqual(provider.requestedFrames.filter { $0 == unreadable }.count, reads, "it is not read again")
    }

    func testScreenshotsThatNeverBecomeReadyAreGivenUpOnEvenWhenMoreWaitThanAScanTakes() async throws {
        // Six frames whose video never becomes final. A scan looks at one of them, and before each scan of
        // screenshots the indexer looks for text waiting to be retried and finds none.
        for index in 0..<6 {
            try await insertFrame(text: "marathon training schedule intervals tempo, day \(index)", topic: 2, status: 4)
        }
        let indexer = makeIndexer(retryTimeout: 0.05, retryScanLimit: 1, retryGrace: 0.2)

        // A frame is given up on once it is past the timeout and this loop has seen it waiting for the grace
        // period. Finding no text to retry must not make the loop forget when it first saw the screenshots.
        _ = await settle(indexer)
        var waiting = try await database.embeddingSources(retryingKind: .image, model: "test-images/64", beforeFrameID: nil, limit: 10)
        XCTAssertEqual(waiting.count, 6)
        let clock = ContinuousClock()
        let deadline = clock.now.advanced(by: .seconds(5))
        repeat {
            await indexer.runPass()
            try await Task.sleep(for: .milliseconds(20))
            waiting = try await database.embeddingSources(retryingKind: .image, model: "test-images/64", beforeFrameID: nil, limit: 10)
        } while !waiting.isEmpty && clock.now < deadline

        XCTAssertTrue(waiting.isEmpty, "all six are given up on, not kept waiting for ever")
        XCTAssertTrue(provider.requestedFrames.isEmpty, "and none of them was read")
        let status = await settle(indexer)
        XCTAssertEqual(status.state, .upToDate)
        XCTAssertEqual(status.embeddedImages, 6, "all six are accounted for")
    }

    func testScreenshotsWaitForPowerWhileTextDoesNot() async throws {
        try await insertFrame(text: "airline flight booking confirmation Lisbon departure gate", topic: 1)

        let onBattery = makeIndexer(onPower: false)
        var status = await settle(onBattery)
        XCTAssertEqual(status.embeddedFrames, 1)
        XCTAssertEqual(status.embeddedImages, 0)
        XCTAssertEqual(status.state, .paused)
        XCTAssertEqual(images.sizes.count, 0)

        status = await settle(makeIndexer(onPower: true))
        XCTAssertEqual(status.embeddedImages, 1)
    }

    func testAnEncoderFailureLeavesScreenshotsForTheNextPass() async throws {
        try await insertFrame(text: "airline flight booking confirmation Lisbon departure gate", topic: 1)
        let indexer = makeIndexer()

        images.failNextCalls = 1
        await indexer.runPass()   // text
        await indexer.runPass()   // screenshot: fails
        var status = await indexer.status()
        XCTAssertEqual(status.state, .failed)
        XCTAssertEqual(status.embeddedImages, 0)
        XCTAssertFalse(provider.holdsCachedFrames, "the video decoder is not left open by a batch that failed")

        status = await settle(indexer)
        XCTAssertEqual(status.embeddedImages, 1)
    }

    func testScreenshotSizeFillsTheTokenBudgetInMultiplesOf48() throws {
        // A 16-inch MacBook Pro screen at 280 tokens.
        let retina = try XCTUnwrap(ImageSizing.targetSize(width: 3456, height: 2234, budget: 280))
        XCTAssertEqual(retina.width, 960)
        XCTAssertEqual(retina.height, 624)
        // 280 tokens are 280 × 9 patches of 16 × 16 pixels; the size never exceeds that.
        XCTAssertLessThanOrEqual(retina.width * retina.height, 280 * 9 * 256)
        XCTAssertNil(ImageSizing.targetSize(width: 0, height: 100, budget: 280))

        let source = TopicImageProvider.makeImage(topic: 1, width: 300, height: 200)
        let sized = try XCTUnwrap(ImageSizing.embeddingImage(from: source, budget: 70))
        XCTAssertEqual(sized.rgb.count, sized.width * sized.height * 3)
        XCTAssertEqual(sized.width % 48, 0)
        // The color survives scaling (packed RGB, no alpha): this image is topic 1's shade of red.
        XCTAssertEqual(sized.rgb[0], TopicImageProvider.red(forTopic: 1))
        XCTAssertEqual(sized.rgb[1], 0)
    }
}

/// Returns a solid-color image per frame; the red channel encodes which topic the "screenshot" shows.
private final class TopicImageProvider: FrameImageProviding, @unchecked Sendable {
    struct Unreadable: Error {}

    private let lock = NSLock()
    private var topicOfFrame: [Int64: Int] = [:]
    private var requested: [Int64] = []
    private var failing: Set<Int64> = []
    private var discards = 0
    private var readSinceDiscard = false
    private var readHook: (@Sendable (Int64) async -> Void)?

    /// Runs just before a frame's image is returned (something happens to the frame while it is being read).
    var onRead: (@Sendable (Int64) async -> Void)? {
        get { lock.withLock { readHook } }
        set { lock.withLock { readHook = newValue } }
    }

    var requestedFrames: [Int64] { lock.withLock { requested } }
    var discardCount: Int { lock.withLock { discards } }
    /// Whether a frame was read since the cache was last discarded (a real provider would still hold a decoder).
    var holdsCachedFrames: Bool { lock.withLock { readSinceDiscard } }

    func discardCachedFrames() {
        lock.withLock {
            discards += 1
            readSinceDiscard = false
        }
    }
    var failFrames: Set<Int64> {
        get { lock.withLock { failing } }
        set { lock.withLock { failing = newValue } }
    }

    func setTopic(_ topic: Int, forFrame frameID: Int64) {
        lock.withLock { topicOfFrame[frameID] = topic }
    }

    static func red(forTopic topic: Int) -> UInt8 { UInt8(40 + topic * 60) }

    static func makeImage(topic: Int, width: Int = 480, height: Int = 300) -> CGImage {
        let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        )!
        context.setFillColor(red: CGFloat(red(forTopic: topic)) / 255, green: 0, blue: 0, alpha: 1)
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        return context.makeImage()!
    }

    func frameImage(for source: FrameEmbeddingSource) async throws -> CGImage {
        let topic: Int = try lock.withLock {
            requested.append(source.frameID)
            readSinceDiscard = true
            if failing.contains(source.frameID) { throw Unreadable() }
            return topicOfFrame[source.frameID] ?? 0
        }
        await onRead?(source.frameID)
        return Self.makeImage(topic: topic)
    }
}

/// Stand-in image encoder: reads the topic from an image's color and returns that topic's word vector.
private final class TopicImageEmbedder: ImageEmbedding, @unchecked Sendable {
    struct Failed: Error {}

    private let words: WordEmbedder
    private let topics: [String]
    private let lock = NSLock()
    private var seen: [(width: Int, height: Int)] = []
    private var failures = 0

    init(words: WordEmbedder, topics: [String]) {
        self.words = words
        self.topics = topics
    }

    var sizes: [(width: Int, height: Int)] { lock.withLock { seen } }
    var failNextCalls: Int {
        get { lock.withLock { failures } }
        set { lock.withLock { failures = newValue } }
    }

    func embed(images: [EmbeddingImage]) async throws -> [[Float]] {
        try lock.withLock {
            if failures > 0 {
                failures -= 1
                throw Failed()
            }
            seen.append(contentsOf: images.map { ($0.width, $0.height) })
        }
        return images.map { image in
            let red = Int(image.rgb[0])
            let topic = min(max(Int((Double(red - 40) / 60).rounded()), 0), topics.count - 1)
            return words.vector(topics[topic])
        }
    }

    func shutdown() {}
}
