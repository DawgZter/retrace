import XCTest
import Foundation
import Shared
import SQLCipher
@testable import Database

/// The `frame_embedding` table against a real SQLite database: the V20 migration, the indexer's cursors, and the
/// triggers that keep a vector from outliving the text it was made from.
final class FrameEmbeddingTests: XCTestCase {
    private let model = "test-model/1"
    private var database: DatabaseManager!
    private var videoID: Int64 = 0
    private var segmentID: Int64 = 0

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
        segmentID = try await database.insertSegment(
            bundleID: "com.test.app",
            startDate: Date(),
            endDate: Date().addingTimeInterval(300),
            windowName: "Quarterly plan",
            browserUrl: nil,
            type: 0
        )
    }

    override func tearDown() async throws {
        try await database.close()
        database = nil
    }

    /// A frame with indexed OCR text, as the indexer would read it.
    private func insertFrame(text: String, index: Int) async throws -> FrameEmbeddingSource {
        let frameID = try await database.insertFrame(
            FrameReference(
                id: FrameID(value: 0),
                timestamp: Date().addingTimeInterval(Double(index)),
                segmentID: AppSegmentID(value: segmentID),
                videoID: VideoSegmentID(value: videoID),
                frameIndexInSegment: index,
                metadata: .empty,
                source: .native
            )
        )
        let docID = try await database.indexFrameText(
            mainText: text,
            chromeText: nil,
            windowTitle: "Quarterly plan",
            segmentId: segmentID,
            frameId: frameID
        )
        return FrameEmbeddingSource(frameID: frameID, docID: docID, windowName: "Quarterly plan", text: text)
    }

    /// Stores one vector per source, as the indexer does after embedding.
    @discardableResult
    private func store(
        _ vectors: [(source: FrameEmbeddingSource, vector: Data)],
        kind: EmbeddingKind = .text,
        model: String? = nil
    ) async throws -> [FrameEmbeddingRecord] {
        try await database.storeFrameEmbeddings(
            vectors.map { FrameEmbeddingRecord(frameID: $0.source.frameID, kind: kind, docID: $0.source.docID, vector: $0.vector) },
            from: vectors.map(\.source),
            model: model ?? self.model
        )
    }

    private func vector(_ value: UInt8) -> Data {
        Data(repeating: value, count: 8)
    }

    /// The frame as the indexer would read it now (its current state and revision).
    private func current(_ source: FrameEmbeddingSource) async throws -> FrameEmbeddingSource {
        let sources = try await database.embeddingSources(beforeDocID: source.docID + 1, limit: 1)
        return try XCTUnwrap(sources.first { $0.frameID == source.frameID })
    }

    private func marker(
        _ source: FrameEmbeddingSource,
        _ state: FrameEmbeddingRecord.State,
        kind: EmbeddingKind = .image
    ) -> FrameEmbeddingRecord {
        FrameEmbeddingRecord(frameID: source.frameID, kind: kind, docID: source.docID, state: state)
    }

    func testSourcesWalkDocumentsInBothDirectionsWithTextAndWindowName() async throws {
        let first = try await insertFrame(text: "alpha budget review", index: 0)
        let second = try await insertFrame(text: "beta hiring plan", index: 1)
        let third = try await insertFrame(text: "gamma launch notes", index: 2)

        let latest = try await database.latestIndexedDocID()
        XCTAssertEqual(latest, third.docID)

        let newestFirst = try await database.embeddingSources(beforeDocID: third.docID + 1, limit: 2)
        XCTAssertEqual(newestFirst.map(\.frameID), [third.frameID, second.frameID])
        XCTAssertEqual(newestFirst.first?.text, "gamma launch notes")
        XCTAssertEqual(newestFirst.first?.windowName, "Quarterly plan")

        let forward = try await database.embeddingSources(afterDocID: first.docID, limit: 10)
        XCTAssertEqual(forward.map(\.docID), [second.docID, third.docID])

        let indexed = try await database.indexedDocumentCount()
        XCTAssertEqual(indexed, 3)
    }

    func testInsertedVectorsRoundTripAndBoundsFollowDocumentIDs() async throws {
        let first = try await insertFrame(text: "alpha budget review", index: 0)
        let second = try await insertFrame(text: "beta hiring plan", index: 1)
        let third = try await insertFrame(text: "x", index: 2)

        let none = try await database.embeddingDocIDBounds(kind: .text, model: model)
        XCTAssertNil(none)

        try await store([(second, vector(2)), (third, Data())])

        let bounds = try await database.embeddingDocIDBounds(kind: .text, model: model)
        XCTAssertEqual(bounds?.min, second.docID)
        XCTAssertEqual(bounds?.max, third.docID)

        // Readers get only real vectors; the count covers every frame looked at.
        let stored = try await database.frameEmbeddings(kind: .text, model: model, afterFrameID: -1, limit: 10)
        XCTAssertEqual(stored, [FrameEmbeddingRecord(frameID: second.frameID, docID: second.docID, vector: vector(2))])
        let count = try await database.frameEmbeddingCount(kind: .text, model: model)
        XCTAssertEqual(count, 2)

        // Embedding a frame again replaces its vector.
        try await store([(second, vector(9))])
        let replaced = try await database.frameEmbeddings(kind: .text, model: model, afterFrameID: -1, limit: 10)
        XCTAssertEqual(replaced.map(\.vector), [vector(9)])

        // Paging by frame id.
        try await store([(first, vector(1))])
        let afterFirst = try await database.frameEmbeddings(kind: .text, model: model, afterFrameID: first.frameID, limit: 10)
        XCTAssertEqual(afterFirst.map(\.frameID), [second.frameID])
    }

    func testVectorForAFrameDeletedMeanwhileIsSkippedNotAnError() async throws {
        let kept = try await insertFrame(text: "alpha budget review", index: 0)
        let deleted = try await insertFrame(text: "beta hiring plan", index: 1)

        // The frame is deleted while its text is being embedded.
        try await database.deleteFrame(id: FrameID(value: deleted.frameID))
        let written = try await store([(kept, vector(1)), (deleted, vector(2))])

        XCTAssertEqual(written.map(\.frameID), [kept.frameID])
        let stored = try await database.frameEmbeddings(kind: .text, model: model, afterFrameID: -1, limit: 10)
        XCTAssertEqual(stored.map(\.frameID), [kept.frameID])
    }

    func testVectorMadeFromTextThatWasReplacedMeanwhileIsNotStored() async throws {
        let source = try await insertFrame(text: "alpha budget review with a secret phrase", index: 0)

        // While the text is being embedded, a redaction rewrites the frame's document. SQLite gives the new
        // document the old one's id here (it was the highest), so the id alone would not reveal the change.
        let newDocID = try await database.indexFrameText(
            mainText: "alpha budget review",
            chromeText: nil,
            windowTitle: "Quarterly plan",
            segmentId: segmentID,
            frameId: source.frameID
        )
        XCTAssertEqual(newDocID, source.docID)

        let written = try await store([(source, vector(1))])

        XCTAssertTrue(written.isEmpty, "a vector made from the redacted text must not be written")
        let stored = try await database.frameEmbeddings(kind: .text, model: model, afterFrameID: -1, limit: 10)
        XCTAssertTrue(stored.isEmpty)
        let present = try await database.framesWithEmbedding(kind: .text, model: model, among: [source.frameID])
        XCTAssertTrue(present.isEmpty)
    }

    func testAFrameWhoseTextChangedInPlaceCanBeMarkedToTryAgainButNotAsHavingNothingToEmbed() async throws {
        let source = try await insertFrame(text: "alpha budget review with a private note", index: 0)

        // The text is rewritten in place while it is being embedded: same document, different text.
        try await database.updateDocument(id: source.docID, content: "alpha budget review")

        var written = try await store([(source, vector(1))])
        XCTAssertTrue(written.isEmpty, "a vector made from the old text is refused")
        // "Nothing to embed" was concluded from the old text too.
        written = try await database.storeFrameEmbeddings(
            [marker(source, .nothingToEmbed, kind: .text)], from: [source], model: model
        )
        XCTAssertTrue(written.isEmpty)
        // "Try again later" says nothing about the text, and is how the frame gets looked at again.
        written = try await database.storeFrameEmbeddings(
            [marker(source, .retryLater, kind: .text)], from: [source], model: model
        )
        XCTAssertEqual(written.map(\.frameID), [source.frameID])
        let waiting = try await database.embeddingSources(retryingKind: .text, model: model, beforeFrameID: nil, limit: 10)
        XCTAssertEqual(waiting.map(\.text), ["alpha budget review"], "it is offered again with its new text")

        // A frame that is gone is not marked.
        try await database.deleteFrame(id: FrameID(value: source.frameID))
        written = try await database.storeFrameEmbeddings(
            [marker(source, .retryLater, kind: .text)], from: [source], model: model
        )
        XCTAssertTrue(written.isEmpty)
    }

    func testAFrameWithALeftoverOlderDocumentIsOfferedOnceWithItsCurrentText() async throws {
        let leftover = try await insertFrame(text: "legacy old text with a private note", index: 0)
        let other = try await insertFrame(text: "beta hiring plan", index: 1)
        // A database from before V15 can hold two documents for one frame. The one with the higher id is the
        // current one: the one keyword search reads.
        let currentDocID = other.docID + 100
        guard let db = await database.getConnection() else { return XCTFail("no connection") }
        for sql in [
            "INSERT INTO searchRanking(rowid, text, otherText, title) VALUES (\(currentDocID), 'legacy current text', NULL, 'Quarterly plan');",
            "INSERT INTO doc_segment (docid, segmentId, frameId) VALUES (\(currentDocID), \(segmentID), \(leftover.frameID));",
        ] {
            XCTAssertEqual(sqlite3_exec(db, sql, nil, nil, nil), SQLITE_OK)
        }

        // Walking in either direction, the frame comes up once, with its current document.
        let forward = try await database.embeddingSources(afterDocID: 0, limit: 10)
        XCTAssertEqual(forward.map(\.docID), [other.docID, currentDocID])
        XCTAssertEqual(forward.last?.frameID, leftover.frameID)
        XCTAssertEqual(forward.last?.text, "legacy current text")
        let backward = try await database.embeddingSources(beforeDocID: currentDocID + 1, limit: 10)
        XCTAssertEqual(backward.map(\.docID), [currentDocID, other.docID])
        let indexed = try await database.indexedDocumentCount()
        XCTAssertEqual(indexed, 2, "two frames, however many documents")

        // A vector made from the leftover document is refused; one made from the current document is stored.
        var written = try await store([(leftover, vector(1))])
        XCTAssertTrue(written.isEmpty)
        written = try await store([(try XCTUnwrap(forward.last), vector(2))])
        XCTAssertEqual(written.map(\.frameID), [leftover.frameID])
    }

    func testScreenshotVectorIsStoredOnlyIfTheFrameIsAsItWasWhenItsPixelsWereRead() async throws {
        let imageModel = "image-model/1"
        let inserted = try await insertFrame(text: "alpha budget review", index: 0)
        try await database.updateFrameProcessingStatus(frameID: inserted.frameID, status: 2)
        let readBeforeRedaction = try await current(inserted)

        // A redaction is queued while the screenshot is being embedded.
        try await database.updateFrameProcessingStatus(frameID: inserted.frameID, status: 5, rewritePurpose: "redaction")
        var written = try await store([(readBeforeRedaction, vector(7))], kind: .image, model: imageModel)
        XCTAssertTrue(written.isEmpty, "these pixels still show what is about to be redacted")

        // A "try again later" marker describes no pixels: it is accepted as long as the frame has its document.
        written = try await database.storeFrameEmbeddings(
            [marker(readBeforeRedaction, .retryLater)],
            from: [readBeforeRedaction],
            model: imageModel
        )
        XCTAssertEqual(written.count, 1)

        // The redaction is burned in. A vector made from pixels read before that is still refused, even though
        // the frame is "final" again; one made from pixels read afterwards is stored.
        try await database.updateFrameProcessingStatus(frameID: inserted.frameID, status: 7, rewritePurpose: "redaction")
        written = try await store([(readBeforeRedaction, vector(7))], kind: .image, model: imageModel)
        XCTAssertTrue(written.isEmpty, "the video was rewritten after these pixels were read")

        let readAfterRedaction = try await current(inserted)
        XCTAssertNotEqual(readAfterRedaction.revision, readBeforeRedaction.revision)
        written = try await store([(readAfterRedaction, vector(8))], kind: .image, model: imageModel)
        XCTAssertEqual(written.count, 1)
        let present = try await database.framesWithEmbedding(kind: .image, model: imageModel, among: [inserted.frameID, 999])
        XCTAssertEqual(present, [inserted.frameID])
    }

    func testAFrameWaitingToBeRetriedIsOfferedAgainAndNotCountedAsDone() async throws {
        let imageModel = "image-model/1"
        let waiting = try await insertFrame(text: "alpha budget review", index: 0)
        let done = try await insertFrame(text: "beta hiring plan", index: 1)
        try await database.updateFrameProcessingStatus(frameID: done.frameID, status: 2)
        let doneNow = try await current(done)
        try await store([(doneNow, vector(3))], kind: .image, model: imageModel)

        let firstMarked = Date(timeIntervalSince1970: 1_800_000_000)
        guard let db = await database.getConnection() else { return XCTFail("no connection") }
        var written = try EmbeddingQueries.store(
            db: db, records: [marker(waiting, .retryLater)], sources: [waiting], model: imageModel, now: firstMarked
        )
        XCTAssertEqual(written.count, 1)

        // Both frames have a row, so the walk has moved past them; only one is done.
        let bounds = try await database.embeddingDocIDBounds(kind: .image, model: imageModel)
        XCTAssertEqual(bounds?.min, waiting.docID)
        XCTAssertEqual(bounds?.max, done.docID)
        let doneCount = try await database.frameEmbeddingCount(kind: .image, model: imageModel)
        XCTAssertEqual(doneCount, 1)
        let vectors = try await database.frameEmbeddings(kind: .image, model: imageModel, afterFrameID: -1, limit: 10)
        XCTAssertEqual(vectors.map(\.frameID), [done.frameID])

        var retry = try await database.embeddingSources(retryingKind: .image, model: imageModel, beforeFrameID: nil, limit: 10)
        XCTAssertEqual(retry.map(\.frameID), [waiting.frameID])
        XCTAssertEqual(retry.first?.retryingSince, firstMarked)
        XCTAssertEqual(retry.first?.imageState, .pending)

        // The waiting frames can be paged through, newest first.
        let pastIt = try await database.embeddingSources(
            retryingKind: .image, model: imageModel, beforeFrameID: waiting.frameID, limit: 10
        )
        XCTAssertTrue(pastIt.isEmpty)

        // Marked again later: it remembers when it was first marked.
        written = try EmbeddingQueries.store(
            db: db,
            records: [marker(waiting, .retryLater)],
            sources: [waiting],
            model: imageModel,
            now: firstMarked.addingTimeInterval(3600)
        )
        retry = try await database.embeddingSources(retryingKind: .image, model: imageModel, beforeFrameID: nil, limit: 10)
        XCTAssertEqual(retry.first?.retryingSince, firstMarked)

        // Its video is finalized and OCR finishes: it is offered as ready, and a vector takes the marker's place.
        try await database.updateFrameProcessingStatus(frameID: waiting.frameID, status: 2)
        try await database.markVideoFinalized(id: videoID, frameCount: 150, fileSize: 1024)
        retry = try await database.embeddingSources(retryingKind: .image, model: imageModel, beforeFrameID: nil, limit: 10)
        XCTAssertEqual(retry.first?.imageState, .ready)
        try await store([(try XCTUnwrap(retry.first), vector(4))], kind: .image, model: imageModel)

        let stillWaiting = try await database.embeddingSources(retryingKind: .image, model: imageModel, beforeFrameID: nil, limit: 10)
        XCTAssertTrue(stillWaiting.isEmpty)
        let finalCount = try await database.frameEmbeddingCount(kind: .image, model: imageModel)
        XCTAssertEqual(finalCount, 2)
    }

    func testAFrameTheUserDeletedLosesItsVectorsAtOnceAndGetsNoNewOnes() async throws {
        let kept = try await insertFrame(text: "alpha budget review", index: 0)
        let deleted = try await insertFrame(text: "beta hiring plan with a private note", index: 1)
        try await store([(kept, vector(1)), (deleted, vector(2))])

        // Deleting from the timeline: the frame row stays until its video has been rewritten.
        let result = try await database.deleteOrScheduleNativeFramesForDeletion(frameIDs: [deleted.frameID])
        XCTAssertEqual(result.scheduledJobs.map(\.frameID), [deleted.frameID], "the frame is only marked for now")

        let stored = try await database.frameEmbeddings(kind: .text, model: model, afterFrameID: -1, limit: 10)
        XCTAssertEqual(stored.map(\.frameID), [kept.frameID], "its vector goes when it is marked, not when the video is rewritten")

        // The indexer still sees the frame's document. It is told the frame is deleted, and a vector for it is
        // refused whether it was made before or after the deletion.
        let asReadNow = try await current(deleted)
        XCTAssertTrue(asReadNow.isMarkedForDeletion)
        XCTAssertEqual(asReadNow.imageState, .unavailable)
        var written = try await store([(deleted, vector(2))])
        XCTAssertTrue(written.isEmpty)
        written = try await store([(asReadNow, vector(2))])
        XCTAssertTrue(written.isEmpty)

        // What is recorded instead is that there is nothing to embed, so the walk moves on.
        written = try await database.storeFrameEmbeddings(
            [marker(asReadNow, .nothingToEmbed, kind: .text)],
            from: [asReadNow],
            model: model
        )
        XCTAssertEqual(written.count, 1)
        let present = try await database.framesWithEmbedding(kind: .text, model: model, among: [kept.frameID, deleted.frameID])
        XCTAssertEqual(present, [kept.frameID])
    }

    func testDeletingAFrameDeletesItsVector() async throws {
        let kept = try await insertFrame(text: "alpha budget review", index: 0)
        let deleted = try await insertFrame(text: "beta hiring plan", index: 1)
        try await store([(kept, vector(1)), (deleted, vector(2))])

        try await database.deleteFrame(id: FrameID(value: deleted.frameID))

        let stored = try await database.frameEmbeddings(kind: .text, model: model, afterFrameID: -1, limit: 10)
        XCTAssertEqual(stored.map(\.frameID), [kept.frameID])
    }

    func testReindexingAFramesTextDeletesItsVectorAndOffersItAgain() async throws {
        let frame = try await insertFrame(text: "alpha budget review with a secret phrase", index: 0)
        try await store([(frame, vector(1))])

        // OCR reprocessing or a redaction rewrite replaces the frame's document.
        let newDocID = try await database.indexFrameText(
            mainText: "alpha budget review",
            chromeText: nil,
            windowTitle: "Quarterly plan",
            segmentId: segmentID,
            frameId: frame.frameID
        )

        let stored = try await database.frameEmbeddings(kind: .text, model: model, afterFrameID: -1, limit: 10)
        XCTAssertTrue(stored.isEmpty, "a vector made from the old text must not survive it")

        // With its vector gone the frame is outside the embedded range again, which is how the indexer finds it.
        // (SQLite may give the new document the old one's id, so the id alone can't tell them apart.)
        let bounds = try await database.embeddingDocIDBounds(kind: .text, model: model)
        XCTAssertNil(bounds)
        let offered = try await database.embeddingSources(beforeDocID: newDocID + 1, limit: 10)
        XCTAssertEqual(offered.map(\.frameID), [frame.frameID])
        XCTAssertEqual(offered.first?.text, "alpha budget review")
    }

    func testTextAndImageVectorsOfAFrameAreSeparateRowsWithSeparateCursors() async throws {
        let first = try await insertFrame(text: "alpha budget review", index: 0)
        let second = try await insertFrame(text: "beta hiring plan", index: 1)

        try await store([(first, vector(1)), (second, vector(2))])
        try await database.updateFrameProcessingStatus(frameID: second.frameID, status: 2)
        try await store([(try await current(second), vector(7))], kind: .image, model: "image-model/1")

        let textBounds = try await database.embeddingDocIDBounds(kind: .text, model: model)
        XCTAssertEqual(textBounds?.min, first.docID)
        XCTAssertEqual(textBounds?.max, second.docID)
        let imageBounds = try await database.embeddingDocIDBounds(kind: .image, model: "image-model/1")
        XCTAssertEqual(imageBounds?.min, second.docID)
        XCTAssertEqual(imageBounds?.max, second.docID)

        let images = try await database.frameEmbeddings(kind: .image, model: "image-model/1", afterFrameID: -1, limit: 10)
        XCTAssertEqual(images, [FrameEmbeddingRecord(frameID: second.frameID, kind: .image, docID: second.docID, vector: vector(7))])
        let texts = try await database.frameEmbeddings(kind: .text, model: model, afterFrameID: -1, limit: 10)
        XCTAssertEqual(texts.map(\.vector), [vector(1), vector(2)])

        // Removing one kind leaves the other.
        try await database.deleteAllFrameEmbeddings(kind: .image)
        let textCount = try await database.frameEmbeddingCount(kind: .text, model: model)
        XCTAssertEqual(textCount, 2)
        let imageCount = try await database.frameEmbeddingCount(kind: .image, model: "image-model/1")
        XCTAssertEqual(imageCount, 0)
    }

    func testEveryWayAFrameLosesItsContentRemovesItsScreenshotVectorToo() async throws {
        let imageModel = "image-model/1"
        var frames: [FrameEmbeddingSource] = []
        for index in 0..<4 {
            let frame = try await insertFrame(text: "alpha budget review number \(index)", index: index)
            try await database.updateFrameProcessingStatus(frameID: frame.frameID, status: 2)
            try await store([(frame, vector(1))])
            try await store([(try await current(frame), vector(9))], kind: .image, model: imageModel)
            frames.append(frame)
        }
        func screenshotVectors() async throws -> Set<Int64> {
            Set(try await database.frameEmbeddings(kind: .image, model: imageModel, afterFrameID: -1, limit: 10).map(\.frameID))
        }
        func textVectors() async throws -> Set<Int64> {
            Set(try await database.frameEmbeddings(kind: .text, model: model, afterFrameID: -1, limit: 10).map(\.frameID))
        }
        var withScreenshot = try await screenshotVectors()
        XCTAssertEqual(withScreenshot, Set(frames.map(\.frameID)))

        // The frame is deleted outright (retention).
        try await database.deleteFrame(id: FrameID(value: frames[0].frameID))
        // The frame is deleted from the timeline (marked, removed from its video later).
        _ = try await database.deleteOrScheduleNativeFramesForDeletion(frameIDs: [frames[1].frameID])
        // The frame's text is replaced (OCR reprocessing, redaction).
        _ = try await database.indexFrameText(
            mainText: "alpha budget review",
            chromeText: nil,
            windowTitle: "Quarterly plan",
            segmentId: segmentID,
            frameId: frames[2].frameID
        )

        withScreenshot = try await screenshotVectors()
        XCTAssertEqual(withScreenshot, [frames[3].frameID], "only the untouched frame keeps its screenshot vector")
        let withText = try await textVectors()
        XCTAssertEqual(withText, [frames[3].frameID])
    }

    func testAFramesScreenshotIsReadyOnlyWhenItsPixelsAreFinal() async throws {
        let frame = try await insertFrame(text: "alpha budget review", index: 0)

        // Freshly captured: OCR hasn't finished and the video is still being written.
        var source = try await database.embeddingSources(beforeDocID: frame.docID + 1, limit: 1).first
        XCTAssertEqual(source?.imageState, .pending)
        XCTAssertEqual(source?.videoPath, "chunks/test")
        XCTAssertEqual(source?.frameIndex, 0)

        // OCR finished, but the video chunk isn't finalized yet.
        try await database.updateFrameProcessingStatus(frameID: frame.frameID, status: 2)
        source = try await database.embeddingSources(beforeDocID: frame.docID + 1, limit: 1).first
        XCTAssertEqual(source?.imageState, .pending)

        try await database.markVideoFinalized(id: videoID, frameCount: 150, fileSize: 1024)
        source = try await database.embeddingSources(beforeDocID: frame.docID + 1, limit: 1).first
        XCTAssertEqual(source?.imageState, .ready)

        // A redaction is waiting to be burned into the video: the current pixels still show what was redacted.
        try await database.updateFrameProcessingStatus(frameID: frame.frameID, status: 5, rewritePurpose: "redaction")
        source = try await database.embeddingSources(beforeDocID: frame.docID + 1, limit: 1).first
        XCTAssertEqual(source?.imageState, .pending)

        try await database.updateFrameProcessingStatus(frameID: frame.frameID, status: 7, rewritePurpose: "redaction")
        source = try await database.embeddingSources(beforeDocID: frame.docID + 1, limit: 1).first
        XCTAssertEqual(source?.imageState, .ready)

        // The redaction could not be applied: this screenshot must not be embedded as it is. The rewrite is
        // retried, so the frame waits rather than being written off.
        try await database.updateFrameProcessingStatus(frameID: frame.frameID, status: 8, rewritePurpose: "redaction")
        source = try await database.embeddingSources(beforeDocID: frame.docID + 1, limit: 1).first
        XCTAssertEqual(source?.imageState, .pending)

        // OCR itself failed: nothing more will come of this frame.
        try await database.updateFrameProcessingStatus(frameID: frame.frameID, status: 3)
        source = try await database.embeddingSources(beforeDocID: frame.docID + 1, limit: 1).first
        XCTAssertEqual(source?.imageState, .unavailable)
    }

    func testChangingADocumentsTextInPlaceEmptiesItsVectorsAndQueuesThemAgain() async throws {
        let kept = try await insertFrame(text: "alpha budget review", index: 0)
        let changed = try await insertFrame(text: "beta hiring plan with a private note", index: 1)
        try await store([(kept, vector(1)), (changed, vector(2))])

        // No document is deleted here, so none of the triggers fires.
        try await database.updateDocument(id: changed.docID, content: "beta hiring plan")

        let present = try await database.framesWithEmbedding(kind: .text, model: model, among: [kept.frameID, changed.frameID])
        XCTAssertEqual(present, [kept.frameID], "the vector made from the old text is not served")
        let vectors = try await database.frameEmbeddings(kind: .text, model: model, afterFrameID: -1, limit: 10)
        XCTAssertEqual(vectors.map(\.frameID), [kept.frameID])
        let waiting = try await database.embeddingSources(retryingKind: .text, model: model, beforeFrameID: nil, limit: 10)
        XCTAssertEqual(waiting.map(\.frameID), [changed.frameID])
        XCTAssertEqual(waiting.first?.text, "beta hiring plan", "and it is offered again with its new text")
    }

    func testReadingVectorsAPageAtATimeWalksThePrimaryKeyInsteadOfSortingTheTable() async throws {
        // Reading a million vectors took 149 seconds instead of one when SQLite chose the (kind, model, docid)
        // index for this query and sorted every row of the kind for each page.
        guard let db = await database.getConnection() else { return XCTFail("no connection") }
        var statement: OpaquePointer?
        defer { sqlite3_finalize(statement) }
        XCTAssertEqual(
            sqlite3_prepare_v2(db, "EXPLAIN QUERY PLAN " + EmbeddingQueries.vectorPageSQL, -1, &statement, nil),
            SQLITE_OK
        )
        var plan: [String] = []
        while sqlite3_step(statement) == SQLITE_ROW {
            plan.append(String(cString: sqlite3_column_text(statement, 3)))
        }
        XCTAssertTrue(plan.contains { $0.contains("PRIMARY KEY") && $0.contains("frameId>") }, "\(plan)")
        XCTAssertFalse(plan.contains { $0.contains("TEMP B-TREE") }, "\(plan)")
    }

    func testVectorsFromAnotherModelAreRemoved() async throws {
        let first = try await insertFrame(text: "alpha budget review", index: 0)
        let second = try await insertFrame(text: "beta hiring plan", index: 1)
        try await store([(first, vector(1))], model: "old-model/1")
        try await store([(second, vector(2))])

        let removed = try await database.deleteFrameEmbeddings(kind: .text, exceptModel: model)
        XCTAssertEqual(removed, 1)

        let oldCount = try await database.frameEmbeddingCount(kind: .text, model: "old-model/1")
        XCTAssertEqual(oldCount, 0)
        let stored = try await database.frameEmbeddings(kind: .text, model: model, afterFrameID: -1, limit: 10)
        XCTAssertEqual(stored.map(\.frameID), [second.frameID])
    }
}
