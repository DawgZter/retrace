import CoreGraphics
import Foundation

// MARK: - Semantic Search

/// What a frame's vector was made from. Both kinds live in one vector space, so one query is compared with both.
public enum EmbeddingKind: Int, Sendable, CaseIterable {
    /// The frame's OCR text (with its window title).
    case text = 0
    /// The frame's screenshot.
    case image = 1
}

/// Settings keys and fixed values for semantic search (search by meaning).
public enum SemanticSearchSettings {
    /// UserDefaults key (suite `io.retrace.app`): whether semantic search is turned on. Off by default.
    public static let enabledDefaultsKey = "semanticSearchEnabled"
    /// UserDefaults key: whether screenshots are embedded too (needs the image encoder). Off by default.
    public static let imagesEnabledDefaultsKey = "semanticSearchImagesEnabled"

    /// The text model's file name inside the models folder.
    public static let modelFileName = "embeddinggemma-2-Q8_0.gguf"
    /// The image (and audio) encoders' file name inside the models folder.
    public static let imageModelFileName = "mmproj-embeddinggemma-2-Q8_0.gguf"

    /// Dimensions kept of the model's 768 (Matryoshka truncation, normalized again).
    public static let dimension = 256

    /// Soft tokens a screenshot is given. 280 is what a retrieval test on real screen history settled on:
    /// 560 found no more and took 2.7 times as long.
    public static let imageTokenBudget = 280

    /// What made a kind's stored vectors: model, quantization, input format and dimension.
    /// Vectors from different values never share an index; changing any part re-embeds that kind.
    public static func modelIdentifier(for kind: EmbeddingKind) -> String {
        switch kind {
        case .text: return "embeddinggemma-2-q8_0/text-v1/\(dimension)"
        case .image: return "embeddinggemma-2-q8_0/image\(imageTokenBudget)-v1/\(dimension)"
        }
    }

    /// Cosine similarity from which a match is taken at its word. Any query has frames that stand out from
    /// the rest, even one about something that was never on screen; what tells the two apart is how similar the
    /// best frames are in absolute terms. Callers show only a few matches below this value.
    ///
    /// It belongs to the model and input format above. Measured on 26,000 frames of one English-language
    /// history with 66 queries that have an answer and 40 that have none: no correct top-10 result of either
    /// kind fell below 0.69, and the best frame for a query with no answer was below it four times in five.
    public static let strongMatchCosine: Float = 0.69

    private static var defaults: UserDefaults {
        UserDefaults(suiteName: "io.retrace.app") ?? .standard
    }

    public static var isEnabled: Bool {
        defaults.bool(forKey: enabledDefaultsKey)
    }

    /// Screenshots are embedded only when semantic search itself is on.
    public static var isImagesEnabled: Bool {
        isEnabled && defaults.bool(forKey: imagesEnabledDefaultsKey)
    }
}

/// A frame to embed: its OCR text, and where its screenshot is.
public struct FrameEmbeddingSource: Sendable, Equatable {
    /// Whether the frame's screenshot can be embedded yet.
    public enum ImageState: Sendable, Equatable {
        /// Encoded, OCR finished and any redaction applied: the pixels are final.
        case ready
        /// Still being encoded, processed or redacted, or waiting for a failed redaction to be retried.
        case pending
        /// Will never be readable or safe to embed (processing failed, the frame was deleted).
        case unavailable
    }

    public let frameID: Int64
    /// The frame's document id in the FTS index. Documents are numbered in the order they were indexed,
    /// so this is the cursor the embedding indexer walks.
    public let docID: Int64
    public let createdAt: Date
    public let windowName: String?
    public let text: String
    public let videoPath: String?
    public let frameIndex: Int
    public let videoFrameRate: Double?
    public let imageState: ImageState
    /// The user has deleted this frame and it is waiting to be removed from its video. Nothing is embedded
    /// from it.
    public let isMarkedForDeletion: Bool
    /// Changes whenever the frame is processed again or its video is rewritten. A screenshot vector is stored
    /// only if this is unchanged since the pixels were read.
    public let revision: String
    /// When the indexer first found it couldn't embed this frame's screenshot yet. Nil unless the frame came
    /// from `embeddingSources(retryingKind:model:beforeFrameID:limit:)`.
    public let retryingSince: Date?

    public init(
        frameID: Int64,
        docID: Int64,
        createdAt: Date = Date(),
        windowName: String?,
        text: String,
        videoPath: String? = nil,
        frameIndex: Int = 0,
        videoFrameRate: Double? = nil,
        imageState: ImageState = .unavailable,
        isMarkedForDeletion: Bool = false,
        revision: String = "",
        retryingSince: Date? = nil
    ) {
        self.frameID = frameID
        self.docID = docID
        self.createdAt = createdAt
        self.windowName = windowName
        self.text = text
        self.videoPath = videoPath
        self.frameIndex = frameIndex
        self.videoFrameRate = videoFrameRate
        self.imageState = imageState
        self.isMarkedForDeletion = isMarkedForDeletion
        self.revision = revision
        self.retryingSince = retryingSince
    }
}

/// One row of `frame_embedding`: what the indexer concluded about a frame for one kind.
public struct FrameEmbeddingRecord: Sendable, Equatable {
    public enum State: Int, Sendable {
        /// `vector` holds `dimension` float16 values, little-endian, L2-normalized.
        case vector = 0
        /// The frame was looked at and has nothing to embed (too little text, a failed redaction, deleted).
        case nothingToEmbed = 1
        /// The frame could not be embedded yet (its screenshot wasn't final or couldn't be read) and should be
        /// tried again later.
        case retryLater = 2
    }

    public let frameID: Int64
    public let kind: EmbeddingKind
    public let docID: Int64
    public let state: State
    /// Empty unless `state` is `.vector`.
    public let vector: Data

    /// A vector, or (when `vector` is empty) a marker that the frame has nothing to embed.
    public init(frameID: Int64, kind: EmbeddingKind = .text, docID: Int64, vector: Data) {
        self.init(
            frameID: frameID,
            kind: kind,
            docID: docID,
            state: vector.isEmpty ? .nothingToEmbed : .vector,
            vector: vector
        )
    }

    public init(frameID: Int64, kind: EmbeddingKind, docID: Int64, state: State, vector: Data = Data()) {
        self.frameID = frameID
        self.kind = kind
        self.docID = docID
        self.state = state
        self.vector = state == .vector ? vector : Data()
    }
}

/// Storage for frame embeddings (the `frame_embedding` table).
/// Owner: DATABASE agent
public protocol EmbeddingStoreProtocol: Actor {
    /// Frames indexed after `docID` (ascending).
    func embeddingSources(afterDocID docID: Int64, limit: Int) async throws -> [FrameEmbeddingSource]

    /// Frames indexed before `docID` (descending).
    func embeddingSources(beforeDocID docID: Int64, limit: Int) async throws -> [FrameEmbeddingSource]

    /// Frames whose `kind` row says "try again later", newest first, as they are now. `beforeFrameID` pages
    /// through them: nil starts from the newest.
    func embeddingSources(
        retryingKind kind: EmbeddingKind,
        model: String,
        beforeFrameID: Int64?,
        limit: Int
    ) async throws -> [FrameEmbeddingSource]

    /// Lowest and highest document id that has a `kind` embedding from `model`, or nil when there are none.
    func embeddingDocIDBounds(kind: EmbeddingKind, model: String) async throws -> (min: Int64, max: Int64)?

    /// Highest document id in the FTS index, or nil when nothing is indexed.
    func latestIndexedDocID() async throws -> Int64?

    /// Stores records made by `model` from `sources`, replacing any the frames already have of the same kind,
    /// and returns the records that were written.
    ///
    /// A vector is written only if its source is unchanged since it was read: the frame still has the same
    /// document with the same text, it has not been deleted, and (for a screenshot) its `revision` is the same.
    /// Embedding takes time; if the frame was redacted, reprocessed or deleted meanwhile, a vector made from
    /// the old text or pixels must not be stored. A "nothing to embed" marker needs the same document and text
    /// too. A "retry later" marker only needs the frame and its document to exist, and keeps the time the frame
    /// was first marked.
    func storeFrameEmbeddings(
        _ records: [FrameEmbeddingRecord],
        from sources: [FrameEmbeddingSource],
        model: String
    ) async throws -> [FrameEmbeddingRecord]

    /// The frames among `frameIDs` that currently have a `kind` vector from `model` and are not deleted.
    func framesWithEmbedding(kind: EmbeddingKind, model: String, among frameIDs: [Int64]) async throws -> Set<Int64>

    /// `kind` vectors from `model` with a frame id above `frameID`, ascending. Markers are left out.
    func frameEmbeddings(
        kind: EmbeddingKind,
        model: String,
        afterFrameID frameID: Int64,
        limit: Int
    ) async throws -> [FrameEmbeddingRecord]

    /// Frames `model` has dealt with for `kind`: a vector, or a marker that there is nothing to embed.
    /// Frames waiting to be retried are not counted.
    func frameEmbeddingCount(kind: EmbeddingKind, model: String) async throws -> Int

    /// Number of frames with OCR text in the FTS index (what there is to embed).
    func indexedDocumentCount() async throws -> Int

    /// Removes `kind` vectors made by any other model. Returns how many were removed.
    func deleteFrameEmbeddings(kind: EmbeddingKind, exceptModel model: String) async throws -> Int

    /// Removes every vector of `kind`, or of all kinds when nil.
    func deleteAllFrameEmbeddings(kind: EmbeddingKind?) async throws
}

/// Reads a frame's screenshot for embedding.
public protocol FrameImageProviding: Sendable {
    func frameImage(for source: FrameEmbeddingSource) async throws -> CGImage

    /// Forgets any decoded or cached video state. Called before each batch, so every screenshot is read from the
    /// video file as it is now: a chunk rewritten to burn in a redaction must never be read through a decoder
    /// that was opened before the rewrite. Called again when the batch ends, so no decoder stays open once
    /// indexing has nothing left to do.
    func discardCachedFrames()
}

extension FrameImageProviding {
    public func discardCachedFrames() {}
}

/// A frame whose meaning stands out for a query.
public struct SemanticMatch: Sendable, Equatable {
    public let frameID: Int64
    /// How far this frame stands out among all frames for this query, in standard deviations.
    public let score: Double
    public let cosine: Float
    /// Which of the frame's vectors matched best.
    public let kind: EmbeddingKind

    public init(frameID: Int64, score: Double, cosine: Float, kind: EmbeddingKind = .text) {
        self.frameID = frameID
        self.score = score
        self.cosine = cosine
        self.kind = kind
    }
}

/// Finds frames by meaning.
/// Owner: SEARCH agent
public protocol SemanticSearchProtocol: Sendable {
    /// Frames whose meaning stands out for `query`, best first. Empty when semantic search is off,
    /// the model is missing, or nothing stands out. Never throws: keyword search must keep working.
    func matches(for query: String, limit: Int) async -> [SemanticMatch]
}

/// Progress of the embedding index, for Settings.
public struct SemanticIndexStatus: Sendable, Equatable {
    public enum State: String, Sendable {
        case off
        case modelMissing
        case helperMissing
        case indexing
        case upToDate
        case paused
        case failed
    }

    public let state: State
    /// Frames whose OCR text has been looked at.
    public let embeddedFrames: Int
    /// Frames whose screenshot has been looked at.
    public let embeddedImages: Int
    /// Frames in the FTS index (what there is to embed).
    public let indexedFrames: Int
    /// Whether screenshots are being embedded too.
    public let imagesEnabled: Bool
    public let detail: String?

    public init(
        state: State,
        embeddedFrames: Int,
        embeddedImages: Int = 0,
        indexedFrames: Int,
        imagesEnabled: Bool = false,
        detail: String? = nil
    ) {
        self.state = state
        self.embeddedFrames = embeddedFrames
        self.embeddedImages = embeddedImages
        self.indexedFrames = indexedFrames
        self.imagesEnabled = imagesEnabled
        self.detail = detail
    }
}
