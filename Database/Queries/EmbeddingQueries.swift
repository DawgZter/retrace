import Foundation
import SQLCipher
import Shared

/// SQL for the `frame_embedding` table (semantic search vectors).
///
/// Every frame the indexer has looked at gets a row per kind, so a kind's resume points are simply its lowest and
/// highest docid. A row is a vector, a marker that the frame has nothing to embed, or a marker that it should be
/// tried again later (`FrameEmbeddingRecord.State`); readers of vectors only see the first.
/// Owner: DATABASE agent
enum EmbeddingQueries {
    private static let SQLITE_TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

    // MARK: - Sources

    /// Changes whenever a frame is processed again or its video is rewritten, so a screenshot read before the
    /// change can be told from one read after it.
    private static let revisionSQL = """
        (COALESCE(f.processedAt, 0) || ':' || COALESCE(f.rewrittenAt, 0) || ':'
            || COALESCE(f.processingStatus, 0) || ':' || COALESCE(f.rewritePurpose, ''))
        """

    private static let sourceColumnsSQL = """
        ds.frameId,
        ds.docid,
        f.createdAt,
        s.windowName,
        sc.c0,
        v.path,
        f.videoFrameIndex,
        v.frameRate,
        f.processingStatus,
        f.rewritePurpose,
        v.processingState,
        \(revisionSQL)
        """

    /// True for `ds` when it is its frame's current document. A database from before V15 can hold more than one
    /// document for a frame; the one with the highest id is the one keyword search reads, and the others are
    /// leftovers whose text must not be embedded.
    private static let isLatestDocumentSQL = """
        NOT EXISTS (
            SELECT 1 FROM doc_segment newer WHERE newer.frameId = ds.frameId AND newer.docid > ds.docid
        )
        """

    private static let sourceJoinsSQL = """
        JOIN searchRanking_content sc ON sc.id = ds.docid
        JOIN frame f ON f.id = ds.frameId
        LEFT JOIN segment s ON s.id = f.segmentId
        LEFT JOIN video v ON v.id = f.videoId
        """

    static func sources(
        db: OpaquePointer,
        afterDocID docID: Int64,
        limit: Int
    ) throws -> [FrameEmbeddingSource] {
        try sources(db: db, comparison: ">", order: "ASC", docID: docID, limit: limit)
    }

    static func sources(
        db: OpaquePointer,
        beforeDocID docID: Int64,
        limit: Int
    ) throws -> [FrameEmbeddingSource] {
        try sources(db: db, comparison: "<", order: "DESC", docID: docID, limit: limit)
    }

    private static func sources(
        db: OpaquePointer,
        comparison: String,
        order: String,
        docID: Int64,
        limit: Int
    ) throws -> [FrameEmbeddingSource] {
        let sql = """
            SELECT \(sourceColumnsSQL), NULL
            FROM doc_segment ds
            \(sourceJoinsSQL)
            WHERE ds.docid \(comparison) ? AND ds.frameId IS NOT NULL
            AND \(isLatestDocumentSQL)
            ORDER BY ds.docid \(order)
            LIMIT ?;
            """
        var statement: OpaquePointer?
        defer { sqlite3_finalize(statement) }
        guard sqlite3_prepare_v2(db, sql, -1, &statement, nil) == SQLITE_OK else {
            throw DatabaseError.queryFailed(query: sql, underlying: String(cString: sqlite3_errmsg(db)))
        }
        sqlite3_bind_int64(statement, 1, docID)
        sqlite3_bind_int64(statement, 2, Int64(limit))
        return readSources(statement)
    }

    /// Frames whose `kind` row says "try again later", newest first, as they are now.
    static func retrySources(
        db: OpaquePointer,
        kind: EmbeddingKind,
        model: String,
        beforeFrameID: Int64?,
        limit: Int
    ) throws -> [FrameEmbeddingSource] {
        let sql = """
            SELECT \(sourceColumnsSQL), e.updatedAt
            FROM frame_embedding e
            JOIN doc_segment ds ON ds.frameId = e.frameId AND ds.docid = e.docid
            \(sourceJoinsSQL)
            WHERE e.kind = ? AND e.state = 2 AND e.model = ? AND e.frameId < ?
            ORDER BY e.frameId DESC
            LIMIT ?;
            """
        var statement: OpaquePointer?
        defer { sqlite3_finalize(statement) }
        guard sqlite3_prepare_v2(db, sql, -1, &statement, nil) == SQLITE_OK else {
            throw DatabaseError.queryFailed(query: sql, underlying: String(cString: sqlite3_errmsg(db)))
        }
        sqlite3_bind_int64(statement, 1, Int64(kind.rawValue))
        sqlite3_bind_text(statement, 2, model, -1, SQLITE_TRANSIENT)
        sqlite3_bind_int64(statement, 3, beforeFrameID ?? Int64.max)
        sqlite3_bind_int64(statement, 4, Int64(limit))
        return readSources(statement)
    }

    private static func readSources(_ statement: OpaquePointer?) -> [FrameEmbeddingSource] {
        var results: [FrameEmbeddingSource] = []
        while sqlite3_step(statement) == SQLITE_ROW {
            let videoPath = sqlite3_column_text(statement, 5).map { String(cString: $0) }
            let rewritePurpose = sqlite3_column_text(statement, 9).map { String(cString: $0) }
            let videoFinalized = sqlite3_column_type(statement, 10) != SQLITE_NULL
                && sqlite3_column_int64(statement, 10) == 0
            results.append(
                FrameEmbeddingSource(
                    frameID: sqlite3_column_int64(statement, 0),
                    docID: sqlite3_column_int64(statement, 1),
                    createdAt: Date(timeIntervalSince1970: Double(sqlite3_column_int64(statement, 2)) / 1000),
                    windowName: sqlite3_column_text(statement, 3).map { String(cString: $0) },
                    text: sqlite3_column_text(statement, 4).map { String(cString: $0) } ?? "",
                    videoPath: videoPath,
                    frameIndex: Int(sqlite3_column_int64(statement, 6)),
                    videoFrameRate: sqlite3_column_type(statement, 7) != SQLITE_NULL
                        ? sqlite3_column_double(statement, 7)
                        : nil,
                    imageState: imageState(
                        processingStatus: Int(sqlite3_column_int64(statement, 8)),
                        rewritePurpose: rewritePurpose,
                        hasVideo: videoPath != nil,
                        videoFinalized: videoFinalized
                    ),
                    isMarkedForDeletion: rewritePurpose == "deletion",
                    revision: sqlite3_column_text(statement, 11).map { String(cString: $0) } ?? "",
                    retryingSince: sqlite3_column_type(statement, 12) != SQLITE_NULL
                        ? Date(timeIntervalSince1970: Double(sqlite3_column_int64(statement, 12)) / 1000)
                        : nil
                )
            )
        }
        return results
    }

    /// Whether a frame's pixels are final. A screenshot is embedded only once OCR has finished and any
    /// redaction has been burned into the video, so a vector can never describe pixels the user had removed.
    /// (processingStatus: 2 completed, 3 failed, 5/6 redaction rewrite pending, 7 rewrite completed, 8 rewrite failed.)
    ///
    /// A failed rewrite counts as pending, not as final: the processing queue retries it, and a retry that
    /// succeeds does not replace the frame's document, so nothing else would bring the frame back.
    static func imageState(
        processingStatus: Int,
        rewritePurpose: String?,
        hasVideo: Bool,
        videoFinalized: Bool
    ) -> FrameEmbeddingSource.ImageState {
        if processingStatus == 3 || rewritePurpose == "deletion" {
            return .unavailable
        }
        if processingStatus == 8 { return .pending }
        guard hasVideo, videoFinalized else { return .pending }
        if processingStatus == 7 { return .ready }
        if processingStatus == 2 { return rewritePurpose == nil ? .ready : .pending }
        return .pending
    }

    static func latestIndexedDocID(db: OpaquePointer) throws -> Int64? {
        try singleInt64(
            db: db,
            sql: "SELECT docid FROM doc_segment WHERE frameId IS NOT NULL ORDER BY docid DESC LIMIT 1;"
        )
    }

    static func indexedDocumentCount(db: OpaquePointer) throws -> Int {
        // Frames, not documents: a frame with a leftover document (see `isLatestDocumentSQL`) counts once.
        Int(try singleInt64(db: db, sql: "SELECT COUNT(DISTINCT frameId) FROM doc_segment WHERE frameId IS NOT NULL;") ?? 0)
    }

    // MARK: - Embeddings

    static func docIDBounds(
        db: OpaquePointer,
        kind: EmbeddingKind,
        model: String
    ) throws -> (min: Int64, max: Int64)? {
        // Two walks of the (kind, model, docid) index rather than one MIN/MAX scan.
        guard let lowest = try singleInt64(
            db: db,
            sql: "SELECT docid FROM frame_embedding WHERE kind = ? AND model = ? ORDER BY docid ASC LIMIT 1;",
            kind: kind,
            text: model
        ), let highest = try singleInt64(
            db: db,
            sql: "SELECT docid FROM frame_embedding WHERE kind = ? AND model = ? ORDER BY docid DESC LIMIT 1;",
            kind: kind,
            text: model
        ) else {
            return nil
        }
        return (lowest, highest)
    }

    /// Writes each record whose source is unchanged since it was read, and returns the ones written.
    ///
    /// "Unchanged" is checked in the same statement as the write. A vector, and a note that there is nothing to
    /// embed, need the frame to still have the same document with the same text (a document's id alone is not
    /// enough: SQLite can give a replacement the id of the document it replaced). A vector also needs the frame
    /// not to be marked for deletion and, for a screenshot, the same revision as when its pixels were read.
    /// "Try again later" says nothing about the frame's content, so it only needs the frame and its document
    /// to exist: it is what gets written when one of the others was refused.
    ///
    /// A row that is already in the state being written keeps its `updatedAt`, so "retry later" remembers when
    /// the frame was first found not ready.
    static func store(
        db: OpaquePointer,
        records: [FrameEmbeddingRecord],
        sources: [FrameEmbeddingSource],
        model: String,
        now: Date = Date()
    ) throws -> [FrameEmbeddingRecord] {
        guard !records.isEmpty else { return [] }
        let sourceOfFrame = Dictionary(sources.map { ($0.frameID, $0) }, uniquingKeysWith: { first, _ in first })

        // ?1 frame, ?2 kind, ?3 state, ?4 docid, ?5 model, ?6 now, ?7 vector, ?8 source text, ?9 source revision
        let sql = """
            INSERT INTO frame_embedding (frameId, kind, state, docid, model, updatedAt, vector)
            SELECT ?1, ?2, ?3, ?4, ?5, ?6, ?7
            WHERE EXISTS (
                SELECT 1
                FROM doc_segment ds
                JOIN searchRanking_content sc ON sc.id = ds.docid
                JOIN frame f ON f.id = ds.frameId
                WHERE ds.frameId = ?1 AND ds.docid = ?4
                AND \(isLatestDocumentSQL)
                AND (?3 = 2 OR COALESCE(sc.c0, '') = ?8)
                AND (
                    ?3 != 0
                    OR (
                        (f.rewritePurpose IS NULL OR f.rewritePurpose != 'deletion')
                        AND (?2 != 1 OR \(revisionSQL) = ?9)
                    )
                )
            )
            ON CONFLICT (frameId, kind) DO UPDATE SET
                updatedAt = CASE
                    WHEN frame_embedding.state = excluded.state AND frame_embedding.state != 0
                        AND frame_embedding.docid = excluded.docid AND frame_embedding.model = excluded.model
                    THEN frame_embedding.updatedAt
                    ELSE excluded.updatedAt
                END,
                state = excluded.state,
                docid = excluded.docid,
                model = excluded.model,
                vector = excluded.vector;
            """
        let nowMs = Int64(now.timeIntervalSince1970 * 1000)
        var stored: [FrameEmbeddingRecord] = []
        try exec(db: db, sql: "SAVEPOINT frame_embedding_store;")
        do {
            var statement: OpaquePointer?
            defer { sqlite3_finalize(statement) }
            guard sqlite3_prepare_v2(db, sql, -1, &statement, nil) == SQLITE_OK else {
                throw DatabaseError.queryFailed(query: sql, underlying: String(cString: sqlite3_errmsg(db)))
            }
            for record in records {
                guard let source = sourceOfFrame[record.frameID], source.docID == record.docID else { continue }
                sqlite3_reset(statement)
                sqlite3_clear_bindings(statement)
                sqlite3_bind_int64(statement, 1, record.frameID)
                sqlite3_bind_int64(statement, 2, Int64(record.kind.rawValue))
                sqlite3_bind_int64(statement, 3, Int64(record.state.rawValue))
                sqlite3_bind_int64(statement, 4, record.docID)
                sqlite3_bind_text(statement, 5, model, -1, SQLITE_TRANSIENT)
                sqlite3_bind_int64(statement, 6, nowMs)
                if record.vector.isEmpty {
                    sqlite3_bind_zeroblob(statement, 7, 0)
                } else {
                    _ = record.vector.withUnsafeBytes { bytes in
                        sqlite3_bind_blob(statement, 7, bytes.baseAddress, Int32(bytes.count), SQLITE_TRANSIENT)
                    }
                }
                sqlite3_bind_text(statement, 8, source.text, -1, SQLITE_TRANSIENT)
                sqlite3_bind_text(statement, 9, source.revision, -1, SQLITE_TRANSIENT)
                guard sqlite3_step(statement) == SQLITE_DONE else {
                    throw DatabaseError.queryFailed(query: sql, underlying: String(cString: sqlite3_errmsg(db)))
                }
                if sqlite3_changes(db) > 0 {
                    stored.append(record)
                }
            }
            try exec(db: db, sql: "RELEASE frame_embedding_store;")
        } catch {
            try? exec(db: db, sql: "ROLLBACK TO frame_embedding_store; RELEASE frame_embedding_store;")
            throw error
        }
        return stored
    }

    static func framesWithEmbedding(
        db: OpaquePointer,
        kind: EmbeddingKind,
        model: String,
        among frameIDs: [Int64]
    ) throws -> Set<Int64> {
        var found = Set<Int64>()
        // In pieces, to stay well under SQLite's bound-parameter limit.
        var start = 0
        while start < frameIDs.count {
            let chunk = Array(frameIDs[start..<min(start + 500, frameIDs.count)])
            start += chunk.count
            let placeholders = chunk.map { _ in "?" }.joined(separator: ", ")
            let sql = """
                SELECT frameId FROM frame_embedding
                WHERE kind = ? AND model = ? AND state = 0 AND frameId IN (\(placeholders));
                """
            var statement: OpaquePointer?
            defer { sqlite3_finalize(statement) }
            guard sqlite3_prepare_v2(db, sql, -1, &statement, nil) == SQLITE_OK else {
                throw DatabaseError.queryFailed(query: sql, underlying: String(cString: sqlite3_errmsg(db)))
            }
            sqlite3_bind_int64(statement, 1, Int64(kind.rawValue))
            sqlite3_bind_text(statement, 2, model, -1, SQLITE_TRANSIENT)
            for (offset, frameID) in chunk.enumerated() {
                sqlite3_bind_int64(statement, Int32(offset + 3), frameID)
            }
            while sqlite3_step(statement) == SQLITE_ROW {
                found.insert(sqlite3_column_int64(statement, 0))
            }
        }
        return found
    }

    /// One page of a kind's vectors, in frame order.
    ///
    /// The unary plus keeps SQLite from answering this through the (kind, model, docid) index, which would
    /// collect and sort every row of the kind for each page. Walking the primary key from `frameId` on is what
    /// makes reading N vectors a page at a time take time proportional to N.
    static let vectorPageSQL = """
        SELECT frameId, docid, vector
        FROM frame_embedding
        WHERE frameId > ? AND +kind = ? AND +model = ? AND state = 0
        ORDER BY frameId ASC
        LIMIT ?;
        """

    static func embeddings(
        db: OpaquePointer,
        kind: EmbeddingKind,
        model: String,
        afterFrameID frameID: Int64,
        limit: Int
    ) throws -> [FrameEmbeddingRecord] {
        let sql = vectorPageSQL
        var statement: OpaquePointer?
        defer { sqlite3_finalize(statement) }
        guard sqlite3_prepare_v2(db, sql, -1, &statement, nil) == SQLITE_OK else {
            throw DatabaseError.queryFailed(query: sql, underlying: String(cString: sqlite3_errmsg(db)))
        }
        sqlite3_bind_int64(statement, 1, frameID)
        sqlite3_bind_int64(statement, 2, Int64(kind.rawValue))
        sqlite3_bind_text(statement, 3, model, -1, SQLITE_TRANSIENT)
        sqlite3_bind_int64(statement, 4, Int64(limit))

        var results: [FrameEmbeddingRecord] = []
        while sqlite3_step(statement) == SQLITE_ROW {
            let byteCount = Int(sqlite3_column_bytes(statement, 2))
            guard byteCount > 0, let bytes = sqlite3_column_blob(statement, 2) else { continue }
            results.append(
                FrameEmbeddingRecord(
                    frameID: sqlite3_column_int64(statement, 0),
                    kind: kind,
                    docID: sqlite3_column_int64(statement, 1),
                    vector: Data(bytes: bytes, count: byteCount)
                )
            )
        }
        return results
    }

    /// Frames dealt with (a vector, or nothing to embed). Both counts are answered from indexes alone.
    static func count(db: OpaquePointer, kind: EmbeddingKind, model: String) throws -> Int {
        let all = try singleInt64(
            db: db,
            sql: "SELECT COUNT(*) FROM frame_embedding WHERE kind = ? AND model = ?;",
            kind: kind,
            text: model
        ) ?? 0
        let waiting = try singleInt64(
            db: db,
            sql: "SELECT COUNT(*) FROM frame_embedding WHERE kind = ? AND state = 2;",
            kind: kind
        ) ?? 0
        return Int(max(0, all - waiting))
    }

    /// Marks the vectors made from document `docID` as out of date: its text was changed in place, which no
    /// trigger sees. The vectors are emptied at once and the frame is embedded again by the indexer's retry pass.
    static func requeue(db: OpaquePointer, docID: Int64, now: Date = Date()) throws {
        let sql = """
            UPDATE frame_embedding
            SET state = 2, vector = zeroblob(0), updatedAt = ?
            WHERE frameId IN (SELECT frameId FROM doc_segment WHERE docid = ? AND frameId IS NOT NULL);
            """
        var statement: OpaquePointer?
        defer { sqlite3_finalize(statement) }
        guard sqlite3_prepare_v2(db, sql, -1, &statement, nil) == SQLITE_OK else {
            throw DatabaseError.queryFailed(query: sql, underlying: String(cString: sqlite3_errmsg(db)))
        }
        sqlite3_bind_int64(statement, 1, Int64(now.timeIntervalSince1970 * 1000))
        sqlite3_bind_int64(statement, 2, docID)
        guard sqlite3_step(statement) == SQLITE_DONE else {
            throw DatabaseError.queryFailed(query: sql, underlying: String(cString: sqlite3_errmsg(db)))
        }
    }

    static func delete(db: OpaquePointer, kind: EmbeddingKind, exceptModel model: String) throws -> Int {
        let sql = "DELETE FROM frame_embedding WHERE kind = ? AND model != ?;"
        var statement: OpaquePointer?
        defer { sqlite3_finalize(statement) }
        guard sqlite3_prepare_v2(db, sql, -1, &statement, nil) == SQLITE_OK else {
            throw DatabaseError.queryFailed(query: sql, underlying: String(cString: sqlite3_errmsg(db)))
        }
        sqlite3_bind_int64(statement, 1, Int64(kind.rawValue))
        sqlite3_bind_text(statement, 2, model, -1, SQLITE_TRANSIENT)
        guard sqlite3_step(statement) == SQLITE_DONE else {
            throw DatabaseError.queryFailed(query: sql, underlying: String(cString: sqlite3_errmsg(db)))
        }
        return Int(sqlite3_changes(db))
    }

    static func deleteAll(db: OpaquePointer, kind: EmbeddingKind?) throws {
        if let kind {
            try exec(db: db, sql: "DELETE FROM frame_embedding WHERE kind = \(kind.rawValue);")
        } else {
            try exec(db: db, sql: "DELETE FROM frame_embedding;")
        }
    }

    // MARK: - Helpers

    private static func exec(db: OpaquePointer, sql: String) throws {
        var errorPointer: UnsafeMutablePointer<CChar>?
        guard sqlite3_exec(db, sql, nil, nil, &errorPointer) == SQLITE_OK else {
            let message = errorPointer.flatMap { String(cString: $0) } ?? "Unknown error"
            sqlite3_free(errorPointer)
            throw DatabaseError.queryFailed(query: sql, underlying: message)
        }
    }

    private static func singleInt64(
        db: OpaquePointer,
        sql: String,
        kind: EmbeddingKind? = nil,
        text: String? = nil
    ) throws -> Int64? {
        var statement: OpaquePointer?
        defer { sqlite3_finalize(statement) }
        guard sqlite3_prepare_v2(db, sql, -1, &statement, nil) == SQLITE_OK else {
            throw DatabaseError.queryFailed(query: sql, underlying: String(cString: sqlite3_errmsg(db)))
        }
        var index: Int32 = 1
        if let kind {
            sqlite3_bind_int64(statement, index, Int64(kind.rawValue))
            index += 1
        }
        if let text {
            sqlite3_bind_text(statement, index, text, -1, SQLITE_TRANSIENT)
        }
        guard sqlite3_step(statement) == SQLITE_ROW, sqlite3_column_type(statement, 0) != SQLITE_NULL else {
            return nil
        }
        return sqlite3_column_int64(statement, 0)
    }
}

// MARK: - DatabaseManager conformance

extension DatabaseManager: EmbeddingStoreProtocol {
    private func embeddingConnection() throws -> OpaquePointer {
        guard let db = getConnection() else {
            throw DatabaseError.connectionFailed(underlying: "Database not initialized")
        }
        return db
    }

    public func embeddingSources(afterDocID docID: Int64, limit: Int) async throws -> [FrameEmbeddingSource] {
        try EmbeddingQueries.sources(db: try embeddingConnection(), afterDocID: docID, limit: limit)
    }

    public func embeddingSources(beforeDocID docID: Int64, limit: Int) async throws -> [FrameEmbeddingSource] {
        try EmbeddingQueries.sources(db: try embeddingConnection(), beforeDocID: docID, limit: limit)
    }

    public func embeddingSources(
        retryingKind kind: EmbeddingKind,
        model: String,
        beforeFrameID: Int64?,
        limit: Int
    ) async throws -> [FrameEmbeddingSource] {
        try EmbeddingQueries.retrySources(
            db: try embeddingConnection(),
            kind: kind,
            model: model,
            beforeFrameID: beforeFrameID,
            limit: limit
        )
    }

    public func embeddingDocIDBounds(
        kind: EmbeddingKind,
        model: String
    ) async throws -> (min: Int64, max: Int64)? {
        try EmbeddingQueries.docIDBounds(db: try embeddingConnection(), kind: kind, model: model)
    }

    public func latestIndexedDocID() async throws -> Int64? {
        try EmbeddingQueries.latestIndexedDocID(db: try embeddingConnection())
    }

    public func storeFrameEmbeddings(
        _ records: [FrameEmbeddingRecord],
        from sources: [FrameEmbeddingSource],
        model: String
    ) async throws -> [FrameEmbeddingRecord] {
        try EmbeddingQueries.store(db: try embeddingConnection(), records: records, sources: sources, model: model)
    }

    public func framesWithEmbedding(
        kind: EmbeddingKind,
        model: String,
        among frameIDs: [Int64]
    ) async throws -> Set<Int64> {
        try EmbeddingQueries.framesWithEmbedding(
            db: try embeddingConnection(),
            kind: kind,
            model: model,
            among: frameIDs
        )
    }

    public func frameEmbeddings(
        kind: EmbeddingKind,
        model: String,
        afterFrameID frameID: Int64,
        limit: Int
    ) async throws -> [FrameEmbeddingRecord] {
        try EmbeddingQueries.embeddings(
            db: try embeddingConnection(),
            kind: kind,
            model: model,
            afterFrameID: frameID,
            limit: limit
        )
    }

    public func frameEmbeddingCount(kind: EmbeddingKind, model: String) async throws -> Int {
        try EmbeddingQueries.count(db: try embeddingConnection(), kind: kind, model: model)
    }

    public func indexedDocumentCount() async throws -> Int {
        try EmbeddingQueries.indexedDocumentCount(db: try embeddingConnection())
    }

    public func deleteFrameEmbeddings(kind: EmbeddingKind, exceptModel model: String) async throws -> Int {
        try EmbeddingQueries.delete(db: try embeddingConnection(), kind: kind, exceptModel: model)
    }

    public func deleteAllFrameEmbeddings(kind: EmbeddingKind?) async throws {
        try EmbeddingQueries.deleteAll(db: try embeddingConnection(), kind: kind)
    }
}
