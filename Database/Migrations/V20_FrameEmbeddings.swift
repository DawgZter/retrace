import Foundation
import SQLCipher
import Shared

/// V20 Migration: `frame_embedding`, the vectors semantic search compares a query with.
///
/// A vector is derived from a frame's OCR text or screenshot, so it must not outlive them. Triggers remove a
/// frame's vectors when the frame is deleted (retention), when the user deletes it and it is waiting to be cut
/// out of its video (quick delete), and when its FTS document is removed (OCR reprocessing, redaction). A frame
/// that gets a new document is embedded again by the indexer.
struct V20_FrameEmbeddings: Migration {
    let version = 20

    func migrate(db: OpaquePointer) async throws {
        Log.info("🧭 Creating V20 frame_embedding table for semantic search...", category: .database)

        // One row per frame and kind, written once the indexer has looked at the frame.
        // kind:      0 = the frame's OCR text, 1 = its screenshot (EmbeddingKind).
        // state:     0 = `vector` holds N × float16 (little-endian, L2-normalized),
        //            1 = nothing to embed, 2 = not embeddable yet, try again (FrameEmbeddingRecord.State).
        // model:     what made the vector, so vectors from different models are never compared.
        // updatedAt: when the row entered its current state (ms). A frame is retried only for so long.
        try execute(
            db: db,
            sql: """
                CREATE TABLE IF NOT EXISTS frame_embedding (
                    frameId   INTEGER NOT NULL REFERENCES frame(id) ON DELETE CASCADE,
                    kind      INTEGER NOT NULL,
                    state     INTEGER NOT NULL DEFAULT 0,
                    docid     INTEGER NOT NULL,
                    model     TEXT NOT NULL,
                    updatedAt INTEGER NOT NULL DEFAULT 0,
                    vector    BLOB NOT NULL,
                    PRIMARY KEY (frameId, kind)
                ) WITHOUT ROWID;
                """
        )

        // The indexer walks documents in the order they were indexed; a kind's lowest and highest docid are
        // its resume points. The same index answers "how many frames are done" without touching the vectors.
        try execute(
            db: db,
            sql: """
                CREATE INDEX IF NOT EXISTS idx_frame_embedding_kind_model_docid
                ON frame_embedding(kind, model, docid);
                """
        )
        // The few frames waiting to be tried again.
        try execute(
            db: db,
            sql: """
                CREATE INDEX IF NOT EXISTS idx_frame_embedding_retry
                ON frame_embedding(kind, frameId)
                WHERE state = 2;
                """
        )

        // A trigger can only be attached to a table that exists. Every real database has these since V1 (and
        // frame.rewritePurpose since V15); a partial legacy schema, as some migration tests build, just goes
        // without the ones it can't have.
        if try tableExists(db: db, name: "frame") {
            try execute(
                db: db,
                sql: """
                    CREATE TRIGGER IF NOT EXISTS frame_embedding_after_frame_delete
                    AFTER DELETE ON frame
                    BEGIN
                        DELETE FROM frame_embedding WHERE frameId = OLD.id;
                    END;
                    """
            )

            // Deleting from the timeline only marks the frame; the row goes when its video has been rewritten.
            // Its vectors go now.
            if try columnExists(db: db, table: "frame", column: "rewritePurpose") {
                try execute(
                    db: db,
                    sql: """
                        CREATE TRIGGER IF NOT EXISTS frame_embedding_after_frame_marked_deleted
                        AFTER UPDATE OF rewritePurpose ON frame
                        WHEN NEW.rewritePurpose = 'deletion'
                        BEGIN
                            DELETE FROM frame_embedding WHERE frameId = NEW.id;
                        END;
                        """
                )
            }
        }

        if try tableExists(db: db, name: "doc_segment") {
            try execute(
                db: db,
                sql: """
                    CREATE TRIGGER IF NOT EXISTS frame_embedding_after_document_delete
                    AFTER DELETE ON doc_segment
                    WHEN OLD.frameId IS NOT NULL
                    BEGIN
                        DELETE FROM frame_embedding WHERE frameId = OLD.frameId;
                    END;
                    """
            )
        }

        Log.info("✅ V20 migration completed: frame_embedding table and cleanup triggers created", category: .database)
    }

    private func tableExists(db: OpaquePointer, name: String) throws -> Bool {
        try hasRow(
            db: db,
            sql: "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1;",
            arguments: [name]
        )
    }

    private func columnExists(db: OpaquePointer, table: String, column: String) throws -> Bool {
        try hasRow(
            db: db,
            sql: "SELECT 1 FROM pragma_table_info(?) WHERE name = ? LIMIT 1;",
            arguments: [table, column]
        )
    }

    private func hasRow(db: OpaquePointer, sql: String, arguments: [String]) throws -> Bool {
        var statement: OpaquePointer?
        defer { sqlite3_finalize(statement) }
        guard sqlite3_prepare_v2(db, sql, -1, &statement, nil) == SQLITE_OK else {
            throw DatabaseError.migrationFailed(
                version: version,
                underlying: "Failed to inspect the schema: \(String(cString: sqlite3_errmsg(db)))"
            )
        }
        for (offset, argument) in arguments.enumerated() {
            sqlite3_bind_text(
                statement,
                Int32(offset + 1),
                argument,
                -1,
                unsafeBitCast(-1, to: sqlite3_destructor_type.self)
            )
        }
        return sqlite3_step(statement) == SQLITE_ROW
    }

    private func execute(db: OpaquePointer, sql: String) throws {
        var errorPointer: UnsafeMutablePointer<CChar>?
        let result = sqlite3_exec(db, sql, nil, nil, &errorPointer)

        if result != SQLITE_OK {
            let errorMessage = errorPointer.flatMap { String(cString: $0) } ?? "Unknown error"
            sqlite3_free(errorPointer)
            throw DatabaseError.migrationFailed(version: version, underlying: "SQL execution failed: \(errorMessage)")
        }
    }
}
