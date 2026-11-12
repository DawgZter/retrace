import XCTest
import Foundation
import Shared
import SQLCipher
import Search
@testable import Database

/// How long search takes to read its vectors from the database, on this machine. Skipped unless asked for:
///
///   RETRACE_SEMANTIC_BENCH_LOAD_ROWS=100000,1000000 swift test --filter FrameEmbeddingLoadBenchmarks
///
/// For each row count it fills `frame_embedding` in a database file in the temporary folder, then times what the
/// first search after launch (or after ten idle minutes) waits for: `SemanticSearchService` reading every vector
/// into memory. The vectors are random bytes; only their number and size matter here.
final class FrameEmbeddingLoadBenchmarks: XCTestCase {
    private struct NoModel: TextEmbedding {
        func embed(_ texts: [String]) async throws -> [[Float]] {
            texts.map { _ in [Float](repeating: 0.0625, count: SemanticSearchSettings.dimension) }
        }
        func shutdown() {}
    }

    func testTimeToLoadStoredVectors() async throws {
        guard let setting = ProcessInfo.processInfo.environment["RETRACE_SEMANTIC_BENCH_LOAD_ROWS"] else {
            throw XCTSkip("set RETRACE_SEMANTIC_BENCH_LOAD_ROWS (for example 100000,1000000) to run")
        }
        let model = SemanticSearchSettings.modelIdentifier(for: .text)
        for rows in setting.split(separator: ",").compactMap({ Int($0.trimmingCharacters(in: .whitespaces)) }) {
            let file = FileManager.default.temporaryDirectory
                .appendingPathComponent("retrace_embedding_bench_\(UUID().uuidString).db")
            defer {
                for suffix in ["", "-wal", "-shm"] { try? FileManager.default.removeItem(atPath: file.path + suffix) }
            }
            let database = DatabaseManager(databasePath: file.path)
            try await database.initialize()
            let maybeConnection = await database.getConnection()
            let db = try XCTUnwrap(maybeConnection)
            // These rows have no frames behind them: that check is switched off for the fill.
            let fill = """
                PRAGMA foreign_keys = OFF;
                INSERT INTO frame_embedding (frameId, kind, state, docid, model, updatedAt, vector)
                WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < \(rows))
                SELECT i, 0, 0, i, '\(model)', 0, randomblob(\(SemanticSearchSettings.dimension * 2)) FROM n;
                PRAGMA foreign_keys = ON;
                PRAGMA wal_checkpoint(TRUNCATE);
                """
            XCTAssertEqual(sqlite3_exec(db, fill, nil, nil, nil), SQLITE_OK, String(cString: sqlite3_errmsg(db)))
            let megabytes = ((try? FileManager.default.attributesOfItem(atPath: file.path)[.size] as? Int) ?? 0) / 1_000_000

            let service = SemanticSearchService(store: database, embedder: NoModel(), isEnabled: { true })
            let started = Date()
            await service.prepareForSearch()
            let seconds = Date().timeIntervalSince(started)
            let stored = try await database.frameEmbeddingCount(kind: .text, model: model)
            XCTAssertEqual(stored, rows)
            print(String(format: "[bench] %d vectors (%d MB database file): read into memory in %.2f s", rows, megabytes, seconds))
            try await database.close()
        }
    }
}
