import Accelerate
import Foundation
import Shared

/// Vector helpers for semantic search: Matryoshka truncation and the float16 storage format.
enum EmbeddingVector {
    /// Keeps the leading `dimension` values of a model vector and L2-normalizes them again.
    /// EmbeddingGemma 2 is trained for this (Matryoshka); a sliced unit vector is no longer unit length.
    static func truncated(_ vector: [Float], to dimension: Int) -> [Float]? {
        guard vector.count >= dimension, dimension > 0 else { return nil }
        var head = Array(vector[0..<dimension])
        var sumOfSquares: Float = 0
        vDSP_svesq(head, 1, &sumOfSquares, vDSP_Length(dimension))
        guard sumOfSquares > 0, sumOfSquares.isFinite else { return nil }
        var scale = 1 / sumOfSquares.squareRoot()
        vDSP_vsmul(head, 1, &scale, &head, 1, vDSP_Length(dimension))
        return head
    }

    /// `dimension` × float16, little-endian: half the bytes of float32, with no measurable effect on ranking.
    static func storageData(_ vector: [Float]) -> Data {
        let halves = vector.map { Float16($0) }
        return halves.withUnsafeBufferPointer { Data(buffer: $0) }
    }
}

/// Every stored vector in memory, scored by brute force.
///
/// Rows stay float16 (half the memory: 512 bytes a row at 256 dimensions) and are converted a block at a time for
/// the dot products. A scan is one matrix-vector product, linear in the number of rows: on an M5 Pro an optimized
/// build takes about 25 ms for a million rows and 90 ms for three million (`VectorIndexBenchmarks` measures it).
/// Not thread-safe: owned by `SemanticSearchService`.
final class VectorIndex {
    static let blockRows = 2048

    let dimension: Int
    private(set) var frameIDs: [Int64] = []
    private var halves: [Float16] = []
    private var rowOfFrame: [Int64: Int] = [:]
    /// A number per row that changes whenever that frame's vector is written. A search notes it when it scores
    /// a vector and checks it again before returning, so a vector replaced in between is noticed.
    private var versions: [UInt32] = []
    private var nextVersion: UInt32 = 1

    init(dimension: Int) {
        self.dimension = dimension
    }

    var count: Int { frameIDs.count }

    func removeAll() {
        frameIDs.removeAll()
        halves.removeAll()
        rowOfFrame.removeAll()
        versions.removeAll()
    }

    /// Makes room for `rows` vectors in one allocation. Without it the storage grows by doubling, which can
    /// leave it holding close to twice what it needs.
    func reserve(rows: Int) {
        guard rows > count else { return }
        frameIDs.reserveCapacity(rows)
        versions.reserveCapacity(rows)
        halves.reserveCapacity(rows * dimension)
        rowOfFrame.reserveCapacity(rows)
    }

    /// Adds a frame's vector, or replaces the one it already has. Ignores data of the wrong size.
    func upsert(frameID: Int64, vector: Data) {
        guard vector.count == dimension * MemoryLayout<Float16>.stride else { return }
        let version = nextVersion
        nextVersion &+= 1
        if let row = rowOfFrame[frameID] {
            versions[row] = version
            vector.withUnsafeBytes { bytes in
                let source = bytes.bindMemory(to: Float16.self)
                for offset in 0..<dimension {
                    halves[row * dimension + offset] = source[offset]
                }
            }
            return
        }
        rowOfFrame[frameID] = frameIDs.count
        frameIDs.append(frameID)
        versions.append(version)
        vector.withUnsafeBytes { bytes in
            halves.append(contentsOf: bytes.bindMemory(to: Float16.self))
        }
    }

    /// Removes a frame's vector, if it has one (its text was redacted or reprocessed, or the frame was deleted).
    func remove(frameID: Int64) {
        guard let row = rowOfFrame.removeValue(forKey: frameID) else { return }
        let last = frameIDs.count - 1
        if row != last {
            // Move the last row into the gap.
            let moved = frameIDs[last]
            frameIDs[row] = moved
            versions[row] = versions[last]
            rowOfFrame[moved] = row
            for offset in 0..<dimension {
                halves[row * dimension + offset] = halves[last * dimension + offset]
            }
        }
        frameIDs.removeLast()
        versions.removeLast()
        halves.removeLast(dimension)
    }

    /// Cosine similarity of `query` (unit length, `dimension` values) with every row.
    func scores(for query: [Float]) -> [Float] {
        let rows = frameIDs.count
        var scores = [Float](repeating: 0, count: rows)
        guard rows > 0, query.count == dimension else { return scores }

        var block = [Float](repeating: 0, count: Self.blockRows * dimension)
        halves.withUnsafeBufferPointer { halves in
            scores.withUnsafeMutableBufferPointer { scores in
                block.withUnsafeMutableBufferPointer { block in
                    var row = 0
                    while row < rows {
                        let blockCount = min(Self.blockRows, rows - row)
                        let values = blockCount * dimension
                        var source = vImage_Buffer(
                            data: UnsafeMutableRawPointer(mutating: halves.baseAddress! + row * dimension),
                            height: 1,
                            width: vImagePixelCount(values),
                            rowBytes: values * MemoryLayout<Float16>.stride
                        )
                        var destination = vImage_Buffer(
                            data: block.baseAddress!,
                            height: 1,
                            width: vImagePixelCount(values),
                            rowBytes: values * MemoryLayout<Float>.stride
                        )
                        vImageConvert_Planar16FtoPlanarF(&source, &destination, vImage_Flags(kvImageNoFlags))
                        // (blockCount × dimension) · (dimension × 1)
                        vDSP_mmul(
                            block.baseAddress!, 1,
                            query, 1,
                            scores.baseAddress! + row, 1,
                            vDSP_Length(blockCount), 1, vDSP_Length(dimension)
                        )
                        row += blockCount
                    }
                }
            }
        }
        return scores
    }

    /// Every row scored for one query, with what is needed to say how far a row stands out.
    struct Scan {
        fileprivate let scores: [Float]
        fileprivate let mean: Float
        fileprivate let spread: Double

        fileprivate func standing(ofRow row: Int) -> Double {
            Double(scores[row] - mean) / spread
        }
    }

    /// Scores every row for `query`. The result describes the index as it is now: use it before changing the index.
    func scan(for query: [Float]) -> Scan {
        let scores = scores(for: query)
        var mean: Float = 0
        var deviation: Float = 0
        if !scores.isEmpty {
            vDSP_normalize(scores, 1, nil, 1, &mean, &deviation, vDSP_Length(scores.count))
        }
        // A floor on the spread: with a handful of near-identical frames, tiny differences would look like outliers.
        return Scan(scores: scores, mean: mean, spread: Double(max(deviation, 0.02)))
    }

    /// How far `frameID` stands out in `scan` (standard deviations above the mean of all rows), its cosine
    /// similarity, and the version of the vector that was scored, or nil if the frame has no row.
    func standing(of frameID: Int64, in scan: Scan) -> (score: Double, cosine: Float, version: UInt32)? {
        guard let row = rowOfFrame[frameID], row < scan.scores.count else { return nil }
        return (scan.standing(ofRow: row), scan.scores[row], versions[row])
    }

    /// The version of the vector `frameID` has now, or nil if it has none.
    func version(of frameID: Int64) -> UInt32? {
        rowOfFrame[frameID].map { versions[$0] }
    }

    /// The rows of `scan` that stand out by at least `gate`, best first, at most `limit`.
    func matches(in scan: Scan, limit: Int, gate: Double, kind: EmbeddingKind = .text) -> [SemanticMatch] {
        guard !scan.scores.isEmpty, limit > 0 else { return [] }
        var candidates: [(row: Int, z: Double)] = []
        for row in scan.scores.indices {
            let z = scan.standing(ofRow: row)
            if z >= gate { candidates.append((row, z)) }
        }
        candidates.sort { $0.z > $1.z || ($0.z == $1.z && frameIDs[$0.row] > frameIDs[$1.row]) }
        return candidates.prefix(limit).map {
            SemanticMatch(frameID: frameIDs[$0.row], score: $0.z, cosine: scan.scores[$0.row], kind: kind)
        }
    }

    /// Every frame of `scan` that stands out by at least `gate`, in no particular order.
    func frames(standingOutIn scan: Scan, gate: Double) -> [Int64] {
        var frames: [Int64] = []
        for row in scan.scores.indices where scan.standing(ofRow: row) >= gate {
            frames.append(frameIDs[row])
        }
        return frames
    }

    /// Frames whose similarity stands out for this query, best first.
    ///
    /// Every row is scored anyway, so the distribution comes for free: a frame's score is how many standard
    /// deviations its cosine sits above the mean of all frames. Only frames at or above `gate` count as matches.
    /// That puts every query on one scale; it does not tell a real match from the best of nothing (see
    /// `SemanticSearchSettings.strongMatchCosine`).
    func matches(for query: [Float], limit: Int, gate: Double, kind: EmbeddingKind = .text) -> [SemanticMatch] {
        matches(in: scan(for: query), limit: limit, gate: gate, kind: kind)
    }
}
