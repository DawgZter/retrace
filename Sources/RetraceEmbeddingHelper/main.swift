import Foundation

// RetraceEmbeddingHelper --model <embeddinggemma-2-Q8_0.gguf> [--media <mmproj-embeddinggemma-2-Q8_0.gguf>]
//
// The embedding model as a separate process, so its memory (about half a gigabyte for text, a gigabyte with the
// image encoder) goes back to the system when Retrace stops it, and a crash in llama.cpp can't take the app
// down. Speaks JSON lines on stdin/stdout:
//
//   → (on start)                           {"ready":true,"dim":768,"images":false,"load_seconds":0.9}
//                                          or {"ready":false,"error":"…"}
//   ← {"id":1,"op":"embed","inputs":[…]}   {"id":1,"ok":true,"dim":768,"count":2,"vectors":"<base64 float32 LE>"}
//   ← {"id":2,"op":"image","images":[{"width":960,"height":624,"rgb":"<base64 packed RGB>"}]}
//                                          {"id":2,"ok":true,"dim":768,"count":1,"vectors":"…"}
//   ← {"id":3,"op":"quit"}                 {"id":3,"ok":true}
//
// Vectors are L2-normalized, one row per input. Images must already be at their final size (see `ImageSizing` in
// the Search module). The helper exits when stdin closes (its parent quit).

func emit(_ object: [String: Any]) {
    guard var data = try? JSONSerialization.data(withJSONObject: object) else { return }
    data.append(0x0A)
    FileHandle.standardOutput.write(data)
}

setvbuf(stdout, nil, _IONBF, 0)

var modelPath: String?
var mediaPath: String?
var arguments = CommandLine.arguments.dropFirst().makeIterator()
while let argument = arguments.next() {
    if argument == "--model" { modelPath = arguments.next() }
    if argument == "--media" { mediaPath = arguments.next() }
}

guard let modelPath else {
    emit(["ready": false, "error": "usage: RetraceEmbeddingHelper --model <path> [--media <path>]"])
    exit(2)
}
for path in [modelPath, mediaPath].compactMap({ $0 }) where !FileManager.default.fileExists(atPath: path) {
    emit(["ready": false, "error": "no model at \(path)"])
    exit(1)
}

let started = Date()
let embedder: Embedder
do {
    embedder = try Embedder(modelPath: modelPath, mediaPath: mediaPath)
} catch {
    emit(["ready": false, "error": "\(error)"])
    exit(1)
}
emit([
    "ready": true,
    "dim": embedder.dimension,
    "images": embedder.embedsImages,
    "load_seconds": (Date().timeIntervalSince(started) * 100).rounded() / 100,
])

func reply(id: Any, vectors: [[Float]]) {
    let floats = vectors.flatMap { $0 }
    emit([
        "id": id,
        "ok": true,
        "dim": embedder.dimension,
        "count": vectors.count,
        "vectors": floats.withUnsafeBufferPointer { Data(buffer: $0) }.base64EncodedString(),
    ])
}

func fail(id: Any, _ error: Error) {
    emit(["id": id, "ok": false, "error": String("\(error)".prefix(500))])
}

/// Handles one request line. Returns false when the helper should exit.
func handle(_ line: String) -> Bool {
    guard let data = line.data(using: .utf8),
          let request = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return true }
    let id: Any = request["id"] ?? NSNull()
    switch request["op"] as? String {
    case "quit":
        emit(["id": id, "ok": true])
        return false
    case "embed":
        do {
            reply(id: id, vectors: try embedder.embed(request["inputs"] as? [String] ?? []))
        } catch {
            fail(id: id, error)
        }
    case "image":
        let images = request["images"] as? [[String: Any]] ?? []
        do {
            let vectors = try images.map { image -> [Float] in
                guard let width = image["width"] as? Int, let height = image["height"] as? Int,
                      let encoded = image["rgb"] as? String, let rgb = Data(base64Encoded: encoded),
                      width > 0, height > 0, rgb.count == width * height * 3 else {
                    throw Embedder.Failure(description: "an image needs width, height and width × height × 3 bytes of rgb")
                }
                return try rgb.withUnsafeBytes { bytes in
                    try embedder.embedImage(
                        width: width,
                        height: height,
                        rgb: bytes.bindMemory(to: UInt8.self).baseAddress!
                    )
                }
            }
            reply(id: id, vectors: vectors)
        } catch {
            fail(id: id, error)
        }
    case let op:
        emit(["id": id, "ok": false, "error": "unknown op \(op ?? "(none)")"])
    }
    return true
}

// Each request runs in its own autorelease pool. A command-line loop has no run loop to drain one, and a
// screenshot request carries megabytes of JSON and pixel data that would otherwise pile up for the life of
// the process.
var running = true
while running, let line = readLine(strippingNewline: true) {
    running = autoreleasepool { handle(line) }
}
embedder.close()
