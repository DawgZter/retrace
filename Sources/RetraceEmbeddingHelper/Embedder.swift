import CLlamaEmbedding
import Foundation

/// EmbeddingGemma 2 on llama.cpp (Metal): mean-pooled, L2-normalized 768-dimension vectors for text and, when the
/// image encoder is loaded, for screenshots, in one shared space.
///
/// Ported from the text and image paths of DigUp's `LlamaEmbedder` (https://github.com/ARahim3/DigUp, MIT,
/// © Abdur Rahim), which checked this framing against the reference implementation: text is `<bos> text <eos>`,
/// an image is `<bos> <|image> soft tokens <image|> <eos>`, each input is its own sequence, and a whole batch goes
/// through one `llama_process` call (the model is non-causal and has no KV cache).
///
/// Not thread-safe: one call at a time.
final class Embedder {
    struct Failure: Error, CustomStringConvertible {
        let description: String
    }

    /// One input must fit one micro-batch. 2048 tokens is the reference's cut; it holds a full screen of OCR text
    /// or the largest image (1120 soft tokens).
    static let maxTokens = 2048
    private static let maxSequences = 16

    let dimension: Int
    /// Whether the image encoder is loaded.
    var embedsImages: Bool { media != nil }
    private var model: OpaquePointer?
    private var context: OpaquePointer?
    private var media: OpaquePointer?
    private var batch: OpaquePointer?
    private let vocab: OpaquePointer?
    private let rowSize: Int   // the width of the encoder rows fed to the text model
    private var sequences: Int32 = 0
    private var tokens = 0
    private var position: llama_pos = 0

    /// - Parameter mediaPath: the image encoder (`mmproj-…gguf`), or nil for text only (about half the memory:
    ///   roughly 0.5 GB resident against 1 GB, measured on an M5 Pro).
    init(modelPath: String, mediaPath: String? = nil) throws {
        Self.setUpBackend()
        let cpuOnly = ProcessInfo.processInfo.environment["RETRACE_EMBEDDING_DEVICE"] == "cpu"

        var modelParams = llama_model_default_params()
        modelParams.n_gpu_layers = cpuOnly ? 0 : 999
        guard let model = llama_model_load_from_file(modelPath, modelParams) else {
            throw Failure(description: "can't load the model at \(modelPath)")
        }

        var params = llama_context_default_params()
        params.n_ctx = UInt32(Self.maxTokens)
        params.n_batch = UInt32(Self.maxTokens)
        params.n_ubatch = UInt32(Self.maxTokens)
        params.n_seq_max = UInt32(Self.maxSequences)
        params.kv_unified = true   // each input may use the whole context
        params.embeddings = true
        params.pooling_type = LLAMA_POOLING_TYPE_MEAN
        guard let context = llama_init_from_model(model, params) else {
            llama_model_free(model)
            throw Failure(description: "can't create a llama.cpp context")
        }

        var media: OpaquePointer?
        if let mediaPath {
            var mediaParams = mtmd_context_params_default()
            mediaParams.use_gpu = !cpuOnly
            mediaParams.print_timings = false
            mediaParams.warmup = true
            mediaParams.image_min_tokens = 1      // never rescale: images arrive at their final size
            mediaParams.image_max_tokens = 1120
            media = mtmd_init_from_file(mediaPath, model, mediaParams)
            if media == nil {
                llama_free(context)
                llama_model_free(model)
                throw Failure(description: "can't load the image encoder at \(mediaPath)")
            }
        }

        self.model = model
        self.context = context
        self.media = media
        batch = llama_batch_ext_init(context)
        vocab = llama_model_get_vocab(model)
        dimension = Int(llama_model_n_embd_out(model))
        rowSize = Int(llama_model_n_embd_inp(model))

        // The first calls build the Metal pipelines: pay for that now rather than on the first real request.
        do {
            for _ in 0..<2 { _ = try embed(["task: search result | query: warm up"]) }
        } catch {
            close()
            throw error
        }
    }

    deinit { close() }

    func close() {
        if let batch { llama_batch_ext_free(batch) }
        if let media { mtmd_free(media) }
        if let context { llama_free(context) }
        if let model { llama_model_free(model) }
        (batch, media, context, model) = (nil, nil, nil, nil)
    }

    /// One L2-normalized vector per text, in order.
    func embed(_ texts: [String]) throws -> [[Float]] {
        reset()
        var vectors: [[Float]] = []
        for text in texts {
            let tokens = tokenize(text)
            if sequences == Self.maxSequences || self.tokens + tokens.count > Self.maxTokens {
                vectors += try run()
            }
            sequences += 1
            position = 0
            for token in tokens { try add(token) }
        }
        if sequences > 0 { vectors += try run() }
        return vectors
    }

    /// The vector of one screenshot: `rgb` is `width` × `height` packed RGB, already at the size the caller chose
    /// for its token budget (the encoder is told never to rescale).
    func embedImage(width: Int, height: Int, rgb: UnsafePointer<UInt8>) throws -> [Float] {
        guard let media else { throw Failure(description: "this helper was started without the image encoder") }
        guard width > 0, height > 0, let bitmap = mtmd_bitmap_init(UInt32(width), UInt32(height), rgb) else {
            throw Failure(description: "can't read a \(width) × \(height) image")
        }
        defer { mtmd_bitmap_free(bitmap) }
        let chunks = mtmd_input_chunks_init()
        defer { mtmd_input_chunks_free(chunks) }

        var bitmaps: [OpaquePointer?] = [bitmap]
        let status = String(cString: mtmd_default_marker()).withCString { marker in
            var text = mtmd_input_text(text: marker, text_len: strlen(marker), add_special: true, parse_special: true)
            return mtmd_tokenize(media, chunks, &text, &bitmaps, 1)
        }
        guard status == 0 else { throw Failure(description: "can't tokenize the image (\(status))") }

        reset()
        sequences += 1
        position = 0
        for index in 0..<mtmd_input_chunks_size(chunks) {
            let chunk = mtmd_input_chunks_get(chunks, index)
            if mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT {
                var count = 0
                guard let tokens = mtmd_input_chunk_get_tokens_text(chunk, &count) else { continue }
                for i in 0..<count { try add(tokens[i]) }
            } else {
                guard mtmd_encode_chunk(media, chunk) == 0, let rows = mtmd_get_output_embd(media) else {
                    throw Failure(description: "the image encoder failed")
                }
                try add(rows: rows, count: mtmd_input_chunk_get_n_tokens(chunk))
            }
        }
        guard let vector = try run().first else { throw Failure(description: "no vector for the image") }
        return vector
    }

    // MARK: - Private

    /// `<bos> text <eos>`, cut to `maxTokens` keeping both special tokens. Special tokens are added, never
    /// parsed out of the text: OCR text or a query that happens to contain `<eos>` or `<|image>` is just text.
    private func tokenize(_ text: String) -> [llama_token] {
        let utf8 = Array(text.utf8CString)
        var tokens = [llama_token](repeating: 0, count: utf8.count + 8)
        var count = llama_tokenize(vocab, utf8, Int32(utf8.count - 1), &tokens, Int32(tokens.count), true, false)
        if count < 0 {
            tokens = [llama_token](repeating: 0, count: Int(-count))
            count = llama_tokenize(vocab, utf8, Int32(utf8.count - 1), &tokens, Int32(tokens.count), true, false)
        }
        tokens.removeLast(tokens.count - Int(max(count, 0)))
        if tokens.count > Self.maxTokens {
            tokens = Array(tokens[0..<(Self.maxTokens - 1)]) + [llama_vocab_eos(vocab)]
        }
        return tokens
    }

    private func reset() {
        llama_batch_ext_clear(batch)
        sequences = 0
        tokens = 0
    }

    private func add(_ token: llama_token) throws {
        try place(llama_batch_ext_add_token(batch, sequences - 1, token))
    }

    /// Adds encoder output rows (an image's soft tokens) to the current input.
    private func add(rows: UnsafeMutablePointer<Float>, count: Int) throws {
        for row in 0..<count {
            let embd = llama_embd(data: rows + row * rowSize, n_rows: 1, n_embd: rowSize)   // copied by the batch
            try place(llama_batch_ext_add_embd(batch, sequences - 1, embd))
        }
    }

    private func place(_ index: Int32) throws {
        guard index >= 0 else { throw Failure(description: "batch full (\(index))") }
        var pos: [llama_pos] = [position, 0, 0, 0]
        guard llama_batch_ext_set_pos(batch, index, &pos), llama_batch_ext_set_output_embd(batch, index, true) else {
            throw Failure(description: "can't place token \(index)")
        }
        position += 1
        tokens += 1
    }

    private func run() throws -> [[Float]] {
        defer { reset() }
        let status = llama_process(context, LLAMA_PROCESS_TYPE_ENCODE, batch)
        guard status == 0 else { throw Failure(description: "llama_process failed (\(status))") }
        return try (0..<sequences).map { sequence in
            guard let pooled = llama_get_embeddings_seq(context, sequence) else {
                throw Failure(description: "no pooled embedding for input \(sequence)")
            }
            var sum = 0.0
            for i in 0..<dimension { sum += Double(pooled[i]) * Double(pooled[i]) }
            let scale = sum > 0 ? Float(1 / sum.squareRoot()) : 0
            return (0..<dimension).map { pooled[$0] * scale }
        }
    }

    private static var backendReady = false

    /// llama.cpp's process-wide setup, once. Its log is noisy at load time: only warnings and errors reach stderr.
    private static func setUpBackend() {
        guard !backendReady else { return }
        backendReady = true
        llama_log_set({ level, text, _ in
            guard let text else { return }
            if level != GGML_LOG_LEVEL_CONT { LogLevel.current = level }
            guard LogLevel.current.rawValue >= GGML_LOG_LEVEL_WARN.rawValue else { return }
            FileHandle.standardError.write(Data(String(cString: text).utf8))
        }, nil)
        llama_backend_init()
    }

    /// CONT lines continue the previous message at its level.
    private enum LogLevel {
        static var current = GGML_LOG_LEVEL_INFO
    }
}
