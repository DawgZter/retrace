# Third-Party Notices

Retrace's optional semantic search uses the third-party software listed below. Each licence requires its copyright and permission notice to accompany copies of the software, so they are reproduced here in full.

## llama.cpp

- **Project**: llama.cpp, https://github.com/ggml-org/llama.cpp
- **Licence**: MIT
- **Where it is used**: `scripts/build_llama_embedding.sh` clones llama.cpp at tag `b11461` (commit `4d756bc72bf00a4aacf410ae15a2d315f3db400d`), applies `scripts/llama-embedding-no-logits.patch`, and builds a static library into `Vendors/llamacpp`. That library is linked into the `RetraceEmbeddingHelper` executable. The llama.cpp source is not in this repository.
- llama.cpp's own tree contains further third-party code under its own licences (see its `vendor/` and `licenses/` directories at that tag). Those notices are not reproduced here.

```
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## DigUp

- **Project**: DigUp, https://github.com/ARahim3/DigUp
- **Licence**: MIT
- **Where it is used**:
  - `Sources/RetraceEmbeddingHelper/Embedder.swift` is ported from the text and image paths of DigUp's `LlamaEmbedder`.
  - `Search/Semantic/ImageSizing.swift` ports the image sizing rule from DigUp's `ImageSizing`.
  - `scripts/build_llama_embedding.sh` follows the approach of DigUp's `scripts/build-llama.sh`.
  - `scripts/llama-embedding-no-logits.patch` is DigUp's `scripts/llama.cpp-no-logits.patch`, unchanged.

```
MIT License

Copyright (c) 2026 Abdur Rahim

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## EmbeddingGemma 2 (model, not bundled)

- **Model**: EmbeddingGemma 2 by Google DeepMind, licensed under Apache 2.0
- Retrace does not bundle the model. When semantic search is turned on in Settings, the app downloads the 8-bit GGUF files (`embeddinggemma-2-Q8_0.gguf` and, for screenshots, `mmproj-embeddinggemma-2-Q8_0.gguf`) from Hugging Face (`ggml-org/embeddinggemma-2-GGUF`, at a pinned revision).
