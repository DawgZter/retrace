#!/bin/bash
# Builds llama.cpp (MIT) for Retrace's semantic search helper (RetraceEmbeddingHelper).
#
# Pins one llama.cpp build tag and commit, applies scripts/llama-embedding-no-logits.patch, and leaves a static
# library and headers in Vendors/llamacpp (not tracked by git):
#
#   Vendors/llamacpp/lib/libllamacpp.a
#   Vendors/llamacpp/include/llama.h, mtmd.h (+ ggml headers)
#   Vendors/llamacpp/BUILD_STAMP      the tag, patch hash and deployment target this build was made from
#
# Package.swift adds the helper target only when that library exists, so `swift build` works without running this:
# the app then builds without semantic search.
#
# The approach (pinned tag, the no-logits patch, embedded Metal shaders) follows DigUp's scripts/build-llama.sh
# (https://github.com/ARahim3/DigUp, MIT).
#
# Usage: scripts/build_llama_embedding.sh        (needs git, cmake and Xcode; takes a few minutes the first time)
set -euo pipefail
cd "$(dirname "$0")/.."

TAG=b11461   # EmbeddingGemma 2 needs b11452 or later
COMMIT=4d756bc72bf00a4aacf410ae15a2d315f3db400d   # TAG's commit: a tag can be moved, a commit can't
DEPLOYMENT_TARGET=13.0
PATCH="$PWD/scripts/llama-embedding-no-logits.patch"
SOURCE="${LLAMA_SOURCE:-$PWD/Vendors/llamacpp/source}"
BUILD="${LLAMA_BUILD:-$PWD/Vendors/llamacpp/cmake-build}"
OUT="$PWD/Vendors/llamacpp"

STAMP="$TAG $(shasum -a 256 "$PATCH" | cut -c1-12) macos$DEPLOYMENT_TARGET mtmd"
if [[ -f "$OUT/lib/libllamacpp.a" && "$(cat "$OUT/BUILD_STAMP" 2>/dev/null)" == "$STAMP" ]]; then
  echo "llama.cpp $TAG is already built in Vendors/llamacpp"
  exit 0
fi

if ! command -v cmake > /dev/null; then
  echo "cmake not found: install it (brew install cmake) and run this again" >&2
  exit 1
fi
if [[ -z "${DEVELOPER_DIR:-}" && -d /Applications/Xcode.app ]]; then
  export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
fi

if [[ ! -d "$SOURCE" ]]; then
  git -c advice.detachedHead=false clone --quiet --depth 1 --branch "$TAG" \
    https://github.com/ggml-org/llama.cpp "$SOURCE"
elif [[ "$(git -C "$SOURCE" describe --tags --exact-match 2>/dev/null)" != "$TAG" ]]; then
  git -C "$SOURCE" fetch --quiet --depth 1 origin tag "$TAG"
  git -C "$SOURCE" checkout --quiet --force "$TAG"
fi
if [[ "$(git -C "$SOURCE" rev-parse HEAD)" != "$COMMIT" ]]; then
  echo "llama.cpp's $TAG isn't commit $COMMIT here: not building it" >&2
  exit 1
fi

# Without the patch, an embedding context reserves and clears a vocabulary's worth of logits for every input token
# (1 MiB per token with Gemma's 262k vocabulary), which an embedding model never produces.
if git -C "$SOURCE" apply --check "$PATCH" 2>/dev/null; then
  git -C "$SOURCE" apply "$PATCH"
elif ! git -C "$SOURCE" apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "$PATCH doesn't apply to $SOURCE, and isn't applied already: not building it" >&2
  exit 1
fi

GENERATOR=$(command -v ninja > /dev/null && echo Ninja || echo "Unix Makefiles")
# Only the libraries: llama plus mtmd (the image encoder), no tools, server or anything downloaded at build time.
# No BLAS backend: Accelerate's new cblas entry points need macOS 13.3, and Metal does the work anyway.
cmake -S "$SOURCE" -B "$BUILD" -G "$GENERATOR" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="$DEPLOYMENT_TARGET" -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DBUILD_SHARED_LIBS=OFF -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_NATIVE=OFF -DGGML_BLAS=OFF \
  -DLLAMA_BUILD_MTMD=ON -DLLAMA_BUILD_COMMON=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_UI=OFF \
  -DLLAMA_OPENSSL=OFF -DLLAMA_CURL=OFF > /dev/null
cmake --build "$BUILD" --config Release -j "$(sysctl -n hw.ncpu)" --target llama mtmd > /dev/null

mkdir -p "$OUT/lib" "$OUT/include"
LIBS=()
while IFS= read -r lib; do LIBS+=("$lib"); done < <(find "$BUILD" -name '*.a' \
  \( -name libllama.a -o -name libmtmd.a -o -name 'libggml*.a' -o -name 'libvendor-hash.a' \))
xcrun libtool -static -no_warning_for_no_symbols -o "$OUT/lib/libllamacpp.a" "${LIBS[@]}"
cp "$SOURCE/include/llama.h" "$SOURCE/tools/mtmd/mtmd.h" "$SOURCE"/ggml/include/*.h "$OUT/include/"
echo "$STAMP" > "$OUT/BUILD_STAMP"
echo "Built llama.cpp $TAG (patched, Metal, macOS $DEPLOYMENT_TARGET+) into Vendors/llamacpp"
