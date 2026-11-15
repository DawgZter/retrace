#!/bin/bash
# Builds llama.cpp (MIT) for Retrace's semantic search: the library RetraceEmbeddingHelper links, and llama-server,
# which runs the result check's decision model.
#
# Pins one llama.cpp tag and commit and leaves these in Vendors/llamacpp (not tracked by git):
#
#   Vendors/llamacpp/lib/libllamacpp.a
#   Vendors/llamacpp/include/llama.h, mtmd.h (+ ggml headers)
#   Vendors/llamacpp/bin/llama-server
#   Vendors/llamacpp/BUILD_STAMP      the tag and deployment target this build was made from
#
# Package.swift adds the helper target only when that library exists, so `swift build` works without running this:
# the app then builds without semantic search. This script ends by clearing SwiftPM's manifest cache (see
# clear_swiftpm_manifest_cache below), so that the next `swift build` sees the library.
#
# The approach (pinned tag, embedded Metal shaders) follows DigUp's scripts/build-llama.sh
# (https://github.com/ARahim3/DigUp, MIT). DigUp also patches llama.cpp so that an embedding context does not
# reserve a vocabulary's worth of logits per token; llama.cpp does that itself from v1.0.0 on, so there is no patch.
#
# Usage: scripts/build_llama_embedding.sh        (needs git, cmake and Xcode; takes a few minutes the first time)
set -euo pipefail
cd "$(dirname "$0")/.."

TAG=v1.0.0   # EmbeddingGemma 2 needs b11452 or later; the vllm-d3 decision models need v1.0.0
COMMIT=b1b770d90d94cad9e9ea5b993bdbc383a522fb28   # TAG's commit: a tag can be moved, a commit can't
DEPLOYMENT_TARGET=13.0
SOURCE="${LLAMA_SOURCE:-$PWD/Vendors/llamacpp/source}"
BUILD="${LLAMA_BUILD:-$PWD/Vendors/llamacpp/cmake-build}"
OUT="$PWD/Vendors/llamacpp"

# SwiftPM caches each evaluated Package.swift in ~/Library/Caches/org.swift.swiftpm/manifests/manifest.db. The cache key
# is the manifest's text and the environment, not the files the manifest looks at, so a Package.swift evaluated before
# libllamacpp.a existed keeps reporting no RetraceEmbeddingHelper target, and `swift build` keeps skipping the helper,
# until that entry is gone. (`swift package --manifest-cache none describe` shows what a fresh evaluation gives.)
# SwiftPM has no command that drops one package's entry. `swift package purge-cache` removes this same database but
# also every cached dependency clone on the machine, so this removes only the database: SwiftPM recreates it and
# evaluates each manifest again, once, the next time it is needed.
# This runs when the library was already built too, so running the script again is the fix for a missing helper target.
clear_swiftpm_manifest_cache() {
  local db="$HOME/Library/Caches/org.swift.swiftpm/manifests/manifest.db"
  rm -f "$db" "$db-shm" "$db-wal"
}

STAMP="$TAG macos$DEPLOYMENT_TARGET mtmd server"
if [[ -f "$OUT/lib/libllamacpp.a" && -x "$OUT/bin/llama-server" && "$(cat "$OUT/BUILD_STAMP" 2>/dev/null)" == "$STAMP" ]]; then
  echo "llama.cpp $TAG is already built in Vendors/llamacpp"
  clear_swiftpm_manifest_cache
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

GENERATOR=$(command -v ninja > /dev/null && echo Ninja || echo "Unix Makefiles")
# The libraries (llama plus mtmd, the image encoder) and llama-server, without its web UI, its other tools or
# anything downloaded at build time. The server only ever listens on this Mac, so it is built without TLS.
# No BLAS backend: Accelerate's new cblas entry points need macOS 13.3, and Metal does the work anyway.
cmake -S "$SOURCE" -B "$BUILD" -G "$GENERATOR" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="$DEPLOYMENT_TARGET" -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DBUILD_SHARED_LIBS=OFF -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_NATIVE=OFF -DGGML_BLAS=OFF \
  -DLLAMA_BUILD_MTMD=ON -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_UI=OFF \
  -DLLAMA_OPENSSL=OFF -DLLAMA_CURL=OFF > /dev/null
cmake --build "$BUILD" --config Release -j "$(sysctl -n hw.ncpu)" --target llama mtmd llama-server > /dev/null

mkdir -p "$OUT/lib" "$OUT/include" "$OUT/bin"
cp "$BUILD/bin/llama-server" "$OUT/bin/llama-server"
LIBS=()
while IFS= read -r lib; do LIBS+=("$lib"); done < <(find "$BUILD" -name '*.a' \
  \( -name libllama.a -o -name libmtmd.a -o -name 'libggml*.a' -o -name 'libvendor-hash.a' \))
xcrun libtool -static -no_warning_for_no_symbols -o "$OUT/lib/libllamacpp.a" "${LIBS[@]}"
cp "$SOURCE/include/llama.h" "$SOURCE/tools/mtmd/mtmd.h" "$SOURCE"/ggml/include/*.h "$OUT/include/"
echo "$STAMP" > "$OUT/BUILD_STAMP"
clear_swiftpm_manifest_cache
# SwiftPM does not notice that the headers in Vendors/llamacpp changed, and a helper compiled against the old ones
# crashes on start when it is linked with the new library (the structs it passes have a different layout). Marking
# its sources as changed makes the next `swift build` compile them again.
touch Sources/RetraceEmbeddingHelper/*.swift
echo "Built llama.cpp $TAG (Metal, macOS $DEPLOYMENT_TARGET+) into Vendors/llamacpp"
