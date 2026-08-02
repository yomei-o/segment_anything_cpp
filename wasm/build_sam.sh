#!/bin/sh
# Build the MobileSAM WASM module (needs emsdk active: `source /c/emsdk/emsdk_env.sh`).
set -e
cd "$(dirname "$0")"
emcc -O3 -std=c++20 -DNDEBUG -msimd128 -I../pure -I../pure/third_party sam_wasm.cpp \
  -sEXPORTED_FUNCTIONS=_fn_ready,_fn_encode,_fn_decode,_fn_scale,_fn_iou,_malloc,_free \
  -sEXPORTED_RUNTIME_METHODS=ccall,cwrap,HEAPU8,HEAPF32,FS \
  -sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=268435456 \
  -sMODULARIZE=1 -sEXPORT_NAME=createSam -sENVIRONMENT=web,node -o sam.js
echo "built: sam.js sam.wasm"
