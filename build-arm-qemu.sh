set -e

SRC=/mnt/c/Users/Jack/Documents/luau-fork
BUILD="$HOME/luau-arm-build"

cmake -S "$SRC" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_SYSTEM_NAME=Linux \
  -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
  -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++

cmake --build "$BUILD" --target Luau.Repl.CLI -j"$(nproc)"

QEMU="qemu-aarch64 -L /usr/aarch64-linux-gnu"
$QEMU "$BUILD/luau"            "$SRC/simd_validate.luau"
$QEMU "$BUILD/luau" --codegen  "$SRC/simd_validate.luau"
echo "Both must print: CHECKSUM 1165363688  (matches x64 interp+native)"
