set -e
cmake -B build-arm -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-arm --target Luau.Repl.CLI -j"$(nproc)"

./build-arm/luau           simd_validate.luau
./build-arm/luau --codegen simd_validate.luau
echo "Both must print: CHECKSUM 1165363688  (matches x64 interp+native)"
