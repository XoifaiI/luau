# SIMD Value Types

Four builtin libraries, `u32x4`, `f32x4`, `u32x8`, and `f32x8`, that operate on a fixed width vector of 32 bit lanes. A value holds 4 lanes (128 bits) or 8 lanes (256 bits) and is operated on as a whole. At optimization level 2 the compiler keeps these values in vector registers and lowers each operation to one native instruction: SSE2 or AVX2 on x64, NEON on ARM.

These libraries are for the structure of arrays case, many independent 32 bit values that all take the same operation. The existing `vector` type already covers a single three component value. Typical uses are cryptography (ChaCha20, Blake3, SHA2), terrain and voxel fields, pathfinding cost relaxation, particle physics, and image or signal processing.

## Model

There is one runtime value holding raw lane bits. The library name decides how those bits are read:

* `u32x4` and `u32x8` read the lanes as unsigned 32 bit integers.
* `f32x4` and `f32x8` read them as 32 bit floats.

`type` and `typeof` both return `"simd"` for any vector value, because the element type is interpretation, not identity. Reinterpreting between integer and float therefore costs nothing: you just call the operation from the other library on the same value, no conversion and no copy. For example `f32x4.add` on a value built with `u32x4.splat(0x3f800000)` (the bits of `1.0`) gives the bits of `2.0`.

Converting is different from reinterpreting. `u32x4.tofloat` converts each integer lane to the nearest float, and `f32x4.toint` truncates each float lane to an integer. Both emit real convert instructions, so `u32x4.tofloat(u32x4.splat(5))` is the float `5.0`.

The width is checked, the element type is not. The type checker has two types, `simd` for the 128 bit libraries and `simd256` for the 256 bit libraries, and reports an error if you pass one width to an operation of the other. Within a width `u32x4` and `f32x4` share `simd`, which is what lets the free reinterpretation typecheck. Mixing integer and float views of the same width is the caller's responsibility.

Two values compare equal when their lane bits are identical. Equality, `rawequal`, and table keys all compare the full payload, like `number` and `vector` rather than by reference. The compare is bitwise, so negative zero does not equal positive zero, and two NaN lanes are equal when their bits match.

Lanes are indexed from 0: `0` to `3` for a 128 bit value, `0` to `7` for a 256 bit value. `tostring` and `print` render a value as `simd(...)` with eight lanes shown as unsigned integers; a 128 bit value shows zeros in its high four lanes.

Values stay in a register in straight line or unrolled code and never allocate there. A value becomes a 32 byte heap object only when it must outlive the register file, for example when captured by a closure or stored where the collector can reach it. A vector carried across a loop back edge keeps one box for the whole loop and reuses it, so it does not allocate per iteration, but it does store and reload its lanes across the edge the same way a loop carried number round trips through its stack slot. Unrolling the hot section removes that round trip by keeping the value purely in a register.

## Library

Every operation works lane by lane with no interaction between lanes, and unless noted takes whole vectors and returns a whole vector. The 128 and 256 bit libraries mirror each other, so the signatures below show the `u32x4` and `f32x4` forms; the `u32x8` and `f32x8` forms take eight lanes and return `simd256`.

### Integer: `u32x4` and `u32x8`

```
function u32x4.create(a: number, b: number, c: number, d: number): simd
function u32x4.splat(n: number): simd
function u32x4.extract(v: simd, i: number): number
function u32x4.unpack(v: simd): (number, number, number, number)
```

`create` builds a value from individual lanes taken as unsigned 32 bit integers, `splat` sets every lane to `n`, `extract` returns one lane, and `unpack` returns every lane at once.

```
function u32x4.add(a: simd, b: simd): simd
function u32x4.sub(a: simd, b: simd): simd
function u32x4.mul(a: simd, b: simd): simd
```

Lane wise add, subtract, and multiply, wrapping to the low 32 bits on overflow.

```
function u32x4.band(a: simd, b: simd): simd
function u32x4.bor(a: simd, b: simd): simd
function u32x4.bxor(a: simd, b: simd): simd
function u32x4.bnot(a: simd): simd
```

Bitwise and, or, exclusive or, and complement.

```
function u32x4.shl(v: simd, count: number): simd
function u32x4.shr(v: simd, count: number): simd
function u32x4.rotl(v: simd, count: number): simd
```

Per lane left shift, logical right shift, and left rotate. A literal `count` in `0 .. 31` lowers to a single instruction. A shift count outside `0 .. 31` produces zero, the way the hardware and `bit32` behave, and a rotate count is taken modulo 32.

```
function u32x4.shuffle(v: simd, control: number): simd
function u32x4.tofloat(v: simd): simd
```

`shuffle` permutes the lanes within each 128 bit block by a compile time control byte that picks a source lane for each of the four destinations; on a 256 bit value the same control applies to each half. `tofloat` converts each integer lane to the nearest float, used through the matching float library.

```
function u32x4.eq(a: simd, b: simd): simd
function u32x4.lt(a: simd, b: simd): simd
function u32x4.gt(a: simd, b: simd): simd
function u32x4.select(mask: simd, a: simd, b: simd): simd
```

`eq`, `lt`, and `gt` compare lanes as unsigned integers and set each result lane to all ones where the test holds and all zeros where it does not. `select` blends bitwise, taking `a` where the `mask` lane is set and `b` where it is clear, so `select(u32x4.lt(a, b), a, b)` is a lane wise minimum.

```
function u32x4.sum(v: simd): number
function u32x4.hmin(v: simd): number
function u32x4.hmax(v: simd): number
function u32x4.hband(v: simd): number
function u32x4.hbor(v: simd): number
function u32x4.hbxor(v: simd): number
```

Horizontal reductions collapse all lanes to one `number`. `sum` adds (wrapping to 32 bits), `hmin` and `hmax` take the unsigned minimum and maximum, and `hband`, `hbor`, and `hbxor` fold with bitwise and, or, and exclusive or.

### Float: `f32x4` and `f32x8`

```
function f32x4.create(a: number, b: number, c: number, d: number): simd
function f32x4.splat(n: number): simd
function f32x4.extract(v: simd, i: number): number
function f32x4.unpack(v: simd): (number, number, number, number)
```

The float counterparts of the integer constructors and accessors. `extract` and `unpack` return floats.

```
function f32x4.add(a: simd, b: simd): simd
function f32x4.sub(a: simd, b: simd): simd
function f32x4.mul(a: simd, b: simd): simd
function f32x4.div(a: simd, b: simd): simd
function f32x4.min(a: simd, b: simd): simd
function f32x4.max(a: simd, b: simd): simd
function f32x4.sqrt(v: simd): simd
function f32x4.fma(a: simd, b: simd, c: simd): simd
```

Lane wise float arithmetic. `fma` computes `a * b + c` with one fused instruction where the hardware has one. `min` and `max` are a lane wise compare and select, with the same result on every platform including against NaN.

```
function f32x4.toint(v: simd): simd
function f32x4.eq(a: simd, b: simd): simd
function f32x4.lt(a: simd, b: simd): simd
function f32x4.gt(a: simd, b: simd): simd
function f32x4.select(mask: simd, a: simd, b: simd): simd
function f32x4.sum(v: simd): number
function f32x4.hmin(v: simd): number
function f32x4.hmax(v: simd): number
```

`toint` truncates each float lane toward zero to a 32 bit integer with saturation: a lane outside the signed range clamps to the nearest end and a NaN lane becomes zero, identically on every platform. The comparisons are ordered float compares producing the same all ones or all zeros mask, `select` blends bitwise, and `sum`, `hmin`, and `hmax` reduce across the lanes. There is no float `hband`, `hbor`, or `hbxor`.

### Buffers

```
function buffer.readu32x4(b: buffer, offset: number): simd
function buffer.writeu32x4(b: buffer, offset: number, value: simd): ()
function buffer.readu32x8(b: buffer, offset: number): simd256
function buffer.writeu32x8(b: buffer, offset: number, value: simd256): ()
```

A 128 bit access moves 16 bytes and a 256 bit access moves 32 bytes. The same bytes can be read back through either the integer or the float library.

```
function buffer.transpose(target: buffer, targetOffset: number, source: buffer, sourceOffset: number, rows: number, columns: number): ()
function buffer.transposexor(target: buffer, targetOffset: number, data: buffer, dataOffset: number, source: buffer, sourceOffset: number, rows: number, columns: number): ()
```

`transpose` writes the transpose of a `rows` by `columns` matrix of 32 bit words. On x64 with AVX2 it works in eight by eight register tiles, otherwise it uses a scalar path. `transposexor` does the same transpose and exclusive ors the result with the `data` matrix in one pass, so `target` becomes `transpose(source) ~ data`. All offsets must be multiples of four and `target` must not overlap the inputs.

## Native code generation

On x64 the 128 bit operations use SSE2, present on every 64 bit x86 processor, and the 256 bit operations use AVX2. When AVX2 is absent the 256 bit operations run in the interpreter instead, decided from the processor's reported features when native code is generated, so results stay correct everywhere and only the speed differs. On ARM64 everything uses NEON; since NEON has no 256 bit register, a 256 bit value occupies two 128 bit registers and a 256 bit operation becomes two 128 bit instructions.

Operations lower to native instructions at optimization level 2, where a nested call argument such as `u32x8.bxor(v, u32x8.rotl(v, 7))` still inlines to a single instruction. At level 1 that nested form falls back to a slower library call, so performance sensitive code should compile at level 2.

AVX2 has no packed integer rotate. `rotl` by a byte multiple (8, 16, 24) lowers to a single `vpshufb`, and any other count is synthesized as a shift, a shift, and an or. Choosing a byte aligned rotate where the algorithm allows it is faster.

### Measured throughput

From `bench_simd_vs_scalar.luau` on x64 with AVX2 at optimization level 2. These are one machine's numbers and show the relationships between paths, not absolute guarantees. Byte rates count 32 bit lanes times four bytes, so they measure data transformed rather than buffer traffic except where a row streams memory.

The first kernel is the intended shape: a fully unrolled, register resident mix of add, rotate, and xor over eight independent accumulators, with no buffer access and no loop, so nothing is gated by bounds checks or a loop back edge.

| path | throughput |
| --- | --- |
| scalar with `bit32` | about 2.1 GB/s |
| `u32x4` | about 32 GB/s |
| `u32x8` | about 64 GB/s |

The 256 bit path runs about 1.95 times the 128 bit path, close to the exact doubling of width. The scalar baseline carries the cost of working through Luau numbers, so it understates how well the native scalar code generator does on plain integer work. The ceiling for this mix is the operation itself, three vector instructions per step on a short dependency chain. A pure add chain on the same value reaches about 190 GB/s, which shows the limit is the kernel and not the code generator.

A hand written loop that streams whole vectors through memory one element at a time sustains about 7 GB/s, because each element pays a bounds check and writes its value back into a register slot before memory bandwidth becomes the limit. To reach memory bandwidth, use the bulk `buffer` operations, which run the whole buffer in one native call:

| bulk op over 4 MB | throughput |
| --- | --- |
| `buffer.bxor` | about 62 GB/s |
| `buffer.rotl` | about 75 GB/s |
| `buffer.fmul` | about 59 GB/s |
| `buffer.copy` | about 32 GB/s |

On ARM under emulation, where absolute numbers are distorted but ratios hold, the 256 bit path keeps the same roughly 1.9 times relationship over the 128 bit path.

## Notes

Operations are functions, not operators, so a dense kernel reads as a sequence of calls. This keeps the cost of each operation explicit.

Portable code that must be fast on every machine should prefer the 128 bit libraries, since the 256 bit path needs AVX2 on x64 while 128 bit relies only on SSE2 and NEON.
