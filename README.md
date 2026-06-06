# SIMD Value Types

## Summary

Four builtin value types that each hold a short fixed width vector of 32 bit lanes, exposed through the `u32x4`, `f32x4`, `u32x8`, and `f32x8` libraries. One value packs four lanes (128 bits) or eight lanes (256 bits) and is operated on as a whole. The compiler keeps these values in vector registers and lowers each operation to a single native SIMD instruction on x64 (SSE2 and AVX2) and on ARM (NEON), so data parallel loops run many times faster than the equivalent scalar code.

## Motivation

A large class of Luau workloads spends most of its time applying the same 32 bit operation across a long run of values. Cryptographic permutations mix arrays of 32 bit words with add, xor, and rotate. Voxel and terrain systems sweep density and noise fields. Pathfinding relaxes the cost of many grid cells at once. Physics integrates many particles per step. In every one of these the work is identical across neighbouring elements, which is exactly the shape that SIMD hardware was built to accelerate.

Today this work is written one number at a time. Even with the native code generator, a scalar loop issues one arithmetic instruction per element and pays the loop overhead on every iteration. The hardware on every platform Luau targets can apply the same operation to four or eight 32 bit lanes in a single instruction, but Luau has had no value that maps onto a vector register, so that throughput was unreachable from script.

Userland workarounds exist. Packing lanes into the existing `vector` type, into pairs of numbers, or into a buffer that is indexed by hand all recover some of the structure but none of the speed. Each operation still decomposes into scalar work, each intermediate still risks a heap allocation, and the type system has no idea any of it represents a vector. The result is code that is harder to read and still slower than the machine allows.

A first class vector value addresses all three problems at once. It fits in a register, it is recognised by the compiler so every operation becomes one instruction, and it gives a clear vocabulary for writing data parallel code.

## Design

There is one underlying value. It carries either four 32 bit lanes (128 bits) or eight 32 bit lanes (256 bits) of raw data. The four libraries are four views over that data:

* `u32x4` and `u32x8` interpret the lanes as unsigned 32 bit integers.
* `f32x4` and `f32x8` interpret the lanes as 32 bit floats.

The element type lives in the library name rather than in the runtime value, in the same spirit as the WebAssembly `v128` type. This keeps the value small and keeps the native type guards simple, because there is a single runtime tag to check rather than one per element type and width. `type` and `typeof` both return the string `"simd"` for every vector value, since the element type is a matter of interpretation rather than identity.

The width, by contrast, is visible to the type checker. There are two static types, `simd` for the 128 bit libraries `u32x4` and `f32x4`, and `simd256` for the 256 bit libraries `u32x8` and `f32x8`. The two are distinct, so the checker reports an error if a value of one width is passed to an operation of the other, even though both share the one runtime tag. The element type within a width stays unchecked on purpose, so `u32x4` and `f32x4` share `simd`, which is what makes the free reinterpretation between integer and float views typecheck.

A direct consequence of the shared representation is that reinterpreting the bits of a value costs nothing. Viewing the integer result of a mix step as floats means calling the operation from the float library on the same value, with no conversion instruction and no copy. There is no separate reinterpret function because none is required. For example, `f32x4.add` applied to a value built with `u32x4.splat(0x3f800000)`, the bit pattern of `1.0`, produces the bit pattern of `2.0`.

Reinterpreting is distinct from converting. `u32x4.tofloat` performs a real numeric conversion of each integer lane to the nearest float, and `f32x4.toint` performs a real truncating conversion in the other direction. These emit hardware convert instructions. `u32x4.tofloat` applied to `u32x4.splat(5)` yields the float `5.0`, not the value `5` reinterpreted as a float.

Values are register resident. In a straight line or unrolled body the compiler holds the value in a vector register across every operation and never allocates. A value only becomes a 32 byte heap object when it must outlive the register file, for example when it is captured by a closure or stored where the garbage collector can reach it. A vector that is carried across a loop back edge, such as an accumulator or a hoisted constant, keeps a single box for the whole loop and reuses it in place, so it does not allocate per iteration; the cost that remains is storing the lanes to that box and reloading them across the edge, the same round trip a loop carried number makes through its stack slot. A value that genuinely escapes on every iteration, for instance one stored into a table inside the loop, still allocates each time. The fastest kernels keep the hot section unrolled so the carried value stays purely in a register, which is how high performance cryptography is written in any case.

Two values are equal when their lane bits are identical. Equality, `rawequal`, and table key lookup all compare the full lane payload, so a vector computed twice from the same inputs is equal to its twin even though it is a separate heap object, the way `number` and the existing `vector` type behave rather than comparing by reference. The comparison is bitwise, which only matters at the edges: in float lanes a negative zero and a positive zero are not equal because their bits differ, and two NaN lanes are equal when their bits match. Since the value carries no element type, a bitwise compare is the only definition that is well formed for both the integer and the float view.

Lanes are indexed from zero. For a 128 bit value the lanes are `0` through `3`, and for a 256 bit value they are `0` through `7`.

`tostring` and `print` render a value as `simd(...)` listing its eight lanes as unsigned integers, so a value can be inspected directly rather than showing an address. Since the value carries no width, a 128 bit value shows the zeros it stores in its high four lanes.

## Library

Unless noted, every operation takes whole vectors and returns a whole vector, working lane by lane with no interaction between lanes. The 128 bit and 256 bit libraries mirror each other.

### Integer lanes: `u32x4` and `u32x8`

```
function u32x4.create(a: number, b: number, c: number, d: number): simd
function u32x8.create(a, b, c, d, e, f, g, h: number): simd
```

Builds a value from individual lane values, taken as unsigned 32 bit integers.

```
function u32x4.splat(n: number): simd
```

Builds a value with every lane set to `n`.

```
function u32x4.extract(v: simd, i: number): number
```

Returns lane `i` as an unsigned 32 bit integer, with `i` in `0 .. 3` for `u32x4` and `0 .. 7` for `u32x8`.

```
function u32x4.unpack(v: simd): (number, number, number, number)
function u32x8.unpack(v: simd256): (number, number, number, number, number, number, number, number)
```

Returns every lane at once as unsigned 32 bit integers, four from a 128 bit value and eight from a 256 bit value. This is the inverse of `create` and avoids calling `extract` once per lane.

```
function u32x4.add(a: simd, b: simd): simd
function u32x4.sub(a: simd, b: simd): simd
function u32x4.mul(a: simd, b: simd): simd
```

Lane wise addition, subtraction, and multiplication of 32 bit integers, wrapping to the low 32 bits on overflow. All three are provided at both widths.

```
function u32x4.band(a: simd, b: simd): simd
function u32x4.bor(a: simd, b: simd): simd
function u32x4.bxor(a: simd, b: simd): simd
function u32x4.bnot(a: simd): simd
```

Bitwise and, or, exclusive or, and complement across all lanes.

```
function u32x4.shl(v: simd, count: number): simd
function u32x4.shr(v: simd, count: number): simd
function u32x4.rotl(v: simd, count: number): simd
```

Per lane left shift, logical right shift, and left rotate of each 32 bit lane. A `count` that is a literal in `0 .. 31` lowers to a single instruction. Any other count still works: a shift count outside `0 .. 31` shifts every bit out and gives zero, the way the hardware shift and `bit32` behave, and a rotate count is taken modulo 32, so every integer is a valid cyclic rotation.

```
function u32x4.shuffle(v: simd, control: number): simd
```

Permutes the lanes within each 128 bit block according to a compile time `control` byte that selects, for each of the four destination lanes, which source lane to copy. For a 256 bit value the same four lane control is applied to each 128 bit half independently.

```
function u32x4.tofloat(v: simd): simd
```

Converts each integer lane to the nearest 32 bit float. The result is used through the matching float library, `f32x4` or `f32x8`.

```
function u32x4.eq(a: simd, b: simd): simd
function u32x4.lt(a: simd, b: simd): simd
function u32x4.gt(a: simd, b: simd): simd
```

Lane wise comparison as unsigned 32 bit integers. Each lane of the result is all ones (`0xFFFFFFFF`) where the comparison holds and all zeros where it does not, which is the mask shape that `select` consumes. `eq` tests equality, `lt` and `gt` test unsigned ordering.

```
function u32x4.select(mask: simd, a: simd, b: simd): simd
```

Lane wise blend: each result lane takes the bits of `a` where the corresponding `mask` lane is set and the bits of `b` where it is clear. The mask is normally produced by a comparison, so `select(u32x4.lt(a, b), a, b)` is a lane wise minimum, but the blend is purely bitwise and works with any mask.

```
function u32x4.sum(v: simd): number
function u32x4.hmin(v: simd): number
function u32x4.hmax(v: simd): number
function u32x4.hband(v: simd): number
function u32x4.hbor(v: simd): number
function u32x4.hbxor(v: simd): number
```

Horizontal reductions: these collapse all lanes of one value to a single `number` rather than returning a vector. `sum` adds the lanes, wrapping to 32 bits; `hmin` and `hmax` take the unsigned minimum and maximum; `hband`, `hbor`, and `hbxor` fold the lanes with bitwise and, or, and exclusive or. The `h` prefix on the bitwise and ordering reductions keeps them distinct from the lane wise `band` and `min` style operations. A reduction is most useful at the end of a vectorized loop, for example `u32x4.hbxor` to finish a checksum or `u32x4.hbor(u32x4.eq(a, b)) ~= 0` to test whether any lane matched. All six are provided at both widths.

### Float lanes: `f32x4` and `f32x8`

```
function f32x4.create(a: number, b: number, c: number, d: number): simd
function f32x8.create(a, b, c, d, e, f, g, h: number): simd
function f32x4.splat(n: number): simd
function f32x4.extract(v: simd, i: number): number
```

The float counterparts of the integer constructors and lane accessor. `extract` returns the lane as a float.

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

Lane wise float arithmetic, minimum, maximum, square root, and a multiply add computing `a * b + c` that uses a fused instruction on hardware that provides one. `min` and `max` are defined as a lane wise compare and select that returns the smaller or larger value of each lane pair, with the same behaviour on every platform including how an unordered comparison against NaN resolves.

```
function f32x4.toint(v: simd): simd
```

Converts each float lane to a 32 bit integer by truncation toward zero, with saturation: a lane outside the signed 32 bit range clamps to the nearest end, and a NaN lane becomes zero, identically on every platform. The result is used through the matching integer library.

```
function f32x4.unpack(v: simd): (number, number, number, number)
function f32x8.unpack(v: simd256): (number, number, number, number, number, number, number, number)
function f32x4.eq(a: simd, b: simd): simd
function f32x4.lt(a: simd, b: simd): simd
function f32x4.gt(a: simd, b: simd): simd
function f32x4.select(mask: simd, a: simd, b: simd): simd
function f32x4.sum(v: simd): number
function f32x4.hmin(v: simd): number
function f32x4.hmax(v: simd): number
```

The float counterparts of the integer lane accessor, comparisons, blend, and reductions. `unpack` returns the lanes as floats. The comparisons are ordered float compares that produce the same all ones or all zeros mask, and `select` blends bitwise, exactly as in the integer library. `sum` adds the lanes in order as 32 bit floats, and `hmin` and `hmax` take the floating point minimum and maximum across the lanes. The integer only `hband`, `hbor`, and `hbxor` have no float form.

### Buffers

The `buffer` library reads and writes whole vectors to contiguous bytes, so vectors stream to and from memory without going lane by lane.

```
function buffer.readu32x4(b: buffer, offset: number): simd
function buffer.writeu32x4(b: buffer, offset: number, value: simd): ()
function buffer.readu32x8(b: buffer, offset: number): simd256
function buffer.writeu32x8(b: buffer, offset: number, value: simd256): ()
```

A 128 bit access moves 16 bytes and a 256 bit access moves 32 bytes. The same bytes can be read back through either the integer or the float library, since the element type is chosen by the caller.

```
function buffer.transpose(target: buffer, targetOffset: number, source: buffer, sourceOffset: number, rows: number, columns: number): ()
function buffer.transposexor(target: buffer, targetOffset: number, data: buffer, dataOffset: number, source: buffer, sourceOffset: number, rows: number, columns: number): ()
```

`transpose` reads a `rows` by `columns` matrix of 32 bit words from `source` and writes its `columns` by `rows` transpose to `target`. On x64 with AVX2 it works in eight by eight register tiles and falls back to a scalar path otherwise. `transposexor` does the same transpose of `source` and exclusive ors it with the `data` matrix on the way out, so `target` becomes `transpose(source) ~ data` in one pass, which is the shape a transposed keystream or bitsliced layout wants when it is combined with a message block. All offsets must be multiples of four, and `target` must not overlap the inputs.

## Native code generation

Each operation maps to a hardware instruction rather than a library call. On x64 the 128 bit operations use SSE2, which is present on every 64 bit x86 processor, and the 256 bit operations use AVX2. When the processor does not support AVX2, the 256 bit operations run in the interpreter instead, decided from the processor's reported features at the time the native code is generated, so behaviour stays correct everywhere and only the speed differs. On ARM64 every operation uses NEON, which is always available. NEON has no 256 bit register, so a 256 bit value occupies a pair of 128 bit registers and a 256 bit operation becomes two 128 bit instructions.

Operations are lowered to native instructions when the script is compiled at optimization level 2. At level 2 an argument that is itself a call, such as `u32x8.bxor(v, u32x8.rotl(v, 7))`, still inlines to a single instruction. At level 1 that nested form falls back to a library call, which is correct but much slower, so performance sensitive code should be compiled at level 2. Shipped Roblox code is compiled at level 2.

### Measured throughput

The numbers below come from one kernel, a small mix of read, rotate, xor, and add over a one mebibyte buffer. They are a snapshot from a single machine and are meant to show the relationships between the paths rather than to be absolute guarantees.

On x64, with every path compiled at optimization level 2 so the comparison is fair:

| path | throughput |
| --- | --- |
| scalar with `bit32` | about 1.7 GB/s |
| `u32x4`, four lanes | about 6.5 GB/s |
| `u32x8`, eight lanes | about 12.3 GB/s |
| same kernel in the interpreter | about 0.1 GB/s |

The 128 bit path runs about four times the native scalar path, and the 256 bit path about seven times the native scalar path and about 1.9 times the 128 bit path, which is close to the doubling of width. Both run well over a hundred times the same kernel in the interpreter. The scalar baseline here is the native one, which the code generator already optimizes well, so these are gains over fast scalar code rather than over the interpreter. On ARM under emulation, where absolute numbers are distorted but ratios still hold, the 256 bit path keeps the same roughly 1.9 times relationship over the 128 bit path that it has on x64.

## Use cases

The common thread is processing many independent 32 bit values that all take the same operation, which is the structure of arrays layout rather than a single small vector. Luau already has the `vector` type for a single three component value, so these libraries are aimed at the many values at once case.

**Cryptography.** Stream ciphers and hashes such as ChaCha20, Blake3, and the SHA2 family are built almost entirely from 32 bit add, xor, and rotate over fixed size state. A `u32x8` value carries eight words of state and advances all of them per instruction, and the buffer reads and writes stream the message through without unpacking. This is the workload that drove the design, and an unrolled round function is allocation free.

**Terrain and voxels.** Density fields, noise generation, and run length or palette processing sweep large grids of 32 bit values. Eight samples of value noise or eight voxels of a column can be evaluated together, and the float library covers the arithmetic for smoothing and gradient steps.

**Pathfinding.** Grid based search and flow fields relax many cells at once. Because the operations work lane by lane, a single `min` updates a whole row of eight cell costs against the matching row of neighbour costs in one instruction, and integer lanes carry packed cell indices or flags alongside.

**Physics.** Particle systems, cloth, and broad phase work integrate or test many bodies that share the same update. Holding eight x coordinates, eight velocities, and eight masses as separate vectors lets a step advance eight bodies together, and the multiply add fits the integration math directly.

**Image and signal processing.** Convolutions, tone mapping, colour conversion, and resampling apply a fixed kernel across a buffer of pixels or samples, which maps cleanly onto lane wise float and integer math with buffer streaming.

**General bulk math.** Any loop that applies a fixed 32 bit operation across an array, for example checksums, packing and unpacking, comparisons, or reductions, gains from processing four or eight elements per iteration.

## Drawbacks

Operations are library functions rather than operators, so expressions are more verbose than scalar arithmetic and a dense kernel reads as a sequence of calls. This matches the choice made for other specialized numeric work in Luau and keeps the cost of each operation explicit.

The 256 bit path depends on AVX2 on x64. On a processor without it those operations run in the interpreter, so portable code that must be fast on every machine should prefer the 128 bit libraries, which rely only on SSE2 and NEON and are native everywhere Luau runs natively.

A vector carried across a loop back edge no longer allocates per iteration, but it does still round trip its lanes through its box on every iteration, the same way a loop carried number round trips through its stack slot. Keeping the hot section unrolled removes that round trip by holding the value purely in a register, which also tends to be the faster shape for other reasons.

At runtime `type` cannot distinguish element type or width, since every vector shares one tag, and the runtime does not stop you mixing the namespaces. The type checker is stronger: it uses the two distinct static types `simd` and `simd256`, so passing a 128 bit value to a 256 bit operation, or the reverse, is a type error rather than a silent treatment of the missing lanes. What it still cannot catch is an element mismatch within one width, because `u32x4` and `f32x4` deliberately share `simd` so that reading an integer vector through the float library is the intended free reinterpretation. That reinterpretation is indistinguishable from a mistake, so the choice of element view within a width remains the caller's responsibility, the same trade made by `v128` in WebAssembly.

Full speed depends on compiling at optimization level 2. At level 1 nested call arguments fall back to a slower path with no warning, which can surprise someone benchmarking an unoptimized build.

## Alternatives

**Do nothing.** The userland workarounds keep working, but they cannot reach the throughput of a register backed value, and they leave the type system blind to the vector structure. For the workloads above the gap is large enough to be the difference between viable and not.

**Operators.** Overloaded operators would shorten expressions, but they were rejected for the same reasons that bitwise heavy numeric work in Luau already favours functions. Polymorphic operators invite silent performance cliffs, much of this code uses bitwise operations that have no operator syntax in Luau in any case, and explicit functions let the caller choose the exact operation, for example a signed versus unsigned interpretation, without ambiguity.

**A single dynamic width type.** A value that knows its own width and element type at runtime would allow stronger checking, at the cost of a larger value, a runtime tag per kind, and more complex native guards. Keeping one representation with the interpretation supplied by the library name keeps the value small and the generated code simple, which is what makes the register resident, allocation free fast path possible.
