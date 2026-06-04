--!native
--!optimize 2

-- computes out = (a ^ b) + c, lanewise, entirely through register-resident simd values
local function kernel(out: buffer, a: buffer, b: buffer, c: buffer)
	local va = buffer.readu32x4(a, 0)
	local vb = buffer.readu32x4(b, 0)
	local vc = buffer.readu32x4(c, 0)
	local r = simd.add(simd.bxor(va, vb), vc)
	buffer.writeu32x4(out, 0, r)
end

local function also(out: buffer, a: buffer, b: buffer)
	-- exercise and/or/sub/not too
	local va = buffer.readu32x4(a, 0)
	local vb = buffer.readu32x4(b, 0)
	local r = simd.bor(simd.band(va, vb), simd.bnot(simd.sub(va, vb)))
	buffer.writeu32x4(out, 0, r)
end

local function u32(x) return x % 4294967296 end

local a, b, c = buffer.create(16), buffer.create(16), buffer.create(16)
local va, vb, vc = {}, {}, {}
for i = 0, 3 do
	va[i] = u32((i * 2654435761) + 7)
	vb[i] = u32((i * 40503) * 131 + 99)
	vc[i] = u32(i * 1013904223 + 12345)
	buffer.writeu32(a, i * 4, va[i])
	buffer.writeu32(b, i * 4, vb[i])
	buffer.writeu32(c, i * 4, vc[i])
end

local out = buffer.create(16)
kernel(out, a, b, c)

local pass = true
for i = 0, 3 do
	local want = u32(bit32.bxor(va[i], vb[i]) + vc[i])
	local got = buffer.readu32(out, i * 4)
	if got ~= want then
		pass = false
		print(string.format("kernel lane %d: got %X want %X", i, got, want))
	end
end

local out2 = buffer.create(16)
also(out2, a, b)
for i = 0, 3 do
	local want = bit32.bor(bit32.band(va[i], vb[i]), bit32.band(bit32.bnot(u32(va[i] - vb[i])), 0xFFFFFFFF))
	local got = buffer.readu32(out2, i * 4)
	if got ~= want then
		pass = false
		print(string.format("also lane %d: got %X want %X", i, got, want))
	end
end

print(pass and "native simd: PASS" or "native simd: FAIL")
