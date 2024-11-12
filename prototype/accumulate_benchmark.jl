using BenchmarkTools
using Metal
import AcceleratedKernels as AK

using Random
Random.seed!(0)


function akacc(v)
    va = AK.accumulate(+, v, init=zero(eltype(v)), block_size=512)
    Metal.synchronize()
    va
end


function baseacc(v)
    va = accumulate(+, v, init=zero(eltype(v)))
    Metal.synchronize()
    va
end


v = MtlArray(rand(1:100, 1_000_000))

# Correctness checks
va = akacc(v) |> Array
vb = baseacc(v) |> Array
# @assert va == vb

# Benchmarks
println("Base vs AK")
display(@benchmark baseacc($v))
display(@benchmark akacc($v))
