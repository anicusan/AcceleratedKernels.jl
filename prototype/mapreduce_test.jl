
using Random
using BenchmarkTools
using Profile
using PProf

using KernelAbstractions
using oneAPI

import AcceleratedKernels as AK


Random.seed!(0)


function redmin(s)
    d = AK.mapreduce(
        abs,
        (x, y) -> x < y ? x : y,
        s;
        init=typemax(eltype(s)),
        block_size=256,
        switch_below=10_000,
    )
end


s = oneArray(shuffle(-100:1_000_000))
d = redmin(s)
@assert d == 0
println("Simple correctness check passed")

println("AcceleratedKernels minimum:")
display(@benchmark(redmin(s)))

println("oneAPI minimum:")
display(@benchmark(
    mapreduce(
        abs,
        (x, y) -> x < y ? x : y,
        s; init=typemax(eltype(s)),
    )
))

