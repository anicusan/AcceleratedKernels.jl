
using Random
using BenchmarkTools
using Profile
using PProf

using KernelAbstractions
using oneAPI

import AcceleratedKernels as AK


Random.seed!(0)


function akcopyto!(x)
    AK.foreachindex(x) do i
        @inbounds x[i] = i
    end
end


x = oneArray(ones(Int32, 100_000))
akcopyto!(x)

xh = Array(x)
@assert all(xh .== 1:length(xh))
println("Simple correctness check passed")

println("AcceleratedKernels foreachindex copy:")
display(@benchmark(akcopyto!(x)))

println("oneAPI copyto!:")
arange = Array(1:length(x))
display(@benchmark(copyto!(x, arange)))

