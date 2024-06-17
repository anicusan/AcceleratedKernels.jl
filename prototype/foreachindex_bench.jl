
using Random
using BenchmarkTools
using Profile
using PProf

using KernelAbstractions
using oneAPI

import AcceleratedKernels as AK


Random.seed!(0)


function akcopyto!(x, scheduler)
    v = similar(x)
    AK.foreachindex(x, scheduler=scheduler) do i
        @inbounds v[i] = x[i] * 2 + 1
    end
end


x = (ones(Int32, 100_000))

println("AcceleratedKernels foreachindex :polyester copy:")
display(@benchmark(akcopyto!(x, :polyester)))

println("AcceleratedKernels foreachindex :threads copy:")
display(@benchmark(akcopyto!(x, :threads)))

println("Base copyto!:")
arange = Array(1:length(x))
display(@benchmark(copyto!(x, arange)))

