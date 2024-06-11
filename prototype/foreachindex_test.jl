
using Random
using BenchmarkTools
using Profile
using PProf

using KernelAbstractions
using oneAPI

import AcceleratedKernels as AK


Random.seed!(0)


function akcopyto!(y, x)
    @assert length(x) == length(y)
    AK.foreachindex(x, max_tasks=1) do i
        @inbounds y[i] = x[i]
    end
end


x = ones(Int32, 100_000)
y = similar(x)
akcopyto!(y, x)

yh = Array(y)
@assert all(yh .== 1)
println("Simple correctness check passed")

println("AcceleratedKernels foreachindex copy:")
display(@benchmark(akcopyto!(y, x)))

println("oneAPI copyto!:")
display(@benchmark(copyto!(y, x)))

