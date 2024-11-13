
# using Random
# using BenchmarkTools
# using Profile
# using PProf

# using KernelAbstractions
# using Metal

# import AcceleratedKernels as AK


# Random.seed!(0)




# s = MtlArray(rand(Int32(1):Int32(100), 3, 1000))
# d = AK.reduce(+, s; init=zero(eltype(s)), dims=3)
# KernelAbstractions.synchronize(get_backend(s))
# d


using pocl_jll
using OpenCL

using KernelAbstractions
import AcceleratedKernels as AK

using BenchmarkTools
using Random
Random.seed!(0)


function sum_base(s; dims)
    d = reduce(+, s; init=zero(eltype(s)), dims=dims)
    KernelAbstractions.synchronize(get_backend(s))
    d
end


function sum_ak(s; dims)
    d = AK.reduce(+, s; init=zero(eltype(s)), dims=dims, block_size=16)
    KernelAbstractions.synchronize(get_backend(s))
    d
end


# Make array with highly unequal per-axis sizes
sh = rand(Int32(-100):Int32(100), 10, 100_000)
s = CLArray(sh)


# Correctness
@assert sum_base(sh, dims=1) == Array(sum_ak(s, dims=1))
@assert sum_base(sh, dims=2) == Array(sum_ak(s, dims=2))

# Benchmarks
println("\nReduction over small axis - AK vs Base")
display(@benchmark sum_ak($s, dims=1))
display(@benchmark sum_base($sh, dims=1))

println("\nReduction over long axis - AK vs Base")
display(@benchmark sum_ak($s, dims=2))
display(@benchmark sum_base($sh, dims=2))





# function redmin(s)
#     d = AK.reduce(
#         (x, y) -> x < y ? x : y,
#         s;
#         init=typemax(eltype(s)),
#         block_size=256,
#         switch_below=0,
#     )
# end
# 
# 
# s = CuArray(shuffle(1:1_000_000))
# d = redmin(s)
# @assert d == 1
# println("Simple correctness check passed")
# 
# println("AcceleratedKernels minimum:")
# display(@benchmark(redmin(s)))
# 
# println("oneAPI minimum:")
# display(@benchmark(minimum(s)))

