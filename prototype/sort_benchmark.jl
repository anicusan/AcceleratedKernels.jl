using Random
using CUDA
using BUC       # Exposed CUDA Thrust functions, precompiled for common integers
import AcceleratedKernels as AK


Random.seed!(0)


# Generate random numbers
n = 10_000_000
d = CuArray{Int64}(undef, n);

# Benchmark
using BenchmarkTools

println("CPU Sort:")
display(@benchmark sort!(h) setup=(h=rand(eltype(d), n)))

println("CUDA.jl Sort:")
display(@benchmark sort!($d) setup=(rand!(d)))

println("AcceleratedKernels Sort:")
temp = similar(d)
display(@benchmark AK.merge_sort!($d, temp=temp, block_size=256) setup=(rand!(d)))

println("BUC / CUDA Thrust Sort:")
display(@benchmark buc_sort!($d) setup=(rand!(d)))

