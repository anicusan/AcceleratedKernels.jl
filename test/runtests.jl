using AcceleratedKernels
using Test
using Random
using CUDA


@testset "merge_sort" begin
    # Write your tests here.
    Random.seed!(0)

    # Fuzzy testing
    for _ in 1:1000
        num_elems = rand(1:100_000)
        v = CuArray(rand(Int32, num_elems))
        merge_sort!(v)
        vh = Array(v)
        @assert issorted(vh)
    end

    for _ in 1:1000
        num_elems = rand(1:100_000)
        v = CuArray(rand(UInt32, num_elems))
        merge_sort!(v)
        vh = Array(v)
        @assert issorted(vh)
    end

    for _ in 1:1000
        num_elems = rand(1:100_000)
        v = CuArray(rand(Float32, num_elems))
        merge_sort!(v)
        vh = Array(v)
        @assert issorted(vh)
    end
end
