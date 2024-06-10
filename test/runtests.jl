using AcceleratedKernels
using Test
using Random
using CUDA


@testset "merge_sort" begin
    Random.seed!(0)

    # Fuzzy testing
    for _ in 1:10_000
        num_elems = rand(1:100_000)
        v = CuArray(rand(Int32, num_elems))
        merge_sort!(v)
        vh = Array(v)
        @assert issorted(vh)
    end

    for _ in 1:10_000
        num_elems = rand(1:100_000)
        v = CuArray(rand(UInt32, num_elems))
        merge_sort!(v)
        vh = Array(v)
        @assert issorted(vh)
    end

    for _ in 1:10_000
        num_elems = rand(1:100_000)
        v = CuArray(rand(Float32, num_elems))
        merge_sort!(v)
        vh = Array(v)
        @assert issorted(vh)
    end
end
