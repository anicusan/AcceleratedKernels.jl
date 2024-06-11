import AcceleratedKernels as AK
using Test
using Random
using CUDA


@testset "merge_sort" begin
    Random.seed!(0)

    # Fuzzy correctness testing
    for _ in 1:10_000
        num_elems = rand(1:100_000)
        v = CuArray(rand(Int32, num_elems))
        AK.merge_sort!(v)
        vh = Array(v)
        @assert issorted(vh)
    end

    for _ in 1:10_000
        num_elems = rand(1:100_000)
        v = CuArray(rand(UInt32, num_elems))
        AK.merge_sort!(v)
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

    # Testing different settings
    AK.merge_sort!(CuArray{Float32}(1:10_000))
    AK.merge_sort!(CuArray{Int32}(1:10_000))

    AK.merge_sort(CuArray{Float32}(1:10_000))
    AK.merge_sort(CuArray{Int32}(1:10_000))
end


@testset "merge_sort_by_key" begin
    Random.seed!(0)

    # Fuzzy correctness testing
    for _ in 1:10_000
        num_elems = rand(1:100_000)
        k = CuArray(rand(Int32, num_elems))
        v = copy(k) .- 1
        AK.merge_sort_by_key!(k, v)
        kh = Array(k)
        vh = Array(v)
        @assert issorted(kh)
        @assert issorted(vh)
    end

    for _ in 1:10_000
        num_elems = rand(1:100_000)
        k = CuArray(rand(UInt32, num_elems))
        v = copy(k) .- 1
        AK.merge_sort_by_key!(k, v)
        kh = Array(k)
        vh = Array(v)
        @assert issorted(kh)
        @assert issorted(vh)
    end

    for _ in 1:10_000
        num_elems = rand(1:100_000)
        k = CuArray(rand(Float32, num_elems))
        v = copy(k) .- 1
        AK.merge_sort_by_key!(k, v)
        kh = Array(k)
        vh = Array(v)
        @assert issorted(kh)
        @assert issorted(vh)
    end

    # Testing different settings
    AK.merge_sort_by_key!(CuArray{Float32}(1:10_000), CuArray{Int32}(1:10_000))
    AK.merge_sort_by_key!(CuArray{Int32}(1:10_000), CuArray{Float32}(1:10_000))

    AK.merge_sort_by_key(CuArray{Float32}(1:10_000), CuArray{Int32}(1:10_000))
    AK.merge_sort_by_key(CuArray{Int32}(1:10_000), CuArray{Float32}(1:10_000))
end


@testset "merge_sortperm" begin
    Random.seed!(0)

    # Fuzzy correctness testing
    for _ in 1:10_000
        num_elems = rand(1:100_000)
        ix = CuArray{Int32}(undef, num_elems)
        v = CuArray(rand(Int32, num_elems))
        AK.merge_sortperm!(ix, v)
        ixh = Array(ix)
        vh = Array(v)
        @assert issorted(vh[ixh])
    end

    for _ in 1:10_000
        num_elems = rand(1:100_000)
        ix = CuArray{Int32}(undef, num_elems)
        v = CuArray(rand(UInt32, num_elems))
        AK.merge_sortperm!(ix, v)
        ixh = Array(ix)
        vh = Array(v)
        @assert issorted(vh[ixh])
    end

    for _ in 1:10_000
        num_elems = rand(1:100_000)
        ix = CuArray{Int32}(undef, num_elems)
        v = CuArray(rand(Float32, num_elems))
        AK.merge_sortperm!(ix, v)
        ixh = Array(ix)
        vh = Array(v)
        @assert issorted(vh[ixh])
    end

    # Testing different settings
    AK.merge_sortperm(CuArray{Int32}(1:10_000), CuArray{Int32}(1:10_000))
    AK.merge_sortperm(CuArray{Int32}(1:10_000), CuArray{Float32}(1:10_000))
end


@testset "reduce" begin
    Random.seed!(0)

    function redmin(s)
        # Reduction-based minimum finder
        AK.reduce(
            (x, y) -> x < y ? x : y,
            s;
            init=typemax(eltype(s)),
        )
    end

    # Fuzzy correctness testing
    for _ in 1:10_000
        num_elems = rand(1:100_000)
        v = CuArray(rand(Int32, num_elems))
        s = redmin(v)
        vh = Array(v)
        @assert s == minimum(vh)
    end

    for _ in 1:10_000
        num_elems = rand(1:100_000)
        v = CuArray(rand(UInt32, num_elems))
        s = redmin(v)
        vh = Array(v)
        @assert s == minimum(vh)
    end

    for _ in 1:10_000
        num_elems = rand(1:100_000)
        v = CuArray(rand(Float32, num_elems))
        s = redmin(v)
        vh = Array(v)
        @assert s == minimum(vh)
    end

    function redsum(s)
        # Reduction-based summation
        AK.reduce(
            (x, y) -> x + y,
            s;
            init=zero(eltype(s)),
        )
    end

    # Fuzzy correctness testing
    for _ in 1:10_000
        num_elems = rand(1:100_000)
        v = CuArray(rand(Int32, num_elems))
        s = redsum(v)
        vh = Array(v)
        @assert s == sum(vh)
    end

    for _ in 1:10_000
        num_elems = rand(1:100_000)
        v = CuArray(rand(UInt32, num_elems))
        s = redsum(v)
        vh = Array(v)
        @assert s == sum(vh)
    end

    for _ in 1:10_000
        num_elems = rand(1:100_000)
        v = CuArray(rand(Float32, num_elems))
        s = redsum(v)
        vh = Array(v)
        @assert s == sum(vh)
    end
end

