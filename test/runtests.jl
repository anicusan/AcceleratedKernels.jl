import AcceleratedKernels as AK
using KernelAbstractions
using Test
using Random
import Pkg


# Pass command-line argument to test suite to install the right backend, e.g.
#   julia> import Pkg
#   julia> Pkg.test(test_args=["--oneAPI"])
if "--CUDA" in ARGS
    Pkg.add("CUDA")
    using CUDA
    const backend = CUDABackend()
elseif "--oneAPI" in ARGS
    Pkg.add("oneAPI")
    using oneAPI
    const backend = oneAPIBackend()
elseif "--AMDGPU" in ARGS
    Pkg.add("AMDGPU")
    using AMDGPU
    const backend = ROCBackend()
elseif "--Metal" in ARGS
    Pkg.add("Metal")
    using Metal
    const backend = MetalBackend()
else
    # Otherwise do CPU tests
    const backend = CPU()
end


function array_from_host(h_arr::AbstractArray, dtype=nothing)
    d_arr = KernelAbstractions.zeros(backend, isnothing(dtype) ? eltype(h_arr) : dtype, size(h_arr))
    copyto!(d_arr, h_arr isa Array ? h_arr : Array(h_arr))      # Allow unmaterialised types, e.g. ranges
    d_arr
end


@testset "Aqua" begin
    using Aqua
    Aqua.test_all(AK)
end


@testset "TaskPartitioner" begin
    # All tasks needed
    tp = AK.TaskPartitioner(10, 4, 1)
    @test tp.num_tasks == 4
    @test length(tp) == tp.num_tasks
    @test tp[1] === 1:3
    @test tp[2] === 4:6
    @test tp[3] === 7:8
    @test tp[4] === 9:10

    # Not all tasks needed
    tp = AK.TaskPartitioner(20, 6, 5)
    @test tp.num_tasks == 4
    @test length(tp) == tp.num_tasks
    @test tp[1] === 1:5
    @test tp[2] === 6:10
    @test tp[3] === 11:15
    @test tp[4] === 16:20
end


@testset "task_partition" begin
    Random.seed!(0)

    # Single-threaded
    x = zeros(Int, 1000)
    AK.task_partition(length(x), 1, 1) do irange
        for i in irange
            x[i] = i
        end
    end
    @test all(x .== 1:length(x))

    # Multi-threaded
    x = zeros(Int, 1000)
    AK.task_partition(length(x), 10, 1) do irange
        for i in irange
            x[i] = i
        end
    end
    @test all(x .== 1:length(x))
end


@testset "foreachindex" begin
    Random.seed!(0)

    # CPU
    if backend == CPU()
        x = zeros(Int, 1000)
        AK.foreachindex(x) do i
            x[i] = i
        end
        @test all(x .== 1:length(x))

        x = zeros(Int, 1000)
        AK.foreachindex(x, max_tasks=1, min_elems=1) do i
            x[i] = i
        end
        @test all(x .== 1:length(x))

        x = zeros(Int, 1000)
        AK.foreachindex(x, max_tasks=10, min_elems=1) do i
            x[i] = i
        end
        @test all(x .== 1:length(x))

        x = zeros(Int, 1000)
        AK.foreachindex(x, max_tasks=10, min_elems=10, scheduler=:threads) do i
            x[i] = i
        end
        @test all(x .== 1:length(x))

        x = zeros(Int, 1000)
        AK.foreachindex(x, max_tasks=10, min_elems=10, scheduler=:polyester) do i
            x[i] = i
        end
        @test all(x .== 1:length(x))

    # GPU
    else
        x = array_from_host(zeros(Int, 10_000))
        f1(x) = AK.foreachindex(x) do i     # This must be inside a function to have a known type!
            x[i] = i
        end
        f1(x)
        xh = Array(x)
        @test all(xh .== 1:length(xh))

        x = array_from_host(zeros(Int, 10_000))
        f2(x) = AK.foreachindex(x, block_size=64) do i
            x[i] = i
        end
        f2(x)
        xh = Array(x)
        @test all(xh .== 1:length(xh))
    end
end


@testset "map" begin
    Random.seed!(0)

    # CPU
    if backend == CPU()
        x = Array(1:1000)
        y = AK.map(x) do i
            i^2
        end
        @test y == map(i -> i^2, x)

        x = Array(1:1000)
        y = zeros(Int, 1000)
        AK.map!(y, x) do i
            i^2
        end
        @test y == map(i -> i^2, x)

        x = rand(Float32, 1000)
        y = AK.map(x, scheduler=:threads, max_tasks=2, min_elems=100) do i
            i > 0.5 ? i : 0
        end
        @test y == map(i -> i > 0.5 ? i : 0, x)

        x = rand(Float32, 1000)
        y = AK.map(x, scheduler=:polyester, max_tasks=4, min_elems=500) do i
            i > 0.5 ? i : 0
        end
        @test y == map(i -> i > 0.5 ? i : 0, x)

    # GPU
    else
        x = array_from_host(1:1000)
        y = AK.map(x) do i
            i^2
        end
        @test Array(y) == map(i -> i^2, 1:1000)

        x = array_from_host(1:1000)
        y = array_from_host(zeros(Int, 1000))
        AK.map!(y, x) do i
            i^2
        end
        @test Array(y) == map(i -> i^2, 1:1000)

        x = array_from_host(rand(Float32, 1000))
        y = AK.map(x, block_size=64) do i
            i > 0.5 ? i : 0
        end
        @test Array(y) == map(i -> i > 0.5 ? i : 0, Array(x))
    end
end


if backend != CPU()
@testset "merge_sort" begin
    Random.seed!(0)

    # Fuzzy correctness testing
    for _ in 1:1000
        num_elems = rand(1:100_000)
        v = array_from_host(rand(Int32, num_elems))
        AK.merge_sort!(v)
        vh = Array(v)
        @test issorted(vh)
    end

    for _ in 1:1000
        num_elems = rand(1:100_000)
        v = array_from_host(rand(UInt32, num_elems))
        AK.merge_sort!(v)
        vh = Array(v)
        @test issorted(vh)
    end

    for _ in 1:1000
        num_elems = rand(1:100_000)
        v = array_from_host(rand(Float32, num_elems))
        AK.merge_sort!(v)
        vh = Array(v)
        @test issorted(vh)
    end

    # Testing different settings
    v = array_from_host(1:10_000, Float32)
    AK.merge_sort!(v, lt=(>), by=abs, rev=true,
                block_size=64, temp=array_from_host(1:10_000, Float32))
    @test issorted(Array(v))

    v = array_from_host(1:10_000, Int32)
    AK.merge_sort!(v, lt=(>), rev=true,
                block_size=64, temp=array_from_host(1:10_000, Int32))
    @test issorted(Array(v))

    v = array_from_host(1:10_000, Float32)
    AK.merge_sort(v, lt=(>), by=abs, rev=true,
                block_size=64, temp=array_from_host(1:10_000, Float32))
    @test issorted(Array(v))

    v = array_from_host(1:10_000, Int32)
    AK.merge_sort(v, lt=(>), by=abs, rev=true,
                block_size=64, temp=array_from_host(1:10_000, Int32))
    @test issorted(Array(v))
end
end


@testset "sort" begin
    Random.seed!(0)

    # Fuzzy correctness testing
    for _ in 1:100
        num_elems = rand(1:100_000)
        v = array_from_host(rand(Int32, num_elems))
        AK.sort!(v)
        vh = Array(v)
        @test issorted(vh)
    end

    for _ in 1:100
        num_elems = rand(1:100_000)
        v = array_from_host(rand(UInt32, num_elems))
        AK.sort!(v)
        vh = Array(v)
        @test issorted(vh)
    end

    for _ in 1:100
        num_elems = rand(1:100_000)
        v = array_from_host(rand(Float32, num_elems))
        AK.sort!(v)
        vh = Array(v)
        @test issorted(vh)
    end

    # Testing different settings
    v = array_from_host(1:10_000, Float32)
    AK.sort!(v, lt=(>), by=abs, rev=true,
             block_size=64, temp=array_from_host(1:10_000, Float32))
    @test issorted(Array(v))

    v = array_from_host(1:10_000, Int32)
    AK.sort!(v, lt=(>), rev=true,
             block_size=64, temp=array_from_host(1:10_000, Int32))
    @test issorted(Array(v))

    v = array_from_host(1:10_000, Float32)
    AK.sort(v, lt=(>), by=abs, rev=true,
            block_size=64, temp=array_from_host(1:10_000, Float32))
    @test issorted(Array(v))

    v = array_from_host(1:10_000, Int32)
    AK.sort(v, lt=(>), by=abs, rev=true,
            block_size=64, temp=array_from_host(1:10_000, Int32))
    @test issorted(Array(v))
end


if backend != CPU()
@testset "merge_sort_by_key" begin
    Random.seed!(0)

    # Fuzzy correctness testing
    for _ in 1:1000
        num_elems = rand(1:100_000)
        k = array_from_host(rand(Int32, num_elems))
        v = copy(k) .- 1
        AK.merge_sort_by_key!(k, v)
        kh = Array(k)
        vh = Array(v)
        @test issorted(kh)
        @test issorted(vh)
    end

    for _ in 1:1000
        num_elems = rand(1:100_000)
        k = array_from_host(rand(UInt32, num_elems))
        v = copy(k) .- 1
        AK.merge_sort_by_key!(k, v)
        kh = Array(k)
        vh = Array(v)
        @test issorted(kh)
        @test issorted(vh)
    end

    for _ in 1:1000
        num_elems = rand(1:100_000)
        k = array_from_host(rand(Float32, num_elems))
        v = copy(k) .- 1
        AK.merge_sort_by_key!(k, v)
        kh = Array(k)
        vh = Array(v)
        @test issorted(kh)
        @test issorted(vh)
    end

    # Testing different settings
    k = array_from_host(1:10_000, Float32)
    v = array_from_host(1:10_000, Int32)
    AK.merge_sort_by_key!(k, v,
                        lt=(>), by=abs, rev=true,
                        block_size=64,
                        temp_keys=array_from_host(1:10_000, Float32),
                        temp_values=array_from_host(1:10_000, Int32))
    @test issorted(Array(k))
    @test issorted(Array(v))

    k = array_from_host(1:10_000, Int32)
    v = array_from_host(1:10_000, Float32)
    AK.merge_sort_by_key!(k, v,
                        lt=(>), by=abs, rev=true,
                        block_size=64,
                        temp_keys=array_from_host(1:10_000, Int32),
                        temp_values=array_from_host(1:10_000, Float32))
    @test issorted(Array(k))
    @test issorted(Array(v))

    k = array_from_host(1:10_000, Float32)
    v = array_from_host(1:10_000, Int32)
    AK.merge_sort_by_key(k, v,
                        lt=(>), by=abs, rev=true,
                        block_size=64,
                        temp_keys=array_from_host(1:10_000, Float32),
                        temp_values=array_from_host(1:10_000, Int32))
    @test issorted(Array(k))
    @test issorted(Array(v))

    k = array_from_host(1:10_000, Int32)
    v = array_from_host(1:10_000, Float32)
    AK.merge_sort_by_key(k, v,
                        lt=(>), by=abs, rev=true,
                        block_size=64,
                        temp_keys=array_from_host(1:10_000, Int32),
                        temp_values=array_from_host(1:10_000, Float32))
    @test issorted(Array(k))
    @test issorted(Array(v))
end
end


if backend != CPU()
@testset "merge_sortperm" begin
    Random.seed!(0)

    # Fuzzy correctness testing
    for _ in 1:1000
        num_elems = rand(1:100_000)
        ix = array_from_host(zeros(Int32, num_elems))
        v = array_from_host(rand(Int32, num_elems))
        AK.merge_sortperm!(ix, v)
        ixh = Array(ix)
        vh = Array(v)
        @test issorted(vh[ixh])
    end

    for _ in 1:1000
        num_elems = rand(1:100_000)
        ix = array_from_host(zeros(Int32, num_elems))
        v = array_from_host(rand(UInt32, num_elems))
        AK.merge_sortperm!(ix, v)
        ixh = Array(ix)
        vh = Array(v)
        @test issorted(vh[ixh])
    end

    for _ in 1:1000
        num_elems = rand(1:100_000)
        ix = array_from_host(zeros(Int32, num_elems))
        v = array_from_host(rand(Float32, num_elems))
        AK.merge_sortperm!(ix, v)
        ixh = Array(ix)
        vh = Array(v)
        @test issorted(vh[ixh])
    end

    # Testing different settings
    ix = array_from_host(1:10_000, Int32)
    v = array_from_host(1:10_000, Float32)
    AK.merge_sortperm!(ix,
                    v,
                    lt=(>), by=abs, rev=true,
                    inplace=true, block_size=64,
                    temp_ix=array_from_host(1:10_000, Int32),
                    temp_v=array_from_host(1:10_000, Float32))
    ixh = Array(ix)
    vh = Array(v)
    @test issorted(vh[ixh])

    v = array_from_host(1:10_000, Float32)
    ix = AK.merge_sortperm(v,
                        lt=(>), by=abs, rev=true,
                        inplace=true, block_size=64,
                        temp_ix=array_from_host(1:10_000, Int),
                        temp_v=array_from_host(1:10_000, Float32))
    ixh = Array(ix)
    vh = Array(v)
    @test issorted(vh[ixh])
end
end


if backend != CPU()
@testset "merge_sortperm_lowmem" begin
    Random.seed!(0)

    # Fuzzy correctness testing
    for _ in 1:1000
        num_elems = rand(1:100_000)
        ix = array_from_host(zeros(Int32, num_elems))
        v = array_from_host(rand(Int32, num_elems))
        AK.merge_sortperm_lowmem!(ix, v)
        ixh = Array(ix)
        vh = Array(v)
        @test issorted(vh[ixh])
    end

    for _ in 1:1000
        num_elems = rand(1:100_000)
        ix = array_from_host(zeros(Int32, num_elems))
        v = array_from_host(rand(UInt32, num_elems))
        AK.merge_sortperm_lowmem!(ix, v)
        ixh = Array(ix)
        vh = Array(v)
        @test issorted(vh[ixh])
    end

    for _ in 1:1000
        num_elems = rand(1:100_000)
        ix = array_from_host(zeros(Int32, num_elems))
        v = array_from_host(rand(Float32, num_elems))
        AK.merge_sortperm_lowmem!(ix, v)
        ixh = Array(ix)
        vh = Array(v)
        @test issorted(vh[ixh])
    end

    # Testing different settings
    ix = array_from_host(1:10_000, Int32)
    v = array_from_host(1:10_000, Float32)
    AK.merge_sortperm_lowmem!(ix,
                            v,
                            lt=(>), by=abs, rev=true,
                            block_size=64,
                            temp=array_from_host(1:10_000, Int32))
    ixh = Array(ix)
    vh = Array(v)
    @test issorted(vh[ixh])

    v = array_from_host(1:10_000, Float32)
    ix = AK.merge_sortperm_lowmem(v,
                                lt=(>), by=abs, rev=true,
                                block_size=64,
                                temp=array_from_host(1:10_000, Int))
    ixh = Array(ix)
    vh = Array(v)
    @test issorted(vh[ixh])
end
end


@testset "sortperm" begin
    Random.seed!(0)

    # Fuzzy correctness testing
    for _ in 1:1000
        num_elems = rand(1:100_000)
        ix = array_from_host(zeros(Int32, num_elems))
        v = array_from_host(rand(Int32, num_elems))
        AK.sortperm!(ix, v)
        ixh = Array(ix)
        vh = Array(v)
        @test issorted(vh[ixh])
    end

    for _ in 1:1000
        num_elems = rand(1:100_000)
        ix = array_from_host(zeros(Int32, num_elems))
        v = array_from_host(rand(UInt32, num_elems))
        AK.sortperm!(ix, v)
        ixh = Array(ix)
        vh = Array(v)
        @test issorted(vh[ixh])
    end

    for _ in 1:1000
        num_elems = rand(1:100_000)
        ix = array_from_host(zeros(Int32, num_elems))
        v = array_from_host(rand(Float32, num_elems))
        AK.sortperm!(ix, v)
        ixh = Array(ix)
        vh = Array(v)
        @test issorted(vh[ixh])
    end

    # Testing different settings
    ix = array_from_host(1:10_000, Int32)
    v = array_from_host(1:10_000, Float32)
    AK.sortperm!(ix,
                 v,
                 lt=(>), by=abs, rev=true,
                 block_size=64,
                 temp=array_from_host(1:10_000, Int32))
    ixh = Array(ix)
    vh = Array(v)
    @test issorted(vh[ixh])

    v = array_from_host(1:10_000, Float32)
    ix = AK.sortperm(v,
                     lt=(>), by=abs, rev=true,
                     block_size=64,
                     temp=array_from_host(1:10_000, Int))
    ixh = Array(ix)
    vh = Array(v)
    @test issorted(vh[ixh])
end


@testset "reduce_1d" begin
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
    for _ in 1:1000
        num_elems = rand(1:100_000)
        v = array_from_host(rand(Int32, num_elems))
        s = redmin(v)
        vh = Array(v)
        @test s == minimum(vh)
    end

    for _ in 1:1000
        num_elems = rand(1:100_000)
        v = array_from_host(rand(UInt32, num_elems))
        s = redmin(v)
        vh = Array(v)
        @test s == minimum(vh)
    end

    for _ in 1:1000
        num_elems = rand(1:100_000)
        v = array_from_host(rand(Float32, num_elems))
        s = redmin(v)
        vh = Array(v)
        @test s == minimum(vh)
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
    for _ in 1:1000
        num_elems = rand(1:100_000)
        v = array_from_host(rand(1:100, num_elems), Int32)
        s = redsum(v)
        vh = Array(v)
        @test s == sum(vh)
    end

    for _ in 1:1000
        num_elems = rand(1:100_000)
        v = array_from_host(rand(1:100, num_elems), UInt32)
        s = redsum(v)
        vh = Array(v)
        @test s == sum(vh)
    end

    for _ in 1:1000
        num_elems = rand(1:100_000)
        v = array_from_host(rand(Float32, num_elems))
        s = redsum(v)
        vh = Array(v)
        @test s ≈ sum(vh)
    end

    # Allowing N-dimensional arrays, still reduced as 1D
    for _ in 1:100
        n1 = rand(1:100)
        n2 = rand(1:100)
        n3 = rand(1:100)
        vh = rand(Float32, n1, n2, n3)
        v = array_from_host(vh)
        s = redsum(v)
        @test s ≈ sum(vh)
    end

    # Testing different settings
    AK.reduce(
        (x, y) -> x + 1,
        array_from_host(rand(Int32, 10_000)),
        init=Int64(0),
        block_size=64,
        temp=array_from_host(zeros(Int32, 10_000)),
        switch_below=50,
        scheduler=:dynamic,
        max_tasks=10,
        min_elems=100,
    )
    AK.reduce(
        (x, y) -> x + 1,
        rand(Int32, 10_000),
        init=Int64(0),
        scheduler=:greedy,
        max_tasks=16,
        min_elems=1000,
    )
end


@testset "reduce_nd" begin
    Random.seed!(0)

    # Test all possible corner cases against Base.reduce
    for dims in 1:4
        for isize in 0:3
            for jsize in 0:3
                for ksize in 0:3
                    sh = rand(Int32(1):Int32(100), isize, jsize, ksize)
                    s = array_from_host(sh)
                    d = AK.reduce(+, s; init=Int32(0), dims=dims)
                    dh = Array(d)
                    @test dh == sum(sh, init=Int32(0), dims=dims)
                    @test eltype(dh) == eltype(sum(sh, init=Int32(0), dims=dims))
                end
            end
        end
    end

    # Fuzzy correctness testing
    for _ in 1:100
        for dims in 1:3
            n1 = rand(1:100)
            n2 = rand(1:100)
            n3 = rand(1:100)
            vh = rand(Int32(1):Int32(100), n1, n2, n3)
            v = array_from_host(vh)
            s = AK.reduce(+, v; init=Int32(0), dims=dims)
            sh = Array(s)
            @test sh == sum(vh, dims=dims)
        end
    end

    for _ in 1:100
        for dims in 1:3
            n1 = rand(1:100)
            n2 = rand(1:100)
            n3 = rand(1:100)
            vh = rand(UInt32(1):UInt32(100), n1, n2, n3)
            v = array_from_host(vh)
            s = AK.reduce(+, v; init=UInt32(0), dims=dims)
            sh = Array(s)
            @test sh == sum(vh, dims=dims)
        end
    end

    for _ in 1:100
        for dims in 1:3
            n1 = rand(1:100)
            n2 = rand(1:100)
            n3 = rand(1:100)
            vh = rand(Float32, n1, n2, n3)
            v = array_from_host(vh)
            s = AK.reduce(+, v; init=Float32(0), dims=dims)
            sh = Array(s)
            @test sh ≈ sum(vh, dims=dims)
        end
    end

    # Testing different settings
    AK.reduce(
        (x, y) -> x + 1,
        array_from_host(rand(Int32, 3, 4, 5)),
        init=Int32(0),
        dims=2,
        block_size=64,
        temp=array_from_host(zeros(Int32, 3, 1, 5)),
        switch_below=50,
        scheduler=:dynamic,
        max_tasks=10,
        min_elems=100,
    )
    AK.reduce(
        (x, y) -> x + 1,
        array_from_host(rand(Int32, 3, 4, 5)),
        init=Int32(0),
        dims=3,
        block_size=64,
        temp=array_from_host(zeros(Int32, 3, 4, 1)),
        switch_below=50,
        scheduler=:greedy,
        max_tasks=16,
        min_elems=1000,
    )
end


@testset "mapreduce_1d" begin
    Random.seed!(0)

    struct Point
        x::Float32
        y::Float32
    end
    # Only for backend-agnostic initialisation with KernelAbstractions.zero
    Base.zero(::Type{Point}) = Point(0.0f0, 0.0f0)

    function minbox(s)
        # Extract coordinates into tuple and reduce to find dimensionwise minima
        AK.mapreduce(
            p -> (p.x, p.y),
            (a, b) -> (a[1] < b[1] ? a[1] : b[1], a[2] < b[2] ? a[2] : b[2]),
            s;
            init=(typemax(Float32), typemax(Float32)),
        )
    end

    function minbox_base(s)
        # Extract coordinates into tuple and reduce to find dimensionwise minima
        Base.mapreduce(
            p -> (p.x, p.y),
            (a, b) -> (a[1] < b[1] ? a[1] : b[1], a[2] < b[2] ? a[2] : b[2]),
            s;
            init=(typemax(Float32), typemax(Float32)),
        )
    end

    # Fuzzy correctness testing
    for _ in 1:1000
        num_elems = rand(1:100_000)
        v = array_from_host([Point(rand(Float32), rand(Float32)) for _ in 1:num_elems])
        mgpu = minbox(v)

        vh = Array(v)
        mcpu = minbox(vh)
        mbase = minbox_base(vh)

        @test typeof(mgpu) === typeof(mcpu) === typeof(mbase)
        @test mgpu[1] ≈ mcpu[1] ≈ mbase[1]
        @test mgpu[2] ≈ mcpu[2] ≈ mbase[2]
    end

    # Allowing N-dimensional arrays, still reduced as 1D
    for _ in 1:100
        n1 = rand(1:100)
        n2 = rand(1:100)
        n3 = rand(1:100)

        v = array_from_host([Point(rand(Float32), rand(Float32)) for _ in 1:n1, _ in 1:n2, _ in 1:n3])
        mgpu = minbox(v)

        vh = Array(v)
        mcpu = minbox(vh)
        mbase = minbox_base(vh)

        @test typeof(mgpu) === typeof(mcpu) === typeof(mbase)
        @test mgpu[1] ≈ mcpu[1] ≈ mbase[1]
        @test mgpu[2] ≈ mcpu[2] ≈ mbase[2]
    end

    # Testing different settings, enforcing change of type between f and op
    f(s, temp) = AK.mapreduce(
        p -> (p.x, p.y),
        (a, b) -> (a[1] < b[1] ? a[1] : b[1], a[2] < b[2] ? a[2] : b[2]),
        s,
        init=(typemax(Float32), typemax(Float32)),
        block_size=64,
        temp=temp,
        switch_below=50,
        scheduler=:dynamic,
        max_tasks=10,
        min_elems=100,
    )
    v = array_from_host([Point(rand(Float32), rand(Float32)) for _ in 1:10_042])
    temp = similar(v, Tuple{Float32, Float32})
    f(v, temp)
end


@testset "mapreduce_nd" begin
    Random.seed!(0)

    # Test all possible corner cases against Base.reduce
    for dims in 1:4
        for isize in 0:3
            for jsize in 0:3
                for ksize in 0:3
                    sh = rand(Int32(1):Int32(100), isize, jsize, ksize)
                    s = array_from_host(sh)
                    d = AK.mapreduce(-, +, s; init=Int32(0), dims=dims)
                    dh = Array(d)
                    @test dh == mapreduce(-, +, sh, init=Int32(0), dims=dims)
                    @test eltype(dh) == eltype(sum(sh, init=Int32(0), dims=dims))
                end
            end
        end
    end

    # Fuzzy correctness testing
    for _ in 1:100
        for dims in 1:3
            n1 = rand(1:100)
            n2 = rand(1:100)
            n3 = rand(1:100)
            vh = rand(Int32(1):Int32(100), n1, n2, n3)
            v = array_from_host(vh)
            s = AK.mapreduce(-, +, v; init=Int32(0), dims=dims)
            sh = Array(s)
            @test sh == mapreduce(-, +, vh, init=Int32(0), dims=dims)
        end
    end

    struct Point2
        x::Float32
        y::Float32
    end

    # Only for backend-agnostic initialisation with KernelAbstractions.zero
    Base.zero(::Type{Point2}) = Point2(0.0f0, 0.0f0)

    function minbox(s, dims)
        # Extract coordinates into tuple and reduce to find dimensionwise minima
        AK.mapreduce(
            p -> (p.x, p.y),
            (a, b) -> (a[1] < b[1] ? a[1] : b[1], a[2] < b[2] ? a[2] : b[2]),
            s;
            init=(typemax(Float32), typemax(Float32)),
            dims=dims,
        )
    end

    function minbox_base(s, dims)
        # Extract coordinates into tuple and reduce to find dimensionwise minima
        Base.mapreduce(
            p -> (p.x, p.y),
            (a, b) -> (a[1] < b[1] ? a[1] : b[1], a[2] < b[2] ? a[2] : b[2]),
            s;
            init=(typemax(Float32), typemax(Float32)),
            dims=dims,
        )
    end

    # Fuzzy correctness testing
    for _ in 1:100
        for dims in 1:3
            n1 = rand(1:100)
            n2 = rand(1:100)
            n3 = rand(1:100)
            v = array_from_host([Point(rand(Float32), rand(Float32)) for _ in 1:n1, _ in 1:n2, _ in 1:n3])
            mgpu = minbox(v, dims)

            vh = Array(v)
            mcpu = minbox(vh, dims)
            mbase = minbox_base(vh, dims)

            @test eltype(mgpu) === eltype(mcpu) === eltype(mbase)
            @test all([
                (mgpu_red[1] ≈ mcpu[i][1] ≈ mbase[i][1]) && (mgpu_red[2] ≈ mcpu[i][2] ≈ mbase[i][2])
                for (i, mgpu_red) in enumerate(Array(mgpu))
            ])
        end
    end

    # Testing different settings
    AK.mapreduce(
        -,
        (x, y) -> x + 1,
        array_from_host(rand(Int32, 3, 4, 5)),
        init=Int32(0),
        dims=2,
        block_size=64,
        temp=array_from_host(zeros(Int32, 3, 1, 5)),
        switch_below=50,
        scheduler=:dynamic,
        max_tasks=10,
        min_elems=100,
    )
    AK.mapreduce(
        -,
        (x, y) -> x + 1,
        array_from_host(rand(Int32, 3, 4, 5)),
        init=Int32(0),
        dims=3,
        block_size=64,
        temp=array_from_host(zeros(Int32, 3, 4, 1)),
        switch_below=50,
        scheduler=:greedy,
        max_tasks=16,
        min_elems=1000,
    )

end


@testset "accumulate" begin

    Random.seed!(0)

    # Single block exlusive scan (each block processes two elements)
    for num_elems in 1:256
        x = array_from_host(ones(Int32, num_elems))
        y = copy(x)
        AK.accumulate!(+, y; init=0, inclusive=false, block_size=128)
        yh = Array(y)
        @test all(yh .== 0:length(yh) - 1)
    end

    # Single block inclusive scan
    for num_elems in 1:256
        x = array_from_host(rand(1:1000, num_elems), Int32)
        y = copy(x)
        AK.accumulate!(+, y; init=0, block_size=128)
        @test all(Array(y) .== accumulate(+, Array(x)))
    end

    # Large exclusive scan
    for _ in 1:1000
        num_elems = rand(1:100_000)
        x = array_from_host(ones(Int32, num_elems))
        y = copy(x)
        AK.accumulate!(+, y; init=0, inclusive=false)
        yh = Array(y)
        @test all(yh .== 0:length(yh) - 1)
    end

    # Large inclusive scan
    for _ in 1:1000
        num_elems = rand(1:100_000)
        x = array_from_host(rand(1:1000, num_elems), Int32)
        y = copy(x)
        AK.accumulate!(+, y; init=0)
        @test all(Array(y) .== accumulate(+, Array(x)))
    end

    # Testing different settings
    AK.accumulate!(+, array_from_host(ones(Int32, 1000)), init=0, inclusive=false,
                   block_size=128,
                   temp=array_from_host(zeros(Int32, 1000)),
                   temp_flags=array_from_host(zeros(Int8, 1000)))
    AK.accumulate(+, array_from_host(ones(Int32, 1000)), init=0, inclusive=false,
                  block_size=128,
                  temp=array_from_host(zeros(Int32, 1000)),
                  temp_flags=array_from_host(zeros(Int8, 1000)))
end


@testset "searchsorted" begin

    Random.seed!(0)

    # Fuzzy correctness testing of searchsortedfirst
    for _ in 1:100
        num_elems_v = rand(1:100_000)
        num_elems_x = rand(1:100_000)

        # Ints
        v = array_from_host(sort(rand(Int32, num_elems_v)))
        x = array_from_host(rand(Int32, num_elems_x))
        ix = similar(x, Int32)
        AK.searchsortedfirst!(ix, v, x)

        vh = Array(v)
        xh = Array(x)
        ixh = AK.searchsortedfirst(vh, xh)
        ixh_base = [searchsortedfirst(vh, e) for e in xh]

        @test all(Array(ix) .== ixh .== ixh_base)

        # Floats
        v = array_from_host(sort(rand(Float32, num_elems_v)))
        x = array_from_host(rand(Float32, num_elems_x))
        ix = similar(x, Int32)
        AK.searchsortedfirst!(ix, v, x)

        vh = Array(v)
        xh = Array(x)
        ixh = AK.searchsortedfirst(vh, xh)
        ixh_base = [searchsortedfirst(vh, e) for e in xh]

        @test all(Array(ix) .== ixh .== ixh_base)
    end

    # Fuzzy correctness testing of searchsortedlast
    for _ in 1:100
        num_elems_v = rand(1:100_000)
        num_elems_x = rand(1:100_000)

        # Ints
        v = array_from_host(sort(rand(Int32, num_elems_v)))
        x = array_from_host(rand(Int32, num_elems_x))
        ix = similar(x, Int32)
        AK.searchsortedlast!(ix, v, x)

        vh = Array(v)
        xh = Array(x)
        ixh = AK.searchsortedlast(vh, xh)
        ixh_base = [searchsortedlast(vh, e) for e in xh]

        @test all(Array(ix) .== ixh .== ixh_base)

        # Floats
        v = array_from_host(sort(rand(Float32, num_elems_v)))
        x = array_from_host(rand(Float32, num_elems_x))
        ix = similar(x, Int32)
        AK.searchsortedlast!(ix, v, x)

        vh = Array(v)
        xh = Array(x)
        ixh = AK.searchsortedlast(vh, xh)
        ixh_base = [searchsortedlast(vh, e) for e in xh]

        @test all(Array(ix) .== ixh .== ixh_base)
    end

    # Testing different settings
    v = array_from_host(sort(rand(Int32, 100_000)))
    x = array_from_host(rand(Int32, 10_000))
    ix = similar(x, Int32)

    AK.searchsortedfirst!(ix, v, x, by=abs, lt=(>), rev=true, block_size=64)
    AK.searchsortedfirst(v, x, by=abs, lt=(>), rev=true, block_size=64)
    AK.searchsortedlast!(ix, v, x, by=abs, lt=(>), rev=true, block_size=64)
    AK.searchsortedlast(v, x, by=abs, lt=(>), rev=true, block_size=64)
 
    vh = Array(v)
    xh = Array(x)
    ixh = similar(xh, Int32)

    AK.searchsortedfirst!(ixh, vh, xh, by=abs, lt=(>), rev=true, max_tasks=10, min_elems=100)
    AK.searchsortedfirst(vh, xh, by=abs, lt=(>), rev=true, max_tasks=10, min_elems=100)
    AK.searchsortedlast!(ixh, vh, xh, by=abs, lt=(>), rev=true, max_tasks=10, min_elems=100)
    AK.searchsortedlast(vh, xh, by=abs, lt=(>), rev=true, max_tasks=10, min_elems=100)
end


@testset "truth" begin

    Random.seed!(0)

    # Simple correctness tests
    v = array_from_host(1:100)

    # TODO: remove cooperative on CUDA
    @test AK.any(x->x<0, v, cooperative=false) === false
    @test AK.any(x->x>99, v, cooperative=false) === true

    @test AK.all(x->x>0, v, cooperative=false) === true
    @test AK.all(x->x<100, v, cooperative=false) === false

    for _ in 1:100
        num_elems = rand(1:100_000)
        v = array_from_host(rand(Float32, num_elems))
        @test AK.any(x->x<0, v, cooperative=false) === false
        @test AK.any(x->x<1, v, cooperative=false) === true
        @test AK.all(x->x<1, v, cooperative=false) === true
        @test AK.all(x->x<0, v, cooperative=false) === false
    end

    # Testing different settings
    v = array_from_host(rand(-5:5, 100_000))
    AK.any(x->x<5, v, cooperative=false, block_size=64)
    AK.all(x->x<5, v, cooperative=false, block_size=64)
end

