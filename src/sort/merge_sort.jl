@kernel inbounds=true function _merge_sort_block!(vec, @Const(comp))

    N = @groupsize()[1]
    s_buf = @localmem eltype(vec) (N * 2,)

    T = eltype(vec)
    I = typeof(N)
    len = length(vec)

    # NOTE: for many index calculations in this library, computation using zero-indexing leads to
    # fewer operations (also code is transpiled to CUDA / ROCm / oneAPI / Metal code which do zero
    # indexing). Internal calculations will be done using zero indexing except when actually
    # accessing memory. As with C, the lower bound is inclusive, the upper bound exclusive.

    # Group (block) and local (thread) indices
    iblock = @index(Group, Linear) - 1
    ithread = @index(Local, Linear) - 1

    i = ithread + iblock * N * 2
    i < len && (s_buf[ithread + 1] = vec[i + 1])

    i = ithread + N + iblock * N * 2
    i < len && (s_buf[ithread + N + 1] = vec[i + 1])

    @synchronize()

    half_size_group = 1
    size_group = 2

    while half_size_group <= N
        gid = ithread ÷ half_size_group

        local v1::T
        local v2::T
        pos1 = typemax(I)
        pos2 = typemax(I)

        i = gid * size_group + half_size_group + iblock * N * 2
        if i < len
            tid = gid * size_group + ithread % half_size_group
            v1 = s_buf[tid + 1]

            i = (gid + 1) * size_group + iblock * N * 2
            n = i < len ? half_size_group : len - iblock * N * 2 - gid * size_group - half_size_group
            lo = gid * size_group + half_size_group
            hi = lo + n
            pos1 = ithread % half_size_group + _lower_bound_s0(s_buf, v1, lo, hi, comp) - lo
        end

        tid = gid * size_group + half_size_group + ithread % half_size_group
        i = tid + iblock * N * 2
        if i < len
            v2 = s_buf[tid + 1]
            lo = gid * size_group
            hi = lo + half_size_group
            pos2 = ithread % half_size_group + _upper_bound_s0(s_buf, v2, lo, hi, comp) - lo
        end

        @synchronize()

        pos1 != typemax(I) && (s_buf[gid * size_group + pos1 + 1] = v1)
        pos2 != typemax(I) && (s_buf[gid * size_group + pos2 + 1] = v2)

        @synchronize()

        half_size_group = half_size_group << 1
        size_group = size_group << 1
    end

    i = ithread + iblock * N * 2
    i < len && (vec[i + 1] = s_buf[ithread + 1])

    i = ithread + N + iblock * N * 2
    i < len && (vec[i + 1] = s_buf[ithread + N + 1])
end


@kernel inbounds=true function _merge_sort_global!(@Const(vec_in), vec_out, @Const(comp), @Const(half_size_group))

    len = length(vec_in)
    N = @groupsize()[1]

    # NOTE: for many index calculations in this library, computation using zero-indexing leads to
    # fewer operations (also code is transpiled to CUDA / ROCm / oneAPI / Metal code which do zero
    # indexing). Internal calculations will be done using zero indexing except when actually
    # accessing memory. As with C, the lower bound is inclusive, the upper bound exclusive.

    # Group (block) and local (thread) indices
    iblock = @index(Group, Linear) - 1
    ithread = @index(Local, Linear) - 1

    idx = ithread + iblock * N
    size_group = half_size_group * 2
    gid = idx ÷ half_size_group

    # Left half
    pos_in = gid * size_group + idx % half_size_group
    lo = gid * size_group + half_size_group

    if lo >= len
        # Incomplete left half, nothing to swap on the right, simply copy elements to be sorted
        # in next iteration
        pos_in < len && (vec_out[pos_in + 1] = vec_in[pos_in + 1])
    else

        hi = (gid + 1) * size_group
        hi > len && (hi = len)

        pos_out = pos_in + _lower_bound_s0(vec_in, vec_in[pos_in + 1], lo, hi, comp) - lo
        vec_out[pos_out + 1] = vec_in[pos_in + 1]

        # Right half
        pos_in = gid * size_group + half_size_group + idx % half_size_group

        if pos_in < len
            lo = gid * size_group
            hi = lo + half_size_group
            pos_out = pos_in - half_size_group + _upper_bound_s0(vec_in, vec_in[pos_in + 1], lo, hi, comp) - lo
            vec_out[pos_out + 1] = vec_in[pos_in + 1]
        end
    end
end


function merge_sort!(
    v::AbstractGPUVector;

    lt=(<),
    by=identity,
    rev::Bool=false,
    order::Base.Order.Ordering=Base.Order.Forward,

    block_size::Int=128,
    temp::Union{Nothing, AbstractGPUVector}=nothing,
)
    # Simple sanity checks
    @argcheck block_size > 0
    if !isnothing(temp)
        @argcheck length(temp) == length(v)
        @argcheck eltype(temp) === eltype(v)
    end

    # Construct comparator
    ord = Base.Order.ord(lt, by, rev, order)
    comp = (x, y) -> Base.Order.lt(ord, x, y)

    # Block level
    backend = get_backend(v)
    blocks = (length(v) + block_size * 2 - 1) ÷ (block_size * 2)
    _merge_sort_block!(backend, block_size)(v, comp, ndrange=(block_size * blocks,))

    # Global level
    half_size_group = block_size * 2
    size_group = half_size_group * 2
    len = length(v)
    if len > half_size_group
        p1 = v
        p2 = isnothing(temp) ? similar(v) : temp

        kernel! = _merge_sort_global!(backend, block_size)

        niter = 0
        while len > half_size_group
            blocks = ((len + half_size_group - 1) ÷ half_size_group + 1) ÷ 2 * (half_size_group ÷ block_size)
            kernel!(p1, p2, comp, half_size_group, ndrange=(block_size * blocks,))

            half_size_group = half_size_group << 1;
            size_group = size_group << 1;
            p1, p2 = p2, p1

            niter += 1
        end

        if isodd(niter)
            copyto!(v, p1)
        end
    end

    nothing
end


function merge_sort(
    v::AbstractGPUVector;

    lt=(<),
    by=identity,
    rev::Bool=false,
    order::Base.Order.Ordering=Base.Order.Forward,

    block_size::Int=128,
    temp::Union{Nothing, AbstractGPUVector}=nothing,
)
    v_copy = copy(v)
    merge_sort!(
        v_copy,
        lt=lt, by=by, rev=rev, order=order,
        block_size=block_size, temp=temp,
    )
    v_copy
end
