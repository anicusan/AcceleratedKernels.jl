@kernel inbounds=true function _merge_sort_by_key_block!(keys, values, @Const(comp))

    N = @groupsize()[1]
    s_keys = @localmem eltype(keys) (N * 2,)
    s_values = @localmem eltype(values) (N * 2,)

    I = typeof(N)
    len = length(keys)

    # NOTE: for many index calculations in this library, computation using zero-indexing leads to
    # fewer operations (also code is transpiled to CUDA / ROCm / oneAPI / Metal code which do zero
    # indexing). Internal calculations will be done using zero indexing except when actually
    # accessing memory. As with C, the lower bound is inclusive, the upper bound exclusive.

    # Group (block) and local (thread) indices
    iblock = @index(Group, Linear) - 1
    ithread = @index(Local, Linear) - 1

    i = ithread + iblock * N * 2
    if i < len
        s_keys[ithread + 1] = keys[i + 1]
        s_values[ithread + 1] = values[i + 1]
    end

    i = ithread + N + iblock * N * 2
    if i < len
        s_keys[ithread + N + 1] = keys[i + 1]
        s_values[ithread + N + 1] = values[i + 1]
    end

    @synchronize()

    half_size_group = 1
    size_group = 2

    while half_size_group <= N
        gid = ithread ÷ half_size_group

        local k1::eltype(keys)
        local k2::eltype(keys)
        local v1::eltype(values)
        local v2::eltype(values)
        pos1 = typemax(I)
        pos2 = typemax(I)

        i = gid * size_group + half_size_group + iblock * N * 2
        if i < len
            tid = gid * size_group + ithread % half_size_group
            k1 = s_keys[tid + 1]
            v1 = s_values[tid + 1]

            i = (gid + 1) * size_group + iblock * N * 2
            n = i < len ? half_size_group : len - iblock * N * 2 - gid * size_group - half_size_group
            lo = gid * size_group + half_size_group
            hi = lo + n
            pos1 = ithread % half_size_group + _lower_bound_s0(s_keys, k1, lo, hi, comp) - lo
        end

        tid = gid * size_group + half_size_group + ithread % half_size_group
        i = tid + iblock * N * 2
        if i < len
            k2 = s_keys[tid + 1]
            v2 = s_values[tid + 1]
            lo = gid * size_group
            hi = lo + half_size_group
            pos2 = ithread % half_size_group + _upper_bound_s0(s_keys, k2, lo, hi, comp) - lo
        end

        @synchronize()

        if pos1 != typemax(I)
            s_keys[gid * size_group + pos1 + 1] = k1
            s_values[gid * size_group + pos1 + 1] = v1
        end
        if pos2 != typemax(I)
            s_keys[gid * size_group + pos2 + 1] = k2
            s_values[gid * size_group + pos2 + 1] = v2
        end

        @synchronize()

        half_size_group = half_size_group << 1
        size_group = size_group << 1
    end

    i = ithread + iblock * N * 2
    if i < len
        keys[i + 1] = s_keys[ithread + 1]
        values[i + 1] = s_values[ithread + 1]
    end

    i = ithread + N + iblock * N * 2
    if i < len
        keys[i + 1] = s_keys[ithread + N + 1]
        values[i + 1] = s_values[ithread + N + 1]
    end
end


@kernel inbounds=true function _merge_sort_by_key_global!(@Const(keys_in), keys_out,
                                                          @Const(values_in), values_out,
                                                          @Const(comp), @Const(half_size_group))

    len = length(keys_in)
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
        if pos_in < len
            keys_out[pos_in + 1] = keys_in[pos_in + 1]
            values_out[pos_in + 1] = values_in[pos_in + 1]
        end
    else

        hi = (gid + 1) * size_group
        hi > len && (hi = len)

        pos_out = pos_in + _lower_bound_s0(keys_in, keys_in[pos_in + 1], lo, hi, comp) - lo
        keys_out[pos_out + 1] = keys_in[pos_in + 1]
        values_out[pos_out + 1] = values_in[pos_in + 1]

        # Right half
        pos_in = gid * size_group + half_size_group + idx % half_size_group

        if pos_in < len
            lo = gid * size_group
            hi = lo + half_size_group
            pos_out = pos_in - half_size_group + _upper_bound_s0(keys_in, keys_in[pos_in + 1], lo, hi, comp) - lo
            keys_out[pos_out + 1] = keys_in[pos_in + 1]
            values_out[pos_out + 1] = values_in[pos_in + 1]
        end
    end
end


function merge_sort_by_key!(
    keys::AbstractGPUVector,
    values::AbstractGPUVector;

    lt=(<),
    by=identity,
    rev::Bool=false,
    order::Base.Order.Ordering=Base.Order.Forward,

    block_size::Int=128,
    temp_keys::Union{Nothing, AbstractGPUVector}=nothing,
    temp_values::Union{Nothing, AbstractGPUVector}=nothing,
)
    # Simple sanity checks
    @assert block_size > 0
    @assert length(keys) == length(values)
    if !isnothing(temp_keys)
        @assert length(temp_keys) == length(keys)
        @assert eltype(temp_keys) === eltype(keys)
    end
    if !isnothing(temp_values)
        @assert length(temp_values) == length(values)
        @assert eltype(temp_values) === eltype(values)
    end

    # Construct comparator
    ord = Base.Order.ord(lt, by, rev, order)
    comp = (x, y) -> Base.Order.lt(ord, x, y)

    # Block level
    backend = get_backend(keys)
    blocks = (length(keys) + block_size * 2 - 1) ÷ (block_size * 2)
    _merge_sort_by_key_block!(backend, block_size)(keys, values, comp, ndrange=(block_size * blocks,))

    # Global level
    half_size_group = block_size * 2
    size_group = half_size_group * 2
    len = length(keys)
    if len > half_size_group
        pk1 = keys
        pk2 = isnothing(temp_keys) ? similar(keys) : temp_keys

        pv1 = values
        pv2 = isnothing(temp_values) ? similar(values) : temp_values

        kernel! = _merge_sort_by_key_global!(backend, block_size)

        niter = 0
        while len > half_size_group
            blocks = ((len + half_size_group - 1) ÷ half_size_group + 1) ÷ 2 * (half_size_group ÷ block_size)
            kernel!(pk1, pk2, pv1, pv2, comp, half_size_group, ndrange=(block_size * blocks,))

            half_size_group = half_size_group << 1;
            size_group = size_group << 1;
            pk1, pk2 = pk2, pk1
            pv1, pv2 = pv2, pv1

            niter += 1
        end

        if isodd(niter)
            keys .= pk1
            values .= pv1
        end
    end

    synchronize(backend)
    nothing
end


function merge_sort_by_key(
    keys::AbstractGPUVector,
    values::AbstractGPUVector;

    lt=(<),
    by=identity,
    rev::Bool=false,
    order::Base.Order.Ordering=Base.Order.Forward,

    block_size::Int=128,
    temp_keys::Union{Nothing, AbstractGPUVector}=nothing,
    temp_values::Union{Nothing, AbstractGPUVector}=nothing,
)
    keys_copy = copy(keys)
    values_copy = copy(values)

    merge_sort_by_key!(
        keys_copy, values_copy,
        lt=lt, by=by, rev=rev, order=order,
        block_size=block_size, temp_keys=temp_keys, temp_values=temp_values,
    )

    keys_copy, values_copy
end

