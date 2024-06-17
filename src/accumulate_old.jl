const ACC_SHMEM_SIZE = 2048
const ACC_NUM_BANKS = 1024
const ACC_LOG_NUM_BANKS = 10


@inline function conflict_free_offset(n)
    # Two possible offsets
    # n >> ACC_LOG_NUM_BANKS
    n >> ACC_NUM_BANKS + n >> (2 * ACC_LOG_NUM_BANKS)
end


@kernel cpu=false inbounds=true function _preaccumulate_block!(@Const(op), dst, @Const(src),
                                                               @Const(init), @Const(seed),
                                                               @Const(inclusive), @Const(next_pow2))

    # NOTE: shmem_size MUST be greater than 2 * next_pow2
    len = length(src)
    temp = @localmem eltype(dst) (ACC_SHMEM_SIZE,)      # NOTE: will use 2 * next_pow2 elements

    # NOTE: for many index calculations in this library, computation using zero-indexing leads to
    # fewer operations (also code is transpiled to CUDA / ROCm / oneAPI / Metal code which do zero
    # indexing). Internal calculations will be done using zero indexing except when actually
    # accessing memory. As with C, the lower bound is inclusive, the upper bound exclusive.

    # Group (block) and local (thread) indices
    iblock = @index(Group, Linear) - 1
    ithread = @index(Local, Linear) - 1

    ai = ithread
    bi = ithread + len ÷ 2

    bank_offset_a = conflict_free_offset(ai)
    bank_offset_b = conflict_free_offset(bi)

    if ithread < len
        temp[ai + bank_offset_a + 1] = src[ai + 1]
        temp[bi + bank_offset_b + 1] = src[bi + 1]
    else
        temp[ai + bank_offset_a + 1] = init
        temp[bi + bank_offset_b + 1] = init
    end

    offset = 1
    d = next_pow2 >> 1
    while d > 0
        @synchronize()

        if ithread < d
            _ai = offset * (2 * ithread + 1) - 1
            _bi = offset * (2 * ithread + 2) - 1
            _ai += conflict_free_offset(_ai)
            _bi += conflict_free_offset(_bi)

            temp[_bi + 1] = op(temp[_bi + 1], temp[_ai + 1])
        end

        offset = offset * 2
        d = d >> 1
    end

    if ithread == 0
        offset0 = conflict_free_offset(next_pow2 - 1)
        temp[next_pow2 - 1 + offset0 + 1] = inclusive ? src[1] : init
    end

    d = 1
    while d < next_pow2
        offset = offset >> 1
        @synchronize()

        if ithread < d
            _ai = offset * (2 * ithread + 1) - 1
            _bi = offset * (2 * ithread + 2) - 1
            _ai += conflict_free_offset(_ai)
            _bi += conflict_free_offset(_bi)

            t = temp[_ai + 1]
            temp[_ai + 1] = temp[_bi + 1]
            temp[_bi + 1] = op(temp[_bi + 1], t)
        end

        d = d * 2
    end

    @synchronize()

	if ithread < len
        if isnothing(seed)
            dst[ai + 1] = temp[ai + bank_offset_a + 1]
            dst[bi + 1] = temp[bi + bank_offset_b + 1]
        else
            dst[ai + 1] = op(seed[], temp[ai + bank_offset_a + 1])
            dst[bi + 1] = op(seed[], temp[bi + bank_offset_b + 1])
        end
    end
end


# TODO: I think the _preaccumulate_block is pretty good, but we may be able to do a faster
# inter-block  scan like in "Single-pass Parallel Prefix Scan with Decoupled Look-back" from NVidia
@kernel cpu=false inbounds=true function _preaccumulate_large!(op, dst, src, init, accm)

    # NOTE: shmem_size MUST be greater than 2 * elems_per_block
    N = @groupsize()[1]
    temp = @localmem eltype(dst) (ACC_SHMEM_SIZE,)      # NOTE: will use (2 * elems_per_block)

    # NOTE: for many index calculations in this library, computation using zero-indexing leads to
    # fewer operations (also code is transpiled to CUDA / ROCm / oneAPI / Metal code which do zero
    # indexing). Internal calculations will be done using zero indexing except when actually
    # accessing memory. As with C, the lower bound is inclusive, the upper bound exclusive.

    # Group (block) and local (thread) indices
    iblock = @index(Group, Linear) - 1
    ithread = @index(Local, Linear) - 1

    elems_per_block = N * 2
    block_offset = iblock * elems_per_block

    ai = ithread
    bi = ithread + N
    bank_offset_a = conflict_free_offset(ai)
    bank_offset_b = conflict_free_offset(bi)

    temp[ai + bank_offset_a + 1] = src[block_offset + ai + 1]
    temp[bi + bank_offset_b + 1] = src[block_offset + bi + 1]

    offset = 1
    d = elems_per_block >> 1
    while d > 0
        @synchronize()
        if ithread < d
            _ai = offset * (2 * ithread + 1) - 1
            _bi = offset * (2 * ithread + 2) - 1
            _ai += conflict_free_offset(_ai)
            _bi += conflict_free_offset(_bi)
            temp[_bi + 1] = op(temp[_bi + 1], temp[_ai + 1])
        end

        offset = offset * 2
        d = d >> 1
    end
    @synchronize()

    if ithread == 0
        offset0 = conflict_free_offset(elems_per_block - 1)
        accm[iblock + 1] = temp[elems_per_block - 1 + offset0 + 1]
        temp[elems_per_block - 1 + offset0 + 1] = init
    end

    d = 1
    while d < elems_per_block
        offset = offset >> 1

        @synchronize()
        if ithread < d
            _ai = offset * (2 * ithread + 1) - 1
            _bi = offset * (2 * ithread + 2) - 1
            _ai += conflict_free_offset(_ai)
            _bi += conflict_free_offset(_bi)

            t = temp[_ai + 1]
            temp[_ai + 1] = temp[_bi + 1]
            temp[_bi + 1] = op(temp[_bi + 1], t)
        end

        d = d * 2
    end
    @synchronize()

    dst[block_offset + ai + 1] = temp[ai + bank_offset_a + 1]
    dst[block_offset + bi + 1] = temp[bi + bank_offset_b + 1]
end


@kernel cpu=false inbounds=true function _accumulate_elem_per_block(op, dst, eblocks)
    iblock = @index(Group, Linear)
    i = @index(Global, Linear)
    dst[i] = op(dst[i], eblocks[(iblock + 1) ÷ 2])
end


@kernel cpu=false inbounds=true function _accumulate_include_first(op, dst, src0)
    i = @index(Global, Linear)
    dst[i] = op(src0[], dst[i])
end


@inline function accumulate_small!(
    op,
    dst,
    src;
    init,
    seed=nothing,
    inclusive=false,
    block_size=128,
)
    # Correctness checks
    next_pow2 = nextpow(2, length(src))
    @assert length(src) <= 2 * block_size
    @assert ACC_SHMEM_SIZE >= 2 * next_pow2

    backend = get_backend(src)
    kernel! = _preaccumulate_block!(backend, block_size)
    kernel!(op, dst, src, init, seed, inclusive, next_pow2, ndrange=block_size)
end


@inline function accumulate_large_even!(
    op,
    dst,
    src;
    init,
    inclusive=false,
    block_size=128,
)
    @assert length(src) % (2 * block_size) == 0

    elems_per_block = 2 * block_size
    num_blocks = length(src) ÷ elems_per_block

    accm = similar(src, eltype(src), num_blocks)
    incr = similar(src, eltype(src), num_blocks)

    backend = get_backend(src)
    kernel! = _preaccumulate_large!(backend, block_size)
    kernel!(op, dst, src, init, accm, ndrange=num_blocks * block_size)

    threads_needed = (num_blocks + 1) ÷ 2
    if threads_needed > block_size
        accumulate_large!(op, incr, accm; init=init, block_size=block_size)
    else
        accumulate_small!(op, incr, accm; init=init, block_size=block_size)
    end

    _accumulate_elem_per_block(backend, block_size)(
        op, dst, incr,
        ndrange=num_blocks * elems_per_block,
    )
end


@inline function accumulate_large!(
    op,
    dst,
    src;
    init,
    block_size=128,
)
    elems_per_block = block_size * 2

    len = length(src)
    remainder = len % elems_per_block
    if remainder == 0
        accumulate_large_even!(op, dst, src; init=init,
                               block_size=block_size)
    else
        length_multiple = len - remainder
        src_multiple = @view src[1:length_multiple]
        dst_multiple = @view dst[1:length_multiple]
        accumulate_large_even!(op, dst_multiple, src_multiple; init=init,
                               block_size=block_size)

        # Accumulate remainder starting from the even section's final accumulated value
        src_remainder = @view src[length_multiple + 1:end]
        dst_remainder = @view dst[length_multiple + 1:end]
        seed = @view(dst[length_multiple])
        accumulate_small!(op, dst_remainder, src_remainder; init=init, seed=seed,
                          inclusive=true, block_size=block_size)
    end
end


@inline function accumulate!(
    op,
    dst,
    src;
    init,
    inclusive::Bool=true,
    block_size::Int=128,
)
    @assert block_size > 0
    @assert length(dst) == length(src)

    # Each thread will process two elements
    elems_per_block = block_size * 2
    backend = get_backend(dst)

    if length(dst) > elems_per_block
        accumulate_large!(op, dst, src; init=init, block_size=block_size)

        # Add first element
        if inclusive
            _accumulate_include_first(backend, block_size)(
                op, dst, @view(src[1]), ndrange=length(dst),
            )
        end
    else
        # Single block accumulation
        accumulate_small!(op, dst, src; init=init, inclusive=inclusive, block_size=block_size)
    end

    synchronize(backend)
end


function accumulate(
    op,
    src;
    init,
    inclusive::Bool=true,
    block_size::Int=128,
)
    dst = similar(src)
    accumulate!(op, dst, src; init=init, inclusive=inclusive, block_size=block_size)
    dst
end

