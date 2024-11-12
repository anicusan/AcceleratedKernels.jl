const ACC_NUM_BANKS = 32
const ACC_LOG_NUM_BANKS = 5

const ACC_FLAG_A::Int8 = 0          # Aggregate of all previous prefixes finished
const ACC_FLAG_P::Int8 = 1          # Only current block's prefix available


@inline function conflict_free_offset(n)
    # Two possible offsets
    n >> ACC_LOG_NUM_BANKS
    # n >> ACC_NUM_BANKS + n >> (2 * ACC_LOG_NUM_BANKS)
end


@kernel cpu=false inbounds=true function _accumulate_block!(op, v, init,
                                                            inclusive,
                                                            flags, prefixes)  # one per block

    # NOTE: shmem_size MUST be greater than 2 * block_size
    # NOTE: block_size MUST be a power of 2
    len = length(v)
    block_size = @groupsize()[1]
    temp = @localmem eltype(v) (2 * block_size + conflict_free_offset(2 * block_size),)

    # NOTE: for many index calculations in this library, computation using zero-indexing leads to
    # fewer operations (also code is transpiled to CUDA / ROCm / oneAPI / Metal code which do zero
    # indexing). Internal calculations will be done using zero indexing except when actually
    # accessing memory. As with C, the lower bound is inclusive, the upper bound exclusive.

    # Group (block) and local (thread) indices
    iblock = @index(Group, Linear) - 1
    ithread = @index(Local, Linear) - 1

    num_blocks = @ndrange()[1] ÷ block_size
    block_offset = iblock * block_size * 2              # Processing two elements per thread

    # Copy two elements from the main array; offset indices to avoid bank conflicts
    ai = ithread
    bi = ithread + block_size

    bank_offset_a = conflict_free_offset(ai)
    bank_offset_b = conflict_free_offset(bi)

    if block_offset + ai < len
        temp[ai + bank_offset_a + 1] = v[block_offset + ai + 1]
    else
        temp[ai + bank_offset_a + 1] = init
    end

    if block_offset + bi < len
        temp[bi + bank_offset_b + 1] = v[block_offset + bi + 1]
    else
        temp[bi + bank_offset_b + 1] = init
    end

    # Build block reduction down
    offset = 1
    next_pow2 = block_size * 2
    d = next_pow2 >> 1
    while d > 0             # TODO: unroll this like in reduce.jl ?
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

    # Flush last element
    if ithread == 0
        offset0 = conflict_free_offset(next_pow2 - 1)
        temp[next_pow2 - 1 + offset0 + 1] = init
    end

    # Build block accumulation up
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

    # Later blocks should always be inclusively-scanned
    if inclusive || iblock != 0
        # To compute an inclusive scan, shift elements left...
        @synchronize()
        t1 = temp[ai + bank_offset_a + 1]
        t2 = temp[bi + bank_offset_b + 1]
        @synchronize()

        if ai > 0
            temp[ai - 1 + conflict_free_offset(ai - 1) + 1] = t1
        end
        temp[bi - 1 + conflict_free_offset(bi - 1) + 1] = t2

        # ...and accumulate the last value too
        if bi == 2 * block_size - 1
            if iblock < num_blocks - 1
                temp[bi + bank_offset_b + 1] = op(t2, v[(iblock + 1) * block_size * 2])
            else
                temp[bi + bank_offset_b + 1] = op(t2, v[len])
            end
        end
    end

    @synchronize()

    # Write this block's final prefix to global array and set flag to "block prefix computed"
    if bi == 2 * block_size - 1
        prefixes[iblock + 1] = temp[bi + bank_offset_b + 1]
        flags[iblock + 1] = ACC_FLAG_P
    end

    if block_offset + ai < len
        v[block_offset + ai + 1] = temp[ai + bank_offset_a + 1]
    end
    if block_offset + bi < len
        v[block_offset + bi + 1] = temp[bi + bank_offset_b + 1]
    end
end


@kernel cpu=false inbounds=true function _accumulate_previous!(op, v, init, flags, @Const(prefixes))

    len = length(v)
    block_size = @groupsize()[1]

    # NOTE: for many index calculations in this library, computation using zero-indexing leads to
    # fewer operations (also code is transpiled to CUDA / ROCm / oneAPI / Metal code which do zero
    # indexing). Internal calculations will be done using zero indexing except when actually
    # accessing memory. As with C, the lower bound is inclusive, the upper bound exclusive.

    # Group (block) and local (thread) indices
    iblock = @index(Group, Linear) - 1 + 1              # Skipping first block
    ithread = @index(Local, Linear) - 1
    block_offset = iblock * block_size * 2              # Processing two elements per thread

    # Each block looks back to find running prefix sum
    running_prefix = init
    inspected_block = iblock - 1
    while inspected_block >= 0

        # Opportunistic: a previous block finished everything
        if flags[inspected_block + 1] == ACC_FLAG_A
            # Previous blocks (except last) always have filled values in v, so index is inbounds
            running_prefix = op(running_prefix, v[(inspected_block + 1) * block_size * 2])
            break
        else
            running_prefix = op(running_prefix, prefixes[inspected_block + 1])
        end

        inspected_block -= 1
    end

    # Now we have aggregate prefix of all previous blocks, add it to all our elements
    ai = ithread
    if block_offset + ai < len
        v[block_offset + ai + 1] = op(running_prefix, v[block_offset + ai + 1])
    end

    bi = ithread + block_size
    if block_offset + bi < len
        v[block_offset + bi + 1] = op(running_prefix, v[block_offset + bi + 1])
    end

    # Set flag for "aggregate of all prefixes up to this block finished"
    @synchronize()      # This is needed so that the flag is not set before copying into v, but
                        # there should be better memory fences to guarantee ordering without
                        # thread synchronization...
    if ithread == 0
        flags[iblock + 1] = ACC_FLAG_A
    end
end


function accumulate!(
    op,
    v::AbstractGPUVector;
    init,
    inclusive::Bool=true,

    block_size::Int=128,
    temp::Union{Nothing, AbstractGPUVector}=nothing,
    temp_flags::Union{Nothing, AbstractGPUVector}=nothing,
)

    # Correctness checks
    @argcheck block_size > 0
    @argcheck ispow2(block_size)

    # Nothing to accumulate
    if length(v) == 0
        return v
    end

    # Each thread will process two elements
    elems_per_block = block_size * 2
    num_blocks = (length(v) + elems_per_block - 1) ÷ elems_per_block

    if isnothing(temp)
        prefixes = similar(v, eltype(v), num_blocks)
    else
        @argcheck eltype(temp) === eltype(v)
        @argcheck length(temp) >= num_blocks
        prefixes = temp
    end

    if isnothing(temp_flags)
        flags = similar(v, Int8, num_blocks)
    else
        @argcheck eltype(temp_flags) <: Integer
        @argcheck length(temp_flags) >= num_blocks
        flags = temp_flags
    end

    backend = get_backend(v)
    kernel1! = _accumulate_block!(backend, block_size)
    kernel1!(op, v, init, inclusive, flags, prefixes,
             ndrange=num_blocks * block_size)

    if num_blocks > 1
        kernel2! = _accumulate_previous!(backend, block_size)
        kernel2!(op, v, init, flags, prefixes,
                 ndrange=(num_blocks - 1) * block_size)
    end

    return v
end


function accumulate!(
    op,
    v::AbstractVector;
    init,
    inclusive::Bool=true,

    block_size::Int=128,
    temp::Union{Nothing, AbstractVector}=nothing,
    temp_flags::Union{Nothing, AbstractVector}=nothing,
)
    # Simple single-threaded CPU implementation - FIXME: implement taccumulate in OhMyThreads.jl
    if length(v) == 0
        return v
    end
    if inclusive
        running = v[begin]
        for i in firstindex(v) + 1:lastindex(v)
            running = op(running, v[i])
            v[i] = running
        end
    else
        running = init
        for i in eachindex(v)
            v[i], running = running, op(running, v[i])
        end
    end
    return v
end


function accumulate(
    op,
    v::AbstractVector;
    init,
    inclusive::Bool=true,

    block_size::Int=128,
    temp::Union{Nothing, AbstractVector}=nothing,
    temp_flags::Union{Nothing, AbstractVector}=nothing,
)
    vcopy = copy(v)
    accumulate!(op, vcopy; init=init, inclusive=inclusive,
                block_size=block_size, temp=temp, temp_flags=temp_flags)
    vcopy
end

