@kernel inbounds=true cpu=false function _reduce_nd_by_thread!(@Const(src), dst, op, init, dims)

    # One thread per output element, when there are more outer elements than in the reduced dim
    # e.g. reduce(+, rand(3, 1000), dims=1) => only 3 elements in the reduced dim
    src_sizes = size(src)
    src_strides = strides(src)
    dst_sizes = size(dst)
    dst_strides = strides(dst)

    output_size = length(dst)
    reduce_size = src_sizes[dims]

    ndims = length(src_sizes)

    N = @groupsize()[1]

    # NOTE: for many index calculations in this library, computation using zero-indexing leads to
    # fewer operations (also code is transpiled to CUDA / ROCm / oneAPI / Metal code which do zero
    # indexing). Internal calculations will be done using zero indexing except when actually
    # accessing memory. As with C, the lower bound is inclusive, the upper bound exclusive.

    # Group (block) and local (thread) indices
    iblock = @index(Group, Linear) - 0x1
    ithread = @index(Local, Linear) - 0x1

    tid = ithread + iblock * N

    # Each thread handles one output element
    tid = ithread + iblock * N
    if tid < output_size

        # # Sometimes slightly faster method using additional memory with
        # # output_idx = @private typeof(iblock) (ndims,)
        # tmp = tid
        # KernelAbstractions.Extras.@unroll for i in ndims:-1:1
        #     output_idx[i] = tmp ÷ dst_strides[i]
        #     tmp = tmp % dst_strides[i]
        # end
        # # Compute the base index in src (excluding the reduced axis)
        # input_base_idx = 0
        # KernelAbstractions.Extras.@unroll for i in 1:ndims
        #     i == dims && continue
        #     input_base_idx += output_idx[i] * src_strides[i]
        # end

        # Compute the base index in src (excluding the reduced axis)
        input_base_idx = typeof(ithread)(0)
        tmp = tid
        KernelAbstractions.Extras.@unroll for i in ndims:-1:1
            if i != dims
                input_base_idx += (tmp ÷ dst_strides[i]) * src_strides[i]
            end
            tmp = tmp % dst_strides[i]
        end

        # Go over each element in the reduced dimension; this implementation assumes that there
        # are so many outer elements (each processed by an independent thread) that we afford to
        # loop sequentially over the reduced dimension (e.g. reduce(+, rand(3, 1000), dims=1))
        res = init
        for i in 0x0:reduce_size - 0x1
            src_idx = input_base_idx + i * src_strides[dims]
            res = op(res, src[src_idx + 0x1])
        end
        dst[tid + 0x1] = res
    end
end


@kernel inbounds=true cpu=false function _reduce_nd_by_block!(@Const(src), dst, op, init, dims)

    # One block per output element, when there are more elements in the reduced dim than in outer
    # e.g. reduce(+, rand(3, 1000), dims=2) => only 3 elements in outer dimensions
    src_sizes = size(src)
    src_strides = strides(src)
    dst_sizes = size(dst)
    dst_strides = strides(dst)

    output_size = length(dst)
    reduce_size = src_sizes[dims]

    ndims = length(src_sizes)

    N = @groupsize()[1]
    sdata = @localmem eltype(dst) (N,)

    # NOTE: for many index calculations in this library, computation using zero-indexing leads to
    # fewer operations (also code is transpiled to CUDA / ROCm / oneAPI / Metal code which do zero
    # indexing). Internal calculations will be done using zero indexing except when actually
    # accessing memory. As with C, the lower bound is inclusive, the upper bound exclusive.

    # Group (block) and local (thread) indices
    iblock = @index(Group, Linear) - 0x1
    ithread = @index(Local, Linear) - 0x1

    # Each block handles one output element
    if iblock < output_size

        # # Sometimes slightly faster method using additional memory with
        # # output_idx = @private typeof(iblock) (ndims,)
        # tmp = iblock
        # KernelAbstractions.Extras.@unroll for i in ndims:-1:1
        #     output_idx[i] = tmp ÷ dst_strides[i]
        #     tmp = tmp % dst_strides[i]
        # end
        # # Compute the base index in src (excluding the reduced axis)
        # input_base_idx = 0
        # KernelAbstractions.Extras.@unroll for i in 1:ndims
        #     i == dims && continue
        #     input_base_idx += output_idx[i] * src_strides[i]
        # end

        # Compute the base index in src (excluding the reduced axis)
        input_base_idx = typeof(ithread)(0)
        tmp = iblock
        KernelAbstractions.Extras.@unroll for i in ndims:-1:1
            if i != dims
                input_base_idx += (tmp ÷ dst_strides[i]) * src_strides[i]
            end
            tmp = tmp % dst_strides[i]
        end

        # We have a block of threads to process the whole reduced dimension. First do pre-reduction
        # in strides of N
        partial = init
        i = ithread
        while i < reduce_size
            src_idx = input_base_idx + i * src_strides[dims]
            partial = op(partial, src[src_idx + 0x1])
            i += N
        end

        # Store partial result in shared memory; now we are down to a single block to reduce within
        sdata[ithread + 0x1] = partial
        @synchronize()

        # Set integer literals as u16 to ensure no indices are promoted beyond the ithread base type.
        # For example, Metal uses UInt32 indices, but if it is mixed with a Julia integer literal
        # (Int64 by default), we incur a type cast
        if N >= 512u16
            ithread < 256u16 && (sdata[ithread + 0x1] = op(sdata[ithread + 0x1], sdata[ithread + 256u16 + 0x1]))
            @synchronize()
        end
        if N >= 256u16
            ithread < 128u16 && (sdata[ithread + 0x1] = op(sdata[ithread + 0x1], sdata[ithread + 128u16 + 0x1]))
            @synchronize()
        end
        if N >= 128u16
            ithread < 64u16 && (sdata[ithread + 0x1] = op(sdata[ithread + 0x1], sdata[ithread + 64u16 + 0x1]))
            @synchronize()
        end
        if N >= 64u16
            ithread < 32u16 && (sdata[ithread + 0x1] = op(sdata[ithread + 0x1], sdata[ithread + 32u16 + 0x1]))
            @synchronize()
        end
        if N >= 32u16
            ithread < 16u16 && (sdata[ithread + 0x1] = op(sdata[ithread + 0x1], sdata[ithread + 16u16 + 0x1]))
            @synchronize()
        end
        if N >= 16u16
            ithread < 0x8 && (sdata[ithread + 0x1] = op(sdata[ithread + 0x1], sdata[ithread + 0x8 + 0x1]))
            @synchronize()
        end
        if N >= 0x8
            ithread < 0x4 && (sdata[ithread + 0x1] = op(sdata[ithread + 0x1], sdata[ithread + 0x4 + 0x1]))
            @synchronize()
        end
        if N >= 0x4
            ithread < 0x2 && (sdata[ithread + 0x1] = op(sdata[ithread + 0x1], sdata[ithread + 0x2 + 0x1]))
            @synchronize()
        end
        if N >= 0x2
            ithread < 0x1 && (sdata[ithread + 0x1] = op(sdata[ithread + 0x1], sdata[ithread + 0x1 + 0x1]))
            @synchronize()
        end
    
        if ithread == 0x0
            dst[iblock + 0x1] = sdata[0x1]
        end
    end
end


function reduce_nd(
    op, src::AbstractGPUArray;
    init,
    dims::Int,
    block_size::Int=256,
    temp::Union{Nothing, AbstractGPUArray}=nothing,
)
    @argcheck 1 <= block_size <= 1024

    # Degenerate cases begin; order of priority matters

    # Invalid dims
    if dims < 1
        throw(ArgumentError("region dimension(s) must be ≥ 1, got $dims"))
    end

    # If dims > number of dimensions, return copy of src, but with the type of init, e.g.:
    #   julia> x = rand(Float64, 3, 5);
    #   julia> reduce(+, x, dims=3, init=Float32(0))
    #   3×5 Matrix{Float32}
    src_sizes = size(src)
    if dims > length(src_sizes)
        if isnothing(temp)
            dst = similar(src, typeof(init), src_sizes)
        else
            @argcheck size(temp) == src_sizes
            @argcheck eltype(temp) == typeof(init)
            dst = temp
        end
        copyto!(dst, src)
        return dst
    end

    # The per-dimension sizes of the destination array; construct tuple without allocations
    dst_sizes = unrolled_map_index(src_sizes) do i
        i == dims ? 1 : src_sizes[i]
    end

    # If any dimension except dims is zero, return empty similar array except with the dims
    # dimension = 1. Weird, see example below:
    #   julia> x = rand(3, 0, 5);
    #   julia> reduce(+, x, dims=3)
    #   3×0×1 Array{Float64, 3}
    for isize in eachindex(src_sizes)
        isize == dims && continue
        if src_sizes[isize] == 0
            if isnothing(temp)
                dst = similar(src, typeof(init), dst_sizes)
            else
                @argcheck size(temp) == dst_sizes
                @argcheck eltype(temp) == typeof(init)
                dst = temp
            end
            return dst
        end
    end

    # If sizes[dims] == 0, return array filled with init; same shape except sizes[dims] = 1:
    #   julia> x = rand(3, 0, 5);
    #   julia> reduce(+, x, dims=2)
    #   3×1×5 Array{Float64, 3}:
    #   [:, :, 1] =
    #    0.0
    #    0.0
    #    0.0
    #   [...]
    len = src_sizes[dims]
    if len == 0
        if isnothing(temp)
            dst = similar(src, typeof(init), dst_sizes)
        else
            @argcheck size(temp) == dst_sizes
            @argcheck eltype(temp) == typeof(init)
            dst = temp
        end
        fill!(dst, init)
        return dst
    end

    # If sizes[dims] == 1, return same array; Base.reduce returns a copy, so let's keep the same
    # semantics. Again, keep same type as init
    if len == 1
        if isnothing(temp)
            dst = similar(src, typeof(init), src_sizes)
        else
            @argcheck size(temp) == src_sizes
            @argcheck eltype(temp) == typeof(init)
            dst = temp
        end
        copyto!(dst, src)
        return dst
    end

    # Degenerate cases end

    # Allocate destination array
    if isnothing(temp)
        dst = similar(src, typeof(init), dst_sizes)
    else
        @argcheck size(temp) == dst_sizes
        @argcheck eltype(temp) == typeof(init)
        dst = temp
    end
    dst_size = length(dst)

    # We have two parallelisation approaches, based on which dimension has more elements:
    #   - If the dimension we are reducing has more elements, (e.g. reduce(+, rand(3, 1000), dims=2)),
    #     we use a block of threads per dst element - thus, a block of threads reduces the dims axis
    #   - If the other dimensions have more elements (e.g. reduce(+, rand(3, 1000), dims=1)), we
    #     use a single thread per dst element - thus, a thread reduces the dims axis sequentially,
    #     while the other dimensions are processed in parallel, independently
    backend = get_backend(dst)
    if dst_size >= src_sizes[dims]
        blocks = (dst_size + block_size - 1) ÷ block_size
        kernel! = _reduce_nd_by_thread!(backend, block_size)
        kernel!(
            src, dst, op, init, dims,
            ndrange=(block_size * blocks,),
        )
    else
        # One block per output element
        blocks = dst_size
        kernel! = _reduce_nd_by_block!(backend, block_size)
        kernel!(
            src, dst, op, init, dims,
            ndrange=(block_size * blocks,),
        )
    end

    return dst
end
