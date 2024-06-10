

# NOTE: for many index calculations in this library, computation using zero-indexing leads to
# fewer operations (also code is transpiled to CUDA / ROCm / oneAPI / Metal code which do zero
# indexing). Internal calculations will be done using zero indexing except when actually
# accessing memory. As with C, the lower bound is inclusive, the upper bound exclusive.
function _lower_bound_s0(arr, value, left=0, right=length(arr), comp=(<))
    right <= 0 && return 0
    comp(arr[right], value) && return right
    while right > left + 1
        mid = left + ((right - left) >> 1)
        if comp(arr[mid], value)
            left = mid
        else
            right = mid
        end
    end
    return left
end


function _upper_bound_s0(arr, value, left=0, right=length(arr), comp=(<))
    right <= 0 && return 0
    comp(value, arr[1]) && return 0
    while right > left + 1
        mid = left + ((right - left) >> 1)
        if comp(value, arr[mid + 1])
            right = mid
        else
            left = mid
        end
    end
    return right
end



@kernel cpu=false function _merge_sort_block!(vec, comp)

    T = eltype(vec)
    len = length(vec)

    N = @uniform @groupsize()[1]
    s_buf = @localmem eltype(vec) (N * 2,)

    # NOTE: for many index calculations in this library, computation using zero-indexing leads to
    # fewer operations (also code is transpiled to CUDA / ROCm / oneAPI / Metal code which do zero
    # indexing). Internal calculations will be done using zero indexing except when actually
    # accessing memory. As with C, the lower bound is inclusive, the upper bound exclusive.

    # Group (block) and local (thread) indices
    iblock = @index(Group, Linear) - 1
    ithread = @index(Local, Linear) - 1

    i = ithread + iblock * N * 2
    if i < len
        s_buf[ithread + 1] = vec[i + 1]
    end

    i = ithread + N + iblock * N * 2
    if i < len
        s_buf[ithread + N + 1] = vec[i + 1]
    end

    @synchronize()

    half_size_group = 1
    size_group = 2

    niter = 0

    while half_size_group <= N
        gid = ithread ÷ half_size_group

        local v1::T
        local v2::T
        pos1 = typemax(Int)
        pos2 = typemax(Int)

        i = gid * size_group + half_size_group + iblock * N * 2
        if i < len
            tid = gid * size_group + ithread ÷ half_size_group
            v1 = s_buf[tid + 1]
            p_search = @view s_buf[gid * size_group + half_size_group + 1:end]
            i = (gid + 1) * size_group + iblock * N * 2
            n = i < len ? half_size_group : len - iblock * N * 2 - gid * size_group - half_size_group
            pos1 = ithread ÷ half_size_group + _lower_bound_s0(p_search, v1, 0, n, comp)
        end

        # if ithread == 0 && iblock == 0
        #     @print("s_buf (ithread ", ithread, " niter ", niter, ")\n")
        #     for i in 1:N * 2
        #         @print(unsafe_trunc(Int, s_buf[i]), " ")
        #     end
        #     @print("\n")
        #     @print("gid ", gid, " | tid ", tid, " | v1 ", v1, " | move_to ", gid * size_group + pos1 + 1, "\n\n")
        # end

        tid = gid * size_group + half_size_group + ithread ÷ half_size_group
        i = tid + iblock * N * 2
        if i < len
            v2 = s_buf[tid + 1]
            p_search = @view s_buf[gid * size_group + 1:end]
            pos2 = ithread ÷ half_size_group + _upper_bound_s0(p_search, v2, 0, half_size_group, comp)
        end

        @synchronize()

        if pos1 != typemax(Int)
            s_buf[gid * size_group + pos1 + 1] = v1
        end
        if pos2 != typemax(Int)
            s_buf[gid * size_group + pos2 + 1] = v2
        end

        @synchronize()

        half_size_group = half_size_group << 1
        size_group = size_group << 1

        niter += 1
    end

    # Debug: print s_buf
    # @print("ithread ", ithread, " | iblock ", iblock, " | s_buf[ithread + 1] ", s_buf[ithread + 1], " | s_buf[ithread + N + 1] ", s_buf[ithread + N + 1], "\n")

    i = ithread + iblock * N * 2
    if i < len
        vec[i + 1] = s_buf[ithread + 1]
    end

    i = ithread + N + iblock * N * 2
    if i < len
        vec[i + 1] = s_buf[ithread + N + 1]
    end
end


@kernel cpu=false function _merge_sort_global!(vec_out, vec_in, comp, half_size_group)

    T = eltype(vec_in)
    len = length(vec_in)

    N = @uniform @groupsize()[1]

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
    pos_in = gid * size_group + idx ÷ half_size_group
    left = gid * size_group + half_size_group
    right = (gid + 1) * size_group
    if right > len
        right = len
    end

    pos_out = pos_in + _lower_bound_s0(vec_in, vec_in[pos_in + 1], left, right, comp) - left
    vec_out[pos_out + 1] = vec_in[pos_in + 1]

    # Right half
    pos_in = gid * size_group + half_size_group + idx ÷ half_size_group
    if pos_in < len
        left = gid * size_group
        right = left + half_size_group
        pos_out = pos_in - half_size_group + _upper_bound_s0(vec_in, vec_in[pos_in + 1], left, right, comp) - left
        vec_out[pos_out + 1] = vec_in[pos_in + 1]
    end
end



function merge_sort!(vec, comp=(<))


    block_size = 16
    blocks = (length(vec) + block_size * 2 - 1) ÷ (block_size * 2)

    backend = get_backend(vec)

    # Block level
    _merge_sort_block!(backend, block_size)(vec, comp, ndrange=(block_size * blocks,))
    synchronize(backend)

    # Global level
    half_size_group = block_size * 2
    size_group = half_size_group * 2
    len = length(vec)
    if len > half_size_group
        p1 = vec
        p2 = similar(vec)

        kernel! = _merge_sort_global!(backend, block_size)

        niter = 0
        while len > half_size_group
            blocks = ((len + half_size_group - 1) ÷ half_size_group ÷ 2) * (half_size_group ÷ block_size)
            kernel!(p2, p1, comp, half_size_group, ndrange=(blocks * block_size,))
            synchronize(backend)

            half_size_group = half_size_group << 1;
            size_group = size_group << 1;
            p1, p2 = p2, p1

            niter += 1
        end

        display([p1 p2])
        
        if isodd(niter)
            vec .= p1
        end
    end

    nothing
end



