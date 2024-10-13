# TODO: this hangs / dies on oneAPI. Test on CUDA
@kernel cpu=false inbounds=true function _any_global!(out, pred, @Const(v))
    temp = @localmem Int8 (1,)
    i = @index(Global, Linear)

    # Technically this is a race, but it doesn't matter as all threads would write the same value.
    # For example, CUDA F4.2 says "If a non-atomic instruction executed by a warp writes to the
    # same location in global memory for more than one of the threads of the warp, only one thread
    # performs a write and which thread does it is undefined."
    temp[1] = 0
    @synchronize()

    # The ndrange check already protects us from out of bounds access
    if pred(v[i])
        temp[1] = 1
    end

    @synchronize()
    if temp[1] != 0
        out[1] = 1
    end
end


function any(
    pred,
    v::AbstractGPUVector;
    block_size::Int=256,
    cooperative::Bool=true,
)
    @argcheck block_size > 0

    # Some platforms crash when multiple threads write to the same memory location in a global
    # array (e.g. old Intel Graphics); if it is the same value, it is well-defined on others (e.g.
    # CUDA). If not cooperative, we need to do a mapreduce
    if cooperative
        backend = get_backend(v)
        out = KernelAbstractions.zeros(backend, Int8, 1)
        _any_global!(backend, block_size)(out, pred, v, ndrange=length(v))
        outh = Array(out)
        return outh[1] == 0 ? false : true
    else
        return mapreduce(
            pred,
            (x, y) -> x || y,
            v;
            init=false,
            block_size=block_size,
        )
    end
end


function all(
    pred,
    v::AbstractGPUVector;
    block_size::Int=256,
    cooperative::Bool=true,
)
    @argcheck block_size > 0

    # Some platforms crash when multiple threads write to the same memory location in a global
    # array (e.g. old Intel Graphics); if it is the same value, it is well-defined on others (e.g.
    # CUDA). If not cooperative, we need to do a mapreduce
    if cooperative
        backend = get_backend(v)
        out = KernelAbstractions.zeros(backend, Int8, 1)
        _any_global!(backend, block_size)(out, (!pred), v, ndrange=length(v))
        outh = Array(out)
        return outh[1] == 0 ? true : false
    else
        return mapreduce(
            pred,
            (x, y) -> x && y,
            v;
            init=true,
            block_size=block_size,
        )
    end
end

