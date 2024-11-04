include("utils.jl")
include("reduce_1d.jl")
include("reduce_nd.jl")
include("mapreduce_1d.jl")
include("mapreduce_nd.jl")


"""
    reduce(
        op, src::AbstractGPUArray;
        init,
        dims::Union{Nothing, Int}=nothing,

        block_size::Int=256,
        temp::Union{Nothing, AbstractGPUArray}=nothing,
        switch_below::Int=0,
    )

Reduce `src` along dimensions `dims` using the binary operator `op`. If `dims` is `nothing`, reduce
`src` to a scalar. If `dims` is an integer, reduce `src` along that dimension. The `init` value is
used as the initial value for the reduction.

The `block_size` parameter controls the number of threads per block.

The `temp` parameter can be used to pass a pre-allocated temporary array. For reduction to a scalar
(`dims=nothing`), `length(temp) >= 2 * (length(src) + 2 * block_size - 1) ÷ (2 * block_size)` is
required. For reduction along a dimension (`dims` is an integer), `temp` is used as the destination
array, and thus must have the exact dimensions required - i.e. same dimensionwise sizes as `src`,
except for the reduced dimension which becomes 1; there are some corner cases when one dimension is
zero, check against `Base.reduce` for CPU arrays for exact behavior.

The `switch_below` parameter controls the threshold below which the reduction is performed on the
CPU and is only used for 1D reductions (i.e. `dims=nothing`).

# Example
Computing a sum, reducing down to a scalar that is copied to host:
```julia
import AcceleratedKernels as AK
using CUDA

v = CuArray{Int16}(rand(1:1000, 100_000))
vsum = AK.reduce((x, y) -> x + y, v; init=zero(eltype(v)))
```

Computing dimensionwise sums in a 2D matrix:
```julia
import AcceleratedKernels as AK
using Metal

m = MtlArray(rand(Int32(1):Int32(100), 10, 100_000))
mrowsum = AK.reduce(+, m; init=zero(eltype(m)), dims=1)
mcolsum = AK.reduce(+, m; init=zero(eltype(m)), dims=2)
```
"""
function reduce(
    op, src::AbstractGPUArray;
    init,
    dims::Union{Nothing, Int}=nothing,

    block_size::Int=256,
    temp::Union{Nothing, AbstractGPUArray}=nothing,
    switch_below::Int=0,
)
    if isnothing(dims)
        return reduce_1d(
            op, src;
            init=init,
            block_size=block_size,
            temp=temp,
            switch_below=switch_below,
        )
    else
        return reduce_nd(
            op, src;
            init=init,
            dims=dims,
            block_size=block_size,
            temp=temp,
        )
    end
end


function reduce(
    op, src::AbstractArray;
    init,
    dims::Union{Nothing, Int}=nothing,
)
    # Fallback to Base
    if isnothing(dims)
        return Base.reduce(op, src; init=init)
    else
        return Base.reduce(op, src; init=init, dims=dims)
    end
end


"""
    mapreduce(
        f, op, src::AbstractGPUArray;
        init,
        dims::Union{Nothing, Int}=nothing,

        block_size::Int=256,
        temp::Union{Nothing, AbstractGPUArray}=nothing,
        switch_below::Int=0,
    )

Reduce `src` along dimensions `dims` using the binary operator `op` after applying `f` elementwise.
If `dims` is `nothing`, reduce `src` to a scalar. If `dims` is an integer, reduce `src` along that
dimension. The `init` value is used as the initial value for the reduction (i.e. after mapping).

The `block_size` parameter controls the number of threads per block.

The `temp` parameter can be used to pass a pre-allocated temporary array. For reduction to a scalar
(`dims=nothing`), `length(temp) >= 2 * (length(src) + 2 * block_size - 1) ÷ (2 * block_size)` is
required. For reduction along a dimension (`dims` is an integer), `temp` is used as the destination
array, and thus must have the exact dimensions required - i.e. same dimensionwise sizes as `src`,
except for the reduced dimension which becomes 1; there are some corner cases when one dimension is
zero, check against `Base.reduce` for CPU arrays for exact behavior.

The `switch_below` parameter controls the threshold below which the reduction is performed on the
CPU and is only used for 1D reductions (i.e. `dims=nothing`).

# Example
Computing a sum of squares, reducing down to a scalar that is copied to host:
```julia
import AcceleratedKernels as AK
using CUDA

v = CuArray{Int16}(rand(1:1000, 100_000))
vsumsq = AK.mapreduce(x -> x * x, (x, y) -> x + y, v; init=zero(eltype(v)))
```

Computing dimensionwise sums of squares in a 2D matrix:
```julia
import AcceleratedKernels as AK
using Metal

f(x) = x * x
m = MtlArray(rand(Int32(1):Int32(100), 10, 100_000))
mrowsumsq = AK.mapreduce(f, +, m; init=zero(eltype(m)), dims=1)
mcolsumsq = AK.mapreduce(f, +, m; init=zero(eltype(m)), dims=2)
```
"""
function mapreduce(
    f, op, src::AbstractGPUArray;
    init,
    dims::Union{Nothing, Int}=nothing,

    block_size::Int=256,
    temp::Union{Nothing, AbstractGPUArray}=nothing,
    switch_below::Int=0,
)
    if isnothing(dims)
        return mapreduce_1d(
            f, op, src;
            init=init,
            block_size=block_size,
            temp=temp,
            switch_below=switch_below,
        )
    else
        return mapreduce_nd(
            f, op, src;
            init=init,
            dims=dims,
            block_size=block_size,
            temp=temp,
        )
    end
end


function mapreduce(
    f, op, src::AbstractArray;
    init,
    dims::Union{Nothing, Int}=nothing,
)
    # Fallback to Base
    if isnothing(dims)
        return Base.mapreduce(f, op, src; init=init)
    else
        return Base.mapreduce(f, op, src; init=init, dims=dims)
    end
end
