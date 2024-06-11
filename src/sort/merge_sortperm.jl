function merge_sortperm!(
    ix::AbstractGPUVector,
    v::AbstractGPUVector;

    lt=(<),
    by=identity,
    rev::Bool=false,
    order::Base.Order.Ordering=Base.Order.Forward,

    inplace::Bool=false,
    block_size::Int=128,
    temp_ix::Union{Nothing, AbstractGPUVector}=nothing,
    temp_v::Union{Nothing, AbstractGPUVector}=nothing,
)
    # Simple sanity checks
    @assert block_size > 0
    @assert length(ix) == length(v)
    if !isnothing(temp_ix)
        @assert length(temp_ix) == length(ix)
        @assert eltype(temp_ix) === eltype(ix)
    end

    if !isnothing(temp_v)
        @assert length(temp_v) == length(v)
        @assert eltype(temp_v) === eltype(v)
    end

    # Initialise indices that will be sorted by the keys in v
    ix .= 1:length(ix)
    keys = inplace ? v : copy(v)

    merge_sort_by_key!(
        keys, ix;
        lt=lt, by=by, rev=rev, order=order,
        block_size=block_size,
        temp_keys=temp_v, temp_values=temp_ix,
    )
end


function merge_sortperm(
    v::AbstractGPUVector;

    lt=(<),
    by=identity,
    rev::Bool=false,
    order::Base.Order.Ordering=Base.Order.Forward,

    inplace::Bool=false,
    block_size::Int=128,
    temp_ix::Union{Nothing, AbstractGPUVector}=nothing,
    temp_v::Union{Nothing, AbstractGPUVector}=nothing,
)
    ix = similar(v, Int)
    merge_sortperm!(
        ix, v,
        lt=lt, by=by, rev=rev, order=order,
        inplace=inplace, block_size=block_size, temp_ix=temp_ix, temp_v=temp_v,
    )
    ix
end

