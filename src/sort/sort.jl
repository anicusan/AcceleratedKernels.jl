include("utils.jl")
include("merge_sort.jl")
include("merge_sort_by_key.jl")
include("merge_sortperm.jl")


# All other algorithms have the same naming convention as Julia Base ones; provide similar
# interface here too. Maybe include a CPU parallel merge sort with each thread using the Julia
# Base radix sort before merging in parallel. We are shadowing the Base definitions, should we not?
# Should we add an `alg` keyword argument like the Base one? I think we can leave that until we
# have multiple sorting algorithms; it would not be a breaking change.
function sort!(
    v::AbstractGPUVector;

    lt=isless,
    by=identity,
    rev::Bool=false,
    order::Base.Order.Ordering=Base.Order.Forward,

    block_size::Int=128,
    temp::Union{Nothing, AbstractGPUVector}=nothing,
)
    merge_sort!(
        v,
        lt=lt, by=by, rev=rev, order=order,
        block_size=block_size, temp=temp,
    )
end


function sort!(
    v::AbstractVector;

    lt=isless,
    by=identity,
    rev::Bool=false,
    order::Base.Order.Ordering=Base.Order.Forward,

    block_size::Int=128,
    temp::Union{Nothing, AbstractVector}=nothing,
)
    # Fallback to Base before we have a CPU parallel sort
    Base.sort!(v; lt=lt, by=by, rev=rev, order=order)
end


function sort(
    v::AbstractGPUVector;

    lt=isless,
    by=identity,
    rev::Bool=false,
    order::Base.Order.Ordering=Base.Order.Forward,

    block_size::Int=128,
    temp::Union{Nothing, AbstractGPUVector}=nothing,
)
    merge_sort(
        v,
        lt=lt, by=by, rev=rev, order=order,
        block_size=block_size, temp=temp,
    )
end


function sort(
    v::AbstractVector;

    lt=isless,
    by=identity,
    rev::Bool=false,
    order::Base.Order.Ordering=Base.Order.Forward,

    block_size::Int=128,
    temp::Union{Nothing, AbstractVector}=nothing,
)
    # Fallback to Base before we have a CPU parallel sort
    Base.sort(v; lt=lt, by=by, rev=rev, order=order)
end


function sortperm!(
    ix::AbstractGPUVector,
    v::AbstractGPUVector;

    lt=isless,
    by=identity,
    rev::Bool=false,
    order::Base.Order.Ordering=Base.Order.Forward,

    block_size::Int=128,
    temp::Union{Nothing, AbstractGPUVector}=nothing,
)
    merge_sortperm_lowmem!(
        ix, v,
        lt=lt, by=by, rev=rev, order=order,
        block_size=block_size, temp=temp,
    )
end


function sortperm!(
    ix::AbstractVector,
    v::AbstractVector;

    lt=isless,
    by=identity,
    rev::Bool=false,
    order::Base.Order.Ordering=Base.Order.Forward,

    block_size::Int=128,
    temp::Union{Nothing, AbstractVector}=nothing,
)
    # Fallback to Base before we have a CPU parallel sortperm
    Base.sortperm!(ix, v; lt=lt, by=by, rev=rev, order=order)
end


function sortperm(
    v::AbstractGPUVector;

    lt=isless,
    by=identity,
    rev::Bool=false,
    order::Base.Order.Ordering=Base.Order.Forward,

    block_size::Int=128,
    temp::Union{Nothing, AbstractGPUVector}=nothing,
)
    merge_sortperm_lowmem(
        v,
        lt=lt, by=by, rev=rev, order=order,
        block_size=block_size, temp=temp,
    )
end


function sortperm(
    v::AbstractVector;

    lt=isless,
    by=identity,
    rev::Bool=false,
    order::Base.Order.Ordering=Base.Order.Forward,

    block_size::Int=128,
    temp::Union{Nothing, AbstractVector}=nothing,
)
    # Fallback to Base before we have a CPU parallel sortperm
    Base.sortperm(v; lt=lt, by=by, rev=rev, order=order)
end
