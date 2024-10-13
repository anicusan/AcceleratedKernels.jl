# Taken from julia.Base to ensure consistent results with the Base CPU version
# License is MIT: https://julialang.org/license
function _searchsortedfirst(v, x, lo::T, hi::T, comp) where T<:Integer
    hi = hi + T(1)
    len = hi - lo
    @inbounds while len != 0
        half_len = len >>> 0x01
        m = lo + half_len
        if comp(v[m], x)
            lo = m + 1
            len -= half_len + 1
        else
            hi = m
            len = half_len
        end
    end
    return lo
end


function _searchsortedlast(v, x, lo::T, hi::T, comp) where T<:Integer
    u = T(1)
    lo = lo - u
    hi = hi + u
    @inbounds while lo < hi - u
        m = lo + ((hi - lo) >>> 0x01)
        if comp(x, v[m])
            hi = m
        else
            lo = m
        end
    end
    return lo
end


function searchsortedfirst!(
    ix::AbstractVector,
    v::AbstractVector,
    x::AbstractVector;

    by=identity, lt=isless, rev::Bool=false,

    # CPU settings
    scheduler=:threads,
    max_tasks::Int=Threads.nthreads(),
    min_elems::Int=1000,

    # GPU settings
    block_size::Int=256,
)
    # Simple sanity checks
    @argcheck length(ix) == length(x)

    # Construct comparator
    ord = Base.Order.ord(lt, by, rev)
    comp = (x, y) -> Base.Order.lt(ord, x, y)

    foreachindex(
        x,
        scheduler=scheduler, max_tasks=max_tasks, min_elems=min_elems,
        block_size=block_size,
    ) do i
        @inbounds ix[i] = _searchsortedfirst(v, x[i], firstindex(v), lastindex(v), comp)
    end
end


function searchsortedfirst(
    v::AbstractVector,
    x::AbstractVector;

    by=identity, lt=isless, rev::Bool=false,

    # CPU settings
    scheduler=:threads,
    max_tasks::Int=Threads.nthreads(),
    min_elems::Int=1000,

    # GPU settings
    block_size::Int=256,
)
    ix = similar(x, Int)
    searchsortedfirst!(
        ix, v, x;
        by=by, lt=lt, rev=rev,
        scheduler=scheduler, max_tasks=max_tasks, min_elems=min_elems,
        block_size=block_size,
    )
    ix
end


function searchsortedlast!(
    ix::AbstractVector,
    v::AbstractVector,
    x::AbstractVector;

    by=identity, lt=isless, rev::Bool=false,

    # CPU settings
    scheduler=:threads,
    max_tasks::Int=Threads.nthreads(),
    min_elems::Int=1000,

    # GPU settings
    block_size::Int=256,
)

    # Simple sanity checks
    @argcheck block_size > 0
    @argcheck length(ix) == length(x)

    # Construct comparator
    ord = Base.Order.ord(lt, by, rev)
    comp = (x, y) -> Base.Order.lt(ord, x, y)

    foreachindex(
        x,
        scheduler=scheduler, max_tasks=max_tasks, min_elems=min_elems,
        block_size=block_size,
    ) do i
        @inbounds ix[i] = _searchsortedlast(v, x[i], firstindex(v), lastindex(v), comp)
    end
end


function searchsortedlast(
    v::AbstractVector,
    x::AbstractVector;

    by=identity, lt=isless, rev::Bool=false,

    # CPU settings
    scheduler=:threads,
    max_tasks::Int=Threads.nthreads(),
    min_elems::Int=1000,

    # GPU settings
    block_size::Int=256,
)
    ix = similar(x, Int)
    searchsortedlast!(
        ix, v, x;
        by=by, lt=lt, rev=rev,
        scheduler=scheduler, max_tasks=max_tasks, min_elems=min_elems,
        block_size=block_size,
    )
    ix
end

