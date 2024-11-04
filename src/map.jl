function map!(
    f, dst::AbstractArray, src::AbstractArray;

    # CPU settings
    scheduler=:threads,
    max_tasks=Threads.nthreads(),
    min_elems=1,

    # GPU settings
    block_size=256,    
)
    @argcheck length(dst) == length(src)
    foreachindex(
        src,
        scheduler=scheduler,
        max_tasks=max_tasks,
        min_elems=min_elems,
        block_size=block_size,
    ) do idx
        dst[idx] = f(src[idx])
    end
    dst
end


function map(
    f, src::AbstractArray;

    # CPU settings
    scheduler=:threads,
    max_tasks=Threads.nthreads(),
    min_elems=1,

    # GPU settings
    block_size=256,
)
    dst = similar(src)
    map!(
        f, dst, src,
        scheduler=scheduler,
        max_tasks=max_tasks,
        min_elems=min_elems,
        block_size=block_size,
    )
end
