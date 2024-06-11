@kernel function _foreachindex_global!(f, indices)
    i = @index(Global, Linear)
    itr_index = @inbounds indices[i]
    f(itr_index)
end


function foreachindex(
    f,
    itr,
    backend::GPU;

    block_size::Int=256,
)
    # GPU implementation
    @assert block_size > 0
    _foreachindex_global!(backend, block_size)(f, eachindex(itr), ndrange=length(itr))
    synchronize(backend)
    nothing
end


function foreachindex(
    f,
    itr,
    backend::CPU;

    max_tasks=Threads.nthreads(),
    min_elems=1,
)
    # Fallback CPU implementation
    task_partition(length(itr), max_tasks, min_elems) do irange
        itr_indices = eachindex(itr)
        for i in irange
            @inline f(itr_indices[i])
        end
    end

    nothing
end


"""
    foreachindex(f, itr, [backend::GPU]; block_size::Int=256)
    foreachindex(f, itr, [backend::CPU]; max_tasks=Threads.nthreads(), min_elems=1)

Parallelised `for` loop over the indices of an iterable. It allows you to run normal Julia code on
a GPU over multiple e.g. CuArray, ROCArray, MtlArray, oneArray.

# Examples
Normally you would write a for loop like this:
```julia
x = Array(1:100)
y = similar(x)
for i in eachindex(x)
    @inbounds y[i] = 2 * x[i] + 1
end
```

Using this function you can have the same for loop body over a GPU array:
```julia
using CUDA
x = CuArray(1:100)
y = similar(x)
foreachindex(x) do i
    @inbounds y[i] = 2 * x[i] + 1
end
```

Note that the above code is pure arithmetic, which you can write directly (and on some platforms
it may be faster) as:
```julia
using CUDA
x = CuArray(1:100)
y = 2 .* x .+ 1
```

This construct is also optimised for CPU threads.
"""
function foreachindex(f, itr, backend=get_backend(itr); kwargs...)
    @assert backend isa Backend
    foreachindex(f, itr, backend; kwargs...)
end

