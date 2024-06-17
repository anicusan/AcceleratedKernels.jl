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

    scheduler=:polyester,
    max_tasks=Threads.nthreads(),
    min_elems=1,
)
    # CPU implementation
    if scheduler === :threads
        task_partition(length(itr), max_tasks, min_elems) do irange
            itr_indices = eachindex(itr)
            for i in irange
                @inbounds itr_index = itr_indices[i]
                @inline f(itr_index)
            end
        end
    elseif scheduler === :polyester
        @batch minbatch=min_elems per=thread for i in eachindex(itr)
            @inline f(i)
        end
    else
        throw(ArgumentError("`scheduler` must be `:threads` or `:polyester`. Received $scheduler"))
    end

    nothing
end


"""
    foreachindex(f, itr, [backend::GPU]; block_size::Int=256)
    foreachindex(f, itr, [backend::CPU]; scheduler=:polyester, max_tasks=Threads.nthreads(), min_elems=1)
    foreachindex(f, itr, backend=get_backend(itr); kwargs...)

Parallelised `for` loop over the indices of an iterable.

It allows you to run normal Julia code on a GPU over multiple arrays - e.g. CuArray, ROCArray,
MtlArray, oneArray - with one GPU thread per index.

On CPUs at most `max_tasks` threads are launched, or fewer such that each thread processes at least
`min_elems` indices; if a single task ends up being needed, `f` is inlined and no thread is
launched. Tune it to your function - the more expensive it is, the fewer elements are needed to
amortise the cost of launching a thread (which is a few μs). The scheduler can be `:polyester`
to use Polyester.jl cheap threads or `:threads` to use normal Julia threads.

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
const x = CuArray(1:100)
const y = similar(x)
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

**Important note**: to use this function on a GPU, the objects referenced inside the loop body must
have known types - i.e. be inside a function, or `const` global objects; but you shouldn't use
global objects anyways. For example:
```julia
using oneAPI

x = oneArray(1:100)

# CRASHES - typical error message: "Reason: unsupported dynamic function invocation"
# foreachindex(x) do i
#     x[i] = i
# end

function somecopy!(v)
    # Because it is inside a function, the type of `v` will be known
    foreachindex(v) do i
        v[i] = i
    end
end

somecopy!(x)    # This works
```
"""
function foreachindex(f, itr, backend=get_backend(itr); kwargs...)
    @assert backend isa Backend     # To avoid calling this function recursively
    foreachindex(f, itr, backend; kwargs...)
end

