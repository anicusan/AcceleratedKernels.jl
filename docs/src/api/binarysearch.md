### Binary Search

Find the indices where some elements `x` should be inserted into a sorted sequence `v` to maintain the sorted order. Effectively applying the Julia.Base functions in parallel on a GPU using `foreachindex`.
- `searchsortedfirst!` (in-place), `searchsortedfirst` (allocating): index of first element in `v` >= `x[j]`.
- `searchsortedlast!`, `searchsortedlast`: index of last element in `v` <= `x[j]`.
- **Other names**: `thrust::upper_bound`, `std::lower_bound`.

Function signature:
```julia
# GPU
searchsortedfirst!(ix::AbstractGPUVector, v::AbstractGPUVector, x::AbstractGPUVector;
                   by=identity, lt=(<), rev::Bool=false,
                   block_size::Int=256)
searchsortedfirst(v::AbstractGPUVector, x::AbstractGPUVector;
                  by=identity, lt=(<), rev::Bool=false,
                  block_size::Int=256)
searchsortedlast!(ix::AbstractGPUVector, v::AbstractGPUVector, x::AbstractGPUVector;
                  by=identity, lt=(<), rev::Bool=false,
                  block_size::Int=256)
searchsortedlast(v::AbstractGPUVector, x::AbstractGPUVector;
                 by=identity, lt=(<), rev::Bool=false,
                 block_size::Int=256)

# CPU
searchsortedfirst!(ix::AbstractVector, v::AbstractVector, x::AbstractVector;
                   by=identity, lt=(<), rev::Bool=false,
                   max_tasks::Int=Threads.nthreads(), min_elems::Int=1000)
searchsortedfirst(v::AbstractVector, x::AbstractVector;
                  by=identity, lt=(<), rev::Bool=false,
                  max_tasks::Int=Threads.nthreads(), min_elems::Int=1000)
searchsortedlast!(ix::AbstractVector, v::AbstractVector, x::AbstractVector;
                  by=identity, lt=(<), rev::Bool=false,
                  max_tasks::Int=Threads.nthreads(), min_elems::Int=1000)
searchsortedlast(v::AbstractVector, x::AbstractVector;
                 by=identity, lt=(<), rev::Bool=false,
                 max_tasks::Int=Threads.nthreads(), min_elems::Int=1000)
```

Example:
```julia
import AcceleratedKernels as AK
using Metal

# Sorted array
v = MtlArray(rand(Float32, 100_000))
AK.merge_sort!(v)

# Elements `x` to place within `v` at indices `ix`
x = MtlArray(rand(Float32, 10_000))
ix = MtlArray{Int}(undef, 10_000)

AK.searchsortedfirst!(ix, v, x)
```
