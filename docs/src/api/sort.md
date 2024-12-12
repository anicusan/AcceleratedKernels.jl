###  `sort` and friends

Sorting algorithms with similar interface and default settings as the Julia Base ones, on GPUs:
- `sort!` (in-place), `sort` (out-of-place)
- `sortperm!`, `sortperm`
- **Other names**: `sort`, `sort_team`, `sort_team_by_key`, `stable_sort` or variations in Kokkos, RAJA, Thrust that I know of.

Function signature:
```julia
sort!(v::AbstractGPUVector;
      lt=isless, by=identity, rev::Bool=false, order::Base.Order.Ordering=Base.Order.Forward,
      block_size::Int=256, temp::Union{Nothing, AbstractGPUVector}=nothing)

sortperm!(ix::AbstractGPUVector, v::AbstractGPUVector;
          lt=isless, by=identity, rev::Bool=false, order::Base.Order.Ordering=Base.Order.Forward,
          block_size::Int=256, temp::Union{Nothing, AbstractGPUVector}=nothing)
```

Specific implementations that the interfaces above forward to:
- `merge_sort!` (in-place), `merge_sort` (out-of-place) - sort arbitrary objects with custom comparisons.
- `merge_sort_by_key!`, `merge_sort_by_key` - sort a vector of keys along with a "payload", a vector of corresponding values.
- `merge_sortperm!`, `merge_sortperm`, `merge_sortperm_lowmem!`, `merge_sortperm_lowmem` - compute a sorting index permutation. 

Function signature:
```julia
merge_sort!(v::AbstractGPUVector;
            lt=(<), by=identity, rev::Bool=false, order::Ordering=Forward,
            block_size::Int=256, temp::Union{Nothing, AbstractGPUVector}=nothing)

merge_sort_by_key!(keys::AbstractGPUVector, values::AbstractGPUVector;
                   lt=(<), by=identity, rev::Bool=false, order::Ordering=Forward,
                   block_size::Int=256,
                   temp_keys::Union{Nothing, AbstractGPUVector}=nothing,
                   temp_values::Union{Nothing, AbstractGPUVector}=nothing)

merge_sortperm!(ix::AbstractGPUVector, v::AbstractGPUVector;
                lt=(<), by=identity, rev::Bool=false, order::Ordering=Forward,
                inplace::Bool=false, block_size::Int=256,
                temp_ix::Union{Nothing, AbstractGPUVector}=nothing,
                temp_v::Union{Nothing, AbstractGPUVector}=nothing)

merge_sortperm_lowmem!(ix::AbstractGPUVector, v::AbstractGPUVector;
                       lt=(<), by=identity, rev::Bool=false, order::Ordering=Forward,
                       block_size::Int=256,
                       temp::Union{Nothing, AbstractGPUVector}=nothing)
```

Example:
```julia
import AcceleratedKernels as AK
using AMDGPU

v = ROCArray(rand(Int32, 100_000))
AK.sort!(v)
```

As GPU memory is more expensive, all functions in AcceleratedKernels.jl expose any temporary arrays they will use (the `temp` argument); you can supply your own buffers to make the algorithms not allocate additional GPU storage, e.g.:
```julia
v = ROCArray(rand(Float32, 100_000))
temp = similar(v)
AK.sort!(v, temp=temp)
```
