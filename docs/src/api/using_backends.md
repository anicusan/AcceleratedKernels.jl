### Using Different Backends

For any of the examples here, simply use a different GPU array and AcceleratedKernels.jl will pick the right backend:
```julia
# Intel Graphics
using oneAPI
v = oneArray{Int32}(undef, 100_000)             # Empty array

# AMD ROCm
using AMDGPU
v = ROCArray{Float64}(1:100_000)                # A range converted to Float64

# Apple Metal
using Metal
v = MtlArray(rand(Float32, 100_000))            # Transfer from host to device

# NVidia CUDA
using CUDA
v = CuArray{UInt32}(0:5:100_000)                # Range with explicit step size

# Transfer GPU array back
v_host = Array(v)
```

All publicly-exposed functions have CPU implementations with unified parameter interfaces:

```julia
import AcceleratedKernels as AK
v = Vector(-1000:1000)                          # Normal CPU array
AK.reduce(+, v, max_tasks=Threads.nthreads())
```

Note the `reduce` and `mapreduce` CPU implementations forward arguments to [OhMyThreads.jl](https://github.com/JuliaFolds2/OhMyThreads.jl), an excellent package for multithreading. The focus of AcceleratedKernels.jl is to provide a unified interface to high-performance implementations of common algorithmic kernels, for both CPUs and GPUs - if you need fine-grained control over threads, scheduling, communication for specialised algorithms (e.g. with highly unequal workloads), consider using [OhMyThreads.jl](https://github.com/JuliaFolds2/OhMyThreads.jl) or [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) directly.

There is ongoing work on multithreaded CPU `sort` and `accumulate` implementations - at the moment, they fall back to single-threaded algorithms; the rest of the library is fully parallelised for both CPUs and GPUs.
