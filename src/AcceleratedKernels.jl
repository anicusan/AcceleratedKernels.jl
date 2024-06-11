# File   : AcceleratedKernels.jl
# License: MIT
# Author : Andrei Leonard Nicusan <a.l.nicusan@gmail.com>
# Date   : 09.06.2024


module AcceleratedKernels


# No functionality exported by this package before discussion with community


# Internal dependencies
using GPUArrays: AbstractGPUVector, @allowscalar
using KernelAbstractions
using DocStringExtensions


# Include code from other files
include("utils.jl")
include("task_partitioner.jl")
include("sort/sort.jl")
include("reduce.jl")
include("mapreduce.jl")
include("foreachindex.jl")


# TODO: searchsorted, scan, any, scatter


end     # module AcceleratedKernels
