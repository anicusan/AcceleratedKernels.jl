# File   : AcceleratedKernels.jl
# License: MIT
# Author : Andrei Leonard Nicusan <a.l.nicusan@gmail.com>
# Date   : 09.06.2024


module AcceleratedKernels


# No functionality exported by this package before discussion with community


# Internal dependencies
using ArgCheck
using GPUArraysCore: AbstractGPUVector, AbstractGPUArray, @allowscalar
using KernelAbstractions
using Polyester: @batch
import OhMyThreads as OMT
using Unrolled
using DocStringExtensions


# Include code from other files
include("utils.jl")
include("task_partitioner.jl")
include("foreachindex.jl")
include("map.jl")
include("sort/sort.jl")
include("reduce/reduce.jl")
include("accumulate.jl")
include("searchsorted.jl")
include("truth.jl")


end     # module AcceleratedKernels
