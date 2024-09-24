# File   : AcceleratedKernels.jl
# License: MIT
# Author : Andrei Leonard Nicusan <a.l.nicusan@gmail.com>
# Date   : 09.06.2024


module AcceleratedKernels


# No functionality exported by this package before discussion with community


# Internal dependencies
using ArgCheck
using GPUArrays: AbstractGPUVector, @allowscalar
using KernelAbstractions
using Polyester: @batch
using DocStringExtensions


# Include code from other files
include("utils.jl")
include("task_partitioner.jl")
include("foreachindex.jl")
include("sort/sort.jl")
include("reduce.jl")
include("mapreduce.jl")
include("accumulate.jl")
include("searchsorted.jl")
include("truth.jl")


# TODO: add commented backends to tests and docs


end     # module AcceleratedKernels
