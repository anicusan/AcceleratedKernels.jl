# File   : AcceleratedKernels.jl
# License: MIT
# Author : Andrei Leonard Nicusan <a.l.nicusan@gmail.com>
# Date   : 09.06.2024


module AcceleratedKernels

# Functionality exported by this package by default
export merge_sort!


# Internal dependencies
using GPUArrays: AbstractGPUVector
using KernelAbstractions


# Include code from other files
include("utils.jl")
include("merge_sort.jl")


end     # module AcceleratedKernels
