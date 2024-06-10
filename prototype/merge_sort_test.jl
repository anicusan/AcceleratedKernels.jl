
using Random

using KernelAbstractions
using oneAPI

using AcceleratedKernels


Random.seed!(0)

# v1 = oneArray(Float32[2, 1, 4, 3, 5, 2])
v1 = oneArray(Int32.(shuffle(1:50)))

v2 = copy(v1)
merge_sort!(v2)



