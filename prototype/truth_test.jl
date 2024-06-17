
using Random
using BenchmarkTools
using Profile
using PProf

using KernelAbstractions
using oneAPI

import AcceleratedKernels as AK


Random.seed!(0)


v = oneArray(1:100)

@assert AK.any(x->x<0, v, cooperative=false) === false
@assert AK.any(x->x>99, v, cooperative=false) === true
println("simple any tests passed")

@assert AK.all(x->x>0, v, cooperative=false) === true
@assert AK.all(x->x<100, v, cooperative=false) === false
println("simple all tests passed")


v = oneArray(1:10_000_000)

println("AcceleratedKernels any:")
display(@benchmark(AK.any(x->x>9_999_999, v, cooperative=false)))

println("oneAPI minimum:")
display(@benchmark(any(x->x>9_999_999, v)))

println("CPU minimum:")
vh = Array(v)
display(@benchmark(any(x->x>9_999_999, vh)))

