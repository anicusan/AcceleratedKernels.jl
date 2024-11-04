
using Random
using BenchmarkTools
using Profile
using PProf

using KernelAbstractions
using Metal

import AcceleratedKernels as AK


Random.seed!(0)




# s = MtlArray(rand(Int32(1):Int32(100), 3, 1000))
# d = AK.reduce(+, s; init=zero(eltype(s)), dims=3)
# KernelAbstractions.synchronize(get_backend(s))
# d


f(x::T) where T = T(2) * x + T(1)



function map_base(s)
    d = map(f, s)
    KernelAbstractions.synchronize(get_backend(s))
    d
end


function map_ak(s)
    d = AK.map(f, s, block_size=512)
    KernelAbstractions.synchronize(get_backend(s))
    d
end


s = MtlArray(rand(Int32(1):Int32(100), 10, 100_000))
@assert map_base(s) == map_ak(s)

display(@benchmark map_base($s))
display(@benchmark map_ak($s))





# function redmin(s)
#     d = AK.reduce(
#         (x, y) -> x < y ? x : y,
#         s;
#         init=typemax(eltype(s)),
#         block_size=256,
#         switch_below=0,
#     )
# end
# 
# 
# s = CuArray(shuffle(1:1_000_000))
# d = redmin(s)
# @assert d == 1
# println("Simple correctness check passed")
# 
# println("AcceleratedKernels minimum:")
# display(@benchmark(redmin(s)))
# 
# println("oneAPI minimum:")
# display(@benchmark(minimum(s)))

