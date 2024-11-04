
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





function redadd_base(s)
    d = reduce(+, s; init=zero(eltype(s)), dims=1)
    KernelAbstractions.synchronize(get_backend(s))
    d
end


function redadd_ak(s)
    d = AK.reduce(+, s; init=zero(eltype(s)), dims=1)
    KernelAbstractions.synchronize(get_backend(s))
    d
end


s = MtlArray(rand(Int32(1):Int32(100), 10, 100_000))
@assert redadd_base(s) == redadd_ak(s)

display(@benchmark redadd_base($s))
display(@benchmark redadd_ak($s))





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

