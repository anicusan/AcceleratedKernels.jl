
using Random
using BenchmarkTools
using Profile
using PProf

using KernelAbstractions
using CUDA

using AcceleratedKernels


Random.seed!(0)

# # v1 = CuArray(Float32[2, 1, 4, 3, 5, 2])
# v1 = CuArray(Int32.(shuffle(1:42)))
# 
# v2 = copy(v1)
# merge_sort!(v2)
# 
# v2h = Array(v2)
# @assert issorted(v2h)


for _ in 1:1000
    num_elems = rand(1:1_000_000)
    v1 = CuArray(rand(Float32, num_elems))

    v2 = copy(v1)
    merge_sort!(v2)

    v2h = Array(v2)
    if !issorted(v2h)
        println("not sorted!")
        display(num_elems)
        @assert false
    end
end

println("correctness checks passed")


println("KernelAbstractions merge_sort on CUDA:")
display(@benchmark merge_sort!(v1) setup=(v1 = CuArray(rand(Float32, 100_000_000))))

println("Julia.Base sort:")
display(@benchmark sort!(v1) setup=(v1 = rand(Float32, 100_000_000)))


v1 = CuArray(rand(Int32, 100_000_000))
merge_sort!(copy(v1))

# Collect a profile
Profile.clear()
@profile merge_sort!(v1)

# Export pprof profile and open interactive profiling web interface.
pprof()

