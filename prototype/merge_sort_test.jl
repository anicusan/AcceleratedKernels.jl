
using Random
using BenchmarkTools
using Profile
using PProf

using KernelAbstractions
using CUDA

import AcceleratedKernels as AK


Random.seed!(0)



# Int sorters
function buc_sort!(v::CuVector{Int16})
    @ccall "libBUCLib.so".buc_sort_int16(v::CuPtr{Int16}, length(v)::Cint)::Cvoid
end

function buc_sort!(v::CuVector{Int32})
    @ccall "libBUCLib.so".buc_sort_int32(v::CuPtr{Int32}, length(v)::Cint)::Cvoid
end

function buc_sort!(v::CuVector{Int64})
    @ccall "libBUCLib.so".buc_sort_int64(v::CuPtr{Int64}, length(v)::Cint)::Cvoid
end


# UInt sorters
function buc_sort!(v::CuVector{UInt16})
    @ccall "libBUCLib.so".buc_sort_uint16(v::CuPtr{UInt16}, length(v)::Cint)::Cvoid
end

function buc_sort!(v::CuVector{UInt32})
    @ccall "libBUCLib.so".buc_sort_uint32(v::CuPtr{UInt32}, length(v)::Cint)::Cvoid
end

function buc_sort!(v::CuVector{UInt64})
    @ccall "libBUCLib.so".buc_sort_uint64(v::CuPtr{UInt64}, length(v)::Cint)::Cvoid
end


# Float sorters
function buc_sort!(v::CuVector{Float32})
    @ccall "libBUCLib.so".buc_sort_float32(v::CuPtr{Float32}, length(v)::Cint)::Cvoid
end

function buc_sort!(v::CuVector{Float64})
    @ccall "libBUCLib.so".buc_sort_float64(v::CuPtr{Float64}, length(v)::Cint)::Cvoid
end



# # v1 = CuArray(Float32[2, 1, 4, 3, 5, 2])
# v1 = CuArray(Int32.(shuffle(1:42)))
# 
# v2 = copy(v1)
# merge_sort!(v2)
# 
# v2h = Array(v2)
# @assert issorted(v2h)


for _ in 1:100
    num_elems = rand(1:1_000_000)
    v1 = CuArray(rand(Float32, num_elems))

    v2 = copy(v1)
    AK.merge_sort!(v2)

    v2h = Array(v2)
    if !issorted(v2h)
        println("not sorted!")
        display(num_elems)
        @assert false
    end
end

println("correctness checks passed")


println("KernelAbstractions merge_sort on CUDA:")
display(@benchmark AK.merge_sort!(v1) setup=(v1 = CuArray(rand(Int32, 100_000_000))))

println("CUDA Thrust Sort:")
display(@benchmark buc_sort!(v1) setup=(v1 = CuArray(rand(Int32, 100_000_000))))

println("CUDA.jl QuickSort:")
display(@benchmark sort!(v1) setup=(v1 = CuArray(rand(Int32, 100_000_000))))

println("Julia.Base sort:")
display(@benchmark sort!(v1) setup=(v1 = rand(Int32, 100_000_000)))


# v1 = oneArray(rand(Int32, 10_000_000))
# merge_sort!(copy(v1))
# 
# # Collect a profile
# Profile.clear()
# @profile merge_sort!(v1)
# 
# # Export pprof profile and open interactive profiling web interface.
# pprof()

