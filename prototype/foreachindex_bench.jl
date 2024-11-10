using BenchmarkTools
import AcceleratedKernels as AK
import OhMyThreads as OMT

using Random
Random.seed!(0)

function simplecopyto!(x)
    v = similar(x)
    for i in eachindex(x)
        @inbounds v[i] = x[i]
    end
end

function akcopyto!(x, scheduler)
    v = similar(x)
    AK.foreachindex(x, scheduler=scheduler) do i
        @inbounds v[i] = x[i]
    end
end

function omtcopyto!(x)
    v = similar(x)
    OMT.tforeach(eachindex(x), scheduler=:static) do i
        @inbounds v[i] = x[i]
    end
end


x = rand(Int32, 1_000_000)

println("\nSimple serial copy:")
display(@benchmark(simplecopyto!(x)))

println("\nAcceleratedKernels foreachindex :polyester copy:")
display(@benchmark(akcopyto!(x, :polyester)))

println("\nAcceleratedKernels foreachindex :threads copy:")
display(@benchmark(akcopyto!(x, :threads)))

println("\nOhMyThreads tforeach :static copy:")
display(@benchmark(omtcopyto!(x)))

