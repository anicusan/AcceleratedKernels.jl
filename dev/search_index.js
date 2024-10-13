var documenterSearchIndex = {"docs":
[{"location":"references/#References","page":"References","title":"References","text":"","category":"section"},{"location":"references/","page":"References","title":"References","text":"import AcceleratedKernels as AK # hide\nAK.DocHelpers.readme_section(\"## 10. References\") # hide","category":"page"},{"location":"references/","page":"References","title":"References","text":"","category":"page"},{"location":"references/","page":"References","title":"References","text":"import AcceleratedKernels as AK # hide\nAK.DocHelpers.readme_section(\"## 11. Acknowledgements\") # hide","category":"page"},{"location":"api/sort/#sort-and-friends","page":"Sorting","title":"sort and friends","text":"","category":"section"},{"location":"api/sort/","page":"Sorting","title":"Sorting","text":"import AcceleratedKernels as AK # hide\nAK.DocHelpers.readme_section(\"### 5.3. `sort` and friends\") # hide","category":"page"},{"location":"api/accumulate/#Accumulate-/-Prefix-Sum-/-Scan","page":"Accumulate","title":"Accumulate / Prefix Sum / Scan","text":"","category":"section"},{"location":"api/accumulate/","page":"Accumulate","title":"Accumulate","text":"import AcceleratedKernels as AK # hide\nAK.DocHelpers.readme_section(\"### 5.6. `accumulate`\") # hide","category":"page"},{"location":"api/task_partition/#Multithreaded-Task-Partitioning","page":"Task Partitioning","title":"Multithreaded Task Partitioning","text":"","category":"section"},{"location":"api/task_partition/","page":"Task Partitioning","title":"Task Partitioning","text":"AcceleratedKernels.TaskPartitioner\nAcceleratedKernels.task_partition","category":"page"},{"location":"api/task_partition/#AcceleratedKernels.TaskPartitioner","page":"Task Partitioning","title":"AcceleratedKernels.TaskPartitioner","text":"struct TaskPartitioner\n\nPartitioning num_elems elements / jobs over maximum max_tasks tasks with minimum min_elems elements per task.\n\nMethods\n\nTaskPartitioner(num_elems, max_tasks=Threads.nthreads(), min_elems=1)\n\nFields\n\nnum_elems::Int64\nmax_tasks::Int64\nmin_elems::Int64\nnum_tasks::Int64\ntask_istarts::Vector{Int64}\n\nExamples\n\nusing AcceleratedKernels: TaskPartitioner\n\n# Divide 10 elements between 4 tasks\ntp = TaskPartitioner(10, 4)\nfor i in 1:tp.num_tasks\n    @show tp[i]\nend\n\n# output\ntp[i] = 1:3\ntp[i] = 4:6\ntp[i] = 7:8\ntp[i] = 9:10\n\nusing AcceleratedKernels: TaskPartitioner\n\n# Divide 20 elements between 6 tasks with minimum 5 elements per task.\n# Not all tasks will be required\ntp = TaskPartitioner(20, 6, 5)\nfor i in 1:tp.num_tasks\n    @show tp[i]\nend\n\n# output\ntp[i] = 1:5\ntp[i] = 6:10\ntp[i] = 11:15\ntp[i] = 16:20\n\n\n\n\n\n","category":"type"},{"location":"api/task_partition/#AcceleratedKernels.task_partition","page":"Task Partitioning","title":"AcceleratedKernels.task_partition","text":"task_partition(f, num_elems, max_tasks=Threads.nthreads(), min_elems=1)\ntask_partition(f, tp::TaskPartitioner)\n\nPartition num_elems jobs across at most num_tasks parallel tasks with at least min_elems per task, calling f(start_index:end_index), where the indices are between 1 and num_elems.\n\nExamples\n\nA toy example showing outputs:\n\nnum_elems = 4\ntask_partition(println, num_elems)\n\n# Output, possibly in a different order due to threading order\n1:1\n4:4\n2:2\n3:3\n\nThis function is probably most useful with a do-block, e.g.:\n\ntask_partition(4) do irange\n    some_long_computation(param1, param2, irange)\nend\n\n\n\n\n\n","category":"function"},{"location":"api/foreachindex/#General-Looping","page":"General Loops","title":"General Looping","text":"","category":"section"},{"location":"api/foreachindex/","page":"General Loops","title":"General Loops","text":"AcceleratedKernels.foreachindex","category":"page"},{"location":"api/foreachindex/#AcceleratedKernels.foreachindex","page":"General Loops","title":"AcceleratedKernels.foreachindex","text":"foreachindex(\n    f, itr, backend::Backend=get_backend(itr);\n\n    # CPU settings\n    scheduler=:threads,\n    max_tasks=Threads.nthreads(),\n    min_elems=1,\n\n    # GPU settings\n    block_size=256,\n)\n\nParallelised for loop over the indices of an iterable.\n\nIt allows you to run normal Julia code on a GPU over multiple arrays - e.g. CuArray, ROCArray, MtlArray, oneArray - with one GPU thread per index.\n\nOn CPUs at most max_tasks threads are launched, or fewer such that each thread processes at least min_elems indices; if a single task ends up being needed, f is inlined and no thread is launched. Tune it to your function - the more expensive it is, the fewer elements are needed to amortise the cost of launching a thread (which is a few μs). The scheduler can be :polyester to use Polyester.jl cheap threads or :threads to use normal Julia threads; either can be faster depending on the function, but in general the latter is more composable.\n\nExamples\n\nNormally you would write a for loop like this:\n\nx = Array(1:100)\ny = similar(x)\nfor i in eachindex(x)\n    @inbounds y[i] = 2 * x[i] + 1\nend\n\nUsing this function you can have the same for loop body over a GPU array:\n\nusing CUDA\nconst x = CuArray(1:100)\nconst y = similar(x)\nforeachindex(x) do i\n    @inbounds y[i] = 2 * x[i] + 1\nend\n\nNote that the above code is pure arithmetic, which you can write directly (and on some platforms it may be faster) as:\n\nusing CUDA\nx = CuArray(1:100)\ny = 2 .* x .+ 1\n\nImportant note: to use this function on a GPU, the objects referenced inside the loop body must have known types - i.e. be inside a function, or const global objects; but you shouldn't use global objects anyways. For example:\n\nusing oneAPI\n\nx = oneArray(1:100)\n\n# CRASHES - typical error message: \"Reason: unsupported dynamic function invocation\"\n# foreachindex(x) do i\n#     x[i] = i\n# end\n\nfunction somecopy!(v)\n    # Because it is inside a function, the type of `v` will be known\n    foreachindex(v) do i\n        v[i] = i\n    end\nend\n\nsomecopy!(x)    # This works\n\n\n\n\n\n","category":"function"},{"location":"api/binarysearch/#Binary-Search","page":"Binary Search","title":"Binary Search","text":"","category":"section"},{"location":"api/binarysearch/","page":"Binary Search","title":"Binary Search","text":"import AcceleratedKernels as AK # hide\nAK.DocHelpers.readme_section(\"### 5.7. `searchsorted` and friends\") # hide","category":"page"},{"location":"benchmarks/#Benchmarks","page":"Benchmarks","title":"Benchmarks","text":"","category":"section"},{"location":"benchmarks/","page":"Benchmarks","title":"Benchmarks","text":"import AcceleratedKernels as AK # hide\nAK.DocHelpers.readme_section(\"## 3. Benchmarks\") # hide","category":"page"},{"location":"performance/#Performance-Tips","page":"Performance Tips","title":"Performance Tips","text":"","category":"section"},{"location":"performance/","page":"Performance Tips","title":"Performance Tips","text":"If you just started using AcceleratedKernels.jl, see the Manual first for some examples.","category":"page"},{"location":"performance/#GPU-Block-Size-and-CPU-Threads","page":"Performance Tips","title":"GPU Block Size and CPU Threads","text":"","category":"section"},{"location":"performance/","page":"Performance Tips","title":"Performance Tips","text":"All GPU functions allow you to specify a block size - this is often a power of two (mostly 64, 128, 256, 512); the optimum depends on the algorithm, input data and hardware - you can try the different values and @time or @benchmark them:","category":"page"},{"location":"performance/","page":"Performance Tips","title":"Performance Tips","text":"@time AK.foreachindex(f, itr_gpu, block_size=512)","category":"page"},{"location":"performance/","page":"Performance Tips","title":"Performance Tips","text":"Similarly, for performance on the CPU the overhead of spawning threads should be masked by processing more elements per thread (but there is no reason here to launch more threads than Threads.nthreads(), the number of threads Julia was started with); the optimum depends on how expensive f is - again, benchmarking is your friend:","category":"page"},{"location":"performance/","page":"Performance Tips","title":"Performance Tips","text":"@time AK.foreachindex(f, itr_cpu, max_tasks=16, min_elems=1000)","category":"page"},{"location":"performance/#Temporary-Arrays","page":"Performance Tips","title":"Temporary Arrays","text":"","category":"section"},{"location":"performance/","page":"Performance Tips","title":"Performance Tips","text":"As GPU memory is more expensive, all functions in AcceleratedKernels.jl expose any temporary arrays they will use (the temp argument); you can supply your own buffers to make the algorithms not allocate additional GPU storage, e.g.:","category":"page"},{"location":"performance/","page":"Performance Tips","title":"Performance Tips","text":"v = ROCArray(rand(Float32, 100_000))\ntemp = similar(v)\nAK.sort!(v, temp=temp)","category":"page"},{"location":"api/custom_structs/#Custom-Structs","page":"Custom Structs","title":"Custom Structs","text":"","category":"section"},{"location":"api/custom_structs/","page":"Custom Structs","title":"Custom Structs","text":"import AcceleratedKernels as AK # hide\nAK.DocHelpers.readme_section(\"## 6. Custom Structs\") # hide","category":"page"},{"location":"roadmap/#Roadmap-/-Future-Plans","page":"Roadmap","title":"Roadmap / Future Plans","text":"","category":"section"},{"location":"roadmap/","page":"Roadmap","title":"Roadmap","text":"import AcceleratedKernels as AK # hide\nAK.DocHelpers.readme_section(\"## 9. Roadmap / Future Plans\") # hide","category":"page"},{"location":"debugging/#Debugging-Kernels","page":"Debugging Kernels","title":"Debugging Kernels","text":"","category":"section"},{"location":"debugging/","page":"Debugging Kernels","title":"Debugging Kernels","text":"As the compilation pipeline of GPU kernels is different to that of base Julia, error messages also look different - for example, where Julia would insert an exception when a variable name was not defined (e.g. we had a typo), a GPU kernel throwing exceptions cannot be compiled and instead you'll see some cascading errors like \"[...] compiling [...] resulted in invalid LLVM IR\" caused by \"Reason: unsupported use of an undefined name\" resulting in \"Reason: unsupported dynamic function invocation\", etc.","category":"page"},{"location":"debugging/","page":"Debugging Kernels","title":"Debugging Kernels","text":"Thankfully, there are only about 3 types of such error messages and they're not that scary when you look into them.","category":"page"},{"location":"debugging/#Undefined-Variables-/-Typos","page":"Debugging Kernels","title":"Undefined Variables / Typos","text":"","category":"section"},{"location":"debugging/","page":"Debugging Kernels","title":"Debugging Kernels","text":"If you misspell a variable name, Julia would insert an exception:","category":"page"},{"location":"debugging/","page":"Debugging Kernels","title":"Debugging Kernels","text":"function set_color(v, color)\n    AK.foreachindex(v) do i\n        v[i] = colour           # Grab your porridge\n    end\nend","category":"page"},{"location":"debugging/","page":"Debugging Kernels","title":"Debugging Kernels","text":"However, exceptions cannot be compiled on GPUs and you will see cascading errors like below:","category":"page"},{"location":"debugging/","page":"Debugging Kernels","title":"Debugging Kernels","text":"(Image: Undefined Name Error)","category":"page"},{"location":"debugging/","page":"Debugging Kernels","title":"Debugging Kernels","text":"The key thing to look for is undefined name, then search for it in your code.","category":"page"},{"location":"debugging/#Exceptions-and-Checks-that-throw","page":"Debugging Kernels","title":"Exceptions and Checks that throw","text":"","category":"section"},{"location":"debugging/","page":"Debugging Kernels","title":"Debugging Kernels","text":"As mentioned above, exceptions cannot be compiled in GPU kernels; however, many normal-looking functions that we reference in kernels may contain argument-checking. If it cannot be proved that a check branch would not throw an exception, you will see a similar cascade of errors. For example, casting a Float32 to an Int32 includes an InexactError exception check - see this tame-looking code:","category":"page"},{"location":"debugging/","page":"Debugging Kernels","title":"Debugging Kernels","text":"function mymul!(v)\n    AK.foreachindex(v) do i\n        v[i] *= 2f0\n    end\nend\n\nv = MtlArray(1:1000)\nmymul!(v)","category":"page"},{"location":"debugging/","page":"Debugging Kernels","title":"Debugging Kernels","text":"See any problem with it? The MtlArray(1:1000) creates a GPU vector filled with Int64 values, but within foreachindex we do v[i] *= 2.0. We are multiplying an Int64 by a Float32, resulting in a Float32 value that we try to write back into v - this may throw an exception, like in normal Julia code:","category":"page"},{"location":"debugging/","page":"Debugging Kernels","title":"Debugging Kernels","text":"julia> x = [1, 2, 3];\njulia> x[1] = 42.5\nERROR: InexactError: Int64(42.5)","category":"page"},{"location":"debugging/","page":"Debugging Kernels","title":"Debugging Kernels","text":"On GPUs you will see an error like this:","category":"page"},{"location":"debugging/","page":"Debugging Kernels","title":"Debugging Kernels","text":"(Image: Check Exception Error)","category":"page"},{"location":"debugging/","page":"Debugging Kernels","title":"Debugging Kernels","text":"Note the error stack: setindex!, convert, Int64, box_float32 - because of the exception check, we have a type instability, which in turn results in boxing values behind pointers, in turn resulting in dynamic memory allocation and finally the error we see at the top, unsupported call to gpu_malloc.","category":"page"},{"location":"debugging/","page":"Debugging Kernels","title":"Debugging Kernels","text":"You may need to do your correctness checks manually, without exceptions; in this specific case, if we did want to cast a Float32 to an Int, we could use unsafe_trunc(T, x) - though be careful when using unsafe functions that you understand their behaviour and assumptions (e.g. log has a DomainError check for negative values):","category":"page"},{"location":"debugging/","page":"Debugging Kernels","title":"Debugging Kernels","text":"function mymul!(v)\n    AK.foreachindex(v) do i\n        v[i] = unsafe_trunc(eltype(v), v[i] * 2.5f0)\n    end\nend\n\nv = MtlArray(1:1000)\nmymul!(v)","category":"page"},{"location":"debugging/#Type-Instability-/-Global-Variables","page":"Debugging Kernels","title":"Type Instability / Global Variables","text":"","category":"section"},{"location":"debugging/","page":"Debugging Kernels","title":"Debugging Kernels","text":"Types must be known to be captured and compiled within GPU kernels. Global variables without const are not type-stable, as you could associate a different value later on in a script:","category":"page"},{"location":"debugging/","page":"Debugging Kernels","title":"Debugging Kernels","text":"v = MtlArray(1:1000)\n\nAK.foreachindex(v) do i\n    v[i] *= 2\nend\n\nv = \"potato\"","category":"page"},{"location":"debugging/","page":"Debugging Kernels","title":"Debugging Kernels","text":"The error stack is a bit more difficult here:","category":"page"},{"location":"debugging/","page":"Debugging Kernels","title":"Debugging Kernels","text":"(Image: Type Unstable Error)","category":"page"},{"location":"debugging/","page":"Debugging Kernels","title":"Debugging Kernels","text":"You see a few dynamic function invocation, an unsupported call to gpu_malloc, and a bit further down a box. The more operations you do on the type-unstable object, the more dynamic function invocation errors you'll see. These would also be the steps Base Julia would take to allow dynamically-changing objects: they'd be put in a Box behind pointers, and allocated on the heap. In a way, it is better that we cannot do that on a GPU, as it hurts performance massively.","category":"page"},{"location":"debugging/","page":"Debugging Kernels","title":"Debugging Kernels","text":"There are two ways to solve this - if you really want to use global variables in a script, put them behind a const:","category":"page"},{"location":"debugging/","page":"Debugging Kernels","title":"Debugging Kernels","text":"const v = MtlArray(1:1000)\n\nAK.foreachindex(v) do i\n    v[i] *= 2\nend\n\n# This would give you an error now\n# v = \"potato\"","category":"page"},{"location":"debugging/","page":"Debugging Kernels","title":"Debugging Kernels","text":"Or better, use functions:","category":"page"},{"location":"debugging/","page":"Debugging Kernels","title":"Debugging Kernels","text":"function mymul!(v, x)\n    AK.foreachindex(v) do i\n        v[i] *= x\n    end\nend\n\nv = MtlArray(1:1000)\nmymul!(v, 2)","category":"page"},{"location":"debugging/","page":"Debugging Kernels","title":"Debugging Kernels","text":"Note that Julia's lambda capture is very powerful - inside AK.foreachindex you can references other objects from within the function (like x), without explicitly passing them to the GPU.","category":"page"},{"location":"debugging/#Apple-Metal-Only:-Float64-is-not-Supported","page":"Debugging Kernels","title":"Apple Metal Only: Float64 is not Supported","text":"","category":"section"},{"location":"debugging/","page":"Debugging Kernels","title":"Debugging Kernels","text":"Mac GPUs do not natively support Float64 values; there is a high-level check when trying to create an array:","category":"page"},{"location":"debugging/","page":"Debugging Kernels","title":"Debugging Kernels","text":"julia> x = MtlArray([1.0, 2.0, 3.0])\nERROR: Metal does not support Float64 values, try using Float32 instead","category":"page"},{"location":"debugging/","page":"Debugging Kernels","title":"Debugging Kernels","text":"However, if we tried to use / convert values in a kernel to a Float64:","category":"page"},{"location":"debugging/","page":"Debugging Kernels","title":"Debugging Kernels","text":"function mymul!(v, x)\n    AK.foreachindex(v) do i\n        v[i] *= x\n    end\nend\n\nv = MtlArray{Float32}(1:1000)\nmymul!(v, 2.0)","category":"page"},{"location":"debugging/","page":"Debugging Kernels","title":"Debugging Kernels","text":"Note that we try to multiply Float32 values by 2.0, which is a Float64 - in which case we get:","category":"page"},{"location":"debugging/","page":"Debugging Kernels","title":"Debugging Kernels","text":"ERROR: LoadError: Compilation to native code failed; see below for details.\n[...]\ncaused by: NSError: Compiler encountered an internal error (AGXMetalG15X_M1, code 3)\n[...]","category":"page"},{"location":"debugging/","page":"Debugging Kernels","title":"Debugging Kernels","text":"Change the 2.0 to 2.0f0 or Float32(2); in kernels with generic types (that are supposed to work on multiple possible input types), do use the same types as your inputs, using e.g. T = eltype(v) then zero(T), T(42), etc.","category":"page"},{"location":"debugging/","page":"Debugging Kernels","title":"Debugging Kernels","text":"","category":"page"},{"location":"debugging/","page":"Debugging Kernels","title":"Debugging Kernels","text":"For other library-related problems, feel free to post a GitHub issue. For help implementing new code, or just advice, you can also use the Julia Discourse forum, the community is incredibly helpful.","category":"page"},{"location":"api/predicates/#Predicates","page":"Predicates","title":"Predicates","text":"","category":"section"},{"location":"api/predicates/","page":"Predicates","title":"Predicates","text":"import AcceleratedKernels as AK # hide\nAK.DocHelpers.readme_section(\"### 5.8. `all` / `any`\") # hide","category":"page"},{"location":"api/predicates/","page":"Predicates","title":"Predicates","text":"Note on the cooperative keyword: some older platforms crash when multiple threads write to the same memory location in a global array (e.g. old Intel Graphics); if all threads were to write the same value, it is well-defined on others (e.g. CUDA F4.2 says \"If a non-atomic instruction executed by a warp writes to the same location in global memory for more than one of the threads of the warp, only one thread performs a write and which thread does it is undefined.\"). This \"cooperative\" thread behaviour allows for a faster implementation; if you have a platform - the only one I know is Intel UHD Graphics - that crashes, set cooperative=false to use a safer mapreduce-based implementation.","category":"page"},{"location":"testing/#Testing","page":"Testing","title":"Testing","text":"","category":"section"},{"location":"testing/","page":"Testing","title":"Testing","text":"import AcceleratedKernels as AK # hide\nAK.DocHelpers.readme_section(\"## 7. Testing\") # hide","category":"page"},{"location":"api/mapreduce/#MapReduce","page":"MapReduce","title":"MapReduce","text":"","category":"section"},{"location":"api/mapreduce/","page":"MapReduce","title":"MapReduce","text":"import AcceleratedKernels as AK # hide\nAK.DocHelpers.readme_section(\"### 5.5. `mapreduce`\") # hide","category":"page"},{"location":"api/using_backends/#Using-Different-Backends","page":"Using Different Backends","title":"Using Different Backends","text":"","category":"section"},{"location":"api/using_backends/","page":"Using Different Backends","title":"Using Different Backends","text":"import AcceleratedKernels as AK # hide\nAK.DocHelpers.readme_section(\"### 5.1. Using Different Backends\") # hide","category":"page"},{"location":"","page":"Overview","title":"Overview","text":"(Image: Logo)","category":"page"},{"location":"","page":"Overview","title":"Overview","text":"Parallel algorithm building blocks for the Julia ecosystem, targeting multithreaded CPUs, and GPUs via Intel oneAPI, AMD ROCm, Apple Metal and Nvidia CUDA (and any future backends added to the JuliaGPU organisation).","category":"page"},{"location":"","page":"Overview","title":"Overview","text":"","category":"page"},{"location":"#What's-Different?","page":"Overview","title":"What's Different?","text":"","category":"section"},{"location":"","page":"Overview","title":"Overview","text":"import AcceleratedKernels as AK # hide\nAK.DocHelpers.readme_section(\"## 1. What's Different?\") # hide","category":"page"},{"location":"","page":"Overview","title":"Overview","text":"","category":"page"},{"location":"#Status","page":"Overview","title":"Status","text":"","category":"section"},{"location":"","page":"Overview","title":"Overview","text":"import AcceleratedKernels as AK # hide\nAK.DocHelpers.readme_section(\"## 2. Status\") # hide","category":"page"},{"location":"","page":"Overview","title":"Overview","text":"","category":"page"},{"location":"","page":"Overview","title":"Overview","text":"## License AcceleratedKernels.jl is MIT-licensed. Enjoy.","category":"page"},{"location":"api/reduce/#Reductions","page":"Reduce","title":"Reductions","text":"","category":"section"},{"location":"api/reduce/","page":"Reduce","title":"Reduce","text":"import AcceleratedKernels as AK # hide\nAK.DocHelpers.readme_section(\"### 5.4. `reduce`\") # hide","category":"page"}]
}