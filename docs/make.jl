using AcceleratedKernels
using Documenter


makedocs(;
    modules=[AcceleratedKernels],
    authors="Andrei-Leonard Nicusan <a.l.nicusan@gmail.com> and contributors",
    sitename="AcceleratedKernels.jl",
    format=Documenter.HTML(;
        canonical="https://anicusan.github.io/AcceleratedKernels.jl",
        edit_link="main",
        assets=String[],
        sidebar_sitename=false,

        # Only create web pretty-URLs on the CI
        prettyurls = get(ENV, "CI", nothing) == "true",
    ),
    pages=[
        "Overview" => "index.md",
        "Benchmarks" => "benchmarks.md",
        "Performance Tips" => "performance.md",
        "Manual" =>[
            "Using Different Backends" => "api/using_backends.md",
            "General Loops" => "api/foreachindex.md",
            "Sorting" => "api/sort.md",
            "Reduce" => "api/reduce.md",
            "MapReduce" => "api/mapreduce.md",
            "Accumulate" => "api/accumulate.md",
            "Binary Search" => "api/binarysearch.md",
            "Predicates" => "api/predicates.md",
            "Custom Structs" => "api/custom_structs.md",
            "Task Partitioning" => "api/task_partition.md",
        ],
        "Testing" => "testing.md",
        "Debugging Kernels" => "debugging.md",
        "Roadmap" => "roadmap.md",
        "References" => "references.md",
    ],
    warnonly=true,
)

deploydocs(;
    repo="github.com/anicusan/AcceleratedKernels.jl",
    devbranch="main",
)
