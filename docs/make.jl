using AcceleratedKernels
using Documenter

DocMeta.setdocmeta!(AcceleratedKernels, :DocTestSetup, :(using AcceleratedKernels); recursive=true)

makedocs(;
    modules=[AcceleratedKernels],
    authors="Andrei-Leonard Nicusan <a.l.nicusan@gmail.com> and contributors",
    sitename="AcceleratedKernels.jl",
    format=Documenter.HTML(;
        canonical="https://anicusan.github.io/AcceleratedKernels.jl",
        edit_link="main",
        assets=String[],

        # Only create web pretty-URLs on the CI
        prettyurls = get(ENV, "CI", nothing) == "true",
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/anicusan/AcceleratedKernels.jl",
    devbranch="main",
)
