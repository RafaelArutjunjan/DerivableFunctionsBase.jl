using DerivableFunctionsBase
using Documenter

DocMeta.setdocmeta!(DerivableFunctionsBase, :DocTestSetup, :(using DerivableFunctionsBase); recursive=true)

makedocs(;
    modules=[DerivableFunctionsBase],
    authors="Rafael Arutjunjan",
    repo="https://github.com/RafaelArutjunjan/DerivableFunctionsBase.jl/blob/{commit}{path}#{line}",
    sitename="DerivableFunctionsBase.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://RafaelArutjunjan.github.io/DerivableFunctionsBase.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/RafaelArutjunjan/DerivableFunctionsBase.jl",
    devbranch="main",
)
