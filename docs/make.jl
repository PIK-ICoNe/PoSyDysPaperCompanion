using Documenter
using Literate
using PoSyDysPaperCompanion

# Generate the companion script as markdown
script = joinpath(pkgdir(PoSyDysPaperCompanion), "examples", "analysis.jl")
outdir = joinpath(@__DIR__, "src", "generated")
isdir(outdir) && rm(outdir, recursive=true)
mkpath(outdir)
Literate.markdown(script, outdir)
Literate.script(script, outdir; keep_comments=true)

doc = makedocs(;
    root=joinpath(pkgdir(PoSyDysPaperCompanion), "docs"),
    sitename="PoSyDysPaperCompanion",
    authors="Hans Würfel and contributors",
    modules=[PoSyDysPaperCompanion],
    pages=[
        "Home" => "index.md",
        "Companion Script" => "generated/analysis.md",
    ],
    remotes=nothing,
    format=Documenter.HTML(;
        canonical="https://pik-icone.github.io/PoSyDysPaperCompanion",
        edit_link="main",
        size_threshold=2_000_000,
        size_threshold_warn=1_000_000,
    ),
    draft=haskey(ENV, "DOCUMENTER_DRAFT"),
    warnonly=true,
    debug=true,
)

if haskey(ENV, "GITHUB_ACTIONS")
    deploydocs(;
        repo="github.com/pik-icone/PoSyDysPaperCompanion.git",
        devbranch="main",
        push_preview=true,
    )
end
