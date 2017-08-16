for pkg in ["Utils"]
    Pkg.installed(pkg) !== nothing && (Pkg.checkout(pkg); continue)
    Pkg.clone("https://github.com/AStupidBear/$pkg.jl.git")
    Pkg.build(pkg)
end

# run(`pip install hyperopt`)
