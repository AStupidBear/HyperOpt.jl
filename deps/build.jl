if !isdir(Pkg.dir("Utils"))
  Pkg.clone("https://github.com/AStupidBear/Utils.jl.git")
  Pkg.build("Utils")
end

run(`pip install hyperopt`)
