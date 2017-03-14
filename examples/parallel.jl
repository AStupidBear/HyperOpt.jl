addprocs(3)
using Utils, HyperOpt
bounds = [(0, 1) for i in 1:2]
@time xmin, ymin, prog = BO.minimize(bounds; optim = false, maxevals = 3) do x
  Utils.slow(); Utils.branin(x)
end
