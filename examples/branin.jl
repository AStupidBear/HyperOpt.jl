using Utils, HyperOpt

bounds = [(0, 1) for i in 1:2]; c0 = [0.5, 0.5]
BO.minimize(branin, bounds, c0; optim = true, maxevals = 20)

bounds = [(0, 1), 0.5]
BO.minimize(branin, bounds; optim = true, maxevals = 3)

c0 = [0.5, 0.5]
BO.minimize(branin, bounds, c0; optim = true, maxevals = 3)
