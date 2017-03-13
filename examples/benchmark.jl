using Utils; reload("BO"); reload("Hyperopt")

function benchmark(f, bounds; maxevals = 10, restarts = 10)
  loss = [zeros(maxevals, restarts) for i in 1:3]
  for t in 1:restarts
    xmin, ymin, progress = BO.minimize(f, bounds; maxevals = maxevals, optim=false)
    loss[1][:, t] = progress

    xmin, ymin, progress = BO.minimize(f, bounds; maxevals = maxevals, optim=true)
    loss[2][:, t] = progress

    param, trials = Hyperopt.minimize(f, bounds; maxevals = maxevals)
    loss[3][:, t] = Hyperopt.progress(trials)
  end
  loss = map(x->median(x, 2), loss)
  Main.plot(loss, label = ["BayesOpt_nooptim" "BayesOpt_optim" "Hyperopt"])
end

bounds = [(-1, 1) for i in 1:10]
@time benchmark(x->sum(abs2, x), bounds; maxevals = 20, restarts = 1) |> gui

bounds = [(0, 1) for i in 1:2]
@time benchmark(branin, bounds; maxevals = 20, restarts = 1) |> gui

bounds = [(0, 2) for i in 1:10]
@time benchmark(rosenbrock, bounds; maxevals = 20, restarts = 1) |> gui
