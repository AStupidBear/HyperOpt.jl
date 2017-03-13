addprocs(3)

using Utils; reload("BO")

bounds = [(0, 1) for i in 1:2]
@time xmin, ymin, progress =  BO.minimize(x-> (@repeat 2 Utils.slow(); Utils.branin(x)),
                              bounds; optim = true, maxevals = 3)
plot(progress) |> gui
