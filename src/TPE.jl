"""
    using Utils: branin
    using TPE
    x_min, y_min = TPE.optimize(branin, 0:1e-2:1, 0:1e-2:1)
    foo(x) = x[1] + x[2] + x[3]
    space, trials= TPE.optimize(foo, 0:1e-2:1, 0:1e-2:10, (0, 1))
"""
module TPE

using Utils, PyCall

hopt = pyimport("hyperopt"); export hopt
hp = hopt[:hp]; export hp
STATUS_OK = hopt[:STATUS_OK]; export STATUS_OK
TPESUGGEST = hopt[:tpe][:suggest]; export TPESUGGEST
RANDOMSUGGEST = hopt[:rand][:suggest]; export RANDOMSUGGEST
Trials = hopt[:Trials]; export Trials

uniform = hp[:uniform]; export uniform
choice = hp[:choice]; export choice
randint = hp[:randint]; export randint
quniform = hp[:quniform]; export quniform
loguniform = hp[:loguniform]; export loguniform
qloguniform = hp[:qloguniform]; export qloguniform
normal = hp[:normal]; export normal
qnormal = hp[:qnormal]; export qnormal
lognormal = hp[:lognormal]; export lognormal
qlognormal = hp[:qlognormal]; export qlognormal

losses(trials) = trials[:losses](); export losses
function valswithlosses(trials)
	ts = trials[:trials];
	[(ts[i]["misc"]["vals"], ts[i]["result"]["loss"]) for i=1:length(ts)]
end

results(trials, name) = [res[name] for res in trials["results"]]
best_trial(trials) = findmin(losses(trials))[2]
best_result(trials) = (id = best_trial(trials); trials["results"][id])
progress(trials) = minimums(losses(trials))

setspace(id, b::Number) = quniform("$id", b - 1e-10, b, 1)
setspace(id, b::Range) =  quniform("$id", extrema(b)..., step(b))
setspace(id, b::NTuple{2, Real}) = uniform("$id", extrema(b)...)
setspace(id, b::Vector) = choice("$id", b)

function optimize(f::Function, bounds; maxevals = 1)
  space=[setspace(i, b) for (i,b) in enumerate(bounds)]
  fmin(f; space = space, max_evals = maxevals)
end
minimize(f::Function, bounds; o...) = optimize(f, bounds; o...)
maximize(f::Function, bounds; o...) = optimize(x -> -f(x), bounds; o...)

function fmin(args...; kwargs...)
  trials = Trials()
  space = hopt[:fmin](args...; kwargs..., algo = TPESUGGEST, trials = trials)
  cmin = [space["$i"] for i in 1:length(space)]
  cmin, trials
end

end # module
