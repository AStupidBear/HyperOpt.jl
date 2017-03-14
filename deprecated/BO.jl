
# addprocs(3)
# using Utils
# reload("BO"); reload("TPE")
#
# @everywhere function branin(v)
#     x, y = v
#     x, y = 15x - 5, 15y
#     res = 1/51.95 * ((y - 5.1*x^2 / (4*π^2) + 5x/π - 6)^2 + (10 -10/8π)cos(x) -44.81)
# end
#
# function benchmark(;dim = 2, maxevals = 10, restarts = 10)
#   loss1 = zeros(maxevals, restarts)
#   loss2 = zeros(maxevals, restarts)
#   for t in 1:restarts
#     bounds = [0:1e-2:1 for _ in 1:dim]
#     xmax, ymax, opt = BO.optimize(x->-branin(x), bounds...;
#                                   opt = true, iter = maxevals)
#     loss1[:, t] = -BO.progress(opt)
#
#     param, trials = TPE.optimize(branin, bounds...; maxevals = maxevals)
#     loss2[:, t] = TPE.progress(trials)
#   end
#   loss1 = median(loss1, 2);  loss2 = median(loss2, 2)
#
#   Main.plot([loss1 loss2])#label = ["BayesOpt", "TPE"])
# end
#
# @time p = benchmark(; maxevals = 20, restarts = 1) |> gui
#

"""
    reload("BO")
    using Utils: branin
    bounds = [0:1e-2:1 for _ in 1:2]
    xmax, ymax, opt = BO.optimize(x->-branin(x), bounds...; iter=10)
"""
module BO

using Utils, Logging, Distributions, GaussianProcesses, BlackBoxOptim
export BayesOpt, optimize!, optimize

# Logging.configure(output=STDOUT, level=DEBUG)

"""
A type for storing our search history, bounds and model.
"""
type BayesOpt
    """
    A function which takes the parameters
    as input and returns a value which is to
    be maximized."""
    f::Function

    """
    The history of parameters tried as an N x M matrix
    where N is the number of parameters and M is the number
    observations.
    """
    X::Array

    """ The history of returned values to function `f` """
    y::Array

    """ The best set of parameters tried """
    xmax::Array

    """ The best result found """
    ymax::Float64

    """
    A collection of bounds for each parameter to restrict the search.
    Each bound can be an abstract array, but is typically a Range{T},
    """
    bounds::Array{AbstractArray}

    """
    The internal model of the dataset which can predict the next value.

    Currently, only a [GaussianProcess](https://github.com/STOR-i/GaussianProcesses.jl)
    is used.
    """
    model::Any
end

"""
The default BayesOpt constructor

Args:
- `f`: the function to optimize
- `bounds`: a collection of bounds (1 per parameter) to direct the search.

Returns:
- `BayesOpt`: a new `BayesOpt` instance.
"""
function BayesOpt{T<:AbstractArray}(f::Function, bounds::T...; noise=-1e8)
    # Initialize our parameters matrix
    X = Array{Float64}(length(bounds), 1)

    # Initialize our results array
    y = Array{Float64}(1)

    # Set our initial observations
    # requiring use to run the function once
    xmax = [rand(b) for b in bounds]
    ymax = f(xmax)

    X[:,1] = xmax
    y[1] = ymax

    # return our new BayesOpt instance with these reasonable defaults
    # NOTE: see https://github.com/STOR-i/GaussianProcesses.jl for our
    # Gaussian Process arguments.
    BayesOpt(
        f, X, y, xmax, ymax, collect(bounds),
        GP(X[:,1:1], y[1:1], MeanZero(), SE(0.0, 0.0), noise)
    )
end

"""
Performs the optimization loop on a `BayesOpt` instance for the
specified number of iterations with the provided number of restarts.

Args:
- `opt`: the `BayesOpt` type to optimize
- `iter`: the number of iterations to run the optimization loop
- `noise` : log of the observation noise, default no observation noise

Returns:
- `Array{Real}(nparams, 1)`: the best params found for `f`
- `Real`: the best result found.
"""
function optimize!(opt::BayesOpt, iter=10; noise=-1e8, opt=true)
  np = nprocs() # determine the number of processes available
  i = last_index = length(opt.y) + 1

  # Allocate `iter` number of observation to both `X` and `y`
  opt.X = hcat(opt.X, Array{Float64}(length(opt.bounds), iter))
  append!(opt.y, Array{Float64}(iter))

  nextidx() = (idx=i; i+=1; idx)
  @sync begin
    for p = 1:np
      if p != myid() || np == 1
        @async begin
          while true
            idx = nextidx()
            if idx > last_index + iter - 1
              break
            end
            Logging.debug("iteration=$i, x=$(opt.xmax), y=$(opt.ymax)")

            # Select the next params
            new_x = acquire_max(opt.model, opt.ymax, opt.bounds)

            # Run the next params
            new_x = Utils.discretize.(new_x, opt.bounds)
            new_x = Utils.purturb(new_x, opt.X[:,1:idx], opt.bounds)

            new_y = remotecall_fetch(opt.f, p, new_x)

            Logging.debug("new x=$new_x, y=$new_y")

            # update X and y with next params and run
            opt.X[:,idx] = new_x
            opt.y[idx] = new_y
            # Just rebuild the model cause apparently the above complains if we update with more data.

            opt.model = GP(opt.X[:,1:idx], opt.y[1:idx], MeanConst(mean(opt.y[1:idx])), SE(0.0, 0.0), noise)
            opt && GaussianProcesses.optimize!(opt.model)

            # Update the max x and y if our new
            # result is better.
            if new_y > opt.ymax
                opt.ymax = new_y
                opt.xmax = new_x
            end
          end
        end
      end
    end
  end
  Logging.info("Optimization completed: x=$(opt.xmax), y=$(opt.ymax)")

  # return our best parameters and result respectively.
  return opt.xmax, opt.ymax
end




"""
A helper function which handles creating the `BayesOpt` instance and
running `optimize!`

Args:
- `f`: the function to optimize
- `bounds`: a collection of bounds (1 per parameter) to direct the search.
- `iter`: the number of iterations to run the optimization loop
- `noise` : log of the observation noise, default no observation noise

Returns:
- `Array{Real}(nparams, 1)`: the best params found for `f`
- `Real`: the best result found.
- `BayesOpt`: a new `BayesOpt` instance.
"""
function optimize{T<:AbstractArray}(f::Function, bounds::T...; iter=100, noise=-1e8, o...)
    # Create out BayesOpt instance
    opt = BayesOpt(f, bounds...; noise=noise)

    # call `optimize!` for it.
    optimize!(opt, iter-1; noise=noise, o...)

    # return the x and y maximums like `optimize!` along with the
    # BayesOpt instance.
    return opt.xmax, opt.ymax, opt
end


"""
Acquire function wraps a tradition [acquisition function](http://arxiv.org/pdf/1012.2599.pdf)
which selects the next x value(s) that will produce the minimum
value of y balancing exploration vs exploitation.

NOTE: While currently `expected_improvement` is the only option, future versions will support
different acquisition function described in the linked paper.

Args:
- `model`: the model which the acquisition function (in this case EI) needs to use.
- `

Returns:

"""
function acquire_max(model, ymax, bounds)
    #Find the minimum of minus the acquisition function
    opt = bbsetup(expected_improvement(model, ymax);
    SearchRange = [extrema(b) for b in bounds],
    TraceMode = :silent)
    res = bboptimize(opt)
    x_max = best_candidate(res)
end


"""
The expected improvement acquisition function which balances
exporation vs exploitation by selecting the next x that is not only
the most likely to be improve the current best result, but also
to account for the expected magnitude.

NOTE: Since, all function used with NLopt only take some
data and a gradient (which in this case will always be empty
cause we are using a derivative free optimize method) we use a closure
to generate this function with the model and max y value.

Args:
- `model`: any model that returns the mean and variance when predict is
        called on some sample data.
- 'ymax': the current maximum value

Returns:
- `Function`: The `ei` function to pass to `NLopt.optimize`.
"""
function expected_improvement(model, ymax)
    """
    Inner closure fuction that performs the EI.

    Args:
    - `x`: the observation to pass to the models predict.
    - `grad`: the gradient, which should be an empty array since we are using a
            derivative free optimization method, but is required by NLopt.
    """
    function ei(x)
        # Get the mean and variance from the existing observations
        # by calling predict on a gaussion process.
        mean, var = predict(model, reshape(x, length(x), 1))

        if var == 0
            # If the variance is 0 just return 0
            return 0
        else
            # otherwise calculate the ei and return that.
            Z = (mean - ymax)/sqrt(var)
            res = -((mean - ymax) * cdf(Normal(), Z) + sqrt(var) * pdf(Normal(), Z))[1]
            return res
        end
    end
    return ei
end

progress(opt::BayesOpt) = maximums(opt.model.y)


end # module

# function optimize!(opt::BayesOpt, iter=100; noise=-1e8)
#     # The last index we looked at is equal to the current number of
#     # observation looked at.
#     last_index = length(opt.y) + 1
#
#     # Allocate `iter` number of observation to both `X` and `y`
#     opt.X = hcat(opt.X, Array{Float64}(length(opt.bounds), iter))
#     append!(opt.y, Array{Float64}(iter))
#
#     # Begin our optimization loop
#     for i in last_index:length(opt.y)
#         Logging.debug("iteration=$i, x=$(opt.xmax), y=$(opt.ymax)")
#
#         # Select the next params
#         new_x = acquire_max(opt.model, opt.ymax, opt.bounds)
#
#         # Run the next params
#         new_y = opt.f(new_x)
#
#         Logging.debug("new x=$new_x, y=$new_y")
#
#         # update X and y with next params and run
#         opt.X[:,i] = new_x
#         opt.y[i] = new_y
#
#         # update the observations in the internal model (ie: the GP).
#         # NOTE: This update process is specific to Gaussian Processes,
#         # but could be refactored out.
#         # opt.model.x = opt.X[:,1:i+1]
#         # opt.model.y = opt.y[1:i+1]
#         # GaussianProcesses.update_mll!(opt.model)
#
#         # Just rebuild the model cause apparently the above complains if we update with more data.
#         opt.model = GP(opt.X[:,1:i], opt.y[1:i], MeanConst(mean(opt.y[1:i])), SE(0.0, 0.0), noise)
#         GaussianProcesses.optimize!(opt.model)
#
#         # Update the max x and y if our new
#         # result is better.
#         if new_y > opt.ymax
#             opt.ymax = new_y
#             opt.xmax = new_x
#         end
#     end
#
#     Logging.info("Optimization completed: x=$(opt.xmax), y=$(opt.ymax)")
#
#     # return our best parameters and result respectively.
#     return opt.xmax, opt.ymax
# end
