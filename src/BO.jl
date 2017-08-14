module BO

using Utils, Distributions, GaussianProcesses, BlackBoxOptim, Logging

srand(100)

type BayesOpt
    f::Function
    X::Array{Float64, 2}
    y::Array{Float64, 1}
    xmax::Array{Float64, 1}
    ymax::Float64
    encoder::BoundEncoder
    model::GP
end

function optimize!(opt::BayesOpt, maxevals = 1, optim = false)
  np = nprocs()
  i = last_index = length(opt.y) + 1

  opt.X = hcat(opt.X, zeros(size(opt.X, 1), maxevals))
  append!(opt.y, -Inf * ones(maxevals))

  nextidx() = (idx=i; i+=1; idx)
  @sync begin
    for p = 1:np
      if p != myid() || np == 1
         @async begin
          while true
            idx = nextidx()
            idx > last_index + maxevals - 1 && break

            new_x = idx < np ? [rand(b) for b in opt.encoder.bounds] :
                            acquire_max(opt.model, opt.encoder.bounds)
            new_x = purturb(new_x, opt.X[:, 1:idx], opt.encoder.bounds)
            new_c = transform(opt.encoder, new_x)
            new_y = remotecall_fetch(opt.f, p, new_c)

            debug("\nnew x = $new_x, y = $new_y")
            debug("\niteration = $i, x_max = $(opt.xmax), ymax = $(opt.ymax)")

            opt.X[:, idx] = new_x
            opt.y[idx] = new_y

            yscale = centralize(opt.y[1:idx])
            opt.model = GP(opt.X[:, 1:idx], yscale,
                        MeanConst(mean(yscale)),
                        SE(0.0, 0.0), -5.0)

            optim && GaussianProcesses.optimize!(opt.model)

            if new_y > opt.ymax
              opt.ymax = new_y
              opt.xmax = new_x
            end
            report(opt)
          end
        end
      end
    end
  end
  info("\nOptimization completed:\n xmax = $(opt.xmax), ymax = $(opt.ymax)")
  return opt.xmax, opt.ymax
end

function acquire_max(model, bounds)
  ymax = maximum(model.y)
  opt = bbsetup(expected_improvement(model, ymax);
                SearchRange = bounds,
                TraceMode = :silent)
  res = bboptimize(opt)
  x_max = best_candidate(res)
end

function expected_improvement(model, ymax)
  function ei(x)
    mean, var = GaussianProcesses.predict(model, reshape(x, length(x), 1))
    if var == 0
      return 0
    else
      Z = (mean - ymax) / sqrt(var)
      res = -((mean - ymax) * cdf(Normal(), Z) +
          sqrt(var) * pdf(Normal(), Z))[1]
      return res
    end
  end
  return ei
end

function optimize(f::Function, bounds, c0 = []; maxevals = 100, optim = false)
  encoder = BoundEncoder(bounds)
  X = zeros(length(encoder.bounds), 1)
  y = zeros(1)

  xmax = isempty(c0) ? [rand(b) for b in encoder.bounds] :
                      inverse_transform(encoder, c0)
  cmax = transform(encoder, xmax)
  maxevals == 1 && return cmax, -1e10, [-1e10]
  ymax = f(cmax)
  X[:, 1] = xmax; y[1] = ymax

  opt = BayesOpt(f, X, y, xmax, ymax, encoder,
          GP(X[:, 1:1], [0.0], MeanZero(), SE(0.0, 0.0), -5.0))

  optimize!(opt, maxevals - 1, optim)
  cmax = transform(opt.encoder, opt.xmax)
  return cmax, opt.ymax, progress(opt)
end

progress(opt::BayesOpt) = maximums(opt.y)

maximize(f, args...; kwargs...) = optimize(f, args...; kwargs...)

function minimize(f, args...; kwargs...)
  cmax, ymax, prog = optimize(x -> -f(x), args...; kwargs...)
  cmin = cmax; ymin = -ymax; prog *= -1
  cmin, ymin, prog
end

function report(opt::BayesOpt)
  # if isdefined(Main, :Plots)
  #   p = Main.plot(progress(opt))
  #   name = opt.name * "_BayesOpt"
  #   Main.savefig(p, timename(name))
  # end
end

end # module
