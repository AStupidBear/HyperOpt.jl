# R, η = 27, 3
# smax = log(R) / log(η)
# B = (smax + 1) * R
module Hyperband

using Utils

function budget(maxresource, reduction = 3)
  smax = floor(Int, log(maxresource) / log(reduction))
  B = (smax + 1) * maxresource
  @parallel (+) for s in smax:-1:0
      n = ceil(Int, (B / maxresource) * (reduction^s / (s + 1)))
      r = maxresource / reduction^s
      n * r * s
  end
end

function resource(maxbudget, reduction = 3)
  indmin(abs(maxbudget - budget(i, reduction)) for i in 1:100)
end

function halving(getconfig, getloss, n, r, reduction,  s)
    best = (Inf,)
    T = [ getconfig() for i in 1:n ]
    for i in 0:s-1
        ni = floor(Int, n / reduction^i)
        ri = r * reduction^i
        L = [ getloss(t, ri) for t in T ]
        l = sortperm(L); l1 = l[1]
        L[l1] < best[1] && (best = (L[l1], ri, T[l1]))
        T = T[l[1:floor(Int, ni / reduction)]]
        report(best)
    end
    return best
end

export hyperband
function hyperband(getconfig, getloss, maxbudget = 27, reduction = 3)
    maxresource = resource(maxbudget, reduction)
    debug("max budget = ", budget(maxresource), ", ",
          "max resource = ", maxresource)
    smax = floor(Int, log(maxresource) / log(reduction))
    B = (smax + 1) * maxresource
    best = (Inf,)
    for s in smax:-1:0
        n = ceil(Int, (B / maxresource) * (reduction^s / (s + 1)))
        r = maxresource / reduction^s
        curr = halving(getconfig, getloss, n, r, reduction, s)
        if curr[1] < best[1]; (best = curr); end
    end
    return best
end

function report(best)
  loss, epoch, config = best
  @DataFrame(loss, epoch, config) |> debug
end

end
