# HyperOpt.jl

Hyper-parameter optimization for Julia. It includes three algorithms:
- Gaussian-Process based Bayesian Optimization (BO)
- TPE (by calling [hyperopt](https://github.com/hyperopt/hyperopt) with PyCall.jl)
- [Hyperband](https://arxiv.org/abs/1603.06560)

## Installation

```julia
Pkg.clone("https://github.com/AStupidBear/Utils.jl.git")
Pkg.clone("https://github.com/AStupidBear/Perseus.jl.git")
```

## Usage
See [examples]("examples/")
