# FastTransformsForwardDiff.jl
A Julia package to support forward-mode auto-differentiation for fast transforms


[![Build Status](https://github.com/JuliaApproximation/FastTransformsForwardDiff.jl/workflows/CI/badge.svg)](https://github.com/JuliaApproximation/FastTransformsForwardDiff.jl/actions)
[![codecov](https://codecov.io/gh/JuliaApproximation/FastTransformsForwardDiff.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaApproximation/FastTransformsForwardDiff.jl)


A package for forward-mode auto-differentiation for fast transforms. Currently supports the fft:
```julia
julia> using FastTransformsForwardDiff: derivative

julia> θ = range(0,2π; length=n+1)[1:end-1];

julia> derivative(ω -> fft(exp.(ω .* cos.(θ)))[1]/n, 1)
0.5651591039924849 + 0.0im
```