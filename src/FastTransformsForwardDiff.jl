module FastTransformsForwardDiff
using ForwardDiff, FFTW
using AbstractFFTs
import ForwardDiff: value, partials, npartials, Dual, tagtype, derivative, jacobian, gradient
import AbstractFFTs: plan_fft, plan_ifft, plan_bfft, plan_rfft
import FFTW: r2r, r2r!, plan_r2r, mul!, Plan

@inline tagtype(::Complex{T}) where T = tagtype(T)
@inline tagtype(::Type{Complex{T}}) where T = tagtype(T)

include("fft.jl")

end # module FastTransformsForwardDiff
