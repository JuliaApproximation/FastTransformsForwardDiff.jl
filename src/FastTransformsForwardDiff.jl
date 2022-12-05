module FastTransformsForwardDiff
using ForwardDiff
import AbstractFFTs
import ForwardDiff: value, partials, npartials, Dual, tagtype, derivative, jacobian, gradient

@inline tagtype(::Complex{T}) where T = tagtype(T)
@inline tagtype(::Type{Complex{T}}) where T = tagtype(T)

include("fft.jl")

end # module FastTransformsForwardDiff
