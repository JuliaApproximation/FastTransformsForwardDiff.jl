module FastTransformsForwardDiff
using ForwardDiff, FFTW
using AbstractFFTs
import ForwardDiff: value, partials, npartials, Dual, tagtype, derivative, jacobian, gradient
import AbstractFFTs: plan_fft, plan_ifft, plan_bfft, plan_rfft, plan_brfft, plan_irfft
import FFTW: r2r, r2r!, plan_r2r, mul!, Plan

@inline tagtype(::Complex{T}) where T = tagtype(T)
@inline tagtype(::Type{Complex{T}}) where T = tagtype(T)

dual2array(x::Array{<:Dual{Tag,T}}) where {Tag,T} = reinterpret(reshape, T, x)
dual2array(x::Array{<:Complex{<:Dual{Tag, T}}}) where {Tag,T} = complex.(dual2array(real(x)), dual2array(imag(x)))
array2dual(DT::Type{<:Dual}, x::Array{T}) where T = reinterpret(reshape, DT, real(x))
array2dual(DT::Type{<:Dual}, x::Array{<:Complex{T}}) where T = complex.(array2dual(DT, real(x)), array2dual(DT, imag(x)))

value(x::Complex{<:Dual}) = Complex(x.re.value, x.im.value)

partials(x::Complex{<:Dual}, n::Int) = Complex(partials(x.re, n), partials(x.im, n))

npartials(x::Complex{<:Dual{T,V,N}}) where {T,V,N} = N
npartials(::Type{<:Complex{<:Dual{T,V,N}}}) where {T,V,N} = N


for P in (:Plan, :ScaledPlan)  # need ScaledPlan to avoid ambiguities
    @eval begin
        Base.:*(p::AbstractFFTs.$P, x::AbstractArray{DT}) where DT<:Dual = array2dual(DT, p * dual2array(x))
        Base.:*(p::AbstractFFTs.$P, x::AbstractArray{<:Complex{DT}}) where DT<:Dual = array2dual(DT, p * dual2array(x))
    end
end

mul!(y::AbstractArray{<:Union{Dual,Complex{<:Dual}}}, p::Plan, x::AbstractArray{<:Union{Dual,Complex{<:Dual}}}) = copyto!(y, p*x)


include("fft.jl")

end # module FastTransformsForwardDiff
