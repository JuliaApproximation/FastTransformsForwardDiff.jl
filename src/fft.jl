dual2array(x::Array{<:Dual{Tag,T}}) where {Tag,T} = reinterpret(reshape, T, x)
dual2array(x::Array{<:Complex{<:Dual{Tag, T}}}) where {Tag,T} = complex.(dual2array(real(x)), dual2array(imag(x)))
array2dual(DT::Type{<:Dual}, x::Array{T}) where T = reinterpret(reshape, DT, real(x))
array2dual(DT::Type{<:Dual}, x::Array{<:Complex{T}}) where T = complex.(array2dual(DT, real(x)), array2dual(DT, imag(x)))

value(x::Complex{<:Dual}) = Complex(x.re.value, x.im.value)

partials(x::Complex{<:Dual}, n::Int) = Complex(partials(x.re, n), partials(x.im, n))

npartials(x::Complex{<:Dual{T,V,N}}) where {T,V,N} = N
npartials(::Type{<:Complex{<:Dual{T,V,N}}}) where {T,V,N} = N

# AbstractFFTs.complexfloat(x::AbstractArray{<:Dual}) = float.(x .+ 0im)
AbstractFFTs.complexfloat(x::AbstractArray{<:Dual}) = AbstractFFTs.complexfloat.(x)
AbstractFFTs.complexfloat(d::Dual{T,V,N}) where {T,V,N} = convert(Dual{T,float(V),N}, d) + 0im

AbstractFFTs.realfloat(x::AbstractArray{<:Dual}) = AbstractFFTs.realfloat.(x)
AbstractFFTs.realfloat(d::Dual{T,V,N}) where {T,V,N} = convert(Dual{T,float(V),N}, d)

for plan in (:plan_fft, :plan_ifft, :plan_bfft, :plan_rfft)
    @eval begin
        $plan(x::AbstractArray{<:Dual}, dims=1:ndims(x)) = $plan(dual2array(x), 1 .+ dims)
        $plan(x::AbstractArray{<:Complex{<:Dual}}, dims=1:ndims(x)) = $plan(dual2array(x), 1 .+ dims)
    end
end

plan_r2r(x::AbstractArray{<:Dual}, FLAG, dims=1:ndims(x)) = plan_r2r(dual2array(x), FLAG, 1 .+ dims)
plan_r2r(x::AbstractArray{<:Complex{<:Dual}}, FLAG, dims=1:ndims(x)) = plan_r2r(dual2array(x), FLAG, 1 .+ dims)

for plan in (:plan_irfft, :plan_brfft)  # these take an extra argument, only when complex?
    @eval begin
        $plan(x::AbstractArray{<:Dual}, dims=1:ndims(x)) = $plan(dual2array(x), 1 .+ dims)
        $plan(x::AbstractArray{<:Complex{<:Dual}}, d::Integer, dims=1:ndims(x)) = $plan(dual2array(x), d, 1 .+ dims)
    end
end

r2r(x::AbstractArray{<:Dual}, kinds, region...) = plan_r2r(x, kinds, region...) * x
r2r(x::AbstractArray{<:Complex{<:Dual}}, kinds, region...) = plan_r2r(x, kinds, region...) * x


for P in (:Plan, :ScaledPlan)  # need ScaledPlan to avoid ambiguities
    @eval begin
        Base.:*(p::AbstractFFTs.$P, x::AbstractArray{DT}) where DT<:Dual = array2dual(DT, p * dual2array(x))
        Base.:*(p::AbstractFFTs.$P, x::AbstractArray{<:Complex{DT}}) where DT<:Dual = array2dual(DT, p * dual2array(x))
    end
end

mul!(y::AbstractArray{<:Dual}, p::Plan, x::AbstractArray{<:Dual}) = copyto!(y, p*x)