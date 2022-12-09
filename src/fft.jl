
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

