using FastTransformsForwardDiff, FFTW, Test
using ForwardDiff: Dual, valtype, value, partials, derivative
using AbstractFFTs: complexfloat, realfloat


@testset "fft and rfft" begin
    x1 = Dual.(1:4.0, 2:5, 3:6)

    @test value.(x1) == 1:4
    @test partials.(x1, 1) == 2:5
    @test partials.(x1, 2) == 3:6

    @test complexfloat(x1)[1] === complexfloat(x1[1]) === Dual(1.0, 2.0, 3.0) + 0im
    @test realfloat(x1)[1] === realfloat(x1[1]) === Dual(1.0, 2.0, 3.0)

    @test fft(x1, 1)[1] isa Complex{<:Dual}

    @testset "$f" for f in [fft, ifft, rfft, bfft]
        @test value.(f(x1)) == f(value.(x1))
        @test partials.(f(x1), 1) == f(partials.(x1, 1))
        @test partials.(f(x1), 2) == f(partials.(x1, 2))
    end

    f = x -> real(fft([x; 0; 0])[1])
    @test derivative(f,0.1) ≈ 1

    r = x -> real(rfft([x; 0; 0])[1])
    @test derivative(r,0.1) ≈ 1


    n = 100
    θ = range(0,2π; length=n+1)[1:end-1]
    # emperical from Mathematical
    @test derivative(ω -> fft(exp.(ω .* cos.(θ)))[1]/n, 1) ≈ 0.565159103992485

    # c = x -> dct([x; 0; 0])[1]
    # @test derivative(c,0.1) ≈ 1
end

@testset "r2r" begin
    x1 = Dual.(1:4.0, 2:5, 3:6)

    @test value.(FFTW.r2r(x1, FFTW.R2HC)) == FFTW.r2r(value.(x1), FFTW.R2HC)
    @test partials.(FFTW.r2r(x1, FFTW.R2HC), 1) == FFTW.r2r(partials.(x1, 1), FFTW.R2HC)
    @test partials.(FFTW.r2r(x1, FFTW.R2HC), 2) == FFTW.r2r(partials.(x1, 2), FFTW.R2HC)

    f = ω -> FFTW.r2r([ω; zeros(9)], FFTW.R2HC)[1]
    @test derivative(f, 0.1) ≡ 1.0
end