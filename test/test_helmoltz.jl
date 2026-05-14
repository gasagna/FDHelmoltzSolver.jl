using Printf
using FFTW

@testset "helmoltz solver                        " begin

    # PROBLEM I
    # 3 u''(y) - 2 u(y) = exp(y)
    # u(±1) = exp(±1)
    # with solution u(y) = exp(y)

    # PROBLEM II
    # solve the problem
    # 2 u''(y) - 2 u(y) = -2(1 + π^2)sin(π*x)
    # u(±1) = 0
    # with solution u(y) = sin(π*x)

    # PROBLEM III
    # solve the problem
    # 2 u''(y) + 1 u(y) = -sin(x)
    # u(±1) = sin(±1)
    # with solution u(y) = sin(x)
    
    # degree of the polynomial should be enough for all cases
    P = 21

    # chebychev points
    y = cos.(π*(0:P)/P)

    for (f, u, u₊, u₋, θ₀, θ₁) in ((y -> exp.(y),                y -> exp.(y),   exp(1), exp(-1), 3,  2),
                                   (y -> -2*(1 + π^2)*sin.(π*y), y -> sin.(π*y), 0,      0,       2,  2),
                                   (y -> -sin.(y),               y -> sin.(y),   sin(1), sin(-1), 2, -1))

        # coefficients of the chebychev expansion of the function exp(y)
        f̂ = ChebCoeffs(FFTW.r2r(f.(y), FFTW.REDFT00)/P)
        f̂[0] /= 2

        # create solver
        h = HelmoltzSolver(P)

        # and update coefficients
        update!(h, θ₀, θ₁)

        # solve in place on a copy
        û = solve!(h, copy(f̂), u₊, u₋)

        # transform back to physical space
        û[0] *= 2

        # check
        @test norm(FFTW.r2r(û.data/2, FFTW.REDFT00) .- u.(y)) < 1e-14
    end
end