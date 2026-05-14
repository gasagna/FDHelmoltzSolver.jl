@testset "helmoltz solver                        " begin

    # PROBLEM I
    # 3 u''(y) - 2 u(y) = exp(y)
    # u(±1) = exp(±1)
    # with solution u(y) = exp(y)

    # PROBLEM II
    # 2 u''(y) - 2 u(y) = -2(1 + π²)sin(π*y)
    # u(±1) = 0
    # with solution u(y) = sin(π*y)

    # PROBLEM III
    # 2 u''(y) + u(y) = -sin(y)
    # u(±1) = sin(±1)
    # with solution u(y) = sin(y)

    # uniform grid on [-1, 1] and 7-point stencil (6th-order accurate)
    N     = 201
    width = 7
    y     = collect(range(-1, 1; length=N))

    h = HelmoltzSolver(y, width)

    for (f, u_exact, u_l, u_r, θ₀, θ₁) in (
            (y -> exp.(y),                y -> exp.(y),   exp(-1), exp(1), 3,  2),
            (y -> -2*(1 + π^2)*sin.(π*y), y -> sin.(π*y), 0,       0,      2,  2),
            (y -> -sin.(y),               y -> sin.(y),   sin(-1), sin(1), 2, -1))

        update!(h, θ₀, θ₁)

        r = f.(y)
        solve!(h, r, u_l, u_r)

        @test norm(r .- u_exact.(y)) < 1e-8
    end
end
