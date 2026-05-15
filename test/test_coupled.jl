@testset "coupled helmoltz solver                " begin
    # Solve:
    #   θ₀ u''(x) - θ₁ u(x) = r(x),   x ∈ [-1, 1]
    #   θ₂ v''(x) - θ₃ v(x) = u(x)
    #   v(±1) = v'(±1) = 0
    #
    # Exact solution:
    #   β  = -θ₂π² - θ₃
    #   u  = β cos(πx) - θ₃
    #   v  = cos(πx) + 1
    #   r  = (-θ₀βπ² - θ₁β) cos(πx) + θ₁θ₃

    for (θ₀, θ₁, θ₂, θ₃) in ((1.0, 2.0, 1.5, 0.5), (3.0, 1.0, 2.0, 0.25), (0.5, 4.0, 1.0, 1.0))
        β = -θ₂*π^2 - θ₃

        N     = 101
        width = 7
        xs    = collect(range(-1, 1; length=N))

        rfun(x) = (-θ₀*β*π^2 - θ₁*β)*cos(π*x) + θ₁*θ₃
        vsol(x) = cos(π*x) + 1

        solver = CoupledHelmoltzSolver(DiffMatrix(xs, width, 2), DiffMatrix(xs, width, 1))
        update!(solver, (θ₀, θ₁, θ₂, θ₃))

        r = rfun.(xs)
        solve!(solver, r)

        @test norm(r .- vsol.(xs)) < 1e-7
    end
end
