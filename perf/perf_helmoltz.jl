using BenchmarkTools
using Printf
using FDHelmoltzSolver

# odd sizes
Ps = 9:2:199

for P in Ps
    # create solver
    h = HelmoltzSolver(P)

    # and update coefficients
    update!(h, 1, 1)

    # random forcing
    f = ChebCoeffs(rand(P+1))

    # solve in place on a copy
    # t = @belapsed solve!($h, $f, 0, 1)
    t1 = minimum( @elapsed solve!(h, f, 0.0, 1.0) for i = 1:1000 )
    t2 = minimum( @elapsed solve!(h, f, 0.0, 1.0) for i = 1:1000 )
    t3 = minimum( @elapsed solve!(h, f, 0.0, 1.0) for i = 1:1000 )
    t = min(t1, t2, t3)

    # print time per grid point
    @printf "%04d %.3f\n" P 10^9*t
end
