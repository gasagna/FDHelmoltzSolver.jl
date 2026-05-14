using FDHelmoltzSolver
using BenchmarkTools
using FFTW

# solve this problem
#       / θ₀ u''(y) - θ₁ u(y)        = r(y)
#      |  θ₂ v''(y) - θ₃ v(y) - u(y) = 0
#       \ v(±1) = v'(±1) = 0

# v = cos(π*y) + 1
# u = (-θ₂π^2 - θ₃) * cos(πy) - θ₃ = β * cos(πy) - θ₃
# r = (-θ₀βπ^2 - θ₁β) * cos(πy) + θ₁θ₃

# with these coefficients
θ₀, θ₁, θ₂, θ₃ = 1, 2, 3, 4

# degree of the polynomial should be enough for all cases
P = 121

# chebychev points
y = cos.(π*(0:P)/P)

# functions for the solution
β = (-θ₂*π^2 - θ₃)
rfun(y) = (-θ₀*β*π^2 - θ₁*β) * cos(π*y) + θ₁*θ₃
sol(y) = cos(π*y) + 1

# coefficients of the chebychev expansion of the function exp(y)
r = ChebCoeffs(FFTW.r2r(rfun.(y), FFTW.REDFT00)/P)
r[0] /= 2

# create solver
h = CoupledHelmoltzSolver(P)

@btime FDHelmoltzSolver._ddy($r, Val(:left))

# and update coefficients
@btime solve!($h, (1, 2, 3, 4), $r)