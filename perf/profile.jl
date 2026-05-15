using Profile
using FDHelmoltzSolver
using FDGrids
using LinearAlgebra

const N     = 512
const WIDTH = 7
const θ₀, θ₁, θ₂, θ₃ = 1.0, 2.0, 1.5, 0.5
const θs = (θ₀, θ₁, θ₂, θ₃)

xs = collect(range(-1, 1; length=N))
D₂ = DiffMatrix(xs, WIDTH, 2)
D₁ = DiffMatrix(xs, WIDTH, 1)
r  = randn(N)

h      = HelmoltzSolver(D₂)
solver = CoupledHelmoltzSolver(D₂, D₁)

# ── Warm up (force compilation) ───────────────────────────────────────────────
update!(h, θ₀, θ₁);      solve!(h, copy(r), 0.0, 0.0)
update!(solver, θs);       solve!(solver, copy(r))

# ── Profile HelmoltzSolver.update! ───────────────────────────────────────────
Profile.clear()
@profile for _ in 1:10_000; update!(h, θ₀, θ₁); end

println("=" ^ 72)
println("HelmoltzSolver  update!   N=$N  width=$WIDTH")
println("=" ^ 72)
Profile.print(format=:flat, sortedby=:count, mincount=5)

# ── Profile HelmoltzSolver.solve! ────────────────────────────────────────────
v = copy(r)
Profile.clear()
@profile for _ in 1:10_000; solve!(h, v, 0.0, 0.0); end

println()
println("=" ^ 72)
println("HelmoltzSolver  solve!    N=$N  width=$WIDTH")
println("=" ^ 72)
Profile.print(format=:flat, sortedby=:count, mincount=5)

# ── Profile CoupledHelmoltzSolver.update! ────────────────────────────────────
Profile.clear()
@profile for _ in 1:5_000; update!(solver, θs); end

println()
println("=" ^ 72)
println("CoupledHelmoltzSolver  update!   N=$N  width=$WIDTH")
println("=" ^ 72)
Profile.print(format=:flat, sortedby=:count, mincount=5)

# ── Profile CoupledHelmoltzSolver.solve! ─────────────────────────────────────
v = copy(r)
Profile.clear()
@profile for _ in 1:5_000; solve!(solver, v); end

println()
println("=" ^ 72)
println("CoupledHelmoltzSolver  solve!    N=$N  width=$WIDTH")
println("=" ^ 72)
Profile.print(format=:flat, sortedby=:count, mincount=5)
