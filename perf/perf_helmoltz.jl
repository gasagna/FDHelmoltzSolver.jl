using BenchmarkTools
using FDHelmoltzSolver
using FDGrids
using LinearAlgebra
using Printf

# ── Grid sizes and stencil widths to benchmark ────────────────────────────────
const Ns      = (64, 128, 256, 512, 1024)
const widths  = (3, 5, 7)
const θ₀, θ₁ = 1.0, 2.0

# ── Header ────────────────────────────────────────────────────────────────────
@printf "\n%-6s  %-5s  %12s  %6s  %12s  %6s\n" "N" "width" "update! (ns)" "alloc" "solve! (ns)" "alloc"
println(repeat('─', 64))

for N in Ns, width in widths
    xs = collect(range(-1, 1; length=N))
    D₂ = DiffMatrix(xs, width, 2)
    r  = randn(N)

    h = HelmoltzSolver(D₂)
    update!(h, θ₀, θ₁)          # prime before benchmarking solve!

    b_update = @benchmark update!($h, $θ₀, $θ₁)          evals=1
    b_solve  = @benchmark solve!($h, v, 0.0, 0.0) setup=(v=copy($r)) evals=1

    t_update = minimum(b_update).time          # nanoseconds
    t_solve  = minimum(b_solve).time
    a_update = minimum(b_update).allocs
    a_solve  = minimum(b_solve).allocs

    @printf "%-6d  %-5d  %12.1f  %6d  %12.1f  %6d\n" N width t_update a_update t_solve a_solve
end
println()
