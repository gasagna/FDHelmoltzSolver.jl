using FDHelmoltzSolver, FDGrids, LinearAlgebra, BenchmarkTools, PyPlot

const Ns     = (32, 64, 128, 256, 512, 1024, 2048)
const widths = (3, 5, 7)
const θ₀, θ₁ = 1.0, 2.0
const θs      = (1.0, 2.0, 1.5, 0.5)

# ── Collect timings ───────────────────────────────────────────────────────────
h_update = Dict{Int,Vector{Float64}}()
h_solve  = Dict{Int,Vector{Float64}}()
c_update = Dict{Int,Vector{Float64}}()
c_solve  = Dict{Int,Vector{Float64}}()

for w in widths
    h_update[w] = Float64[]
    h_solve[w]  = Float64[]
    c_update[w] = Float64[]
    c_solve[w]  = Float64[]
    for N in Ns
        xs = collect(range(-1, 1; length=N))
        D₂ = DiffMatrix(xs, w, 2)
        D₁ = DiffMatrix(xs, w, 1)
        r  = randn(N)

        h = HelmoltzSolver(D₂)
        update!(h, θ₀, θ₁)
        push!(h_update[w], minimum(@benchmark update!($h, $θ₀, $θ₁)              evals=1).time * 1e-3)
        push!(h_solve[w],  minimum(@benchmark solve!($h, v, 0.0, 0.0) setup=(v=copy($r)) evals=1).time * 1e-3)

        solver = CoupledHelmoltzSolver(D₂, D₁)
        update!(solver, θs)
        push!(c_update[w], minimum(@benchmark update!($solver, $θs)              evals=1).time * 1e-3)
        push!(c_solve[w],  minimum(@benchmark solve!($solver, v) setup=(v=copy($r)) evals=1).time * 1e-3)
    end
end

# ── Plot ──────────────────────────────────────────────────────────────────────
const COLORS = Dict(3 => "#2196F3", 5 => "#4CAF50", 7 => "#F44336")
const LS     = "-o"

rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["font.family"]  = "serif"
rcParams["font.size"]    = 9
rcParams["axes.spines.top"]   = false
rcParams["axes.spines.right"] = false

fig, axes = subplots(2, 2; figsize=(7, 5), sharex=true)
fig.subplots_adjust(hspace=0.35, wspace=0.35)

Ns_arr = collect(Ns)

titles = ["HelmoltzSolver — update!" "HelmoltzSolver — solve!";
          "CoupledHelmoltzSolver — update!" "CoupledHelmoltzSolver — solve!"]
data   = [h_update h_solve; c_update c_solve]

for (row, col) in Iterators.product(1:2, 1:2)
    ax = axes[row, col]
    d  = data[row, col]
    for w in widths
        ax.loglog(Ns_arr, d[w], LS; color=COLORS[w], markersize=4,
                  linewidth=1.2, label="width=$w")
    end
    ax.set_title(titles[row, col]; fontsize=9)
    ax.set_xlabel("N"; fontsize=8)
    ax.set_ylabel("time (μs)"; fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(true; which="both", linestyle=":", linewidth=0.4, alpha=0.6)
    if row == 1 && col == 2
        ax.legend(; fontsize=8, frameon=false)
    end
end

fig.savefig(joinpath(@__DIR__, "..", "assets", "benchmark.png");
            dpi=150, bbox_inches="tight")
println("Saved assets/benchmark.png")
