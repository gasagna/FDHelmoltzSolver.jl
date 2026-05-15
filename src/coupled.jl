using StaticArrays

export CoupledHelmoltzSolver

"""
    CoupledHelmoltzSolver{T, H, DT, V}

Solver for the coupled one-dimensional boundary value problem

```
  θ₀ u''(x) - θ₁ u(x)        = r(x),   l ≤ x ≤ r
  θ₂ v''(x) - θ₃ v(x) - u(x) = 0
  v(l) = v(r) = v'(l) = v'(r) = 0
```

where `(θ₀, θ₁, θ₂, θ₃)` are real scalar coefficients and `r` is a given
forcing function. The solution `v` and the residual `r` are represented as
vectors of values at the grid points used to construct the solver.

# Algorithm

The system has four boundary conditions on `v` — two Dirichlet (`v(±1) = 0`)
and two Neumann (`v'(±1) = 0`) — but the second-order ODE for `v` can only
accommodate two. The Dirichlet conditions are imposed directly by row
replacement in each inner solve, while the Neumann conditions are enforced
via an **influence matrix** technique.

The solution is decomposed as

```
  v = vₚ + δ₊ v₊ + δ₋ v₋
```

where

- **vₚ** is the particular solution: solve for `u_p` with `u_p(±1) = 0` and
  `r` as the forcing, then solve for `v_p` driven by `u_p` with `v_p(±1) = 0`.
- **v₊** is a homogeneous complement: solve for `u₊` with `u₊(l) = 1`,
  `u₊(r) = 0`, zero forcing, then solve for `v₊` driven by `u₊` with
  `v₊(±1) = 0`.
- **v₋** is the other complement: same but with `u₋(l) = 0`, `u₋(r) = 1`.

The scalars `δ₊` and `δ₋` are determined by enforcing `v'(l) = v'(r) = 0`,
which reduces to the 2×2 linear system

```
  ┌ v₊'(r)  v₋'(r) ┐ ┌ δ₊ ┐   ┌ -vₚ'(r) ┐
  └ v₊'(l)  v₋'(l) ┘ └ δ₋ ┘ = └ -vₚ'(l) ┘
```

solved exactly via `StaticArrays.SMatrix` arithmetic. As a by-product, the
boundary values of `u` are also determined: `u(l) = δ₊`, `u(r) = δ₋`.

Because `v₊`, `v₋`, and the 2×2 influence matrix depend only on the
coefficients `(θ₀, θ₁, θ₂, θ₃)` and not on the right-hand side `r`, they
are computed once in `update!` and reused across all subsequent `solve!` calls
without recomputation. Only the particular solution `vₚ` is recomputed on
each `solve!` call.

# Fields

- `hu`    — `HelmoltzSolver` for the `u` equation; assembled for `(θ₀, θ₁)` on
             each call to `update!` and reused across all inner solves.
- `hv`    — `HelmoltzSolver` for the `v` equation; assembled for `(θ₂, θ₃)` on
             each call to `update!`.
- `D₁`    — first-order finite-difference differentiation matrix; used to
             evaluate `v'` at the boundary grid points for the influence matrix.
- `vs`    — `NTuple{3}` of pre-allocated vectors `(vₚ, v₊, v₋)`. `vₚ` is
             overwritten on every `solve!` call; `v₊` and `v₋` are computed
             once per `update!` call and reused.
- `A_inf` — precomputed 2×2 influence matrix `[v₊'(r) v₋'(r); v₊'(l) v₋'(l)]`;
             stored as a `MMatrix` so it can be updated in-place by `update!`.

# Type parameters

- `T`  — element type (e.g. `Float64`, `ComplexF64`).
- `H`  — concrete `HelmoltzSolver` type; both `hu` and `hv` share the same
          concrete type since they are built from the same second-order stencil.
- `DT` — concrete matrix type of `D₁`.
- `V`  — concrete vector type of the scratch vectors.

# Constructor

    CoupledHelmoltzSolver(D₂::DiffMatrix, D₁::DiffMatrix, [T=Float64])

Build a solver from a second-order `DiffMatrix` `D₂` and a first-order
`DiffMatrix` `D₁`, both built on the same grid with the same stencil width.
`D₁` is stored as a read-only reference; `D₂` is stored inside each
`HelmoltzSolver` as a read-only reference stencil, and each solver allocates
its own working copy for the LU factors.

# Examples

```julia
using FDHelmoltzSolver, FDGrids, LinearAlgebra

xs = collect(range(-1, 1; length=201))
D₂ = DiffMatrix(xs, 7, 2)   # 7-point second-derivative matrix
D₁ = DiffMatrix(xs, 7, 1)   # 7-point first-derivative matrix

solver = CoupledHelmoltzSolver(D₂, D₁)
update!(solver, (1.0, 2.0, 1.5, 0.5))   # assemble for (θ₀, θ₁, θ₂, θ₃)

r = @. sin(π * xs)           # right-hand side for u
solve!(solver, r)             # r is overwritten with the solution v
```

See also: [`update!(::CoupledHelmoltzSolver, ::NTuple)`](@ref),
[`solve!(::CoupledHelmoltzSolver, ::AbstractVector)`](@ref)
"""
struct CoupledHelmoltzSolver{T, H<:HelmoltzSolver, DT<:DiffMatrix, V<:AbstractVector{T}}
    hu    :: H                   # Helmholtz solver for the u equation
    hv    :: H                   # Helmholtz solver for the v equation
    D₁    :: DT                  # first-order differentiation matrix
    vs    :: NTuple{3, V}        # scratch vectors: vₚ (per-solve), v₊ and v₋ (per-update)
    A_inf :: MMatrix{2, 2, T, 4} # precomputed 2×2 influence matrix, filled by update!
    function CoupledHelmoltzSolver(D₂::DiffMatrix, D₁::DiffMatrix, ::Type{T}=Float64) where {T}
        hu    = HelmoltzSolver(D₂, T)
        hv    = HelmoltzSolver(D₂, T)
        vs    = ntuple(i->zeros(T, size(D₂, 1)), 3)
        A_inf = MMatrix{2, 2, T}(undef)
        return new{T, typeof(hu), typeof(D₁), Vector{T}}(hu, hv, D₁, vs, A_inf)
    end
end

"""
    update!(solver::CoupledHelmoltzSolver, θs::NTuple{4, <:Real})

Assemble and factorise the system matrices for both inner solvers, compute the
homogeneous complementary solutions, and precompute the influence matrix,
preparing `solver` for subsequent calls to `solve!`.

The four coefficients `(θ₀, θ₁, θ₂, θ₃)` parameterise the coupled problem:

```
  θ₀ u'' - θ₁ u = r      (assembled into solver.hu)
  θ₂ v'' - θ₃ v = u      (assembled into solver.hv)
```

After factorising both inner `HelmoltzSolver`s, `update!` computes the two
homogeneous complementary solutions `v₊` and `v₋` (which depend only on the
coefficients, not on the right-hand side `r`) and stores them in `solver.vs`.
It also evaluates the boundary derivatives of `v₊` and `v₋` via `D₁` and
assembles the precomputed 2×2 influence matrix into `solver.A_inf`.

Because `v₊`, `v₋`, and `A_inf` are coefficient-dependent but
right-hand-side-independent, `solve!` only needs to compute the particular
solution `vₚ` on each call, reusing the precomputed quantities unchanged.

`update!` must be called at least once before `solve!`, and again whenever any
of the four coefficients change.

# Arguments

- `solver` — the solver to update.
- `θs`     — coefficient tuple `(θ₀, θ₁, θ₂, θ₃)`.

# Examples

```julia
update!(solver, (1.0, 4.0, 1.0, 0.0))   # Helmholtz u'' - 4u = r,  v'' = u
update!(solver, (2.0, 0.0, 1.0, 9.0))   # u'' = r/2,                v'' - 9v = u
```
"""
function update!(solver::CoupledHelmoltzSolver, θs::NTuple{4, <:Real})
    θ₀, θ₁, θ₂, θ₃ = θs

    # Factorise both inner system matrices for the new coefficients.
    update!(solver.hu, θ₀, θ₁)
    update!(solver.hv, θ₂, θ₃)

    vₚ, v₊, v₋ = solver.vs
    N = length(v₊)

    # ---- Homogeneous complement v₊ -------------------------------------------
    # Solve θ₀ u₊'' - θ₁ u₊ = 0 with u₊(l) = 1, u₊(r) = 0 (v₊ starts as the
    # zero forcing for the u equation), then solve θ₂ v₊'' - θ₃ v₊ = u₊ with
    # v₊(±1) = 0. After both calls v₊ holds the homogeneous complement.
    v₊ .= 0
    solve!(solver.hu, v₊, 1, 0)
    solve!(solver.hv, v₊, 0, 0)

    # ---- Homogeneous complement v₋ -------------------------------------------
    # Same as v₊ but with flipped boundary conditions: u₋(l) = 0, u₋(r) = 1.
    v₋ .= 0
    solve!(solver.hu, v₋, 0, 1)
    solve!(solver.hv, v₋, 0, 0)

    # ---- Precompute the influence matrix -------------------------------------
    # mul!(D₁, v, i) evaluates the i-th row of D₁*v, i.e. the finite-difference
    # approximation to v'(xᵢ). Rows 1 and N correspond to the left (l) and
    # right (r) boundary grid points respectively.
    # A_inf is filled column-major: column 1 = [v₊'(r); v₊'(l)],
    #                                column 2 = [v₋'(r); v₋'(l)].
    solver.A_inf[1, 1] = mul!(solver.D₁, v₊, N)
    solver.A_inf[2, 1] = mul!(solver.D₁, v₊, 1)
    solver.A_inf[1, 2] = mul!(solver.D₁, v₋, N)
    solver.A_inf[2, 2] = mul!(solver.D₁, v₋, 1)

    return nothing
end

"""
    solve!(solver::CoupledHelmoltzSolver, r::AbstractVector) -> r

Solve the coupled Helmholtz problem in-place, overwriting `r` with the
solution `v`.

`update!` must have been called prior to `solve!` to assemble and factorise
the system matrices, compute the homogeneous complements `v₊` and `v₋`, and
precompute the influence matrix for the desired coefficients `(θ₀, θ₁, θ₂, θ₃)`.

On entry, `r` contains the right-hand side `r(x)` of the `u` equation
evaluated at the grid points. On exit, `r` holds the solution `v(x)`.

## Algorithm

The homogeneous complements `v₊`, `v₋` and the 2×2 influence matrix are
precomputed by `update!` and reused here without recomputation. Only the
particular solution `vₚ` — which depends on `r` — is computed per call.

**Step 1 — particular solution** (`vₚ`):

Solve `θ₀ u_p'' - θ₁ u_p = r` with homogeneous Dirichlet BCs `u_p(±1) = 0`,
then solve `θ₂ v_p'' - θ₃ v_p = u_p` with `v_p(±1) = 0`. The scratch vector
`vₚ` is initialised to `r` and overwritten first with `u_p`, then with `v_p`.

**Step 2 — influence matrix**:

Evaluate the boundary derivatives `vₚ'(l)` and `vₚ'(r)` via `D₁` and form
the right-hand side `b = [-vₚ'(r); -vₚ'(l)]`. Solve the precomputed 2×2
system `A_inf * [δ₊; δ₋] = b` to obtain the superposition coefficients.

**Step 3 — superposition**:

Write `r .= vₚ + δ₊ v₊ + δ₋ v₋` back into `r`. The result satisfies
`v'(l) = v'(r) = 0` by construction.

# Arguments

- `solver` — a solver prepared by a prior call to `update!`.
- `r`      — on entry: right-hand side `r(x)` of the `u` equation at the grid
             points; on exit: the solution `v(x)`. Length must match the grid
             used to construct `solver`.

# Returns

`r`, overwritten with the solution `v`.

# Examples

```julia
r = @. (-π^2 - 4) * sin(π * xs)   # forcing for a particular u
solve!(solver, r)                   # r now contains v
```
"""
function solve!(solver::CoupledHelmoltzSolver, r::AbstractVector)
    vₚ, v₊, v₋ = solver.vs
    N = length(r)

    # ---- Step 1: particular solution ----------------------------------------
    # Solve θ₀ u_p'' - θ₁ u_p = r with u_p(±1) = 0, then
    # solve θ₂ v_p'' - θ₃ v_p = u_p with v_p(±1) = 0.
    # After both calls vₚ holds v_p.
    vₚ .= r
    solve!(solver.hu, vₚ, 0, 0)
    solve!(solver.hv, vₚ, 0, 0)

    # ---- Step 2: influence matrix -------------------------------------------
    # Use the precomputed A_inf (filled by update!) and the boundary derivatives
    # of vₚ to find δ₊ and δ₋ such that v'(l) = v'(r) = 0.
    b = SVector{2}(-mul!(solver.D₁, vₚ, N), -mul!(solver.D₁, vₚ, 1))
    δ₊, δ₋ = SMatrix(solver.A_inf) \ b

    # ---- Step 3: superposition ----------------------------------------------
    # Write the superposition v = vₚ + δ₊ v₊ + δ₋ v₋ back into r.
    r .= vₚ .+ δ₊ .* v₊ .+ δ₋ .* v₋

    return r
end
