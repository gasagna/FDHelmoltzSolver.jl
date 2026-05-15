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
via an **influence matrix** technique:

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

# Fields

- `hu`  — `HelmoltzSolver` for the `u` equation; assembled for `(θ₀, θ₁)` on
           each call to `update!` and reused across all three inner solves in
           `solve!`.
- `hv`  — `HelmoltzSolver` for the `v` equation; assembled for `(θ₂, θ₃)` on
           each call to `update!`.
- `D₁`  — first-order finite-difference differentiation matrix; used to
           evaluate `v'` at the boundary grid points for the influence matrix.
- `vs`  — `NTuple{3}` of pre-allocated scratch vectors `(vₚ, v₊, v₋)` holding
           the three partial solutions during `solve!`.

# Type parameters

- `T`  — element type (e.g. `Float64`, `ComplexF64`).
- `H`  — concrete `HelmoltzSolver` type; both `hu` and `hv` share the same
          concrete type since they are built from the same second-order stencil.
- `DT` — concrete matrix type of `D₁`.
- `V`  — concrete vector type of the scratch vectors.

# Constructor

    CoupledHelmoltzSolver(D₂, D₁, [T=Float64])

Build a solver from a pre-constructed second-order differentiation matrix `D₂`
and a first-order matrix `D₁`. Both are typically `DiffMatrix` objects from
`FDGrids.jl` built on the same grid with the same stencil width. `D₁` is
stored as a read-only reference; `D₂` is stored inside each `HelmoltzSolver`
as a read-only reference stencil, and each solver allocates its own working
copy for the LU factors.

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
struct CoupledHelmoltzSolver{T, H<:HelmoltzSolver, DT<:AbstractMatrix, V<:AbstractVector{T}}
    hu :: H             # Helmholtz solver for the u equation
    hv :: H             # Helmholtz solver for the v equation
    D₁ :: DT            # first-order differentiation matrix for boundary derivative evaluation
    vs :: NTuple{3, V}  # pre-allocated scratch vectors (vₚ, v₊, v₋)
    function CoupledHelmoltzSolver(D₂::AbstractMatrix, D₁::AbstractMatrix, ::Type{T}=Float64) where {T}
        hu = HelmoltzSolver(D₂, T)
        hv = HelmoltzSolver(D₂, T)
        vs = ntuple(i->zeros(T, size(D₂, 1)), 3)
        return new{T, typeof(hu), typeof(D₁), Vector{T}}(hu, hv, D₁, vs)
    end
end

"""
    update!(solver::CoupledHelmoltzSolver, θs::NTuple{4, <:Real})

Assemble and factorise the system matrices for both inner solvers, preparing
`solver` for subsequent calls to `solve!`.

The four coefficients `(θ₀, θ₁, θ₂, θ₃)` parameterise the coupled problem:

```
  θ₀ u'' - θ₁ u = r      (assembled into solver.hu)
  θ₂ v'' - θ₃ v = u      (assembled into solver.hv)
```

Each inner `HelmoltzSolver` is updated independently: `hu` is assembled for
`(θ₀, θ₁)` and `hv` for `(θ₂, θ₃)`. Both matrices are factorised in-place so
that the three triangular solves in `solve!` can be dispatched without
re-factorising.

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
    update!(solver.hu, θ₀, θ₁)
    update!(solver.hv, θ₂, θ₃)
    return nothing
end

"""
    solve!(solver::CoupledHelmoltzSolver, r::AbstractVector) -> r

Solve the coupled Helmholtz problem in-place, overwriting `r` with the
solution `v`.

`update!` must have been called prior to `solve!` to assemble and factorise
the system matrices for the desired coefficients `(θ₀, θ₁, θ₂, θ₃)`.

On entry, `r` contains the right-hand side `r(x)` of the `u` equation
evaluated at the grid points. On exit, `r` holds the solution `v(x)`.

## Algorithm

The method uses an **influence matrix** technique to enforce all four boundary
conditions on `v` using only second-order inner solvers. Three sub-problems
are solved using the pre-factorised `HelmoltzSolver`s, then a 2×2 linear
system determines the superposition coefficients.

**Step 1 — particular solution** (`vₚ`):

Solve `θ₀ u_p'' - θ₁ u_p = r` with homogeneous Dirichlet BCs `u_p(±1) = 0`,
then solve `θ₂ v_p'' - θ₃ v_p = u_p` with `v_p(±1) = 0`. The scratch vector
`vₚ` is initialised to `r` and overwritten first with `u_p`, then with `v_p`.

**Step 2 — first homogeneous complement** (`v₊`):

Solve `θ₀ u₊'' - θ₁ u₊ = 0` with BCs `u₊(l) = 1`, `u₊(r) = 0`, then solve
`θ₂ v₊'' - θ₃ v₊ = u₊` with `v₊(±1) = 0`. The scratch vector `v₊` is
initialised to zero (the zero forcing for `u₊`) and overwritten accordingly.

**Step 3 — second homogeneous complement** (`v₋`):

Same as Step 2 but with BCs `u₋(l) = 0`, `u₋(r) = 1`.

**Step 4 — influence matrix**:

Evaluate the boundary derivatives `v'(l)` and `v'(r)` for each partial
solution using `D₁`. Enforce `v'(l) = v'(r) = 0` for the superposition
`v = vₚ + δ₊ v₊ + δ₋ v₋` by solving the 2×2 system

```
  ┌ v₊'(r)  v₋'(r) ┐ ┌ δ₊ ┐   ┌ -vₚ'(r) ┐
  └ v₊'(l)  v₋'(l) ┘ └ δ₋ ┘ = └ -vₚ'(l) ┘
```

The final solution `r .= vₚ + δ₊ v₊ + δ₋ v₋` is written back into `r`.

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

    # Load the right-hand side into the particular scratch vector; zero the
    # homogeneous scratch vectors before they are used as u forcing.
    vₚ .= r
    v₊ .= 0
    v₋ .= 0

    # ---- Step 1: particular solution ----------------------------------------
    # Solve θ₀ u_p'' - θ₁ u_p = r with u_p(±1) = 0, then
    # solve θ₂ v_p'' - θ₃ v_p = u_p with v_p(±1) = 0.
    # After both calls vₚ holds v_p.
    solve!(solver.hu, vₚ, 0, 0)
    solve!(solver.hv, vₚ, 0, 0)

    # ---- Step 2: first homogeneous complement --------------------------------
    # Solve θ₀ u₊'' - θ₁ u₊ = 0 with u₊(l) = 1, u₊(r) = 0 (v₊ starts as
    # the zero forcing), then solve θ₂ v₊'' - θ₃ v₊ = u₊ with v₊(±1) = 0.
    # After both calls v₊ holds v₊.
    solve!(solver.hu, v₊, 1, 0)
    solve!(solver.hv, v₊, 0, 0)

    # ---- Step 3: second homogeneous complement -------------------------------
    # Same as Step 2 with flipped boundary conditions: u₋(l) = 0, u₋(r) = 1.
    solve!(solver.hu, v₋, 0, 1)
    solve!(solver.hv, v₋, 0, 0)

    # ---- Step 4: influence matrix -------------------------------------------
    # mul!(D₁, v, i) evaluates the i-th row of D₁*v, i.e. the finite-difference
    # approximation to v'(xᵢ). Rows 1 and N correspond to the left (l) and
    # right (r) boundary points respectively.
    #
    # The 2×2 SMatrix is filled column-major: column 1 holds [v₊'(r); v₊'(l)]
    # and column 2 holds [v₋'(r); v₋'(l)]. The right-hand side b enforces the
    # Neumann conditions: δ₊ v₊' + δ₋ v₋' = -vₚ' at each boundary.
    A = SMatrix{2, 2}(mul!(solver.D₁, v₊, N), mul!(solver.D₁, v₊, 1),
                      mul!(solver.D₁, v₋, N), mul!(solver.D₁, v₋, 1))
    b = SVector{2}(-mul!(solver.D₁, vₚ, N), -mul!(solver.D₁, vₚ, 1))
    δ₊, δ₋ = A \ b

    # Write the superposition v = vₚ + δ₊ v₊ + δ₋ v₋ back into r.
    r .= vₚ .+ δ₊ .* v₊ .+ δ₋ .* v₋

    return r
end
