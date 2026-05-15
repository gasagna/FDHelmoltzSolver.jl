export HelmoltzSolver, solve!, update!

"""
    HelmoltzSolver{T, AT<:DiffMatrix{T}, DT<:DiffMatrix}

Solver for the one-dimensional Helmholtz boundary value problem

```
  θ₀ u''(x) - θ₁ u(x) = r(x),   l ≤ x ≤ r
  u(l) = u_l
  u(r) = u_r
```

where `θ₀` and `θ₁` are real scalar coefficients and `r` is a given forcing
function. The solution `u` and the right-hand side `r` are represented as
vectors of values at the grid points used to construct the solver.

# Discretisation

The second derivative `u''` is approximated by a compact high-order finite
difference stencil stored in the `DiffMatrix` field `D₂`. On each call to
`update!`, the system matrix

```
    A = θ₀ D₂ - θ₁ I
```

is assembled into the working copy `A` and then factorised in-place by a
no-pivoting banded LU routine. Dirichlet boundary conditions are enforced by
replacing the first and last rows of `A` with the corresponding canonical
basis vectors before factorisation, so that the first and last equations
become simply `u[1] = u_l` and `u[N] = u_r`.

# Fields

- `A`  — working copy of the system matrix; overwritten on every `update!` call
          with `θ₀ D₂ - θ₁ I` (interior rows) and identity rows (boundary rows),
          then replaced in-place by its LU factors.
- `D₂` — the second-order finite-difference matrix built from the grid; never
          modified after construction and reused across `update!` calls.

# Type parameters

- `T`  — element type of the system matrix `A` (e.g. `Float64`, `ComplexF64`).
- `AT` — concrete matrix type of `A` (must support in-place `lu!` and `ldiv!`).
- `DT` — concrete matrix type of `D₂` (typically `DiffMatrix{Float64}`, since
          grid geometry is always real, but any `AbstractMatrix` is accepted).

# Constructor

    HelmoltzSolver(D₂::AbstractMatrix, [T=Float64])

Build a solver from a pre-constructed second-order differentiation matrix `D₂`.
Any `AbstractMatrix` is accepted; in practice `D₂` is a `DiffMatrix` from
`FDGrids.jl` so that `lu!` and `ldiv!` dispatch to the fast banded routines.
`D₂` is stored as the reference stencil and is never modified. The working
matrix `A` is allocated via `similar(D₂, T)` and overwritten on the first call
to `update!` before any solve.

# Examples

```julia
using FDHelmoltzSolver, FDGrids, LinearAlgebra

xs = collect(range(-1, 1; length=201))
D₂ = DiffMatrix(xs, 7, 2)         # 7-point (6th-order) second-derivative matrix

h  = HelmoltzSolver(D₂)           # Float64 solver
update!(h, 3.0, 2.0)              # assemble and factorise  3D₂ - 2I

r  = exp.(xs)                      # right-hand side: exp(y)
solve!(h, r, exp(-1), exp(1))      # solve in-place; r is overwritten with u
```

See also: [`update!`](@ref), [`solve!`](@ref)
"""
struct HelmoltzSolver{T, AT<:AbstractMatrix{T}, DT<:AbstractMatrix, LT}
    A  :: AT
    D₂ :: DT
    lu :: LT    # DiffMatrixLU wrapping A; shares A.coeffs — ldiv! uses this
    function HelmoltzSolver(D₂::AbstractMatrix, ::Type{T}=Float64) where {T}
        # Allocate A as an uninitialised DiffMatrix of element type T with the
        # same stencil width and grid size as D₂. A will be overwritten in full
        # on every update! call, so there is no need to zero-initialise it here.
        A  = similar(D₂, T)
        lu = DiffMatrixLU(A)    # wraps A; A.coeffs and lu.factors.coeffs alias the same array
        return new{T, typeof(A), typeof(D₂), typeof(lu)}(A, D₂, lu)
    end
end

"""
    update!(h::HelmoltzSolver{T}, θ₀, θ₁)

Assemble the system matrix for coefficients `(θ₀, θ₁)` and factorise it
in-place, preparing `h` for subsequent calls to `solve!`.

The interior rows of `h.A` are set to `θ₀ * h.D₂ - θ₁ * I`, encoding the
discretised operator `θ₀ u'' - θ₁ u` at each interior grid point. The first
and last rows are then overwritten with canonical basis vectors to enforce
homogeneous-compatible Dirichlet boundary conditions:

```
    row 1: [1, 0, 0, …, 0]   →   u[1]   = u_l  (applied later in solve!)
    row N: [0, 0, 0, …, 1]   →   u[N]   = u_r  (applied later in solve!)
```

After the boundary rows are set, the matrix is factorised in-place by
[`LinearAlgebra.lu!`](@ref), which overwrites `h.A` with the no-pivot banded
LU factors. The right-hand side boundary values `u_l` and `u_r` are not
needed here; they are injected into the right-hand side vector inside
[`solve!`](@ref).

`update!` must be called at least once before the first `solve!`, and again
whenever `θ₀` or `θ₁` change.

# Arguments

- `h`  — the solver to update.
- `θ₀` — coefficient of the second-derivative term.
- `θ₁` — coefficient of the zeroth-derivative (mass) term.

# Examples

```julia
update!(h, 1.0, 0.0)   # pure Laplacian:  u'' = r
update!(h, 1.0, 4.0)   # Helmholtz:       u'' - 4u = r
update!(h, 0.0, -1.0)  # identity:        u = -r  (trivial but valid)
```
"""
function update!(h::HelmoltzSolver{T}, θ₀::Real, θ₁::Real) where {T}
    N = size(h.A, 1)

    # Build the interior operator θ₀ D₂ - θ₁ I by scaling and copying the
    # stencil coefficients, then subtracting θ₁ from each diagonal entry.
    # Operating directly on the underlying Vector avoids the DiffMatrix
    # broadcast machinery (_bc_arg per element) and lets the compiler emit
    # a SIMD-optimised scalar-vector multiply.
    @inbounds h.A.coeffs .= T(θ₀) .* h.D₂.coeffs
    for i in 1:N
        h.A[i, i] -= θ₁
    end

    # Enforce Dirichlet boundary conditions using the tau (row-replacement)
    # method: replace the first and last rows with canonical basis vectors so
    # that the linear system directly imposes u[1] = u_l and u[N] = u_r.
    h.A[1, :] .= zero(T); h.A[1, 1] = one(T)
    h.A[N, :] .= zero(T); h.A[N, N] = one(T)

    # Factorise h.A in-place with a no-pivoting banded LU routine. This
    # overwrites the compact stencil coefficients with the LU factors and
    # prepares the matrix for repeated O(N·WIDTH) triangular solves.
    # h.lu.factors aliases h.A (same coeffs array), so h.lu is implicitly
    # updated and ready for ldiv! without any extra copy.
    lu!(h.A)

    return nothing
end

"""
    solve!(h::HelmoltzSolver, r, u_l, u_r) -> r

Solve the Helmholtz problem in-place, overwriting `r` with the solution `u`.

`update!` must have been called prior to `solve!` to assemble and factorise
the system matrix for the desired coefficients `(θ₀, θ₁)`.

Boundary conditions are enforced by setting `r[1] = u_l` and `r[end] = u_r`
before the triangular solve. This works in conjunction with the row-replacement
performed in `update!`: the first and last rows of the LU-factorised matrix are
identity rows, so the triangular solve simply reads off those values unchanged.

The interior entries `r[2:end-1]` must contain the right-hand side `r(x)` of
the Helmholtz equation evaluated at the interior grid points on entry. On exit,
all entries of `r` hold the solution `u(x)`.

# Arguments

- `h`   — a solver that has been prepared by a prior call to `update!`.
- `r`   — on entry: right-hand side values at the grid points; on exit: the
           solution `u`. Length must match the grid used to construct `h`.
- `u_l` — Dirichlet value at the left boundary `x = l`.
- `u_r` — Dirichlet value at the right boundary `x = r`.

# Returns

`r`, overwritten with the solution `u`.

# Examples

```julia
r = @. -2*(1 + π^2)*sin(π*xs)   # forcing for u = sin(πx)
solve!(h, r, 0.0, 0.0)           # homogeneous Dirichlet BCs
```
"""
function solve!(h::HelmoltzSolver, r::AbstractVector, u_l::Real, u_r::Real)
    size(h.A, 1) == length(r) || throw(ArgumentError("invalid size"))

    # Inject the Dirichlet boundary values into the right-hand side. The
    # corresponding rows of h.A are identity rows (set in update!), so the
    # triangular solve will produce u[1] = u_l and u[end] = u_r exactly.
    @inbounds r[1]   = u_l
    @inbounds r[end] = u_r

    # Solve the banded system in-place using the LU factors stored in h.A.
    # On exit, r holds the solution u at every grid point.
    ldiv!(h.lu, r)

    return r
end
