export HelmoltzSolver, solve!, update!

# Helmoltz solver for differential problems of the form
#   /
#  | θ₀ u''(x) - θ₁ u(x) = r(x), l ≤ x ≤ r
#  | u(l) = u_l
#  | u(r) = u_r
#   \
struct HelmoltzSolver{T, AT<:DiffMatrix{T}, DT<:DiffMatrix}
    A::AT # problem matrix that will be factorised
    D::DT # second order diff matrix (to avoid recomputing it)
    # Construct a solver for a Helmoltz problem on the grid specified by
    # points `xs` (assumed increasing). Use a finite difference stencil of
    # width `width`, corresponding to an order of accuracy `width-1` in most
    # cases. We assume we need to solve a problem where the coefficients
    # are of type `T`, e.g. ComplexF64 or Dual{Float64}.
    function HelmoltzSolver(xs::AbstractVector, width::Int, ::Type{T}=Float64) where {T}
        # create diff matrices. A gets modified and factorised
        # while D is simply stored to avoid recomputing it once A
        # has been factorised if problem coefficients are changed.
        A = DiffMatrix(xs, width, 2; eltype=T)
        D = DiffMatrix(xs, width, 2; eltype=Float64)
        return new{T, typeof(A), typeof(D)}(A, D)
    end
end

# Update the banded matrices with new coefficients and factorise.
function update!(h::HelmoltzSolver{T}, θ₀::Real, θ₁::Real) where {T}
    N = size(h.A, 1)

    # update system
    h.A .= θ₀ .* h.D
    for i in 1:N
        h.A[i, i] -= θ₁
    end

    # apply bc
    h.A[1, :] .= basis_vector(1, N, T)
    h.A[N, :] .= basis_vector(N, N, T)

    # and factorise in place (withouth pivoting)
    lu!(h.A)

    return nothing
end

# Solve the Helmoltz problem with right hand side r
function solve!(h::HelmoltzSolver, r::AbstractVector, u_l::Real, u_r::Real)
    # checks
    size(h.A, 1) == length(r) || throw(ArgumentError("invalid size"))

    # apply booundary conditions
    @inbounds r[1]   = u_l
    @inbounds r[end] = u_r

    # solve in place
    ldiv!(h.A, r)

    return r
end
