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
    # Construct a solver from a second-order DiffMatrix `D`. `A` is an
    # uninitialised copy of `D` with element type `T` that will be overwritten
    # and factorised on each call to `update!`.
    function HelmoltzSolver(D::DiffMatrix, ::Type{T}=Float64) where {T}
        A = similar(D, T)
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
