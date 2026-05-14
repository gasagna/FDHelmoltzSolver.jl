using StaticArrays

export CoupledHelmoltzSolver

# Solve the problem
#
#       / θ₀ u''(x) - θ₁ u(x)        = r(x), l ≤ x ≤ r
#      |  θ₂ v''(x) - θ₃ v(x) - u(x) = 0
#       \ v(l) = v(r) = v'(l) = v'(r) = 0
#
# using a high-order finite difference method with an influence matrix technique,
# and return the solution `v`, overwriting the input argument `r`.

struct CoupledHelmoltzSolver{T, H<:HelmoltzSolver, DT<:AbstractMatrix, V<:AbstractVector{T}}
    hu::H            # helmoltz solver for the u variable
    hv::H            # helmoltz solver for the v variable
     D::DT           # first-order differentiation matrix for boundary derivative evaluation
    vs::NTuple{3, V} # temporary storage vectors
    function CoupledHelmoltzSolver(xs::AbstractVector, width::Int, ::Type{T}=Float64) where {T}
        D2 = DiffMatrix(xs, width, 2; eltype=Float64)
        hu = HelmoltzSolver(D2, T)
        hv = HelmoltzSolver(copy(D2), T)
        D  = DiffMatrix(xs, width, 1; eltype=Float64)
        vs = ntuple(i->zeros(T, length(xs)), 3)
        return new{T, typeof(hu), typeof(D), Vector{T}}(hu, hv, D, vs)
    end
end

function update!(solver::CoupledHelmoltzSolver, θs::NTuple{4, <:Real})
    θ₀, θ₁, θ₂, θ₃ = θs
    update!(solver.hu, θ₀, θ₁)
    update!(solver.hv, θ₂, θ₃)
    return nothing
end

function solve!(solver::CoupledHelmoltzSolver, r::AbstractVector)
    # aliases for the partial solutions
    vₚ, v₊, v₋ = solver.vs
    N = length(r)

    vₚ .= r
    v₊ .= 0
    v₋ .= 0

    # ~~~~ Solve the three sub-problems ~~~~
    # particular: u driven by r, then v driven by u
    solve!(solver.hu, vₚ, 0, 0)
    solve!(solver.hv, vₚ, 0, 0)

    # homogeneous complement v₊: u BCs (1, 0), then v driven by that u
    solve!(solver.hu, v₊, 1, 0)
    solve!(solver.hv, v₊, 0, 0)

    # homogeneous complement v₋: u BCs (0, 1), then v driven by that u
    solve!(solver.hu, v₋, 0, 1)
    solve!(solver.hv, v₋, 0, 0)

    # ~~~~ Influence matrix: enforce v'(l) = v'(r) = 0 ~~~~
    # mul!(D, v, i) evaluates row i of D*v, i.e. the derivative at grid point i.
    # Column-major SMatrix: A[:, 1] from v₊ derivatives, A[:, 2] from v₋ derivatives.
    A = SMatrix{2, 2}(mul!(solver.D, v₊, N), mul!(solver.D, v₊, 1),
                      mul!(solver.D, v₋, N), mul!(solver.D, v₋, 1))
    b = SVector{2}(-mul!(solver.D, vₚ, N), -mul!(solver.D, vₚ, 1))
    δ₊, δ₋ = A\b

    r .= vₚ .+ δ₊ .* v₊ .+ δ₋ .* v₋

    return r
end
