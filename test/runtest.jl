using FDHelmoltzSolver
using BenchmarkTools
using LinearAlgebra
using FFTW
using Test

include("test_chebcoeffs.jl")
include("test_quasitridiag.jl")
include("test_helmoltz.jl")
include("test_coupled.jl")