# ------------------------------------------------------------------
# Licensed under the ISC License. See LICENCE in the project root.
# ------------------------------------------------------------------

module SpectralGaussianSimulation

using GeoStatsBase
import GeoStatsBase: preprocess, solve_single

using Variography
using Statistics
using FFTW

export SpecGaussSim

"""
    SpecGaussSim(var₁=>param₁, var₂=>param₂, ...)

Spectral Gaussian simulation (a.k.a. FFT simulation).

## Parameters

* `variogram` - theoretical variogram (default to GaussianVariogram())
* `mean`      - mean of Gaussian field (default to 0)

### References

TODO
"""
@simsolver SpecGaussSim begin
  @param variogram = GaussianVariogram()
  @param mean = nothing
end

function preprocess(problem::SimulationProblem, solver::SpecGaussSim)
  # retrieve problem info
  pdomain = domain(problem)
  npts = npoints(pdomain)
  dims = size(pdomain)
  center = CartesianIndex(dims .÷ 2)
  c = LinearIndices(dims)[center]

  # result of preprocessing
  preproc = Dict{Symbol,NamedTuple}()

  for (var, V) in variables(problem)
    # get user parameters
    if var ∈ keys(solver.params)
      varparams = solver.params[var]
    else
      varparams = SpecGaussSimParam()
    end

    # determine variogram model
    γ = varparams.variogram

    # check stationarity
    @assert isstationary(γ) "variogram model must be stationary"

    # determine field mean
    μ = isnothing(varparams.mean) : V(0) : varparams.mean

    # compute covariances between centroid and all locations
    C = sill(γ) .- pairwise(γ, pdomain, [c], 1:npts)
    C = reshape(C, dims)

    # compute spectrum of given covariance
    F = sqrt.(abs.(fft(fftshift(C)) ./ npts))
    F[1] = zero(V) # set reference level

    # save preprocessed inputs for var
    preproc[var] = (γ=γ, μ=μ, F=F)
  end

  preproc
end

function solve_single(problem::SimulationProblem, var::Symbol,
                      solver::SeqGaussSim, preproc)
  # retrieve problem info
  pdomain = domain(problem)
  npts = npoints(pdomain)
  dims = size(pdomain)

  # unpack preprocessed parameters
  γ, μ, F = preproc[var]

  # result type
  V = variables(problem)[var]

  # perturbation in frequency domain
  P = F .* exp.(1im .* angle.(fft(rand(V, dims))))
  R = real(ifft(P .* npts))

  # adjust mean and variance
  (sill(γ) / var(R, mean=zero(V))) .* R .+ μ
end
