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

Gutjahr 1997. *General joint conditional simulations using a fast
Fourier transform method.*
"""
@simsolver SpecGaussSim begin
  @param variogram = GaussianVariogram()
  @param mean = 0.0
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

    # determine variogram model and mean
    γ = varparams.variogram
    μ = varparams.mean

    # check stationarity
    @assert isstationary(γ) "variogram model must be stationary"

    # compute covariances between centroid and all locations
    covs = sill(γ) .- pairwise(γ, pdomain, [c], 1:npts)
    C = reshape(covs, dims)

    # move to frequency domain
    F = sqrt.(abs.(fft(fftshift(C))))
    F[1] = zero(V) # set reference level

    # save preprocessed inputs for variable
    preproc[var] = (γ=γ, μ=μ, F=F)
  end

  preproc
end

function solve_single(problem::SimulationProblem, var::Symbol,
                      solver::SpecGaussSim, preproc)
  # retrieve problem info
  pdomain = domain(problem)
  dims = size(pdomain)

  # unpack preprocessed parameters
  γ, μ, F = preproc[var]

  # result type
  V = variables(problem)[var]

  # perturbation in frequency domain
  P = F .* exp.(im .* angle.(fft(rand(V, dims))))

  # move back to time domain
  Z = real(ifft(P))

  # adjust mean and variance
  σ² = Statistics.var(Z, mean=zero(V))
  Z .= √(sill(γ) / σ²) .* Z .+ μ

  # flatten result
  vec(Z)
end

end
