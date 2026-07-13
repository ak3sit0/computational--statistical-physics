"""
    Ising

2D Ising model with Metropolis algorithm. Core functions for equilibration,
measurement of observables (magnetization, energy, specific heat, Binder cumulant),
and finite-size critical analysis.
"""
module Ising

using Random, Statistics
using Base.Threads

export IsingModel, metropolis_step!, equilibrate!, measure_ensemble,
       compute_magnetization, compute_energy, binder_cumulant

"""
    IsingModel

Mutable struct for 2D Ising lattice state.

Fields:
- L::Int: lattice size (L×L)
- J::Float64: coupling strength
- T::Float64: temperature
- spins::Matrix{Int8}: spin configuration (±1)
"""
mutable struct IsingModel
    L::Int
    J::Float64
    T::Float64
    spins::Matrix{Int8}

    function IsingModel(; L::Int, J::Float64=1.0, T::Float64=2.0)
        spins = rand([-1, 1], L, L)
        new(L, J, T, spins)
    end
end

"""
    neighbors(spins, i, j, L)

Sum of neighbors of spin at (i,j) with periodic boundaries.
"""
function neighbors(spins::Matrix, i::Int, j::Int, L::Int)
    ip = i == L ? 1 : i + 1
    im = i == 1 ? L : i - 1
    jp = j == L ? 1 : j + 1
    jm = j == 1 ? L : j - 1
    return spins[ip, j] + spins[im, j] + spins[i, jp] + spins[i, jm]
end

"""
    compute_energy(model)

Total energy of the configuration: E = -J ∑⟨ij⟩ σᵢσⱼ / 2
"""
function compute_energy(model::IsingModel)::Float64
    L, J, spins = model.L, model.J, model.spins
    E = 0.0
    for i in 1:L, j in 1:L
        E -= J * spins[i, j] * neighbors(spins, i, j, L)
    end
    return E / 2  # Each bond counted twice
end

"""
    compute_magnetization(model)

Magnetization: M = |∑ᵢ σᵢ| / L²
"""
function compute_magnetization(model::IsingModel)::Float64
    return abs(sum(model.spins)) / (model.L ^ 2)
end

"""
    metropolis_step!(model, rng)

Single Metropolis step: attempt to flip one random spin.
Returns true if flip was accepted.
"""
function metropolis_step!(model::IsingModel, rng::AbstractRNG=Random.GLOBAL_RNG)::Bool
    L, J, T, spins = model.L, model.J, model.T, model.spins

    # Pick random spin
    i, j = rand(rng, 1:L), rand(rng, 1:L)
    Si = spins[i, j]

    # Energy cost of flip
    dE = 2 * J * Si * neighbors(spins, i, j, L)

    # Metropolis criterion
    if dE < 0 || rand(rng) < exp(-dE / T)
        spins[i, j] = -Si
        return true
    end
    return false
end

"""
    equilibrate!(model, steps; rng, verbose)

Equilibrate model for given steps.
"""
function equilibrate!(model::IsingModel, steps::Int; rng::AbstractRNG=Random.GLOBAL_RNG, verbose::Bool=false)
    for step in 1:steps
        metropolis_step!(model, rng)
        if verbose && step % (steps ÷ 10) == 0
            println("Equilibration: $step / $steps")
        end
    end
    return model
end

"""
    measure_ensemble(L, T, steps_eq, steps_meas, n_realizations; seed)

Measure observables over ensemble of independent realizations.

Returns: (M_mean, M_std, E_mean, E_std, M4_mean)
"""
function measure_ensemble(;
    L::Int,
    T::Float64,
    steps_eq::Int,
    steps_meas::Int,
    n_realizations::Int,
    seed::Int=42
)
    M_vals = Float64[]
    E_vals = Float64[]
    M4_vals = Float64[]

    for realization in 1:n_realizations
        rng = MersenneTwister(seed + realization - 1)

        model = IsingModel(L=L, J=1.0, T=T)
        equilibrate!(model, steps_eq, rng=rng)

        # Measure
        M_meas = Float64[]
        E_meas = Float64[]
        for _ in 1:steps_meas
            metropolis_step!(model, rng)
            push!(M_meas, compute_magnetization(model))
            push!(E_meas, compute_energy(model))
        end

        push!(M_vals, mean(M_meas))
        push!(E_vals, mean(E_meas))
        push!(M4_vals, mean(M_meas .^ 4))
    end

    M_mean = mean(M_vals)
    M_std = std(M_vals) / sqrt(n_realizations)
    E_mean = mean(E_vals)
    E_std = std(E_vals) / sqrt(n_realizations)
    M4_mean = mean(M4_vals)

    return (M_mean, M_std, E_mean, E_std, M4_mean)
end

"""
    binder_cumulant(M_vals, M4_vals)

Binder cumulant: U = 1 - ⟨M⁴⟩ / (3⟨M²⟩²)

The crossing of this cumulant across lattice sizes L identifies the critical temperature Tc.
"""
function binder_cumulant(M2_avg::Float64, M4_avg::Float64)::Float64
    if M2_avg ≈ 0
        return 0.0
    end
    return 1.0 - M4_avg / (3 * M2_avg^2)
end

end  # module Ising
