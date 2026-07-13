"""
    XY

2D XY model (planar spins) with Metropolis algorithm. Exhibits the
Kosterlitz-Thouless phase transition mediated by vortex unbinding.
"""
module XY

using Random, Statistics
using Base.Threads

export XYModel, metropolis_step!, equilibrate!, compute_magnetization,
       compute_energy, detect_vortices

"""
    XYModel

Mutable struct for 2D XY lattice (continuous spin angles).

Fields:
- L::Int: lattice size (L×L)
- J::Float64: coupling strength
- T::Float64: temperature
- theta::Matrix{Float64}: spin angles [0, 2π)
"""
mutable struct XYModel
    L::Int
    J::Float64
    T::Float64
    theta::Matrix{Float64}

    function XYModel(; L::Int, J::Float64=1.0, T::Float64=0.5)
        theta = rand(Float64, L, L) .* 2π
        new(L, J, T, theta)
    end
end

"""
    compute_energy(model)

Energy of XY model: E = -J ∑⟨ij⟩ cos(θᵢ - θⱼ)
"""
function compute_energy(model::XYModel)::Float64
    L, J, theta = model.L, model.J, model.theta
    E = 0.0
    for i in 1:L, j in 1:L
        ip = i == L ? 1 : i + 1
        jp = j == L ? 1 : j + 1
        E -= J * (cos(theta[i,j] - theta[ip,j]) + cos(theta[i,j] - theta[i,jp]))
    end
    return E
end

"""
    compute_magnetization(model)

Magnetization magnitude: M = |∑ᵢ e^(iθᵢ)| / L²
"""
function compute_magnetization(model::XYModel)::Float64
    L = model.L
    mx = sum(cos.(model.theta)) / L^2
    my = sum(sin.(model.theta)) / L^2
    return sqrt(mx^2 + my^2)
end

"""
    metropolis_step!(model, rng)

Single Metropolis step: propose random angle rotation, accept with probability.
"""
function metropolis_step!(model::XYModel, rng::AbstractRNG=Random.GLOBAL_RNG)::Bool
    L, J, T, theta = model.L, model.J, model.T, model.theta

    i, j = rand(rng, 1:L), rand(rng, 1:L)
    ip = i == L ? 1 : i + 1
    im = i == 1 ? L : i - 1
    jp = j == L ? 1 : j + 1
    jm = j == 1 ? L : j - 1

    theta_old = theta[i, j]
    theta_new = theta_old + (rand(rng) - 0.5) * π  # Proposal: ±π/2

    # Energy difference
    E_old = -J * (cos(theta_old - theta[ip,j]) + cos(theta_old - theta[im,j]) +
                   cos(theta_old - theta[i,jp]) + cos(theta_old - theta[i,jm]))
    E_new = -J * (cos(theta_new - theta[ip,j]) + cos(theta_new - theta[im,j]) +
                   cos(theta_new - theta[i,jp]) + cos(theta_new - theta[i,jm]))

    dE = E_new - E_old

    if dE < 0 || rand(rng) < exp(-dE / T)
        theta[i, j] = theta_new
        return true
    end
    return false
end

"""
    equilibrate!(model, steps; rng, verbose)

Equilibrate model.
"""
function equilibrate!(model::XYModel, steps::Int; rng::AbstractRNG=Random.GLOBAL_RNG, verbose::Bool=false)
    for step in 1:steps
        metropolis_step!(model, rng)
        if verbose && step % (steps ÷ 10) == 0
            println("Equilibration: $step / $steps")
        end
    end
    return model
end

"""
    detect_vortices(model)

Count vortices via plaquette analysis.
A vortex is detected where |Q| > π for vorticity Q.

Returns: number of vortex cores
"""
function detect_vortices(model::XYModel)::Int
    L, theta = model.L, model.theta
    n_vortices = 0

    for i in 1:L, j in 1:L
        ip = i == L ? 1 : i + 1
        jp = j == L ? 1 : j + 1

        # Vorticity around plaquette
        Q = theta[i,j] - theta[ip,j] + theta[ip,jp] - theta[i,jp]
        # Unwrap to [-π, π]
        Q = mod(Q + π, 2π) - π

        if abs(Q) > π/2
            n_vortices += 1
        end
    end

    return n_vortices
end

end  # module XY
