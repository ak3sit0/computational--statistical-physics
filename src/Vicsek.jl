"""
    Vicsek

Self-propelled particle model exhibiting order-disorder phase transition out of equilibrium.
Particles move with constant speed and align with their neighbors plus noise.
"""
module Vicsek

using Random, Statistics, LinearAlgebra

export VicsekModel, step!, equilibrate!, measure_ensemble, order_parameter

"""
    VicsekModel

Mutable struct for Vicsek model.

Fields:
- N::Int: number of particles
- L::Float64: box size
- v0::Float64: particle speed
- r::Float64: interaction radius
- η::Float64: noise strength
- x, y, θ::Vector: particle positions and angles
"""
mutable struct VicsekModel
    N::Int
    L::Float64
    v0::Float64
    r::Float64
    η::Float64
    x::Vector{Float64}
    y::Vector{Float64}
    θ::Vector{Float64}

    function VicsekModel(; N::Int, L::Float64, v0::Float64, r::Float64, η::Float64)
        x = rand(Float64, N) .* L
        y = rand(Float64, N) .* L
        θ = rand(Float64, N) .* 2π
        new(N, L, v0, r, η, x, y, θ)
    end
end

"""
    step!(model, rng)

Perform one time step: align with neighbors + noise, then move.
"""
function step!(model::VicsekModel, rng::AbstractRNG=Random.GLOBAL_RNG)
    N, L, v0, r, η = model.N, model.L, model.v0, model.r, model.η
    x, y, θ = model.x, model.y, model.θ

    # Alignment: average angle of neighbors
    θ_new = zeros(Float64, N)
    for i in 1:N
        # Find neighbors within radius r
        neighbors_idx = findall(j -> j ≠ i && sqrt((x[i]-x[j])^2 + (y[i]-y[j])^2) < r, 1:N)

        if isempty(neighbors_idx)
            push!(neighbors_idx, i)  # At least self
        end

        # Average angle of neighbors
        vx = sum(cos(θ[j]) for j in neighbors_idx)
        vy = sum(sin(θ[j]) for j in neighbors_idx)
        θ_new[i] = atan(vy, vx) + (rand(rng) - 0.5) * η
    end

    # Update angles
    model.θ .= θ_new

    # Move
    x_new = x .+ v0 .* cos.(θ_new)
    y_new = y .+ v0 .* sin.(θ_new)

    # Periodic boundaries
    model.x .= mod.(x_new, L)
    model.y .= mod.(y_new, L)

    nothing
end

"""
    order_parameter(model)

Normalized velocity magnitude: φ = |⟨v⟩| / v0
"""
function order_parameter(model::VicsekModel)::Float64
    vx = mean(cos.(model.θ))
    vy = mean(sin.(model.θ))
    return sqrt(vx^2 + vy^2)
end

"""
    equilibrate!(model, steps; rng)

Equilibrate system.
"""
function equilibrate!(model::VicsekModel, steps::Int; rng::AbstractRNG=Random.GLOBAL_RNG)
    for _ in 1:steps
        step!(model, rng)
    end
    return model
end

"""
    measure_ensemble(η, steps_eq, steps_meas, n_realizations; N, L, v0, r, seed)

Measure order parameter over ensemble.
"""
function measure_ensemble(
    η::Float64;
    N::Int=300,
    L::Float64=10.0,
    v0::Float64=0.5,
    r::Float64=1.0,
    steps_eq::Int=500,
    steps_meas::Int=1000,
    n_realizations::Int=50,
    seed::Int=42
)
    φ_vals = Float64[]

    for realization in 1:n_realizations
        rng = MersenneTwister(seed + realization - 1)

        model = VicsekModel(N=N, L=L, v0=v0, r=r, η=η)
        equilibrate!(model, steps_eq, rng=rng)

        # Measure
        φ_sample = Float64[]
        for _ in 1:steps_meas
            step!(model, rng)
            push!(φ_sample, order_parameter(model))
        end

        push!(φ_vals, mean(φ_sample))
    end

    φ_mean = mean(φ_vals)
    φ_std = std(φ_vals) / sqrt(n_realizations)

    return (φ_mean, φ_std)
end

end  # module Vicsek
