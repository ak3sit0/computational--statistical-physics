"""
    Kuramoto

Kuramoto model of coupled phase oscillators exhibiting spontaneous synchronization.
Transitions from incoherent to synchronized state at critical coupling Kc ≈ 2.
"""
module Kuramoto

using Random, Statistics

export order_parameter, kuramoto_step!, simulate, measure_synchronization

"""
    order_parameter(θ)

Kuramoto order parameter: r = |⟨e^(iθ)⟩|
Measures global phase coherence.
"""
function order_parameter(θ::Vector{Float64})::Float64
    N = length(θ)
    r = abs(sum(exp(im * θᵢ) for θᵢ in θ)) / N
    return real(r)
end

"""
    kuramoto_step!(θ, ω, K, dt, rng)

One integration step using Euler-Maruyama for the Kuramoto model:
    dθᵢ/dt = ωᵢ + (K/N) ∑ⱼ sin(θⱼ - θᵢ)

Arguments:
- θ::Vector: current phases
- ω::Vector: natural frequencies
- K::Float64: coupling strength
- dt::Float64: time step
- rng::AbstractRNG: random number generator
"""
function kuramoto_step!(θ::Vector{Float64}, ω::Vector{Float64}, K::Float64, dt::Float64, rng::AbstractRNG=Random.GLOBAL_RNG)
    N = length(θ)
    r = order_parameter(θ)
    ψ = atan(sum(sin.(θ)) / sum(cos.(θ)))  # Mean phase

    for i in 1:N
        dθ = ω[i] + K * sin(ψ - θ[i])
        θ[i] += dθ * dt
        θ[i] = mod(θ[i], 2π)
    end

    nothing
end

"""
    simulate(N; K, T_sim, dt, ω_dist, seed)

Simulate Kuramoto model for time T_sim.

Arguments:
- N::Int: number of oscillators
- K::Float64: coupling strength
- T_sim::Float64: simulation time
- dt::Float64: time step
- ω_dist: function that returns natural frequencies (default: normal distribution)

Returns: (θ_final, r_history, t_history)
"""
function simulate(
    N::Int;
    K::Float64,
    T_sim::Float64,
    dt::Float64=0.1,
    ω_dist=()->randn(),
    seed::Int=42
)
    rng = MersenneTwister(seed)
    Random.seed!(rng, seed)

    # Initialize
    θ = rand(rng, Float64, N) .* 2π
    ω = [ω_dist() for _ in 1:N]

    n_steps = Int(ceil(T_sim / dt))
    r_history = Float64[]
    t_history = Float64[]

    for step in 1:n_steps
        kuramoto_step!(θ, ω, K, dt, rng)
        push!(r_history, order_parameter(θ))
        push!(t_history, step * dt)
    end

    return (θ, r_history, t_history)
end

"""
    measure_synchronization(K_range; N, T_sim, n_realizations)

Measure order parameter r(K) over a range of coupling strengths.

Returns: (K_vals, r_mean, r_std)
"""
function measure_synchronization(
    K_range::Vector{Float64};
    N::Int=100,
    T_sim::Float64=100.0,
    dt::Float64=0.1,
    n_realizations::Int=10,
    seed::Int=42
)
    r_means = Float64[]
    r_stds = Float64[]

    for K in K_range
        r_vals = Float64[]

        for realization in 1:n_realizations
            _, r_hist, _ = simulate(N, K=K, T_sim=T_sim, dt=dt, seed=seed + realization)
            # Use final steady state (last 20% of trajectory)
            idx_start = Int(round(0.8 * length(r_hist)))
            r_steady = mean(r_hist[idx_start:end])
            push!(r_vals, r_steady)
        end

        push!(r_means, mean(r_vals))
        push!(r_stds, std(r_vals) / sqrt(n_realizations))
    end

    return (K_range, r_means, r_stds)
end

end  # module Kuramoto
