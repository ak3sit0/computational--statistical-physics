"""
    RandomWalks

Stochastic processes: Brownian motion, Ornstein-Uhlenbeck, run-and-tumble,
and resetting random walks. Analysis via mean-squared displacement (MSD).
"""
module RandomWalks

using Random, Statistics

export brownian_walk, ornstein_uhlenbeck, run_and_tumble, resetting_walk,
       compute_msd, msd_exponent

"""
    brownian_walk(n_steps; D, dt, seed)

Simple Brownian motion: dx/dt = √(2D) ξ(t)

Returns: trajectory (x, t)
"""
function brownian_walk(n_steps::Int; D::Float64=1.0, dt::Float64=0.01, seed::Int=42)
    rng = MersenneTwister(seed)
    x = cumsum(sqrt(2 * D * dt) .* randn(rng, n_steps))
    t = collect(0:dt:(n_steps-1)*dt)
    return (x, t)
end

"""
    ornstein_uhlenbeck(n_steps; γ, D, dt, seed)

Ornstein-Uhlenbeck process (Brownian motion with velocity damping):
    dx/dt = v
    dv/dt = -γv + √(2γD) ξ(t)

Returns: (x, v, t)
"""
function ornstein_uhlenbeck(n_steps::Int; γ::Float64=1.0, D::Float64=1.0, dt::Float64=0.01, seed::Int=42)
    rng = MersenneTwister(seed)

    x = zeros(n_steps)
    v = zeros(n_steps)
    v[1] = randn(rng)  # Initialize from equilibrium

    for i in 2:n_steps
        dv = -γ * v[i-1] * dt + sqrt(2 * γ * D) * randn(rng) * sqrt(dt)
        v[i] = v[i-1] + dv
        x[i] = x[i-1] + v[i] * dt
    end

    t = collect(0:dt:(n_steps-1)*dt)
    return (x, v, t)
end

"""
    run_and_tumble(n_steps; v0, τ_run, seed)

Run-and-tumble process: particle runs for exponential duration, then tumbles (reorient).

Arguments:
- n_steps: number of discrete steps
- v0: run speed
- τ_run: mean run duration (in steps)

Returns: (x, θ, t)
"""
function run_and_tumble(n_steps::Int; v0::Float64=1.0, τ_run::Float64=10.0, seed::Int=42)
    rng = MersenneTwister(seed)

    x = zeros(n_steps)
    θ = zeros(n_steps)

    current_θ = rand(rng) * 2π
    next_tumble = Int(round(-τ_run * log(rand(rng))))

    for i in 2:n_steps
        if i > next_tumble
            current_θ = rand(rng) * 2π  # Random reorientation
            next_tumble = i + Int(round(-τ_run * log(rand(rng))))
        end

        x[i] = x[i-1] + v0 * cos(current_θ)
        θ[i] = current_θ
    end

    t = collect(1:n_steps)
    return (x, θ, t)
end

"""
    resetting_walk(n_steps; D, reset_rate, dt, seed)

Brownian motion with stochastic resetting: particle resets to origin at random times.

Arguments:
- n_steps: number of steps
- D: diffusion coefficient
- reset_rate: Poisson rate of resets
- dt: time step

Returns: (x, t, n_resets)
"""
function resetting_walk(n_steps::Int; D::Float64=1.0, reset_rate::Float64=0.1, dt::Float64=0.01, seed::Int=42)
    rng = MersenneTwister(seed)

    x = zeros(n_steps)
    n_resets = 0
    reset_prob = reset_rate * dt

    for i in 2:n_steps
        # Brownian step
        dx = sqrt(2 * D * dt) * randn(rng)
        x[i] = x[i-1] + dx

        # Reset with probability reset_prob
        if rand(rng) < reset_prob
            x[i] = 0.0
            n_resets += 1
        end
    end

    t = collect(0:dt:(n_steps-1)*dt)
    return (x, t, n_resets)
end

"""
    compute_msd(trajectories; dt)

Compute mean-squared displacement from ensemble of trajectories.

Arguments:
- trajectories: Vector of position vectors (each a trajectory)
- dt: time step

Returns: (msd, t)
"""
function compute_msd(trajectories::Vector{Vector{Float64}}; dt::Float64=0.01)
    n_traj = length(trajectories)
    max_len = maximum(length(x) for x in trajectories)

    msd = zeros(max_len)
    count = zeros(Int, max_len)

    for x in trajectories
        for i in 1:length(x)
            msd[i] += x[i]^2
            count[i] += 1
        end
    end

    msd = msd ./ count
    t = collect(0:dt:(max_len-1)*dt)
    return (msd, t)
end

"""
    msd_exponent(msd, t)

Estimate diffusion exponent α where MSD ~ t^α.
Uses linear fit on log-log plot over the middle 50% of trajectory.

Returns: α
"""
function msd_exponent(msd::Vector{Float64}, t::Vector{Float64})::Float64
    # Use middle 50% of points (skip initial and final transients)
    n = length(t)
    idx_start = Int(round(n * 0.25))
    idx_end = Int(round(n * 0.75))

    t_fit = t[idx_start:idx_end]
    msd_fit = msd[idx_start:idx_end]

    # Filter out zeros and infinities
    valid = (t_fit .> 0) .& (msd_fit .> 0) .& isfinite.(msd_fit)
    t_fit = t_fit[valid]
    msd_fit = msd_fit[valid]

    if length(t_fit) < 2
        return 1.0  # Default to normal diffusion
    end

    # Linear fit in log-log
    log_t = log.(t_fit)
    log_msd = log.(msd_fit)
    α = (log_msd[end] - log_msd[1]) / (log_t[end] - log_t[1])

    return α
end

end  # module RandomWalks
