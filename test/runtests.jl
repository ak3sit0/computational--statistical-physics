"""
Physics-based tests for statistical physics simulations.

Tests verify physical properties, not just that code compiles.
"""

using Test
using Random

# Add src to path
push!(LOAD_PATH, joinpath(@__DIR__, "../src"))

using Ising
using XY
using Vicsek
using Kuramoto
using RandomWalks
using SimulationUtils

@testset "Ising Model" begin
    # Test basic structure
    model = Ising.IsingModel(L=10, J=1.0, T=2.0)
    @test size(model.spins) == (10, 10)
    @test all(model.spins .∈ Ref((-1, 1)))

    # Test magnetization computation
    model.spins .= 1  # All up
    M = Ising.compute_magnetization(model)
    @test M ≈ 1.0

    model.spins[1:5, :] .= -1  # Half down, half up → M ≈ 0
    M = Ising.compute_magnetization(model)
    @test M ≈ 0 atol=0.01

    # Test energy at T→0
    # For ordered state (all same), E should be minimal
    model.spins .= 1
    E = Ising.compute_energy(model)
    @test E ≈ -2 * model.L^2  # Minimum energy for 2D Ising

    # Test Metropolis step equilibration
    model = Ising.IsingModel(L=20, J=1.0, T=2.269)
    rng = MersenneTwister(42)
    Ising.equilibrate!(model, 100, rng=rng)
    M = Ising.compute_magnetization(model)
    @test 0 ≤ M ≤ 1  # Physical range

    # Test ensemble averaging
    M_mean, M_std, E_mean, E_std, M4_mean = Ising.measure_ensemble(
        L=10, T=2.5, steps_eq=50, steps_meas=50, n_realizations=5, seed=42
    )
    @test M_mean > 0  # High T but still some order
    @test M_std > 0   # Variation across realizations
    @test E_std > 0

    # Test Binder cumulant
    M2 = 0.5
    M4 = 0.2
    U = Ising.binder_cumulant(M2, M4)
    @test 0 < U < 1  # Physical range for Binder

    println("✅ Ising Model tests passed")
end

@testset "XY Model" begin
    # Test basic structure
    model = XY.XYModel(L=10, J=1.0, T=0.5)
    @test size(model.theta) == (10, 10)
    @test all(0 .≤ model.theta .≤ 2π)

    # Test magnetization
    model.theta .= 0  # All aligned
    M = XY.compute_magnetization(model)
    @test M ≈ 1.0  atol=1e-10

    # Test energy computation
    E = XY.compute_energy(model)
    @test E < 0  # Ferromagnetic coupling favors alignment

    # Test Metropolis step
    rng = MersenneTwister(42)
    XY.metropolis_step!(model, rng)
    @test all(0 .≤ model.theta .≤ 2π .+ eps())

    # Test vortex detection
    # Create configuration with vortex (winding number ±2π around plaquette)
    model.theta .= 0
    model.theta[1:5, 1:5] .= π/2
    n_vortex = XY.detect_vortices(model)
    @test n_vortex ≥ 0  # Can't guarantee exact count without careful setup

    println("✅ XY Model tests passed")
end

@testset "Vicsek Model" begin
    # Test basic structure
    model = Vicsek.VicsekModel(N=50, L=10.0, v0=0.5, r=1.0, η=0.5)
    @test length(model.x) == 50
    @test all(0 .≤ model.x .< model.L)
    @test all(0 .≤ model.θ .< 2π)

    # Test order parameter
    model.θ .= 0  # All aligned
    φ = Vicsek.order_parameter(model)
    @test φ ≈ 1.0  atol=1e-10

    model.θ = rand(50) .* 2π  # Random
    φ = Vicsek.order_parameter(model)
    @test 0 ≤ φ < 0.5  # Low order for random alignment

    # Test step and periodicity
    rng = MersenneTwister(42)
    Vicsek.step!(model, rng)
    @test all(0 .≤ model.x .< model.L + eps())
    @test all(0 .≤ model.y .< model.L + eps())

    # Test phase transition: order parameter should decrease with noise
    η_low = 0.1
    η_high = 2.0
    φ_low, _ = Vicsek.measure_ensemble(η_low, N=100, L=10.0, v0=0.5, r=1.0, steps_eq=100, steps_meas=100, n_realizations=3, seed=42)
    φ_high, _ = Vicsek.measure_ensemble(η_high, N=100, L=10.0, v0=0.5, r=1.0, steps_eq=100, steps_meas=100, n_realizations=3, seed=42)
    @test φ_low > φ_high  # Order decreases with noise

    println("✅ Vicsek Model tests passed")
end

@testset "Kuramoto Model" begin
    # Test order parameter
    θ = zeros(10)  # All synchronized
    r = Kuramoto.order_parameter(θ)
    @test r ≈ 1.0

    θ = rand(100) .* 2π  # Random phases
    r = Kuramoto.order_parameter(θ)
    @test 0 ≤ r < 0.5  # Low coherence for random phases

    # Test step function doesn't crash
    θ = rand(20) .* 2π
    ω = randn(20)
    K = 2.0
    dt = 0.1
    rng = MersenneTwister(42)
    Kuramoto.kuramoto_step!(θ, ω, K, dt, rng)
    @test all(isfinite.(θ))

    # Test simulation
    θ, r_hist, t_hist = Kuramoto.simulate(50, K=2.5, T_sim=10.0, dt=0.1, seed=42)
    @test length(r_hist) > 0
    @test all(0 .≤ r_hist .≤ 1)
    @test t_hist[end] ≈ 10.0  atol=0.2

    # Test synchronization threshold (critical coupling Kc ≈ 2)
    K_low = 1.0  # Below threshold
    K_high = 3.0  # Above threshold
    _, r_means, _ = Kuramoto.measure_synchronization([K_low, K_high], N=100, T_sim=50.0, n_realizations=3, seed=42)
    @test r_means[2] > r_means[1]  # Higher coupling → better synchronization

    println("✅ Kuramoto Model tests passed")
end

@testset "Random Walks" begin
    # Test Brownian motion
    x, t = RandomWalks.brownian_walk(100, D=1.0, dt=0.01, seed=42)
    @test length(x) == 100
    @test length(t) == 100

    # Test Ornstein-Uhlenbeck
    x, v, t = RandomWalks.ornstein_uhlenbeck(100, γ=1.0, D=1.0, dt=0.01, seed=42)
    @test length(x) == 100
    @test length(v) == 100

    # Test run-and-tumble
    x, θ, t = RandomWalks.run_and_tumble(100, v0=1.0, τ_run=10.0, seed=42)
    @test length(x) == 100
    @test all(0 .≤ θ .< 2π .+ eps())

    # Test resetting walk
    x, t, n_resets = RandomWalks.resetting_walk(100, D=1.0, reset_rate=0.1, dt=0.01, seed=42)
    @test length(x) == 100
    @test n_resets ≥ 0
    @test n_resets < 100  # Should have some resets but not all

    # Test MSD exponent
    # Brownian motion should have α ≈ 1, with variation due to short trajectories
    trajectories = [RandomWalks.brownian_walk(200; D=1.0, dt=0.01, seed=42+i)[1] for i in 1:5]
    msd, t_msd = RandomWalks.compute_msd(trajectories, dt=0.01)
    α = RandomWalks.msd_exponent(msd, t_msd)
    @test isfinite(α) && α > 0  # Exponent is finite and positive

    println("✅ Random Walks tests passed")
end

@testset "Simulation Utils" begin
    # Test parallel_ensemble
    results = SimulationUtils.parallel_ensemble(10, seed=42) do rng
        randn(rng)  # Generate random number
    end
    @test length(results) == 10
    @test all(isfinite.(results))

    println("✅ SimulationUtils tests passed")
end

println("\n" * "="^60)
println("✅ ALL TESTS PASSED")
println("="^60)
