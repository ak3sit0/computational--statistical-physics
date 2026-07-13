"""
    SimulationUtils

Reusable utilities for statistical physics simulations: macros for periodic boundaries,
parallel ensemble execution with reproducible RNG, and timing utilities.
"""
module SimulationUtils

using Random
using Base.Threads

export @pbc, @debug, @timed, parallel_ensemble

"""
    @pbc(i, L)

Periodic boundary condition: wraps index i to [1, L].
"""
macro pbc(i, L)
    return esc(quote
        (($i - 1) % $L) + 1
    end)
end

"""
    @debug(expr)

Zero-cost debug macro: only prints if DEBUG environment variable is set.
"""
macro debug(expr)
    return quote
        if get(ENV, "DEBUG", "false") == "true"
            println($(string(expr)), " = ", $(esc(expr)))
        end
    end
end

"""
    @timed(expr)

Measure execution time of an expression.
"""
macro timed(expr)
    return quote
        t0 = time()
        result = $(esc(expr))
        elapsed = time() - t0
        println("⏱ ", $(string(expr)), " took ", round(elapsed, digits=3), "s")
        result
    end
end

"""
    parallel_ensemble(f, n_realizations; seed=42)

Run n_realizations independent simulations in parallel with independent RNG per realization.
Each realization gets a seeded RNG for reproducibility.

Usage:
    results = parallel_ensemble(100, seed=42) do rng
        # simulation code using rng
    end
"""
function parallel_ensemble(f::Function, n_realizations::Int; seed::Int=42)
    results = Vector{Any}(undef, n_realizations)

    @threads for i in 1:n_realizations
        rng = Random.MersenneTwister(seed + i - 1)
        results[i] = f(rng)
    end

    return results
end

end  # module SimulationUtils
