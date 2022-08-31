module QuantumAnnealingInterface

import Anneal

using MathOptInterface
const MOI = MathOptInterface

using LinearAlgebra
using QuantumAnnealing

Anneal.@anew Optimizer begin
    name = "Simulated Quantum Annealer"
    sense = :min
    domain = :spin
    version = v"0.2.0"
    attributes = begin
        "num_reads"::Integer = 1_000
        "annealing_time"::Float64 = 1.0
        "annealing_schedule"::QuantumAnnealing.AnnealingSchedule = QuantumAnnealing.AS_LINEAR
        # "steps"::Integer = 0
        # "order"::Integer = 4
        # "mean_tol"::Float64 = 1e-6
        # "max_tol"::Float64=1e-4
        # "iteration_limit"::Integer = 100
        # "state_steps"::Union{Integer,Nothing}=nothing
        # kwargs...
    end
end

function sample_state(P::Vector{Float64}, n::Integer)
    @assert length(P) == (m = 2^n)

    # ~*~ Sample p ~ [0, 1] ~*~ #
    p = rand()

    # ~*~ Run Binary Search ~*~ #
    i = first(searchsorted(P, p))

    return 2 .* digits(Int, i - 1; base=2, pad=n) .- 1
end

function Anneal.sample(annealer::Optimizer{T}) where {T}
    # ~*~ Retrieve Model ~*~ #
    h, J = Anneal.ising(Dict, T, annealer)

    ising_model = merge(
        Dict((i,) => w for (i, w) in h),
        J,
    )

    # ~*~ Retrieve Attributes ~*~ #
    n = MOI.get(annealer, MOI.NumberOfVariables())

    num_reads          = MOI.get(annealer, MOI.RawOptimizerAttribute("num_reads"))
    annealing_time     = MOI.get(annealer, MOI.RawOptimizerAttribute("annealing_time"))
    annealing_schedule = MOI.get(annealer, MOI.RawOptimizerAttribute("annealing_schedule"))
    silent             = MOI.get(annealer, MOI.Silent())

    # ~*~ Timing Information ~*~#
    time_data = Dict{String, Any}()

    # ~*~ Run simulation ~*~ #
    ρ = let results = @timed QuantumAnnealing.simulate(
            ising_model,
            annealing_time,
            annealing_schedule;
            silence=silent
        )
        time_data["simulation"] = results.time

        results.value
    end

    # ~*~ Measure probabilities ~*~ #
    p = let results = @timed cumsum(real.(diag(ρ)))
        time_data["measurement"] = results.time

        results.value
    end

    # ~*~ Sample states ~*~ #
    samples = let results = @timed Vector{Int}[sample_state(p, n) for _ = 1:num_reads]
        time_data["sampling"] = results.time

        results.value
    end

    metadata = Dict{String,Any}(
        "time"   => time_data,
        "origin" => "Quantum Annealing Simulation"
    )

    return Anneal.SampleSet{Int,T}(annealer, samples, metadata)
end

end # module
