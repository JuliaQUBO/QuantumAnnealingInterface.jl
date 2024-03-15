module QuantumAnnealingInterface

using Anneal
using LinearAlgebra
using QuantumAnnealing

Anneal.@anew Optimizer begin
    name       = "Simulated Quantum Annealer"
    sense      = :min
    domain     = :spin
    version    = v"0.2.0"
    attributes = begin
        "num_reads"::Integer                                     = 1_000
        "annealing_time"::Float64                                = 1.0
        "annealing_schedule"::QuantumAnnealing.AnnealingSchedule = QuantumAnnealing.AS_LINEAR
        "steps"::Integer                                         = 0
        "order"::Integer                                         = 4
        "mean_tol"::Float64                                      = 1E-6
        "max_tol"::Float64                                       = 1E-4
        "iteration_limit"::Integer                               = 100
        "state_steps"::Union{Integer,Nothing}                    = nothing
    end
end

const PARAMETER_LIST = [
    :steps,
    :order,
    :mean_tol,
    :max_tol,
    :iteration_limit,
    :state_steps,
]

function Anneal.sample(annealer::Optimizer{T}) where {T}
    # ~*~ Retrieve Model ~*~ #
    h, J, α, β  = Anneal.ising(annealer, Dict, T)
    ising_model = merge(h, J)

    # ~*~ Retrieve Attributes ~*~ #
    n = MOI.get(annealer, MOI.NumberOfVariables())
    m = MOI.get(annealer, MOI.RawOptimizerAttribute("num_reads"))

    annealing_time     = MOI.get(annealer, MOI.RawOptimizerAttribute("annealing_time"))
    annealing_schedule = MOI.get(annealer, MOI.RawOptimizerAttribute("annealing_schedule"))
    
    silent = MOI.get(annealer, MOI.Silent())

    # ~*~ Timing Information ~*~ #
    time_data = Dict{String,Any}()

    params = Dict{Symbol,Any}(
        param => MOI.get(annealer, MOI.RawOptimizerAttribute(string(param)))
        for param in PARAMETER_LIST
    )

    # ~*~ Run simulation ~*~ #
    results = @timed QuantumAnnealing.simulate(
        ising_model,
        annealing_time,
        annealing_schedule;
        silence=silent,
        params...
    )
    ρ = results.value

    time_data["simulation"] = results.time

    # ~*~ Measure probabilities ~*~ #
    P = cumsum(real.(diag(ρ)))

    # ~*~ Sample states ~*~ #
    results = @timed sample_states(P, h, J, α, β, n, m)
    samples = results.value

    time_data["sampling"] = results.time

    time_data["effective"] = sum([
        time_data["simulation"],
        time_data["sampling"],
    ])

    metadata = Dict{String,Any}(
        "time"   => time_data,
        "origin" => "Quantum Annealing Simulation"
    )

    return Anneal.SampleSet{T}(samples, metadata)
end

function sample_states(
    P::Vector{Float64},
    h::Dict{Int,T},
    J::Dict{Tuple{Int,Int},T},
    α::T,
    β::T,
    n::Integer,
    m::Integer,
) where {T}
    samples = Vector{Anneal.Sample{T,Int}}(undef, m)

    for i = 1:m
        ψ = sample_state(P, n)

        samples[i] = Anneal.Sample{T}(ψ, α * (Anneal.energy(h, J, ψ) + β))
    end

    return samples
end

@inline function sample_state(P::Vector{Float64}, n::Integer)
    # ~*~ Sample p ~ [0, 1] ~*~ #
    p = rand()

    # ~*~ Run Binary Search ~*~ #
    i = first(searchsorted(P, p))

    return 2 * digits(Int, i - 1; base=2, pad=n) .- 1
end

end # module
