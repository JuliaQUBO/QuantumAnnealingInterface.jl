module QuantumAnnealingInterface

using LinearAlgebra
using Random
using QuantumAnnealing
import QUBODrivers:
    MOI,
    QUBODrivers,
    QUBOTools,
    Sample,
    SampleSet,
    @setup,
    sample

const DEFAULT_MAX_VARIABLES = 12
const QUANTUM_ANNEALING_VERSION = something(Base.pkgversion(QuantumAnnealing), v"0.2.0")

@setup Optimizer begin
    name       = "Simulated Quantum Annealer"
    version    = QUANTUM_ANNEALING_VERSION
    attributes = begin
        RandomSeed["seed"]::Union{Integer,Nothing}                = nothing
        NumberOfReads["num_reads"]::Integer                      = 1_000
        "annealing_time"::Float64                                = 1.0
        "annealing_schedule"::QuantumAnnealing.AnnealingSchedule = QuantumAnnealing.AS_LINEAR
        "steps"::Integer                                         = 0
        "order"::Integer                                         = 4
        "mean_tol"::Float64                                      = 1E-6
        "max_tol"::Float64                                       = 1E-4
        "iteration_limit"::Integer                               = 100
        "state_steps"::Union{Integer,Nothing}                    = nothing
        MaxVariables["max_variables"]::Integer                   = DEFAULT_MAX_VARIABLES
    end
end

QUBODrivers.honors_final_reads(::Type{<:Optimizer}) = true

const ATTR_LIST = [
    :steps,
    :order,
    :mean_tol,
    :max_tol,
    :iteration_limit,
    :state_steps,
]

function sample(sampler::Optimizer{T}) where {T}
    # Retrieve Model
    n, h, J, α, β = QUBOTools.ising(sampler, :dict; sense = :min)
    _check_problem_size(n, MOI.get(sampler, MaxVariables()))
    ising_model = merge(h, J)

    # Retrieve Attributes
    m                  = MOI.get(sampler, QUBODrivers.FinalNumberOfReads())
    silent             = MOI.get(sampler, MOI.Silent())
    seed               = MOI.get(sampler, QUBODrivers.RandomSeed())
    annealing_time     = MOI.get(sampler, MOI.RawOptimizerAttribute("annealing_time"))
    annealing_schedule = MOI.get(sampler, MOI.RawOptimizerAttribute("annealing_schedule"))
    rng                = _sample_rng(seed)

    _check_final_number_of_reads(m)

    attrs = Dict{Symbol,Any}(
        attr => MOI.get(
            sampler,
            MOI.RawOptimizerAttribute(string(attr))
        )
        for attr in ATTR_LIST
    )

    # Run simulation
    results = @timed QuantumAnnealing.simulate(
        ising_model,
        annealing_time,
        annealing_schedule;
        silence=silent,
        attrs...
    )
    simulation_time = results.time

    # Measurement & Probabilities
    ρ = results.value
    P = cumsum(real.(diag(ρ)))

    # Sample states
    results = @timed sample_states(rng, P, h, J, α, β, n, m)
    samples = results.value

    sampling_time = results.time

    # Write metadata
    metadata = QUBODrivers._sampler_metadata(
        origin                = "QuantumAnnealingInterface.jl",
        algorithm_name        = "Simulated Quantum Annealer",
        backend_name          = "QuantumAnnealing.jl",
        backend_version       = QUANTUM_ANNEALING_VERSION,
        execution_mode        = "state_vector_simulation",
        optimizer_evaluations = m,
        number_of_reads       = m,
        final_number_of_reads = length(samples),
        status                = "locally_solved",
        termination_status    = MOI.LOCALLY_SOLVED,
    )
    metadata["time"] = Dict{String,Any}(
        "effective" => simulation_time + sampling_time,
        "breakdown" => Dict{String,Any}(
            "QuantumAnnealingInterface" => Dict{String,Any}(
                "simulation" => simulation_time,
                "sampling"   => sampling_time,
            ),
        ),
    )

    return SampleSet{T}(samples, metadata; sense = :min, domain = :spin)
end

function sample_states(
    rng::Random.AbstractRNG,
    P::Vector{Float64},
    h::Dict{Int,T},
    J::Dict{Tuple{Int,Int},T},
    α::T,
    β::T,
    n::Integer,
    m::Integer,
) where {T}
    samples = Vector{Sample{T,Int}}(undef, m)

    for i = 1:m
        ψ = sample_state(rng, P, n)
        λ = QUBOTools.value(ψ, h, J, α, β)

        samples[i] = Sample{T,Int}(ψ, λ)
    end

    return samples
end

function sample_state(rng::Random.AbstractRNG, P::Vector{Float64}, n::Integer)
    # Sample p ~ [0, 1]
    p = rand(rng)

    # Run Binary Search
    i = first(searchsorted(P, p))

    # Format as spin vector i.e. ψ ∈ {±1}ⁿ
    ψ = 2 * digits(Int, i - 1; base=2, pad=n) .- 1

    return ψ
end

function _sample_rng(seed::Union{Integer,Nothing})
    if isnothing(seed)
        return Random.default_rng()
    else
        return Random.MersenneTwister(seed)
    end
end

function _check_final_number_of_reads(m::Integer)
    if m < 0
        throw(ArgumentError("Final number of reads must be a non-negative integer"))
    end

    return nothing
end

function _check_problem_size(n::Integer, max_variables::Integer)
    if max_variables < 0
        throw(ArgumentError("max_variables must be a non-negative integer"))
    end

    if n > max_variables
        throw(
            ArgumentError(
                "QuantumAnnealingInterface state-vector simulation supports at most " *
                "$(max_variables) variables by default; got $(n). The backend builds " *
                "a dense 2^n by 2^n density matrix. Set optimizer attribute " *
                "\"max_variables\" higher only when that memory cost is acceptable.",
            ),
        )
    end

    return nothing
end

end # module
