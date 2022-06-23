module QuantumAnnealingInterface

using Anneal
using QuantumAnnealing
using MathOptInterface
const MOI = MathOptInterface
const VI = MOI.VariableIndex
const ROA = MOI.RawOptimizerAttribute

struct Optimizer{T} <: Anneal.AbstractSampler{T}
    x::Dict{VI, Int}
    y::Dict{Int, VI}
    s::T
    Q::Dict{Tuple{Int, Int}, T}
    c::T

    attrs::Dict{String, Any}
end

function Anneal.sample(annealer::Optimizer)
    _, h, J, c = Anneal.ising_normal_form(annealer)
    isingmodel = merge(h, J)

    @show QuantumAnnealing.simulate(
        isingmodel,
        MOI.get(annealer, ROA("annealing_time")),
        MOI.get(annealer, ROA("annealing_schedule")),
    )
end

end # module
