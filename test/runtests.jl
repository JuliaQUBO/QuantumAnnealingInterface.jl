using Test
import TOML

_compat_entries(value::AbstractString) = strip.(split(value, ','))

@testset "README installation docs" begin
    readme = read(joinpath(dirname(@__DIR__), "README.md"), String)
    installation = findfirst("## Installation", readme)
    usage = findfirst("## Usage", readme)

    @test occursin("Pkg.add(\"QuantumAnnealingInterface\")", readme)
    @test occursin("Pkg.add(\"JuMP\")", readme)
    @test occursin("\"max_variables\"", readme)
    @test installation !== nothing
    @test usage !== nothing

    if installation !== nothing && usage !== nothing
        @test first(installation) < first(usage)
    end
end

using QuantumAnnealingInterface: MOI, QUBODrivers, QuantumAnnealingInterface
const QUBOTools = QuantumAnnealingInterface.QUBOTools

const VI = MOI.VariableIndex
const SAT{T} = MOI.ScalarAffineTerm{T}
const SQT{T} = MOI.ScalarQuadraticTerm{T}
const SQF{T} = MOI.ScalarQuadraticFunction{T}

function configure_test_optimizer!(model)
    MOI.set(model, MOI.Silent(), true)
    MOI.set(model, MOI.RawOptimizerAttribute("steps"), 2)
    MOI.set(model, MOI.RawOptimizerAttribute("mean_tol"), 1E-3)
    MOI.set(model, MOI.RawOptimizerAttribute("max_tol"), 1E-3)

    return model
end

function small_qubo_model(::Type{T} = Float64) where {T}
    model = MOI.instantiate(QuantumAnnealingInterface.Optimizer; with_bridge_type = T)
    Q = T[
        1 -2 3
        0 -1 -2
        0 0 2
    ]
    n = size(Q, 1)
    x, _ = MOI.add_constrained_variables(model, fill(MOI.ZeroOne(), n))

    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(
        model,
        MOI.ObjectiveFunction{SQF{T}}(),
        SQF{T}(
            SQT{T}[SQT{T}(Q[i, j], x[i], x[j]) for i = 1:n for j = 1:n if i != j],
            SAT{T}[SAT{T}(Q[i, i], x[i]) for i = 1:n],
            zero(T),
        ),
    )
    configure_test_optimizer!(model)

    return model
end

function solution_sampleset(model)
    raw = MOI.get(model, MOI.RawSolver())

    return QUBOTools.solution(raw)
end

@testset "Compatibility metadata" begin
    root = dirname(@__DIR__)
    project = TOML.parsefile(joinpath(root, "Project.toml"))
    deps = project["deps"]
    compat = project["compat"]
    linearsolve_compat = _compat_entries(compat["LinearSolve"])

    @test "0.6.1" in _compat_entries(compat["QUBODrivers"])
    @test haskey(deps, "LinearSolve")
    @test "3.82" in linearsolve_compat
    @test "4.2" in linearsolve_compat
    @test all(entry -> !startswith(entry, "="), linearsolve_compat)
    @test haskey(deps, "Random")
end

@testset "Benchmark metadata contract" begin
    @test QUBODrivers.supports_seed(QuantumAnnealingInterface.Optimizer)
    @test QUBODrivers.honors_final_reads(QuantumAnnealingInterface.Optimizer)
    @test !QUBODrivers.enforces_time_limit(QuantumAnnealingInterface.Optimizer)

    model = small_qubo_model()
    MOI.set(model, QUBODrivers.FinalNumberOfReads(), 3)
    MOI.set(model, QUBODrivers.RandomSeed(), 1234)
    MOI.optimize!(model)

    sampleset = solution_sampleset(model)
    metadata = QUBOTools.metadata(sampleset)
    time_breakdown = metadata["time"]["breakdown"]["QuantumAnnealingInterface"]

    @test length(sampleset) <= 3
    @test QUBOTools.reads(sampleset) == 3
    @test isempty(QUBODrivers.validate_metadata(metadata))
    @test metadata["origin"] == "QuantumAnnealingInterface.jl"
    @test metadata["algorithm"]["name"] == "Simulated Quantum Annealer"
    @test metadata["backend"]["name"] == "QuantumAnnealing.jl"
    @test metadata["reads"]["number_of_reads"] == 3
    @test metadata["reads"]["final_number_of_reads"] == 3
    @test metadata["seeds"]["sampler"] == 1234
    @test metadata["time"]["effective"] > 0.0
    @test time_breakdown["simulation"] >= 0.0
    @test time_breakdown["sampling"] >= 0.0
end

@testset "Simulation size limit" begin
    model = small_qubo_model()
    MOI.set(model, MOI.RawOptimizerAttribute("max_variables"), 2)

    @test_throws ArgumentError MOI.optimize!(model)
end

QUBODrivers.test(QuantumAnnealingInterface.Optimizer; benchmark_conformance = true) do model
    configure_test_optimizer!(model)
end
