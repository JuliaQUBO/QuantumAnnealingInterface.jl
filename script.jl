using Anneal
using JuMP
using QuantumAnnealingInterface

function build_model(optimizer)
    Q = [-1  2  2
          2 -1  2
          2  2 -1]

    model = Model(optimizer)

    @variable(model, x[1:3], Bin)
    @objective(model, Min, x' * Q * x)

    return model
end

function run_model(model)
    optimize!(model)

    for i = 1:result_count(model)
        xi = value.(model[:x]; result=i)
        yi = objective_value(model; result=i)
        println("f$(xi) = $(yi)")
    end

    println()

    return model
end

qmodel = run_model(build_model(QuantumAnnealingInterface.Optimizer))

emodel = run_model(build_model(ExactSampler.Optimizer))