# QuantumAnnealingInterface.jl
JuMP interface for LANL's [QuantumAnnealing.jl](https://github.com/lanl-ansi/QuantumAnnealing.jl) (ft. [Anneal.jl](https://github.com/psrenergy/Anneal.jl))

## How to
```julia
using JuMP
using QuantumAnnealingInterface

model = Model(QuantumAnnealingInterface.Optimizer)

Q = [ -1  2  2
       2 -1  2
       2  2 -1 ]

@variable(model, x[1:3], Bin)
@objective(model, Min, x' * Q * x)

optimize!(model)
```
