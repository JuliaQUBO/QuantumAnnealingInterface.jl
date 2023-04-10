# QuantumAnnealingInterface.jl
[![QUBODRIVERS](https://img.shields.io/badge/Powered%20by-QUBODrivers.jl-%20%234063d8)](https://github.com/psrenergy/QUBODrivers.jl)

JuMP interface for LANL's [QuantumAnnealing.jl](https://github.com/lanl-ansi/QuantumAnnealing.jl)

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
