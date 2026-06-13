# QuantumAnnealingInterface.jl
[![QUBODRIVERS](https://img.shields.io/badge/Powered%20by-QUBODrivers.jl-%20%234063d8)](https://github.com/JuliaQUBO/QUBODrivers.jl)

JuMP interface for LANL's [QuantumAnnealing.jl](https://github.com/lanl-ansi/QuantumAnnealing.jl)

## Installation
```julia
import Pkg
Pkg.add("JuMP")
Pkg.add("QuantumAnnealingInterface")
```

## Usage
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

## Simulation Size Limit

QuantumAnnealing.jl's state-vector backend builds a dense `2^n` by `2^n`
density matrix. `QuantumAnnealingInterface.Optimizer` rejects models with more
than 12 variables by default through the `"max_variables"` optimizer attribute.
Raise this limit only when the corresponding memory cost is acceptable.
