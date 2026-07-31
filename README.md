# QuantumAnnealingInterface.jl
[![QUBODRIVERS](https://img.shields.io/badge/Powered%20by-QUBODrivers.jl-%20%234063d8)](https://github.com/JuliaQUBO/QUBODrivers.jl)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20434507.svg)](https://doi.org/10.5281/zenodo.20434507)

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

## Citation

`QuantumAnnealingInterface.jl` is the JuMP and QUBODrivers integration layer.
For general use of this integration, cite its
[Zenodo concept DOI](https://doi.org/10.5281/zenodo.20434507). The concept DOI
is the evergreen software identifier and resolves to the latest archived
release. Citation metadata is available in [`CITATION.cff`](CITATION.cff).

For exact-version reproducibility, cite the corresponding Zenodo version DOI.
The archive for `v0.2.1` is
[10.5281/zenodo.21480962](https://doi.org/10.5281/zenodo.21480962).

The simulation engine is the separate LANL
[`QuantumAnnealing.jl`](https://github.com/lanl-ansi/QuantumAnnealing.jl)
project. If your work relies on that simulator, also cite Morrell et al.,
["QuantumAnnealing: A Julia Package for Simulating Dynamics of Transverse Field
Ising Models"](https://arxiv.org/abs/2404.14501), following the upstream citation
guidance. Cite both works when your results depend on the simulator and this
integration.

Maintained releases continue under the existing concept DOI. If this package is
deprecated, its final archive and concept DOI will be frozen rather than deleted
or replaced.
