using QuantumAnnealingInterface: MOI, QUBODrivers, QuantumAnnealingInterface

QUBODrivers.test(QuantumAnnealingInterface.Optimizer) do model
    MOI.set(model, MOI.Silent(), true)
end