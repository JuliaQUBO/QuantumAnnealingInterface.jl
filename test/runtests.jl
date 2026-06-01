using Test

@testset "README installation docs" begin
    readme = read(joinpath(dirname(@__DIR__), "README.md"), String)
    installation = findfirst("## Installation", readme)
    usage = findfirst("## Usage", readme)

    @test occursin("Pkg.add(\"QuantumAnnealingInterface\")", readme)
    @test occursin("Pkg.add(\"JuMP\")", readme)
    @test installation !== nothing
    @test usage !== nothing

    if installation !== nothing && usage !== nothing
        @test first(installation) < first(usage)
    end
end

using QuantumAnnealingInterface: MOI, QUBODrivers, QuantumAnnealingInterface

QUBODrivers.test(QuantumAnnealingInterface.Optimizer) do model
    MOI.set(model, MOI.Silent(), true)
end
