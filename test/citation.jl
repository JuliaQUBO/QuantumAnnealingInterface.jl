@testset "Citation metadata" begin
    root = dirname(@__DIR__)
    read_text(path) = replace(read(path, String), "\r\n" => "\n")

    project   = TOML.parsefile(joinpath(root, "Project.toml"))
    citation  = read_text(joinpath(root, "CITATION.cff"))
    readme    = read_text(joinpath(root, "README.md"))
    checklist = read_text(joinpath(root, ".github", "RELEASE_CHECKLIST.md"))
    has_citation(needle) = occursin(needle, citation)
    has_readme(needle) = occursin(needle, readme)
    has_checklist(needle) = occursin(needle, checklist)

    concept_doi = "10.5281/zenodo.20434507"
    version_doi = "10.5281/zenodo.21480962"
    version     = string(project["version"])

    @test has_citation("doi: \"$concept_doi\"")
    @test has_citation("value: \"$version_doi\"")
    @test has_citation("version: \"$version\"")
    @test has_citation("https://arxiv.org/abs/2404.14501")

    @test has_readme("badge/DOI/$concept_doi.svg")
    @test has_readme("doi.org/$concept_doi")
    @test has_readme("doi.org/$version_doi")
    @test has_readme("https://arxiv.org/abs/2404.14501")

    @test has_checklist(concept_doi)
    @test has_checklist("Can manage")
    @test has_checklist("cffconvert --validate --infile CITATION.cff")
    @test has_checklist("api/records/20434507/versions")
    @test has_checklist("freeze the existing concept")
end
