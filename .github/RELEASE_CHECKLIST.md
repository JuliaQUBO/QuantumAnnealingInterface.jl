# Release checklist

Use this checklist for every tagged release. The Zenodo concept DOI
`10.5281/zenodo.20434507` is the permanent software identifier; publish every
supported release as a new version of that record rather than creating another
concept DOI.

## Before tagging

- [ ] Confirm at least two active JuliaQUBO maintainers have **Can manage**
      access to the Zenodo record through their personal accounts. Never share
      passwords or API tokens. If no second steward is available after
      documented outreach and one direct follow-up, record the temporary
      single-manager exception in JuliaQUBO/QUBO.jl#66 and review it annually
      or whenever maintainership changes.
- [ ] Update the version in `Project.toml`, the version and release date in
      `CITATION.cff`, and the release notes in `CHANGELOG.md`.
- [ ] Keep the concept DOI in `CITATION.cff` and the README badge unchanged.
- [ ] Validate the citation metadata with
      `cffconvert --validate --infile CITATION.cff`.
- [ ] Run the package test suite with
      `julia --project -e 'using Pkg; Pkg.test()'`.

## After publishing the GitHub release

- [ ] Confirm Zenodo created a new version under concept record `20434507`.
      Query the public versions endpoint and check that the latest record's
      `metadata.version` matches the GitHub release tag and its `conceptrecid`
      remains `20434507`:

      ```sh
      curl --fail --location https://zenodo.org/api/records/20434507/versions
      ```

      If Zenodo responds slowly or returns HTTP 429, wait for its `Retry-After`
      interval and retry once before treating the archive as missing.

- [ ] Record the new version DOI in the GitHub release notes, and update the
      version DOI in `CITATION.cff` together with the exact-version paragraph
      in `README.md`, during release preparation or the immediate
      citation-metadata follow-up.
- [ ] Verify the version DOI resolves to the exact release and the concept DOI
      resolves to the latest archived release:

      ```sh
      curl --fail --location --output /dev/null https://doi.org/<version-doi>
      curl --fail --location --output /dev/null https://doi.org/10.5281/zenodo.20434507
      ```

- [ ] Confirm the new record's tag, version, MIT license, repository URL, Julia
      package UUID (`d8651ca0-0f93-43a8-b432-4648ccd57c1f`), and creator names,
      affiliations, and ORCIDs match `Project.toml` and `CITATION.cff`.
- [ ] Download both the Zenodo and official GitHub release archives, compute
      their SHA-256 checksums locally, compare them, and confirm the archived
      `Project.toml` version matches the release tag. Zenodo's file API exposes
      an MD5 checksum, not SHA-256.
- [ ] Treat a missing or mismatched Zenodo version as incomplete release work;
      repair the existing record before announcing archival completion.

Continue adding versions while the package is maintained. If the package is
deprecated, publish its final supported release and freeze the existing concept
record; do not delete it or mint a replacement DOI.
