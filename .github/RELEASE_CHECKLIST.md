# Release checklist

Use this checklist for every tagged release. The Zenodo concept DOI
`10.5281/zenodo.20434507` is the permanent software identifier; publish every
supported release as a new version of that record rather than creating another
concept DOI.

## Before tagging

- [ ] Confirm at least two active JuliaQUBO maintainers have **Can manage**
      access to the Zenodo record through their personal accounts. Never share
      passwords or API tokens.
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

- [ ] Record the new version DOI in the GitHub release notes and update the
      version DOI in `CITATION.cff` during release preparation or the immediate
      citation-metadata follow-up.
- [ ] Verify the version DOI resolves to the exact release and the concept DOI
      resolves to the latest archived release:

      ```sh
      curl --fail --location --output /dev/null https://doi.org/<version-doi>
      curl --fail --location --output /dev/null https://doi.org/10.5281/zenodo.20434507
      ```

- [ ] Download the Zenodo archive, compare its SHA-256 checksum with the
      official GitHub release archive, and confirm its `Project.toml` version
      matches the release tag.
- [ ] Treat a missing or mismatched Zenodo version as incomplete release work;
      repair the existing record before announcing archival completion.

Continue adding versions while the package is maintained. If the package is
deprecated, publish its final supported release and freeze the existing concept
record; do not delete it or mint a replacement DOI.
