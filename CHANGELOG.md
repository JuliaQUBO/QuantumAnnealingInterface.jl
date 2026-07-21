# Changelog

## Unreleased

## v0.2.1 - 2026-07-21

- No breaking changes.
- Support LinearSolve 4.2 while retaining compatibility with LinearSolve 3.82.
- Make compatibility metadata tests robust to future compat expansion.
- Keep the full Julia and operating-system CI matrix while avoiding unnecessary
  precompilation of the complete transitive test environment.
- Harden TagBot permissions and update GitHub Actions dependencies.

## v0.2.0 - 2026-06-13

- Update compatibility to QUBODrivers 0.6.1 and replace the exact LinearSolve
  pin with lower-bounded compat.
- Add QUBODrivers benchmark metadata, seed, final-read, and size-limit support.
- Add README installation instructions for JuMP and QuantumAnnealingInterface.
- Add TagBot automation for registry-triggered tags.
- Add Dependabot maintenance for Julia environments and GitHub Actions.
- Update GitHub Actions dependencies.

## v0.1.0 - 2026-05-28

- Raise the minimum supported Julia version to 1.10.
- Require QUBODrivers 0.5 now that QUBOTools 0.12.1 resolves with the modern SciML stack on Julia 1.10.
- Test Julia 1.10 and the latest stable Julia release in CI.
- Clean the QUBODrivers badge link and author metadata.
