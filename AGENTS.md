# Versioning and releases

- Keep significant user-facing changes in `CHANGELOG.md` under `Unreleased`
  and update the package version in `pyproject.toml` when warranted. Use
  semantic versioning; group related work into a coherent upcoming version.
- Publish GitHub Releases at stable milestones, not after every commit.
  Before publishing, verify the intended `main` commit, tests, lint, package
  builds, CI, and existing Git tags and GitHub Releases.
- At release time, move the existing `Unreleased` changes into one dated version
  section, retain an empty `Unreleased` section, and update comparison links.
  Match the package version, `v`-prefixed Git tag and GitHub Release. Use that
  CHANGELOG section as release notes; do not duplicate entries.
- Keep the README focused on current behavior and onboarding. Use the CHANGELOG,
  Git history and thematic documentation for development history and reports.
