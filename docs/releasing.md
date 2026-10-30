# Releasing the fork

Use atomic conventional commits: `feat`, `fix`, `perf`, `docs`, `build`, `ci`,
`test`, `refactor`, `style`, and `chore`. An optional scope describes the affected
area. `!` or a `BREAKING CHANGE:` footer marks a breaking change. Release notes
bucket these commits, retain other subjects under Other, and include artifact
SHA-256 values. These conventions do not require rewriting existing history.

Before the first fork release, decide its Hex package identity and ownership,
choose a new version, and update Mix, Cargo.toml, Cargo.lock, and the changelog.
The current version is inherited from main; this stack does not claim that a new
0.1.10 release can be published over the existing public package.

`make package` builds the source Hex archive. `scripts/package.py nif --target
TARGET` archives the already tested library in priv/native. `make inventory`
writes a CycloneDX source dependency inventory, including build-time Rust crates.
It is not an inventory of the precise code retained by the linker in each NIF.
ONNX Runtime is consumer supplied and is explicitly outside the package inventory.

`scripts/package.py assemble --all` requires all seven target archives, checks
their member names and binary architectures, builds the Hex package, and writes
artifacts.json and SHA256SUMS. Partial local assembly omits --all and is explicitly
marked incomplete. A release must never use partial assembly. Native archive
metadata is normalized; no claim of reproducible native compilation is made.

Configure required reviewers and tag restrictions on the release environment,
then explicitly enable ORTEX_STAGE_RELEASES in repository variables. Push a
matching vVERSION tag whose commit is already on the fork's main branch.
The staging workflow runs full compatibility and dependency audits, builds docs
and the inventory, assembles the complete bundle, generates notes, and creates a
draft GitHub release after protected-environment approval. Provenance covers the
bundle. Review it before publishing the GitHub release. No publication occurs as
part of developing this stack. Dependency advisories must be addressed before
the release audit can pass.

Hex publication is disabled unless ORTEX_HEX_PUBLISH is explicitly set to true
in repository variables. Before enabling it, configure ownership, a Hex workload
identity restricted to this repository and publish.yml, and required reviewers
and tag restrictions on the GitHub release environment. Public package workload
identities cannot bootstrap a package's first release. The current publisher's
package scope is hexpm/ortex; change that scope and Mix package metadata together
if River chooses a distinct package identity. Do not configure an upstream
package's credentials or ownership as part of foundation work.

Publishing the reviewed GitHub release triggers the protected Hex job. It checks
tag ancestry and matching versions, verifies every downloaded artifact's
attestation and checksums, obtains a short-lived package-scoped token through
OIDC, and publishes the package and docs. No long-lived Hex key is stored.
