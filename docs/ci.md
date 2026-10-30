# CI policy

Checks runs all hygiene checks and the library, model, system, and package suites
on three PR combinations. Main and the nightly use seven native combinations
(pairwise BEAM/Nx/runtime boundaries and five host platforms), plus both Alpine
architectures. This is an explicit selection, not a Cartesian product.
The `ci:full` PR label opts into the complete matrix before merge.
ci/compatibility.json owns the combinations and artifact targets. Toolchain
consistency is checked against mise.toml and rust-toolchain.toml. The current
native implementation requires ORT API 19; API 17 support belongs to the runtime
migration. The declared Nx floor is exercised separately from the locked version.

Nightly checks look for an unexpired full-validation artifact on the exact main
SHA, verify that its run succeeded, and reuse it for at most seven days. A skipped
nightly creates no marker. Manual dispatch forces validation. API errors fail
the decision job instead of incorrectly claiming a successful validation.

PR jobs use ephemeral GitHub-hosted runners, read-only permissions, no publishing
credentials, no persisted checkout credentials, and no shared build caches. All
actions use full commit SHAs with release comments. Release privileges belong in
separate jobs. Job timeouts and artifact retention are explicit. Markdown coverage
is included in the job summary, with HTML and scenario reports attached as artifacts.

## Trusted workflow enforcement after merge

The ordinary pull_request workflow is unprivileged but can be changed by the PR.
To require the definition from main, configure an organization ruleset targeting
only RiverFinancial/ortex's main branch, requiring `.github/workflows/checks.yml`
from RiverFinancial/ortex at refs/heads/main. Enable it after the workflow has
landed. Require the `validated` aggregate and review changes to workflows, tool
configuration, and hooks. Organization ruleset availability and installation are
administrative configuration; repository YAML alone does not enforce this policy.
This stack does not change organization settings or bypass lists. Do not substitute
a privileged pull_request_target checkout of PR code for the required workflow.

Advisories are checked independently every day even when source compatibility
validation is skipped. `make setup.audit audit` runs the same checks locally.
Findings produce a report and fail the audit; they are not suppressed to make the
inherited dependency graph appear clean. Releases must satisfy that gate.

## Alpine reproduction

`make test.alpine` uses the same Dockerfile as CI. The base image is pinned by
digest; Rust/Cargo and ORT package versions are explicit. ORT is obtained from
Alpine edge repositories, whose transitive package selection is mutable. These
images establish compatibility, not bit-for-bit reproducibility. glibc libraries
are tested separately from musl. Published native artifacts include only the NIF;
the consumer supplies ONNX Runtime.
