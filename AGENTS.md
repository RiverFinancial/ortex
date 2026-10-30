# Working on Ortex

Read docs/development.md for setup and supported environments. Use `make help`
for the commands implemented by this branch. Run the relevant checks before
committing; use atomic conventional commits. Keep runtime/API changes separate
from development infrastructure.

The native crate lives in native/ortex. Run Cargo from that directory so its
.cargo/config.toml applies. Changes crossing the BEAM/native boundary need tests
through the public Elixir API. Process-wide runtime configuration requires a
fresh VM. Keep inference workloads and scheduler measurements under bench/.

Fixtures and examples must use public or synthetic inputs, with recorded source,
license, and reference results. Never download models implicitly during tests.

Use the RiverFinancial fork. Do not publish packages or releases unless the
user explicitly requests publication. Do not rewrite other branches as part of
foundation work.
