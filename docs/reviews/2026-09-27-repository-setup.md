# Repository setup review — 2026-09-27

## Scope

This assistant review covers the initial repository baseline and its CI setup.
The existing Policy RAG project was transferred without rewriting application
code, tests, dependency declarations, the lockfile, or the README.

The baseline was organized into five commits: project configuration, ingestion
and retrieval, workflow and web interface, tests and evaluation, and documentation.
These commits record the existing project; they do not represent newly implemented
features or an earlier development timeline.

## Transfer and publication checks

- All 70 copied project and local runtime files matched their source SHA-256 hashes.
- The 20 versioned baseline files matched the source byte for byte.
- Prior Git metadata, the nonportable virtual environment, tool caches, and macOS
  metadata were excluded from the transfer. A fresh environment was installed
  from the unchanged lockfile.
- The original Git exclusions keep `.env`, downloaded PDFs, generated search
  indexes, conversation databases, and evaluation output local.
- A targeted credential-pattern scan found no matches in the versioned baseline.
  This is not a comprehensive security audit.

## Validation

Local validation used Python 3.12.14 and uv 0.12.19:

| Check | Result |
| --- | --- |
| `uv sync --locked --python 3.12` | Passed; lockfile unchanged |
| `python -m pytest -q` | 32 passed; 5 dependency deprecation warnings |
| `ruff check app tests` | Passed |
| `ruff format --check app tests` | Passed; all 10 Python files already formatted |
| Tests from a clean `git archive main`, with model downloads disabled | 32 passed; no local PDFs, indexes, `.env`, or sessions required |
| Workflow YAML, action pins, and permissions | Validated locally |

The tests cover retrieval scoping, PDF validation, index staleness, query parsing,
citation and calculation validation, workflow behavior, and HTTP request boundaries.

## CI design

The proposed workflow runs the existing lint, format, and test commands for pull
requests and pushes to `main`. Actions are pinned to commit hashes, job permissions
are read-only, and model downloads are disabled during tests. Dependency versions
come from the existing lockfile.

## Limits

No full Ollama answer-generation evaluation or live browser session was performed
for this repository transfer. Existing evaluation output was copied as local data;
it is not evidence of a new evaluation run. GitHub-hosted CI results must be checked
on the pull request. This review does not constitute independent human approval.
