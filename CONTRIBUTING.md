# Contributing

Use Python 3.11 or 3.12 and install the locked environment with `uv sync --locked`.
The README covers PDF ingestion, model setup, and running the application.

## Make a change

Start from an updated `main` branch and create a focused branch:

```sh
git switch main
git pull --ff-only
git switch -c codex/describe-the-change
```

Keep commits focused and describe the changes actually made. Include dependency
lockfile changes when updating dependencies. Do not commit `.env`, session
databases, downloaded PDFs, generated indexes, model files, or local evaluation
output; the existing `.gitignore` covers the project's generated data.

## Validate

```sh
uv run --no-sync ruff check app tests
uv run --no-sync ruff format --check app tests
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --no-sync python -m pytest -q
```

Tests use fixtures and mocked models, so they do not require Ollama, downloaded
model weights, or the real PDFs. CI runs these checks with Python 3.12 on Linux.
Its installation step downloads dependencies; test execution disables model
downloads.

For changes to retrieval, generation, or citation behavior, also run the relevant
real-corpus cases described in the README. Unit tests alone do not establish
answer quality. Summarize the results in the pull request without committing local
conversation records.

## Submit for review

Push the branch and open a pull request against `main`. Explain the behavior,
validation performed, and any remaining limits. Review the diff and CI results
before merging. An automated check or assistant review is not a human approval.
