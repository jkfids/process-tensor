# Contributing

Design philosophy, module organization, and all mathematical and style
conventions are fixed in [DESIGN.md](DESIGN.md); read it before writing
code. This document covers only the development workflow.

## Setup

```bash
git clone https://github.com/jkfids/process-tensor
cd process-tensor
pip install -e '.[dev]'
pre-commit install
```

## Workflow

Development proceeds by branch and pull request, including for the
maintainer. Create a branch per change, open a pull request against
`main`, and merge only once CI (tests, lint, formatting) passes. New
behavior comes with tests; the analytic worked examples in `tests/` are
the model.

## Commit messages

Commits follow [Conventional Commits](https://www.conventionalcommits.org):
a type prefix, an imperative subject, and an optional body.

```
feat: add Kraus representation for channels
fix: correct leg order in link_product for multi-leg contractions
docs: expand bond entropy discussion in the notebook
```

Types in use: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`.

## Versioning

Releases follow [semantic versioning](https://semver.org). The version
lives in `src/processtensor/__init__.py`; releases are annotated git tags
of the form `vX.Y.Z`.
