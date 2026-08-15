"""Single source of truth for the package version.

Read at build time by hatchling (``[tool.hatch.version]`` in
``pyproject.toml``) and at runtime by :mod:`processtensor`. Bump it in its own
release change, never on a feature branch.
"""

__version__ = "0.1.0"
