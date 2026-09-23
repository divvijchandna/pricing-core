"""Smoke test: the package is installed and importable."""

import pricing_core


def test_version_is_set_from_installed_metadata() -> None:
    # "+unknown" means the fallback fired, i.e. the package is not installed and
    # the tests are importing a bare source tree.
    assert isinstance(pricing_core.__version__, str)
    assert pricing_core.__version__
    assert "+unknown" not in pricing_core.__version__
