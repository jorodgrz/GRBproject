# Test and validation targets for the GRB classification project.
#
# Usage:
#   make test    -- full pytest suite (skips data-bound tests if Data/ empty)
#   make smoke   -- fast subset (no data, no compas pin), under ~5 s
#   make ci      -- exactly the set CI runs (no slow, no data, no compas pin)
#
# Conda is the canonical dependency manager (see environment.yml); the
# Makefile assumes the active environment already has pytest installed.

.PHONY: test smoke ci clean

test:
	python -m pytest

smoke:
	python -m pytest -m "not slow and not requires_data and not requires_compas"

ci:
	python -m pytest -m "not slow and not requires_data and not requires_compas" --tb=short

clean:
	find tests -type d -name __pycache__ -exec rm -rf {} +
	find . -maxdepth 2 -type d -name .pytest_cache -exec rm -rf {} +
