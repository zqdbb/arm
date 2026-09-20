PYTHON ?= $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python; fi)

.PHONY: compile test validate

compile:
	$(PYTHON) -m compileall src tests scripts

test:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'

validate:
	$(PYTHON) scripts/validate_scaffold.py