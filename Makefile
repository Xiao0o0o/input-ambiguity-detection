PYTHON ?= python

.PHONY: help install install-dev nltk-data test syntax-check quick-eval rules-demo check build dist-check clean

help:
	@echo "make quick-eval    AUROC of the stored quick-start run (CPU, seconds)"
	@echo "make rules-demo    print the black-box rule variants for a few questions"
	@echo "make test          unit tests (pytest)"
	@echo "make check         syntax-check + test + quick-eval"
	@echo "make dist-check    build sdist + wheel into dist/ and run twine check"

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

nltk-data:
	$(PYTHON) -c "import nltk; [nltk.download(x, quiet=True) for x in ('wordnet', 'omw-1.4', 'averaged_perceptron_tagger_eng')]"

test:
	$(PYTHON) -m pytest -q

syntax-check:
	$(PYTHON) -m compileall -q input_ambiguity blackbox whitebox data/build tests

quick-eval:
	PYTHON=$(PYTHON) bash quickstart/run_quickstart.sh auroc

rules-demo:
	$(PYTHON) blackbox/perturbation_rules.py

check: syntax-check test quick-eval

build:
	rm -rf dist
	$(PYTHON) -m build

dist-check: build
	$(PYTHON) -m twine check dist/*

clean:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache quickstart/outputs build dist *.egg-info
