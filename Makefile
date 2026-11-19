# Environment and checks for this repository. Run from the repository root.
ENV ?= swe-akp
SWE_SHA := 7b052c46fca3f85d0ed374e988324081e1bbdb1c
PY := conda run -n $(ENV) python

.PHONY: env-akp pin-swe check

env-akp:  ## conda environment, then the seismology toolkit at the pinned commit
	conda env create -n $(ENV) -f environment-akp.yml
	$(MAKE) pin-swe

pin-swe:
	$(PY) -m pip install --no-deps "seismic-workflow-engine @ git+https://github.com/tjnewton/seismic-workflow-engine.git@$(SWE_SHA)"

check:  ## recompute every number of the paper taken from a table
	$(PY) provenance/check_numbers.py
