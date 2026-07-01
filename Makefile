# FedSSL-MERC — common tasks
PY ?= python3

.PHONY: help install reproduce testbed figures test paper clean

help:
	@echo "FedSSL-MERC make targets:"
	@echo "  make install    Install Python dependencies (requirements.txt)"
	@echo "  make reproduce  Run the full controlled testbed + regenerate figures"
	@echo "  make testbed    Run the testbed experiments only (-> testbed/results/)"
	@echo "  make figures    Regenerate paper figures from testbed results"
	@echo "  make test       Run the testbed sanity checks"
	@echo "  make paper      Build the manuscript PDF (paper/main.pdf)"
	@echo "  make clean      Remove caches and LaTeX build artefacts"

install:
	$(PY) -m pip install -r requirements.txt

reproduce: testbed figures
	@echo "Done. JSON in testbed/results/, figures in paper/figures/."

testbed:
	cd testbed && $(PY) run_experiments.py

figures:
	$(PY) paper/figure_scripts/make_consolidated_figs.py
	$(PY) paper/figure_scripts/make_pipeline.py

test:
	cd testbed && $(PY) tests/test_guard.py

paper:
	cd paper && for i in 1 2 3; do pdflatex -interaction=nonstopmode main.tex >/dev/null; done && echo "Built paper/main.pdf"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -f paper/*.aux paper/*.log paper/*.out paper/*.toc paper/*.pdf
