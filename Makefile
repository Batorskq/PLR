.PHONY: help check clean

help:
	@echo "PLR repository commands"
	@echo ""
	@echo "  make check   Compile Python entry points and utilities"
	@echo "  make clean   Remove generated outputs and Python caches"
	@echo ""
	@echo "Run one method:"
	@echo "  scripts/run_plr.sh mr method_pl_ce /path/to/model --k 16 --subset 1000"
	@echo ""
	@echo "Run all methods:"
	@echo "  scripts/run_all_methods.sh mr /path/to/model --k 16 --subset 1000"

check:
	python -m py_compile *.py utils/*.py

clean:
	rm -rf __pycache__ utils/__pycache__ .pytest_cache .mypy_cache .ruff_cache
	rm -rf results results_*
