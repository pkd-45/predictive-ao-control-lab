.PHONY: test validate clean

test:
	PYTHONPATH=src pytest -q

validate:
	PYTHONPATH=src ./scripts/run_validation.sh

clean:
	rm -rf data/*.h5 models/*.pt models/*.npz models/*.json results/*.json results/*.md results/*.sqlite3 results/baseline results/shifted
