.PHONY: data test
data:
	python -m src.data.make_dataset --prepare
test:
	python -m pytest -q
