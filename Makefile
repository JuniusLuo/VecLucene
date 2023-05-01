
test:
	#PYTHONPATH=. pytest # or python3 -m pytest
	# add -s to print to console, add --log-cli-level=DEBUG to show debug logs
	python3 -m pytest tests/model/
	python3 -m pytest tests/vectorstore/
	python3 -m pytest tests/index/test_vector_index.py
	# This test reports 2000+ DeprecationWarning from pylucene.
	# Catch and filter warnings inside the test only ingore < 100 warnings.
	# most warnings are probably reported when JVM shuts down. Looks not able
	# to explicitly shutdown JVM in the test. So use disable-warnings for this
	# test only.
	python3 -m pytest --disable-warnings tests/index/test_index.py

test_openai:
	# please export your openai key first, OPENAI_API_KEY=your_key
	python3 -m pytest tests/openai/test_model_embedding.py
	python3 -m pytest --disable-warnings tests/openai/test_index.py


PYTHON := python3
site_packages_path := $(shell $(PYTHON) -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
coverage:
	coverage run --omit "$(site_packages_path)/*" -m pytest --disable-warnings
	coverage report --omit "$(site_packages_path)/*"
