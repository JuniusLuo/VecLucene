
.PHONY: test
test:
	#PYTHONPATH=. pytest # or python3 -m pytest
	python3 -m pytest tests/model/
	python3 -m pytest tests/vectorstore/
	python3 -m pytest tests/index/test_vector_index.py
	# This test reports 2000+ DeprecationWarning from pylucene.
	# Catch and filter warnings inside the test only ingore < 100 warnings.
	# most warnings are probably reported when JVM shuts down. Looks not able
	# to explicitly shutdown JVM in the test. So use disable-warnings for this
	# test only.
	python3 -m pytest --disable-warnings tests/index/test_index.py

