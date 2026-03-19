.PHONY: check test

check:
	python -m py_compile train.py train_common_test_i.py train_common_test_i_fft.py train_common_test_i_hog.py train_common_test_i_radial.py run_lsst_mock_pipeline.py

test:
	pytest -q
