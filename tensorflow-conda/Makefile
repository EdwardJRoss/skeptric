environment.yml: index.qmd
	./extract_code.sh

setup-environment.sh: index.qmd
	./extract_code.sh

test_gpu.py: index.qmd
	./extract_code.sh

test_train.py: index.qmd
	./extract_code.sh

test: environment.yml setup-environment.sh test_gpu.py test_train.py
	bash setup-environment.sh
	conda run -n tensorflow python test_gpu.py
	conda run -n tensorflow python test_train.py
