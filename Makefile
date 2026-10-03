install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

run:
	python app.py

lint:
	python -m py_compile app.py

test:
	python -m pytest -vv

clean:
	rm -rf __pycache__
	rm -rf .pytest_cache