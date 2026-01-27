.PHONY: docs-build docs-serve docs-dev

docs-build:
	zola build

docs-serve:
	zola serve --port 8080 --interface 0.0.0.0 --base-url http://127.0.0.1

docs-dev:
	cd public && python3 -m http.server 8080
