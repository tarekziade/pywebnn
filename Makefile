VENV ?= .venv
ACTIVATE = . $(VENV)/bin/activate
RUST_CRATE := pywebnn_rust

.PHONY: venv deps rust test clean

venv:
	@if [ ! -d "$(VENV)" ]; then \
		python3 -m venv "$(VENV)"; \
	fi

deps: venv
	$(ACTIVATE) && pip install --upgrade pip
	$(ACTIVATE) && pip install -r requirements.txt
	$(ACTIVATE) && pip install --no-build-isolation -e .

rust: deps
	$(ACTIVATE) && \
		PYTHONPATH=$(VENV)/lib/python3.11/site-packages \
		LIBTORCH_USE_PYTORCH=1 \
		LIBTORCH_BYPASS_VERSION_CHECK=1 \
		maturin develop --release -m $(RUST_CRATE)/Cargo.toml

test: rust
	$(ACTIVATE) && pytest tests/

clean:
	rm -rf "$(VENV)"
