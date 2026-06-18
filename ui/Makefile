# anima-thor-ui — common dev/ops tasks.  `make help` for the list.
.PHONY: help install test lint fmt smoke run restart deploy
THOR ?= thor
APP  ?= $(HOME)/anima-thor-ui

help:  ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/ —/' | sort

install:  ## set up the local venv
	uv venv --python 3.12 .venv && uv pip install -e . pytest ruff

test:  ## run unit tests (no engine/GPU needed)
	.venv/bin/python -m pytest tests/ -q

lint:  ## ruff check
	.venv/bin/python -m ruff check anima_thor_ui scripts tests

fmt:  ## ruff auto-fix
	.venv/bin/python -m ruff check --fix anima_thor_ui scripts tests

smoke:  ## end-to-end check against a running box (BASE=http://host:7000)
	bash scripts/smoke.sh $(BASE)

run:  ## build + run the container locally (compose)
	docker compose up -d --build

deploy:  ## rsync this dir to Thor and restart the UI container (engine survives)
	rsync -az --exclude '.venv' --exclude '.git' --exclude '__pycache__' ./ $(THOR):$(APP)/
	ssh $(THOR) 'sudo docker restart anima-thor-ui'

restart:  ## restart the UI container on Thor
	ssh $(THOR) 'sudo docker restart anima-thor-ui'
