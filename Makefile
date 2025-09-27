.PHONY: build build-bk up down logs audit

build:
	docker build -t ghcr.io/matatonic/openedai-speech .

build-bk:
	DOCKER_BUILDKIT=1 docker build -t ghcr.io/matatonic/openedai-speech .

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

audit:
	pip-audit
