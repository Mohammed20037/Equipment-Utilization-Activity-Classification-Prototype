.PHONY: install test up down logs ps

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

test:
	pytest -q

up:
	cp -n .env.example .env || true
	docker compose up --build -d

logs:
	docker compose logs -f --tail=150

ps:
	docker compose ps

down:
	docker compose down -v
