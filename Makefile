.PHONY: install test data demo up down logs ps

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

test:
	pytest -q

data:
	python scripts/download_open_source_data.py

demo:
	python scripts/export_demo_gif.py

up:
	cp -n .env.example .env || true
	docker compose up --build -d

logs:
	docker compose logs -f --tail=150

ps:
	docker compose ps

down:
	docker compose down -v
