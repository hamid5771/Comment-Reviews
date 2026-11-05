# Comment Reviews

Collect Google Travel and TripAdvisor reviews for Premier Inn Evesham, analyze hotel and restaurant sentiment with DeBERTa-v3, and explore ratings in a Streamlit dashboard.

## Setup and run

Requires Python 3.13+ and uv. Run from the project root:

```powershell
uv sync
uv run python -m playwright install chromium
uv run python src/comment_reviews/google_scraper.py
uv run python src/comment_reviews/tripadvisor_scraper.py
uv run python src/comment_reviews/Processor.py
uv run streamlit run src/comment_reviews/app.py
```

Both scraper entry points currently request the last 7 days; change `date_range` in their final function calls to adjust the window. The first analysis run downloads the model.

Raw reviews are saved under `data/raw/`; analyzed records under `data/processed/`. Google star-only reviews contribute to public rating averages but have no text sentiment scores. Credibility weights are heuristic, and site pagination can limit coverage.

## Tests

```powershell
uv run python -m unittest discover -s tests -v
```

## Before publishing

Publish `src/`, `tests/`, `pyproject.toml`, `uv.lock`, `.python-version`, `.gitignore`, and this README. Review the author email in `pyproject.toml` before uploading.

Keep `data/`, `comment.txt`, `.venv/`, environment files, and credentials private. Review data and logs contain names and review text. `comment.txt` is already tracked in Git history: excluding it from a new commit does not remove historical copies. Use reviewed files with fresh history, or clean the existing history before publishing.
