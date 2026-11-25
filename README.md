# Comment Reviews — Hospitality Review Intelligence

**Understand what guests think about the hotel and its restaurant, beyond a single overall star rating.**

Comment Reviews combines review collection, aspect-based sentiment analysis (ABSA), and an interactive Streamlit dashboard. The current case study is **Premier Inn Evesham**, using reviews from Google and TripAdvisor.

A guest might praise the room but criticize breakfast. This project separates those signals so operators can investigate service issues and travellers can explore what matters to their stay.

**[Open the hosted demo](https://comment-reviews-demo.onrender.com)** · [Run locally](#run-locally) · [Docker demo](#run-the-docker-demo) · [Limitations](#limitations)

**Built with:** Python · Playwright · DeBERTa-v3 · PyTorch · pandas · SQLite · Streamlit · Plotly · Docker

## Explore the dashboard

The screenshots below show a saved demonstration dataset. Review counts and scores depend on the dataset and selected filters.

### 1. Operator overview and aspect summaries

Compare the submitted public rating with a credibility-weighted rating, the hotel sentiment score, and the restaurant sentiment score. Sidebar controls filter the review window, platform, and submitted star rating, with switches for duplicate exclusion and credibility weighting.

The **Customers Say** panels summarize recurring themes such as room comfort, check-in, breakfast quality, and waiting times.

![Operator dashboard with rating comparisons, sidebar filters, and hotel and restaurant summaries](pictures/Screenshot%202026-11-06%20220259.png)

### 2. Trends and hotel–restaurant comparisons

The monthly chart compares hotel and dining sentiment against submitted public ratings. The scatter plot highlights reviews where the room and stay experience differs from the dining experience; marker size represents the credibility weight.

![Monthly hotel and restaurant sentiment trends beside a scatter plot comparing their scores](pictures/Screenshot%202026-11-06%20220314.png)

### 3. Guest view and personal stay match

Travellers adjust the balance between **room and stay quality** and **restaurant and breakfast quality**. The calculator blends the two aspect scores using that preference and displays a corresponding recommendation. A traveller-type selector filters the supporting insights and reviews.

For example, an 80% hotel / 20% dining preference produces:

```text
Personal match = 0.80 × hotel score + 0.20 × restaurant score
```

This is a preference-based summary of existing review scores, rather than a validated prediction of an individual guest's future experience.

![Guest calculator showing an 80 percent hotel priority, aspect scores, and a booking recommendation](pictures/Screenshot%202026-11-06%20220342.png)

### 4. Touchpoints and supporting review evidence

Explore themes such as cleanliness, bathroom quality, temperature, food quality, and waiting times. Review cards show the underlying comments, submitted ratings, aspect scores, and weighting information. The operator view also provides a sortable review table and CSV export.

![Hotel and dining touchpoint scores with a supporting review card and its aspect scores](pictures/Screenshot%202026-11-06%20220407.png)

## How it works

```text
       Google Travel + TripAdvisor
                    ↓
             Raw review JSONL
                    ↓
     Duplicate detection + DeBERTa-v3 ABSA
                    ↓
          Analyzed review JSONL
                    ↓
           Frozen SQLite snapshot
                    ↓
       Streamlit dashboard / Docker demo
```

The dashboard can also read the analyzed JSONL directly when no frozen database is present.

### Collection

- **Google Travel:** collects ratings, review text, relative dates, and available hotel metadata.
- **TripAdvisor:** collects ratings, review text, contribution counts, and available trip and stay metadata.

Scrapers append records to JSONL files and use review signatures to skip already-saved entries. Adjust the property URLs and `date_range` in the scraper files to change the target or collection window.

### Analysis

`Processor.py` uses `yangheng/deberta-v3-base-absa-v1.1` to score review text against separate hotel and restaurant prompts. Restaurant scoring runs when dining-related keywords are detected. Model polarity is mapped to a 1–5 display scale:

```text
Aspect score = 3 + 2 × (positive probability − negative probability)
```

The processor also flags possible cross-platform duplicates using text overlap and author-name similarity. Contribution counts and Local Guide status feed a configurable-in-code credibility heuristic. That weighting is a design assumption, not a verified measure of reviewer honesty.

Dashboard touchpoint summaries use keyword matching and aggregate existing aspect scores; they are not separate model predictions for every touchpoint.

## Run locally

### 1. Install dependencies

Requires **Python 3.13+**, **uv**, and an internet connection for collection and the initial model download. Run all commands from the project root.

```powershell
uv sync
uv run python -m playwright install chromium
```

### 2. Collect reviews

The processor's default inputs are the Google Travel and TripAdvisor outputs:

```powershell
uv run python src/comment_reviews/google_scraper.py
uv run python src/comment_reviews/tripadvisor_scraper.py
```

Both currently request the last **7 days**. Change `date_range` in their final function calls to collect a different window.

### 3. Analyze and open the dashboard

```powershell
uv run python src/comment_reviews/Processor.py
uv run streamlit run src/comment_reviews/app.py
```

The processor writes `data/processed/analyzed_reviews.jsonl`. The first analysis run downloads the model weights.

The dashboard prefers `data/hotels_frozen.db` when that file exists. To display a new processing run, rebuild the snapshot before opening or restarting the dashboard:

```powershell
uv run python src/comment_reviews/freeze_db.py
```

This command replaces the existing snapshot. Data files are excluded from Git, so a fresh clone needs locally generated data before the dashboard can display results.

## Run the Docker demo

The Docker image runs the dashboard against a frozen SQLite snapshot. Review collection and model inference happen beforehand, so the image installs only Streamlit, pandas, and Plotly.

After generating `data/hotels_frozen.db`, run:

```powershell
docker build -t comment-reviews-demo .
docker run --rm -p 8501:8501 comment-reviews-demo
```

Open **http://localhost:8501**. The existing Dockerfile uses Python 3.11 for this smaller dashboard runtime; the full development pipeline requires Python 3.13+.

The hosted demo uses a snapshot, so it does not scrape or download the NLP model when a visitor opens the app.

## Project structure

```text
src/comment_reviews/
├── google_scraper.py       # Google Travel collection
├── tripadvisor_scraper.py  # TripAdvisor collection
├── Processor.py            # Duplicate flags, aspect scores, and weighting
├── freeze_db.py            # Build the SQLite demo snapshot
└── app.py                  # Operator and guest dashboard views
pictures/                  # Dashboard screenshots and other local captures
tests/                     # Scraper, processing, and dashboard checks
Dockerfile                 # Dashboard-only container
pyproject.toml             # Full project dependencies
```

## Tests

```powershell
uv run python -m unittest discover -s tests -v
```

The existing tests cover the Google Travel scraper and rating-only processing, including dashboard behavior with mocked model results. They do not establish live-site coverage or scraper reliability against future site changes.

## Limitations

- **Collection coverage:** site layouts, consent screens, loading behavior, and pagination can limit the number of reviews collected. Selecting a one-year window does not guarantee a complete year of data.
- **Reviewer metadata:** missing contribution counts currently fall back to `1`, or `15` for detected Local Guides, in the Google Travel scraper. These fallback values are not measured review counts and can affect weighting.
- **Rating-only reviews:** the Google Travel scraper and processor retain ratings without text, with empty aspect scores. The current SQLite freezer excludes rows without review text or a hotel score, so the frozen demo does not include those records.
- **Model interpretation:** aspect scores are model estimates, not submitted star ratings or measured operational performance. Inputs are truncated to 512 tokens, and the project does not report a labelled hospitality evaluation benchmark.
- **Duplicate detection and weighting:** both are heuristics and can make mistakes. A low rating or a small contribution count does not establish that a review is unreliable.

## Repository assets

The README uses four dashboard screenshots from **6 November 2026**. Other captures in `pictures/` include unrelated personal content or visible browser tabs and are not part of this walkthrough. Publish only the intended dashboard assets.

Keep raw datasets, local environment files, credentials, and private notes out of the public repository. The Docker image includes its database snapshot, and the app can display and export review data; review those contents before distributing an image or hosting a demo.
