# veloria-tech-ml-intern-assignment

## Project Overview

This project covers all three tasks from the Veloria Tech ML Internship assignment:

- **Task 1** — Web scraping cricket match data from ESPN Cricinfo
- **Task 2** — Machine learning model to predict IPL match winners
- **Task 3** — Semantic search over match records using vector embeddings

---

## Folder Structure

```
veloria-tech-ml-intern-assignment/
    scraper.py          Task 1 — web scraper
    match_data.csv      Task 1 — scraped output
    matches.csv         Task 2 — Kaggle IPL dataset (2008-2020)
    model.py            Task 2 — ML training script
    model.pkl           Task 2 — saved trained model
    rag_search.py       Task 3 — semantic search script
    requirements.txt    all dependencies
    README.md           this file
```

---

## Setup

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Task 1 — Web Scraping

**Script:** `scraper.py`
**Output:** `match_data.csv`

### What it does

Visits ESPN Cricinfo weekly results page, collects scorecard links, and
scrapes the following fields for each match:

- Match date (YYYY-MM-DD)
- Team 1 and Team 2
- Venue
- Result
- Top scorer and their score
- Gender (Men / Women)

### How to run

```bash
python scraper.py
```

### Notes

- Uses `requests` and `BeautifulSoup` for scraping
- Adds a 1 second delay between requests to avoid overloading the server
- Encoding set to UTF-8 to prevent garbled character issues
- Dates normalized to YYYY-MM-DD format
- Missing values stored as `None`

---

## Task 2 — Match Winner Prediction Model

**Script:** `model.py`
**Output:** `model.pkl`

### Algorithm used

**Random Forest Classifier** (n_estimators=100)

Chosen because IPL match outcomes depend on multiple non-linear factors
like venue, team strength, and toss advantage. Random Forest handles these
well without requiring feature scaling and provides feature importances
out of the box.

### Data

- Primary dataset: `matches.csv` from Kaggle IPL Complete Dataset (2008-2020)
- Merged with scraped `match_data.csv` to demonstrate data pipeline integration

### Features used

| Feature | Description |
|---|---|
| team1_win_rate | Overall historical win % of team1 |
| team2_win_rate | Overall historical win % of team2 |
| team1_home_win_rate | Win % of team1 at this specific venue |
| toss_won_by_team1 | 1 if team1 won the toss, else 0 |
| toss_decision_enc | Toss decision encoded (bat=1, field=0) |

These features directly implement what the assignment describes:
"Team A has won 8 out of their last 10 home matches — the model learns this."

### How to run

```bash
python model.py
```

### Results

```
Accuracy Score : ~63%
F1 Score       : ~0.63
```

Note: IPL matches have genuine unpredictability. Accuracy above 65% on
this dataset without overfitting is not realistic given only toss and
historical win rate data. These results reflect an honest, non-overfit model.

---

## Task 3 — Semantic Search (RAG)

**Script:** `rag_search.py`

### What it does

Converts match records into natural language sentences, generates vector
embeddings, stores them in ChromaDB, and allows semantic search using
natural language queries.

### Pipeline

```
match_data.csv -> text sentences -> embeddings -> ChromaDB -> cosine similarity -> top 3 results
```

### How to run

```bash
python rag_search.py
```

Then type any natural language query:

```
Enter your query: matches where the away team won
Enter your query: high scoring matches
Enter your query: quit
```

### How it works

We converted cricket match records into natural language sentences. We used
SentenceTransformer (all-MiniLM-L6-v2) to generate 384-dimensional embeddings.
We then used ChromaDB with cosine similarity to perform semantic search and
retrieve the most relevant matches based on user queries.

This is the retrieval layer of a RAG system. In production, the retrieved
context would be passed into an LLM to generate a natural language answer.

---

## Challenges Faced

- ESPN Cricinfo page structure is inconsistent across match types — handled
  with fallback regex patterns
- Encoding issues with player names (e.g. garbled UTF-8 characters) — fixed
  by setting response.encoding explicitly
- IPL dataset contains retired/renamed teams — handled by encoding all team
  names from a shared label space

---

## Dependencies

See `requirements.txt`. Install with:

```bash
pip install -r requirements.txt
```
