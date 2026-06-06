# veloria-tech-ml-intern-assignment

## Project Overview

This project covers all three tasks from the Veloria Tech ML Internship assignment:

- **Task 1** — Web scraping cricket match data from ESPN Cricinfo
- **Task 2** — Machine learning model to predict IPL match winners# veloria-tech-ml-intern-assignment

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

## Libraries to Install

```bash
pip install -r requirements.txt
```

**requirements.txt contents:**

```
requests
beautifulsoup4
lxml
pandas
numpy
scikit-learn
joblib
sentence-transformers
chromadb
```

---

## Task 1 — Web Scraping

**Script:** `scraper.py`
**Output:** `match_data.csv`

### Brief Description

Visits ESPN Cricinfo weekly results page, collects scorecard links, and
scrapes the following fields for each match:

- Match date (YYYY-MM-DD)
- Team 1 and Team 2
- Venue
- Result
- Top scorer and their score
- Gender (Men / Women)

### How to Run

```bash
python scraper.py
```

### Notes

- Uses `requests` and `BeautifulSoup` for scraping
- Adds a 1 second delay between requests to avoid overloading the server
- Encoding set to UTF-8 to prevent garbled character issues
- Dates normalized to YYYY-MM-DD format
- Missing values stored as `None`

### Output Preview

```
match_date  team1  team2  venue       result                              top_scorer       top_score  gender
2026-06-04  PAK    AUS    Lahore      Pakistan won by 4 wickets           Josh Inglis      65         Men
2026-06-03  WI     SL     Kingston    Sri Lanka won by 41 runs            Pathum Nissanka  79         Men
2026-06-02  ENG-W  IND-W  Taunton     ENG Women won by 6 wickets          Alice Capsey     82         Women
```

---

## Task 2 — Match Winner Prediction Model

**Script:** `model.py`
**Output:** `model.pkl`

### Brief Description

Trains a Random Forest Classifier to predict IPL match winners using
historical win rates and toss features. Merges Kaggle IPL dataset with
scraped match data for a combined training pipeline.

### Algorithm Used

**Random Forest Classifier** (`n_estimators=200, max_depth=10`)

Chosen because IPL match outcomes depend on multiple non-linear factors
like venue, team strength, and toss advantage. Random Forest handles these
well without requiring feature scaling and provides feature importances out of the box.

### Data

- Primary dataset: `matches.csv` from Kaggle IPL Complete Dataset (2008-2020)
- Merged with scraped `match_data.csv` to demonstrate data pipeline integration

### Features Used

| Feature | Description |
|---|---|
| `team1_win_rate` | Overall historical win % of team1 |
| `team2_win_rate` | Overall historical win % of team2 |
| `team1_home_win_rate` | Win % of team1 at this specific venue |
| `toss_won_by_team1` | 1 if team1 won the toss, else 0 |
| `toss_decision_enc` | Toss decision encoded (bat=1, field=0) |

### How to Run

```bash
python model.py
```

### Results

```
Accuracy Score : 0.6468  (64.68%)
F1 Score       : 0.6419

Confusion Matrix:
                Predicted team2 win   Predicted team1 win
Actual team2 win       72                    38
Actual team1 win       39                    69

Feature Importances:
  team1_home_win_rate       0.3834
  team2_win_rate            0.3265
  team1_win_rate            0.2009
  toss_won_by_team1         0.0495
  toss_decision_enc         0.0397
```

> **Note:** IPL matches have genuine unpredictability. Accuracy above 65% on
> this dataset without overfitting is not realistic given only toss and
> historical win rate data. These results reflect an honest, non-overfit model.

---

## Task 3 — Semantic Search (RAG)

**Script:** `rag_search.py`

### Brief Description

Converts match records into natural language sentences, generates vector
embeddings using SentenceTransformer, stores them in ChromaDB with metadata,
and allows semantic search using natural language queries with keyword pre-filtering.

### Pipeline

```
match_data.csv → text sentences → embeddings → ChromaDB → keyword filter + cosine similarity → top 3 results
```

### How to Run

```bash
python rag_search.py
```

Then type any natural language query:

```
Enter your query: matches where the away team won
Enter your query: high scoring matches
Enter your query: Give Women Matches
Enter your query: quit
```

###  Output Preview

```
Enter your query: Give Women Matches

Top 3 results for: 'Give Women Matches'
----------------------------------------
  1. ENG-W vs IND-W at Taunton on 2026-06-02. ENG Women won by 6 wickets. Top scorer: Alice Capsey with 82 runs.
     Venue: Taunton | Date: 2026-06-02 | Gender: Women
  2. BAN-W vs NL-W at Edinburgh on 2026-06-04. BAN Women won by 13 runs. Top scorer: Dilara Akter with 51 runs.
     Venue: Edinburgh | Date: 2026-06-04 | Gender: Women
  3. PHI-W vs KSA-W at Bangi on 2026-06-04. PHI Women won by 133 runs. Top scorer: Karri Gullem Keen with 67 runs.
     Venue: Bangi | Date: 2026-06-04 | Gender: Women
```

### How It Works

Match records are converted into natural language sentences. SentenceTransformer
(`all-MiniLM-L6-v2`) generates 384-dimensional embeddings. ChromaDB stores
embeddings with structured metadata (team, venue, date, gender). Queries are
first keyword-filtered (e.g. gender) then ranked by cosine similarity.

This is the retrieval layer of a RAG system. In production, the retrieved
context would be passed into an LLM to generate a natural language answer.

---

## Challenges Faced

- **ESPN Cricinfo bot detection** — site returned 403 with plain `requests`.
  Fixed using a session with full browser-like headers and homepage pre-visit to collect cookies.

- **Inconsistent page structure** — match title formats vary across match types.
  Handled with multiple fallback regex patterns for date, venue, and team extraction.

- **Encoding issues** — player names with special characters (e.g. `†` for wicketkeeper)
  caused garbled output. Fixed by explicitly setting `response.encoding = 'utf-8'`.

- **Scraped data missing toss fields** — dropping rows with null toss columns
  eliminated all scraped data from model training. Fixed by filling nulls with `"UNKNOWN"`
  instead of dropping those rows.

- **Semantic search returning wrong results** — pure cosine similarity could not
  filter by team name or gender. Fixed by adding a keyword pre-filter that detects
  intent (e.g. "women", "men") and applies a ChromaDB metadata filter before semantic ranking.

---

## Dependencies

```bash
pip install -r requirements.txt
```
```
