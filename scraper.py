import requests
from bs4 import BeautifulSoup
import csv
import re
import time
from datetime import datetime

#Collect scorecard links from the ESPN Cricinfo results page
def get_scorecard_links():
    """
    Visits the ESPN Cricinfo weekly results index and collects URLs
    for individual match scorecards. Deduplicates links and caps at 15
    to avoid overloading the server.
    """
    url = "https://www.espncricinfo.com/ci/engine/match/index.html?view=week"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print("[ERROR] Failed to fetch results page")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    links = []

    for a in soup.find_all('a', href=re.compile(r'/scorecard/')):
        full_url = "https://www.espncricinfo.com" + a['href'] if a['href'].startswith('/') else a['href']
        # Avoid duplicate links for the same match
        if full_url not in links:
            links.append(full_url)

    return links[:15]

#Helper: normalize date string to YYYY-MM-DD
def parse_date(raw_date):
    """
    Accepts dates in formats found in ESPN page titles and converts them
    to a standard YYYY-MM-DD format for consistency in the CSV.
    Returns None if the date cannot be parsed.

    Handles:
        "June 04, 2026"  ->  "2026-06-04"
        "04 Jun 2026"    ->  "2026-06-04"
    """
    formats = ["%B %d, %Y", "%d %B %Y", "%b %d, %Y", "%d %b %Y"]
    for fmt in formats:
        try:
            return datetime.strptime(raw_date.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


#Helper: clean text encoding issues
def clean_text(text):
    """
    Fixes garbled characters caused by encoding mismatches (e.g. â€  instead
    of a dagger symbol next to a player name). Encodes to UTF-8 and decodes
    back, ignoring any bytes that cannot be represented cleanly.
    """
    if not text:
        return None
    return text.encode('utf-8', errors='ignore').decode('utf-8').strip()

# Helper: detect if a match is a men's or women's match
def get_gender(match_info):
    """
    Checks the match title for known women's team indicators.
    Returns "Women" or "Men" so rows can be filtered consistently later.
    """
    if re.search(r'\bwomen\b|-W\b', match_info, re.I):
        return "Women"
    return "Men"


#Scrape a single scorecard page
def scrape_scorecard(url):
    """
    Visits one match scorecard URL and extracts:
        match_date, team1, team2, venue, result, top_scorer, top_score, gender

    Returns a dict on success, or None if the page could not be parsed.
    """
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        response = requests.get(url, headers=headers, timeout=10)

        # Explicitly set encoding to UTF-8 to prevent garbled character issues
        response.encoding = 'utf-8'

        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.text, 'html.parser')

        # The page <title> contains the match summary in a consistent format:
        # "IRE vs NZ Cricket Scorecard, Only Test at Belfast, May 27 - 29, 2026"
        title_tag = soup.find('title')
        match_info = title_tag.get_text() if title_tag else ""

        #DATE
        # Extract a date-like string from the title and normalize to YYYY-MM-DD
        date_raw = re.search(r'(\w+ \d{1,2},\s*\d{4}|\d{1,2} \w+ \d{4})', match_info)
        match_date = parse_date(date_raw.group(1)) if date_raw else None

        #Gender
        # Detect men's vs women's match for clean dataset filtering
        gender = get_gender(match_info)

        #Teams
        # Title format: "TEAM1 vs TEAM2 Cricket Scorecard, ..."
        # Regex stops before "Cricket Scorecard" or a comma to avoid grabbing extra text
        team1, team2 = None, None
        teams_match = re.search(
            r'^([A-Za-z0-9\-\s]+?)\s+vs\s+([A-Za-z0-9\-\s]+?)\s+(Cricket Scorecard|Scorecard|,)',
            match_info
        )
        if teams_match:
            # Uppercase for consistent naming (PAK, AUS, ENG-W etc.)
            team1 = teams_match.group(1).strip().upper()
            team2 = teams_match.group(2).strip().upper()

        # Veneue
        # Title format: "... at <Venue>, <Month> <Day>, <Year>"
        # Capture everything between "at " and the comma before the date
        venue = None
        venue_match = re.search(
            r'\bat\s+([^,]+),\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|\d{1,2}\s)',
            match_info, re.I
        )
        if venue_match:
            venue = clean_text(venue_match.group(1))


        # Search all text nodes for a result string.
        # Skip anything containing "{" (JSON-LD schema blocks) or over 120 chars.
        result = None
        for elem in soup.find_all(string=re.compile(r'won by|Draw|abandoned|No result', re.I)):
            text = elem.strip()
            if len(text) < 120 and '{' not in text:
                result = clean_text(text)
                break

        #TOP_Scorers
        # Scan all tables for batting rows. Skip non-player rows like Extras/Total.
        # Use column index 2 for runs (standard ESPN batting table layout).
        top_scorer = None
        top_score = None
        max_runs = -1

        # These row labels are not real players and must be excluded
        SKIP_NAMES = {'total', 'extras', 'did not bat', 'fall of wickets', 'dnb', 'yet to bat'}

        batting_tables = soup.find_all(
            'table',
            class_=lambda x: x and 'batsman' in str(x).lower() or 'table' in str(x).lower()
        )

        for table in batting_tables:
            for row in table.find_all('tr'):
                cells = row.find_all('td')
                if len(cells) < 3:
                    continue

                player_raw = cells[0].get_text(strip=True)
                # Remove dismissal info in parentheses e.g. "Tom Blundell (c)"
                player_name = re.sub(r'\s*\(.*?\)', '', player_raw).strip()

                if not player_name or player_name.lower() in SKIP_NAMES:
                    continue

                runs_raw = cells[2].get_text(strip=True)
                if runs_raw.isdigit() and int(runs_raw) > max_runs:
                    max_runs = int(runs_raw)
                    top_scorer = clean_text(player_name)
                    top_score = runs_raw

      # Discard rows where team names are identical (parsing error)
        if team1 and team2 and team1 == team2:
            team1, team2 = None, None

        return {
            'match_date': match_date,
            'team1':      team1,
            'team2':      team2,
            'venue':      venue,
            'result':     result,
            'top_scorer': top_scorer,
            'top_score':  top_score,
            'gender':     gender,
        }

    except Exception as e:
        print(f"[ERROR] Could not scrape {url} : {e}")
        return None


#Main: orchestrate scraping and save to CSV
def main():
    print("[INFO] Fetching match list from ESPN Cricinfo...")
    links = get_scorecard_links()
    print(f"[INFO] Found {len(links)} scorecard links")

    matches = []
    for i, link in enumerate(links):
        print(f"[INFO] Scraping {i+1}/{len(links)}: {link}")
        data = scrape_scorecard(link)
        if data:
            matches.append(data)
        # Polite delay to avoid hammering the server
        time.sleep(1)

    if not matches:
        print("[ERROR] No match data collected. Exiting.")
        return

    fieldnames = ['match_date', 'team1', 'team2', 'venue', 'result', 'top_scorer', 'top_score', 'gender']
    filename = 'match_data.csv'

    # utf-8-sig adds a BOM so the file opens correctly in Excel without encoding issues
    with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(matches)

    print(f"[INFO] Saved {len(matches)} matches to {filename}")

    print("\n[INFO] Preview (first 5 rows):")
    for m in matches[:5]:
        print(m)


if __name__ == "__main__":
    main()
