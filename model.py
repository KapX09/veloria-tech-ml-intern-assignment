import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import joblib

#Load the Kaggle IPL dataset
print("[INFO] Loading matches.csv...")
kaggle_df = pd.read_csv("matches.csv")
print(f"[INFO] Kaggle dataset shape: {kaggle_df.shape}")


#Load and normalize match_data.csv (scraped data)
print("[INFO] Loading match_data.csv...")
scraped_df = pd.read_csv("match_data7.csv")

def extract_winner_from_result(result, team1, team2):
    """
    Parses the result string from scraped data to find the winning team.
    Example: "Pakistan won by 4 wickets" -> "Pakistan"
    Returns None if result is missing or match was abandoned.
    """
    if not isinstance(result, str):
        return None
    result_lower = result.lower()
    if 'abandoned' in result_lower or 'no result' in result_lower:
        return None
    if isinstance(team1, str) and team1.lower() in result_lower:
        return team1
    if isinstance(team2, str) and team2.lower() in result_lower:
        return team2
    return None

# Build a normalized version of scraped data matching Kaggle columns
scraped_normalized = pd.DataFrame()
scraped_normalized['date']          = scraped_df['match_date']
scraped_normalized['team1']         = scraped_df['team1']
scraped_normalized['team2']         = scraped_df['team2']
scraped_normalized['venue']         = scraped_df['venue']
scraped_normalized['toss_winner']   = None
scraped_normalized['toss_decision'] = None
scraped_normalized['winner']        = scraped_df.apply(
    lambda r: extract_winner_from_result(r['result'], r['team1'], r['team2']), axis=1
)
scraped_normalized['city']          = None
scraped_normalized['season']        = pd.to_datetime(
    scraped_df['match_date'], errors='coerce'
).dt.year

keep_cols = ['season', 'city', 'date', 'venue', 'team1', 'team2',
             'toss_winner', 'toss_decision', 'winner']

kaggle_df = kaggle_df[keep_cols]

df = pd.concat([kaggle_df, scraped_normalized[keep_cols]], ignore_index=True)
print(f"[INFO] Combined dataset shape after merge: {df.shape}")
'''We map our scraped columns to match the Kaggle schema so both datasets an be merged into a single training dataframe.'''

#Clean the data
print("[INFO] Cleaning data...")
# Drop rows where key columns are missing.
df = df.dropna(subset=['winner', 'team1', 'team2', 'venue', 'toss_winner', 'toss_decision'])

for col in ['team1', 'team2', 'toss_winner', 'toss_decision', 'winner', 'venue']:
    df[col] = df[col].astype(str).str.strip()

print(f"[INFO] Dataset shape after cleaning: {df.shape}")


#  4 - Feature Engineering
'''team1_win_rate       - overall historical win % of team1
team2_win_rate       - overall historical win % of team2
team1_home_win_rate  - win % of team1 at this specific venue
toss_won_by_team1    - 1 if team1 won the toss, else 0
toss_decision_enc    - bat(1) or field(0)
Target: team1_wins - 1 if team1 won the match, 0 if team2 won
'''
print("[INFO] Engineering features...")

# Target column — needed before computing win rates
df['team1_wins'] = (df['winner'] == df['team1']).astype(int)

def compute_win_rate(df):
    """
    For each team, calculate total wins / total matches played.
    Returns a dict: team name -> win rate (0.0 to 1.0)
    """
    teams = pd.concat([df['team1'], df['team2']]).unique()
    win_rates = {}
    for team in teams:
        total = df[(df['team1'] == team) | (df['team2'] == team)].shape[0]
        wins  = df[df['winner'] == team].shape[0]
        win_rates[team] = wins / total if total > 0 else 0.5
    return win_rates

def compute_home_win_rate(df):
    """
    For each (team, venue) pair, calculate win rate at that ground.
    Returns a dict: (team, venue) -> win rate (0.0 to 1.0)
    """
    home_rates = {}
    all_teams  = pd.concat([df['team1'], df['team2']]).unique()
    all_venues = df['venue'].unique()
    for team in all_teams:
        for venue in all_venues:
            venue_matches = df[
                ((df['team1'] == team) | (df['team2'] == team)) & (df['venue'] == venue)
            ].shape[0]
            venue_wins = df[(df['winner'] == team) & (df['venue'] == venue)].shape[0]
            home_rates[(team, venue)] = venue_wins / venue_matches if venue_matches > 0 else 0.5
    return home_rates

win_rates      = compute_win_rate(df)
home_win_rates = compute_home_win_rate(df)

df['team1_win_rate']      = df['team1'].map(win_rates).fillna(0.5)
df['team2_win_rate']      = df['team2'].map(win_rates).fillna(0.5)
df['team1_home_win_rate'] = df.apply(
    lambda r: home_win_rates.get((r['team1'], r['venue']), 0.5), axis=1
)

# Toss features
df['toss_won_by_team1'] = (df['toss_winner'] == df['team1']).astype(int)
le_toss_d = LabelEncoder()
df['toss_decision_enc'] = le_toss_d.fit_transform(df['toss_decision'])

FEATURES = [
    'team1_win_rate',
    'team2_win_rate',
    'team1_home_win_rate',
    'toss_won_by_team1',
    'toss_decision_enc',
]

X = df[FEATURES]
y = df['team1_wins']

print(f"[INFO] Features: {FEATURES}")
print(f"[INFO] Total samples for training: {len(X)}")


#Train / Test Split
# 80% training, 20% testing. random_state=42 for reproducibility.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"[INFO] Train size: {len(X_train)}, Test size: {len(X_test)}")


#Train Random Forest model
# n_estimators=100 is a solid default. random_state=42 for reproducibility.
print("[INFO] Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


#Evaluate the model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1       = f1_score(y_test, y_pred)
cm       = confusion_matrix(y_test, y_pred)

print("\n[RESULTS] Model Evaluation")
print(f"  Accuracy Score : {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"  F1 Score       : {f1:.4f}")
print(f"\n  Confusion Matrix:")
print(f"                  Predicted team2 win   Predicted team1 win")
print(f"  Actual team2 win       {cm[0][0]}                    {cm[0][1]}")
print(f"  Actual team1 win       {cm[1][0]}                    {cm[1][1]}")

print("\n[RESULTS] Feature Importances:")
for feat, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1]):
    print(f"  {feat:<25} {imp:.4f}")


#Save the trained model
# Saved as model.pkl so it can be loaded in rag_search.py if needed.
joblib.dump(model, "model.pkl")
print("\n[INFO] Model saved to model.pkl")
