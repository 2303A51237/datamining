
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# ---------------------------
# STEP 1: Load and Preprocess Data
# ---------------------------
df = pd.read_csv("Cricket_data.csv")

# Select relevant columns and clean
df = df[['home_team', 'away_team', 'toss_won', 'decision', 'winner']].dropna()
df = df[~df['winner'].isin(['TBA', 'TBC'])]
df = df[~df['home_team'].isin(['TBA', 'TBC'])]
df = df[~df['away_team'].isin(['TBA', 'TBC'])]

# One-hot encoding and target transformation
df_encoded = pd.get_dummies(df, columns=['home_team', 'away_team', 'toss_won', 'decision'])
df_encoded['winner_code'] = df['winner'].astype('category').cat.codes

# ---------------------------
# STEP 2: Train-Test Split
# ---------------------------
X = df_encoded.drop(['winner', 'winner_code'], axis=1)
y = df_encoded['winner_code']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------
# STEP 3: Train Model
# ---------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---------------------------
# STEP 4: Evaluate Model
# ---------------------------
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# ---------------------------
# STEP 5: Feature Importance
# ---------------------------
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns[indices]

plt.figure(figsize=(12,6))
plt.title('Feature Importances')
plt.bar(range(len(features[:10])), importances[indices][:10], align='center')
plt.xticks(range(len(features[:10])), features[:10], rotation=90)
plt.tight_layout()
plt.show()

# ---------------------------
# STEP 6: Dynamic Match Prediction
# ---------------------------
def predict_today_match_dynamic():
    teams = sorted(list(set(df['home_team'].unique()) | set(df['away_team'].unique())))

    print("\n--- IPL Match Prediction ---")
    home = input(f"Enter Home Team ({', '.join(teams)}): ")
    away = input("Enter Away Team (not same as home): ")

    while away == home:
        print("❌ Away team cannot be the same as home team.")
        away = input("Enter a different Away Team: ")

    toss_winner = input(f"Who won the toss? (Must be '{home}' or '{away}'): ")
    while toss_winner not in [home, away]:
        print("❌ Toss winner must be one of the selected teams.")
        toss_winner = input(f"Enter toss winner again ({home}/{away}): ")

    toss_decision = input("Toss decision (BAT FIRST/BOWL FIRST): ").upper()
    while toss_decision not in ["BAT FIRST", "BOWL FIRST"]:
        print("❌ Invalid decision. Choose either 'BAT FIRST' or 'BOWL FIRST'.")
        toss_decision = input("Enter toss decision again: ").upper()

    # Prepare input
    input_df = pd.DataFrame(np.zeros((1, len(X.columns))), columns=X.columns)
    for col in input_df.columns:
        if col == f'home_team_{home}':
            input_df[col] = 1
        if col == f'away_team_{away}':
            input_df[col] = 1
        if col == f'toss_won_{toss_winner}':
            input_df[col] = 1
        if col == f'decision_{toss_decision}':
            input_df[col] = 1

    # Predict winner
    prediction_code = model.predict(input_df)[0]
    predicted_team = df['winner'].astype('category').cat.categories[prediction_code]
    print(f"\n🏏 Predicted Winner: {predicted_team}")

# ---------------------------
# Run Prediction
# ---------------------------
predict_today_match_dynamic()

