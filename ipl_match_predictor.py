
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load and preprocess data
df = pd.read_csv("Cricket_data.csv")
df = df[['home_team', 'away_team', 'toss_won', 'decision', 'winner']].dropna()
df = df[~df['winner'].isin(['TBA', 'TBC'])]
df = df[~df['home_team'].isin(['TBA', 'TBC'])]
df = df[~df['away_team'].isin(['TBA', 'TBC'])]

df_encoded = pd.get_dummies(df, columns=['home_team', 'away_team', 'toss_won', 'decision'])
df_encoded['winner_code'] = df['winner'].astype('category').cat.codes

# Split data and train model
X = df_encoded.drop(['winner', 'winner_code'], axis=1)
y = df_encoded['winner_code']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Prediction function for dynamic user input
def predict_today_match_dynamic():
    teams = sorted(list(set(df['home_team'].unique()) | set(df['away_team'].unique())))

    print("\n--- IPL Match Prediction ---")
    home = input(f"Enter Home Team ({', '.join(teams)}): ")
    away = input(f"Enter Away Team ({', '.join(teams)}): ")
    toss_winner = input(f"Who won the toss? ({home}/{away}): ")
    toss_decision = input("Toss decision (BAT FIRST/BOWL FIRST): ").upper()

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

    prediction_code = model.predict(input_df)[0]
    predicted_team = df['winner'].astype('category').cat.categories[prediction_code]
    print(f"\n🏏 Predicted Winner: {predicted_team}")

# Run the predictor
predict_today_match_dynamic()
