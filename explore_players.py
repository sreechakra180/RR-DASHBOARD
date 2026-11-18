import pandas as pd
import numpy as np

# Load processed data
df = pd.read_parquet('processed_bbb.parquet')

# Apply high pressure definition
# Innings 2: Leverage Index >= 1.2 or (balls_remaining <= 30 and rrr between 7.5 and 24.0 and wickets_in_hand >= 2) or collapse
# Innings 1: Death overs (>= 17) or collapse (w>=3 in <=6 or w>=4 in <=10) or (is_knockout and overid >= 15)
# Super overs

# First train WP model for innings 2
from sklearn.linear_model import LogisticRegression
inn2_mask = (df['innings'] == 2) & (df['chase_won'].notna()) & (df['balls_remaining'] > 0)
features = ['runs_needed', 'balls_remaining', 'wickets_in_hand', 'rrr', 'target']
wp_model = LogisticRegression(max_iter=1000)
wp_model.fit(df.loc[inn2_mask, features], df.loc[inn2_mask, 'chase_won'])
df['wp'] = np.nan
df.loc[inn2_mask, 'wp'] = wp_model.predict_proba(df.loc[inn2_mask, features])[:, 1]
df['li_chase'] = np.nan
df.loc[inn2_mask, 'li_chase'] = 4.0 * df.loc[inn2_mask, 'wp'] * (1.0 - df.loc[inn2_mask, 'wp']) * np.sqrt(120.0 / (df.loc[inn2_mask, 'balls_remaining'] + 6.0))

cond_chase_high = (df['innings'] == 2) & (
    (df['li_chase'] >= 1.2) |
    ((df['balls_remaining'] <= 30) & (df['rrr'] >= 7.5) & (df['rrr'] <= 24.0) & (df['wickets_in_hand'] >= 2)) |
    ((df['overid'] <= 6) & (df['cum_wickets_before'] >= 3)) |
    ((df['overid'] <= 10) & (df['cum_wickets_before'] >= 4))
)

cond_inn1_high = (df['innings'] == 1) & (
    (df['overid'] >= 17) |
    ((df['overid'] <= 6) & (df['cum_wickets_before'] >= 3)) |
    ((df['overid'] <= 10) & (df['cum_wickets_before'] >= 4)) |
    ((df['is_knockout'] == 1) & (df['overid'] >= 15))
)

cond_super = df['innings'] >= 3

df['is_high_pressure'] = cond_chase_high | cond_inn1_high | cond_super

# Check overall benchmarks
print("--- OVERALL DATASET BENCHMARKS ---")
print(f"Total balls: {len(df)}")
print(f"High pressure balls: {df['is_high_pressure'].sum()} ({df['is_high_pressure'].mean()*100:.2f}%)")

# Run rate and boundary rate overall vs pressure
print("\nOverall Run Rate (per 6 balls):", df['ball_runs'].mean() * 6.0)
print("Normal Run Rate:", df.loc[~df['is_high_pressure'], 'ball_runs'].mean() * 6.0)
print("High Pressure Run Rate:", df.loc[df['is_high_pressure'], 'ball_runs'].mean() * 6.0)

print("\nOverall Boundary %:", (df['is_four'] | df['is_six']).mean() * 100)
print("Normal Boundary %:", (df.loc[~df['is_high_pressure'], 'is_four'] | df.loc[~df['is_high_pressure'], 'is_six']).mean() * 100)
print("High Pressure Boundary %:", (df.loc[df['is_high_pressure'], 'is_four'] | df.loc[df['is_high_pressure'], 'is_six']).mean() * 100)

print("\nOverall Wicket %:", df['is_wicket'].mean() * 100)
print("Normal Wicket %:", df.loc[~df['is_high_pressure'], 'is_wicket'].mean() * 100)
print("High Pressure Wicket %:", df.loc[df['is_high_pressure'], 'is_wicket'].mean() * 100)
