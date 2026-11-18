import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

print("Loading processed_bbb.parquet...")
df = pd.read_parquet('processed_bbb.parquet')

# 1. 2nd innings Win Probability
inn2_mask = (df['innings'] == 2) & (df['chase_won'].notna()) & (df['balls_remaining'] > 0)
features_wp = ['runs_needed', 'balls_remaining', 'wickets_in_hand', 'rrr', 'target']
wp_model = LogisticRegression(max_iter=1000)
wp_model.fit(df.loc[inn2_mask, features_wp], df.loc[inn2_mask, 'chase_won'])
df['wp'] = np.nan
df.loc[inn2_mask, 'wp'] = wp_model.predict_proba(df.loc[inn2_mask, features_wp])[:, 1]

# Leverage Index (LI)
df['li_chase'] = np.nan
df.loc[inn2_mask, 'li_chase'] = 4.0 * df.loc[inn2_mask, 'wp'] * (1.0 - df.loc[inn2_mask, 'wp']) * np.sqrt(120.0 / (df.loc[inn2_mask, 'balls_remaining'] + 6.0))

# 2. Situational Pressure Definition
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

# Pressure categorization
def get_pressure_tier(r):
    if r['innings'] >= 3:
        return 'Extreme'
    if r['innings'] == 2:
        if r['li_chase'] >= 2.0 or (r['balls_remaining'] <= 18 and r['rrr'] >= 9.0 and r['rrr'] <= 24.0):
            return 'Extreme'
        elif r['is_high_pressure']:
            return 'High'
        elif r['li_chase'] >= 0.8:
            return 'Medium'
        else:
            return 'Low'
    else: # Innings 1
        if (r['overid'] >= 19) or (r['is_knockout'] == 1 and r['overid'] >= 18) or (r['overid'] <= 4 and r['cum_wickets_before'] >= 3):
            return 'Extreme'
        elif r['is_high_pressure']:
            return 'High'
        elif r['overid'] >= 14 or r['cum_wickets_before'] >= 3:
            return 'Medium'
        else:
            return 'Low'

df['pressure_tier'] = df.apply(get_pressure_tier, axis=1)

# Continuous Pressure Score (0-100)
# We can calibrate continuous pressure score based on leverage index, stage, and phase
def get_continuous_pressure(r):
    if r['innings'] >= 3:
        return 100.0
    tier = r['pressure_tier']
    if tier == 'Extreme':
        base = 80.0
    elif tier == 'High':
        base = 60.0
    elif tier == 'Medium':
        base = 40.0
    else:
        base = 15.0
        
    if r['innings'] == 2 and pd.notna(r['li_chase']):
        bonus = min(20.0, r['li_chase'] * 6.0)
    else:
        bonus = min(20.0, (r['overid'] / 20.0) * 15.0 + (r['is_knockout'] * 5.0))
    return min(100.0, base + bonus)

df['pressure_score'] = df.apply(get_continuous_pressure, axis=1)

# Competition quality tier
def get_comp_tier(name):
    if 'IPL' in str(name):
        return 1.00
    elite_t20i = ['Australia', 'India', 'England', 'South Africa', 'Pakistan', 'New Zealand', 'West Indies', 'Sri Lanka', 'Afghanistan']
    if any(country in str(name) for country in elite_t20i):
        return 0.95
    return 0.85

df['comp_weight'] = df['comp_name'].apply(get_comp_tier)

print("Training Expected Runs & Expected Wickets models...")
exp_features = ['overid', 'innings', 'wickets_in_hand', 'cum_runs_before', 'comp_weight', 'is_knockout']
X_exp = df[exp_features]

reg_bat = HistGradientBoostingRegressor(max_iter=100, min_samples_leaf=50, random_state=42)
reg_bat.fit(X_exp, df['runs'])
df['exp_bat_runs'] = reg_bat.predict(X_exp)

reg_bowl = HistGradientBoostingRegressor(max_iter=100, min_samples_leaf=50, random_state=42)
reg_bowl.fit(X_exp, df['bowler_runs_conceded'])
df['exp_bowl_runs'] = reg_bowl.predict(X_exp)

clf_wkt = HistGradientBoostingClassifier(max_iter=100, min_samples_leaf=50, random_state=42)
clf_wkt.fit(X_exp, df['is_bowler_wicket'].astype(int))
df['exp_bowl_wkt'] = clf_wkt.predict_proba(X_exp)[:, 1]

print("Saving enriched dataset to enriched_bbb.parquet...")
df.to_parquet('enriched_bbb.parquet', index=False)
print("Done! Enriched dataset shape:", df.shape)
print("Pressure tier breakdown:\n", df['pressure_tier'].value_counts())
