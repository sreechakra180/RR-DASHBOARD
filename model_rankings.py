import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# 1. Load data
print("Loading data...")
df = pd.read_parquet('processed_bbb.parquet')

# 2. Train Win Probability model for 2nd innings
inn2_mask = (df['innings'] == 2) & (df['chase_won'].notna()) & (df['balls_remaining'] > 0)
features = ['runs_needed', 'balls_remaining', 'wickets_in_hand', 'rrr', 'target']
wp_model = LogisticRegression(max_iter=1000)
wp_model.fit(df.loc[inn2_mask, features], df.loc[inn2_mask, 'chase_won'])

df['wp'] = np.nan
df.loc[inn2_mask, 'wp'] = wp_model.predict_proba(df.loc[inn2_mask, features])[:, 1]

# Leverage Index for 2nd innings
# Standard baseball/cricket leverage index: 4 * p * (1-p) normalized by time factor
df['li_chase'] = np.nan
df.loc[inn2_mask, 'li_chase'] = 4.0 * df.loc[inn2_mask, 'wp'] * (1.0 - df.loc[inn2_mask, 'wp']) * np.sqrt(120.0 / (df.loc[inn2_mask, 'balls_remaining'] + 6.0))

# 3. Define Pressure Situations
# Innings 2:
# - High leverage: LI >= 1.2
# - Crunch death overs: balls_remaining <= 30 and 7.5 <= RRR <= 24.0 and wickets_in_hand >= 2
# - Top order crisis in chase: overid <= 6 & wickets >= 3, or overid <= 10 & wickets >= 4
cond_chase_high = (df['innings'] == 2) & (
    (df['li_chase'] >= 1.2) |
    ((df['balls_remaining'] <= 30) & (df['rrr'] >= 7.5) & (df['rrr'] <= 24.0) & (df['wickets_in_hand'] >= 2)) |
    ((df['overid'] <= 6) & (df['cum_wickets_before'] >= 3)) |
    ((df['overid'] <= 10) & (df['cum_wickets_before'] >= 4))
)

# Innings 1:
# - Death overs: overs 17-20
# - Early crisis: overid <= 6 & wickets >= 3, or overid <= 10 & wickets >= 4
# - High stakes knockout acceleration: is_knockout and overid >= 15
cond_inn1_high = (df['innings'] == 1) & (
    (df['overid'] >= 17) |
    ((df['overid'] <= 6) & (df['cum_wickets_before'] >= 3)) |
    ((df['overid'] <= 10) & (df['cum_wickets_before'] >= 4)) |
    ((df['is_knockout'] == 1) & (df['overid'] >= 15))
)

# Super overs: innings >= 3
cond_super = df['innings'] >= 3

df['is_high_pressure'] = cond_chase_high | cond_inn1_high | cond_super

# Add competition quality tier weight
# IPL: 1.0, Full Member T20I: 0.95, Others: 0.80
def get_comp_tier(name):
    if 'IPL' in str(name):
        return 1.0
    elite_t20i = ['Australia', 'India', 'England', 'South Africa', 'Pakistan', 'New Zealand', 'West Indies', 'Sri Lanka', 'Afghanistan']
    if any(country in str(name) for country in elite_t20i):
        return 0.95
    return 0.85

df['comp_weight'] = df['comp_name'].apply(get_comp_tier)

print("Computing batting metrics...")
# 4. Batting Evaluation
bat_df = df[df['is_batter_ball']].copy()

def compute_batter_metrics(min_pressure_balls=150):
    p_df = bat_df[bat_df['is_high_pressure']]
    
    # Aggregations for pressure
    p_grp = p_df.groupby(['striker_id', 'striker_name'])
    p_stats = pd.DataFrame({
        'p_balls': p_grp.size(),
        'p_runs': p_grp['runs'].sum(),
        'p_outs': p_grp['is_wicket'].sum(),
        'p_fours': p_grp['is_four'].sum(),
        'p_sixes': p_grp['is_six'].sum(),
        'p_dots': p_grp['runs'].apply(lambda x: (x == 0).sum()),
        'p_matches': p_grp['match_id'].nunique(),
        'p_comp_weight': p_grp['comp_weight'].mean()
    })
    
    # Overall aggregations
    all_grp = bat_df.groupby(['striker_id', 'striker_name'])
    all_stats = pd.DataFrame({
        'all_balls': all_grp.size(),
        'all_runs': all_grp['runs'].sum(),
        'all_outs': all_grp['is_wicket'].sum(),
        'all_fours': all_grp['is_four'].sum(),
        'all_sixes': all_grp['is_six'].sum(),
        'all_dots': all_grp['runs'].apply(lambda x: (x == 0).sum()),
        'all_matches': all_grp['match_id'].nunique(),
    })
    
    merged = p_stats.join(all_stats, how='inner')
    merged = merged[merged['p_balls'] >= min_pressure_balls].copy()
    
    # Raw stats
    merged['p_sr'] = (merged['p_runs'] / merged['p_balls']) * 100
    merged['p_avg'] = merged['p_runs'] / merged['p_outs'].replace(0, 1)
    merged['p_boundary_pct'] = ((merged['p_fours'] + merged['p_sixes']) / merged['p_balls']) * 100
    merged['p_dot_pct'] = (merged['p_dots'] / merged['p_balls']) * 100
    
    merged['all_sr'] = (merged['all_runs'] / merged['all_balls']) * 100
    merged['all_avg'] = merged['all_runs'] / merged['all_outs'].replace(0, 1)
    merged['all_boundary_pct'] = ((merged['all_fours'] + merged['all_sixes']) / merged['all_balls']) * 100
    merged['all_dot_pct'] = (merged['all_dots'] / merged['all_balls']) * 100
    
    # Deltas (Pressure vs Usual)
    merged['delta_sr'] = merged['p_sr'] - merged['all_sr']
    merged['delta_avg'] = merged['p_avg'] - merged['all_avg']
    merged['delta_dot_pct'] = merged['p_dot_pct'] - merged['all_dot_pct'] # negative is good
    
    # Bayesian shrinkage for SR and Avg under pressure
    # Population priors
    prior_sr = (p_df['runs'].sum() / len(p_df)) * 100
    prior_avg = p_df['runs'].sum() / max(1, p_df['is_wicket'].sum())
    
    K_balls = 40.0
    K_outs = 3.0
    
    merged['bayes_p_sr'] = ((merged['p_balls'] * merged['p_sr']) + (K_balls * prior_sr)) / (merged['p_balls'] + K_balls)
    merged['bayes_p_avg'] = ((merged['p_outs'] * merged['p_avg']) + (K_outs * prior_avg)) / (merged['p_outs'] + K_outs)
    
    # Composite Rating:
    # 1. Standardized Pressure Performance: SR (50%) + Avg (30%) + Boundary% (10%) + Low Dot% (10%)
    z_sr = (merged['bayes_p_sr'] - merged['bayes_p_sr'].mean()) / merged['bayes_p_sr'].std()
    z_avg = (merged['bayes_p_avg'] - merged['bayes_p_avg'].mean()) / merged['bayes_p_avg'].std()
    z_bdry = (merged['p_boundary_pct'] - merged['p_boundary_pct'].mean()) / merged['p_boundary_pct'].std()
    z_dot = (merged['p_dot_pct'].mean() - merged['p_dot_pct']) / merged['p_dot_pct'].std()
    
    merged['score_execution'] = (0.45 * z_sr) + (0.35 * z_avg) + (0.10 * z_bdry) + (0.10 * z_dot)
    
    # 2. Standardized Clutch Elevation: delta_sr (60%) + delta_avg (40%)
    z_delta_sr = (merged['delta_sr'] - merged['delta_sr'].mean()) / merged['delta_sr'].std()
    z_delta_avg = (merged['delta_avg'] - merged['delta_avg'].mean()) / merged['delta_avg'].std()
    merged['score_clutch_elevation'] = (0.60 * z_delta_sr) + (0.40 * z_delta_avg)
    
    # 3. Volume and Competition quality factor
    vol_factor = np.log1p(merged['p_balls']) / np.log1p(merged['p_balls'].max())
    
    # Final Pressure Index (0-100 scale)
    raw_final = (0.60 * merged['score_execution']) + (0.40 * merged['score_clutch_elevation'])
    # Multiply by quality and volume dampener
    adjusted_score = raw_final * merged['p_comp_weight'] * (0.85 + 0.15 * vol_factor)
    
    # Scale to 0-100 for presentation
    min_s = adjusted_score.min()
    max_s = adjusted_score.max()
    merged['pressure_rating'] = 50.0 + (adjusted_score - adjusted_score.mean()) / adjusted_score.std() * 15.0
    
    return merged.sort_values('pressure_rating', ascending=False).reset_index()

batters = compute_batter_metrics(min_pressure_balls=150)
print("\n=== TOP 15 BATTERS UNDER PRESSURE ===")
cols_bat = ['striker_id', 'striker_name', 'p_balls', 'p_runs', 'p_sr', 'all_sr', 'delta_sr', 'p_avg', 'all_avg', 'delta_avg', 'pressure_rating']
print(batters[cols_bat].head(15).to_string())

# 5. Bowling Evaluation
print("\nComputing bowling metrics...")
def compute_bowler_metrics(min_pressure_balls=150):
    p_df = df[df['is_high_pressure']]
    
    p_grp = p_df.groupby(['bowler_id', 'bowler_name'])
    p_stats = pd.DataFrame({
        'p_total_balls': p_grp.size(),
        'p_legal_balls': p_grp['is_legal'].sum(),
        'p_runs': p_grp['bowler_runs_conceded'].sum(),
        'p_wickets': p_grp['is_bowler_wicket'].sum(),
        'p_fours': p_grp['is_four'].sum(),
        'p_sixes': p_grp['is_six'].sum(),
        'p_dots': p_grp['bowler_runs_conceded'].apply(lambda x: (x == 0).sum()),
        'p_matches': p_grp['match_id'].nunique(),
        'p_comp_weight': p_grp['comp_weight'].mean()
    })
    
    all_grp = df.groupby(['bowler_id', 'bowler_name'])
    all_stats = pd.DataFrame({
        'all_total_balls': all_grp.size(),
        'all_legal_balls': all_grp['is_legal'].sum(),
        'all_runs': all_grp['bowler_runs_conceded'].sum(),
        'all_wickets': all_grp['is_bowler_wicket'].sum(),
        'all_fours': all_grp['is_four'].sum(),
        'all_sixes': all_grp['is_six'].sum(),
        'all_dots': all_grp['bowler_runs_conceded'].apply(lambda x: (x == 0).sum()),
        'all_matches': all_grp['match_id'].nunique(),
    })
    
    merged = p_stats.join(all_stats, how='inner')
    merged = merged[merged['p_legal_balls'] >= min_pressure_balls].copy()
    
    # Raw stats
    merged['p_econ'] = (merged['p_runs'] / merged['p_legal_balls']) * 6.0
    merged['p_avg'] = merged['p_runs'] / merged['p_wickets'].replace(0, np.nan)
    merged['p_sr'] = merged['p_legal_balls'] / merged['p_wickets'].replace(0, np.nan)
    merged['p_dot_pct'] = (merged['p_dots'] / merged['p_total_balls']) * 100
    merged['p_boundary_pct'] = ((merged['p_fours'] + merged['p_sixes']) / merged['p_total_balls']) * 100
    
    merged['all_econ'] = (merged['all_runs'] / merged['all_legal_balls']) * 6.0
    merged['all_avg'] = merged['all_runs'] / merged['all_wickets'].replace(0, np.nan)
    merged['all_sr'] = merged['all_legal_balls'] / merged['all_wickets'].replace(0, np.nan)
    merged['all_dot_pct'] = (merged['all_dots'] / merged['all_total_balls']) * 100
    merged['all_boundary_pct'] = ((merged['all_fours'] + merged['all_sixes']) / merged['all_total_balls']) * 100
    
    # Deltas
    merged['delta_econ'] = merged['p_econ'] - merged['all_econ'] # negative is good (less increase than baseline)
    merged['delta_sr'] = merged['p_sr'] - merged['all_sr'] # negative is good (takes wickets faster)
    merged['delta_dot_pct'] = merged['p_dot_pct'] - merged['all_dot_pct']
    
    # Bayesian shrinkage for bowling economy and SR
    prior_econ = (p_df['bowler_runs_conceded'].sum() / max(1, p_df['is_legal'].sum())) * 6.0
    prior_sr = p_df['is_legal'].sum() / max(1, p_df['is_bowler_wicket'].sum())
    
    K_balls = 40.0
    K_wkts = 3.0
    
    merged['bayes_p_econ'] = ((merged['p_legal_balls'] * merged['p_econ']) + (K_balls * prior_econ)) / (merged['p_legal_balls'] + K_balls)
    merged['bayes_p_sr'] = ((merged['p_wickets'] * merged['p_sr']) + (K_wkts * prior_sr)) / (merged['p_wickets'] + K_wkts)
    
    # Standardized scores
    # Lower econ is better -> (mean - value)
    z_econ = (merged['bayes_p_econ'].mean() - merged['bayes_p_econ']) / merged['bayes_p_econ'].std()
    z_sr = (merged['bayes_p_sr'].mean() - merged['bayes_p_sr']) / merged['bayes_p_sr'].std()
    z_dot = (merged['p_dot_pct'] - merged['p_dot_pct'].mean()) / merged['p_dot_pct'].std()
    z_bdry = (merged['p_boundary_pct'].mean() - merged['p_boundary_pct']) / merged['p_boundary_pct'].std()
    
    merged['score_execution'] = (0.40 * z_econ) + (0.35 * z_sr) + (0.15 * z_dot) + (0.10 * z_bdry)
    
    # Clutch elevation: beating expectation
    z_delta_econ = (merged['delta_econ'].mean() - merged['delta_econ']) / merged['delta_econ'].std()
    z_delta_sr = (merged['delta_sr'].mean() - merged['delta_sr']) / merged['delta_sr'].std()
    merged['score_clutch_elevation'] = (0.60 * z_delta_econ) + (0.40 * z_delta_sr)
    
    # Volume and competition factor
    vol_factor = np.log1p(merged['p_legal_balls']) / np.log1p(merged['p_legal_balls'].max())
    
    raw_final = (0.65 * merged['score_execution']) + (0.35 * merged['score_clutch_elevation'])
    adjusted_score = raw_final * merged['p_comp_weight'] * (0.85 + 0.15 * vol_factor)
    
    merged['pressure_rating'] = 50.0 + (adjusted_score - adjusted_score.mean()) / adjusted_score.std() * 15.0
    
    return merged.sort_values('pressure_rating', ascending=False).reset_index()

bowlers = compute_bowler_metrics(min_pressure_balls=150)
print("\n=== TOP 15 BOWLERS UNDER PRESSURE ===")
cols_bowl = ['bowler_id', 'bowler_name', 'p_legal_balls', 'p_runs', 'p_wickets', 'p_econ', 'all_econ', 'delta_econ', 'p_sr', 'all_sr', 'pressure_rating']
print(bowlers[cols_bowl].head(15).to_string())
