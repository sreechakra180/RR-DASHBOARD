import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

print("Loading data...")
df = pd.read_parquet('processed_bbb.parquet')

# 1. Identify high pressure
# 2nd innings WP model
inn2_mask = (df['innings'] == 2) & (df['chase_won'].notna()) & (df['balls_remaining'] > 0)
features_wp = ['runs_needed', 'balls_remaining', 'wickets_in_hand', 'rrr', 'target']
wp_model = LogisticRegression(max_iter=1000)
wp_model.fit(df.loc[inn2_mask, features_wp], df.loc[inn2_mask, 'chase_won'])
df['wp'] = np.nan
df.loc[inn2_mask, 'wp'] = wp_model.predict_proba(df.loc[inn2_mask, features_wp])[:, 1]
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

# Add competition quality tier weight
def get_comp_tier(name):
    if 'IPL' in str(name):
        return 1.0
    elite_t20i = ['Australia', 'India', 'England', 'South Africa', 'Pakistan', 'New Zealand', 'West Indies', 'Sri Lanka', 'Afghanistan']
    if any(country in str(name) for country in elite_t20i):
        return 0.95
    return 0.85

df['comp_weight'] = df['comp_name'].apply(get_comp_tier)

print("Training Expected Runs & Expected Wickets models...")
# Features for ball expectation:
# overid, innings, wickets_in_hand, cum_runs_before, comp_weight, is_knockout
exp_features = ['overid', 'innings', 'wickets_in_hand', 'cum_runs_before', 'comp_weight', 'is_knockout']
X_exp = df[exp_features]

# Batting runs expectation model
reg_bat = HistGradientBoostingRegressor(max_iter=100, min_samples_leaf=50, random_state=42)
reg_bat.fit(X_exp, df['runs'])
df['exp_bat_runs'] = reg_bat.predict(X_exp)

# Bowler runs expectation model
reg_bowl = HistGradientBoostingRegressor(max_iter=100, min_samples_leaf=50, random_state=42)
reg_bowl.fit(X_exp, df['bowler_runs_conceded'])
df['exp_bowl_runs'] = reg_bowl.predict(X_exp)

# Bowler wicket expectation model
clf_wkt = HistGradientBoostingClassifier(max_iter=100, min_samples_leaf=50, random_state=42)
clf_wkt.fit(X_exp, df['is_bowler_wicket'].astype(int))
df['exp_bowl_wkt'] = clf_wkt.predict_proba(X_exp)[:, 1]

print("Computing Context-Adjusted Metrics for Batters...")
bat_df = df[df['is_batter_ball']].copy()

def evaluate_batters(min_pressure_balls=150):
    p_df = bat_df[bat_df['is_high_pressure']]
    base_df = bat_df[~bat_df['is_high_pressure']]
    
    # Pressure aggregations
    p_grp = p_df.groupby(['striker_id', 'striker_name'])
    p_stats = pd.DataFrame({
        'p_balls': p_grp.size(),
        'p_runs': p_grp['runs'].sum(),
        'p_exp_runs': p_grp['exp_bat_runs'].sum(),
        'p_outs': p_grp['is_wicket'].sum(),
        'p_fours': p_grp['is_four'].sum(),
        'p_sixes': p_grp['is_six'].sum(),
        'p_dots': p_grp['runs'].apply(lambda x: (x == 0).sum()),
        'p_matches': p_grp['match_id'].nunique(),
        'p_comp_weight': p_grp['comp_weight'].mean()
    })
    
    # Baseline aggregations
    b_grp = base_df.groupby(['striker_id', 'striker_name'])
    b_stats = pd.DataFrame({
        'b_balls': b_grp.size(),
        'b_runs': b_grp['runs'].sum(),
        'b_exp_runs': b_grp['exp_bat_runs'].sum(),
        'b_outs': b_grp['is_wicket'].sum(),
        'b_matches': b_grp['match_id'].nunique(),
    })
    
    all_grp = bat_df.groupby(['striker_id', 'striker_name'])
    all_stats = pd.DataFrame({
        'all_balls': all_grp.size(),
        'all_runs': all_grp['runs'].sum(),
        'all_outs': all_grp['is_wicket'].sum(),
    })
    
    merged = p_stats.join(b_stats, how='inner').join(all_stats, how='inner')
    merged = merged[merged['p_balls'] >= min_pressure_balls].copy()
    
    # Runs Above Expectation per 100 balls
    merged['p_rae_per_100'] = ((merged['p_runs'] - merged['p_exp_runs']) / merged['p_balls']) * 100
    merged['b_rae_per_100'] = ((merged['b_runs'] - merged['b_exp_runs']) / merged['b_balls'].replace(0, 1)) * 100
    merged['delta_rae_per_100'] = merged['p_rae_per_100'] - merged['b_rae_per_100']
    
    # Traditional stats
    merged['p_sr'] = (merged['p_runs'] / merged['p_balls']) * 100
    merged['p_avg'] = merged['p_runs'] / merged['p_outs'].replace(0, 1)
    merged['all_sr'] = (merged['all_runs'] / merged['all_balls']) * 100
    merged['all_avg'] = merged['all_runs'] / merged['all_outs'].replace(0, 1)
    merged['delta_sr'] = merged['p_sr'] - merged['all_sr']
    merged['delta_avg'] = merged['p_avg'] - merged['all_avg']
    
    merged['p_bdry_pct'] = ((merged['p_fours'] + merged['p_sixes']) / merged['p_balls']) * 100
    merged['p_dot_pct'] = (merged['p_dots'] / merged['p_balls']) * 100
    
    # Composite Score:
    # 1. Situational Execution: RAE per 100 (50%) + Pressure Avg (30%) + Bdry% (10%) + Dot% (10%)
    z_rae = (merged['p_rae_per_100'] - merged['p_rae_per_100'].mean()) / merged['p_rae_per_100'].std()
    z_avg = (merged['p_avg'] - merged['p_avg'].mean()) / merged['p_avg'].std()
    z_bdry = (merged['p_bdry_pct'] - merged['p_bdry_pct'].mean()) / merged['p_bdry_pct'].std()
    z_dot = (merged['p_dot_pct'].mean() - merged['p_dot_pct']) / merged['p_dot_pct'].std()
    
    execution_score = (0.50 * z_rae) + (0.30 * z_avg) + (0.10 * z_bdry) + (0.10 * z_dot)
    
    # 2. Clutch Elevation: Delta RAE (60%) + Delta SR (40%)
    z_d_rae = (merged['delta_rae_per_100'] - merged['delta_rae_per_100'].mean()) / merged['delta_rae_per_100'].std()
    z_d_sr = (merged['delta_sr'] - merged['delta_sr'].mean()) / merged['delta_sr'].std()
    elevation_score = (0.60 * z_d_rae) + (0.40 * z_d_sr)
    
    # Volume and comp factor
    vol_factor = np.log1p(merged['p_balls']) / np.log1p(merged['p_balls'].max())
    
    total_raw = (0.65 * execution_score) + (0.35 * elevation_score)
    final_score = total_raw * merged['p_comp_weight'] * (0.85 + 0.15 * vol_factor)
    
    merged['composite_rating'] = 50.0 + (final_score - final_score.mean()) / final_score.std() * 15.0
    return merged.sort_values('composite_rating', ascending=False).reset_index()

batters_eval = evaluate_batters(min_pressure_balls=150)
print("\n=== TOP 20 BATTERS RANKING ===")
cols_b_show = ['striker_id', 'striker_name', 'p_balls', 'p_runs', 'p_sr', 'p_avg', 'p_rae_per_100', 'delta_sr', 'delta_avg', 'composite_rating']
print(batters_eval[cols_b_show].head(20).to_string())

print("\nComputing Context-Adjusted Metrics for Bowlers...")
def evaluate_bowlers(min_pressure_balls=150):
    p_df = df[df['is_high_pressure']]
    base_df = df[~df['is_high_pressure']]
    
    p_grp = p_df.groupby(['bowler_id', 'bowler_name'])
    p_stats = pd.DataFrame({
        'p_total_balls': p_grp.size(),
        'p_legal_balls': p_grp['is_legal'].sum(),
        'p_runs': p_grp['bowler_runs_conceded'].sum(),
        'p_exp_runs': p_grp['exp_bowl_runs'].sum(),
        'p_wickets': p_grp['is_bowler_wicket'].sum(),
        'p_exp_wickets': p_grp['exp_bowl_wkt'].sum(),
        'p_fours': p_grp['is_four'].sum(),
        'p_sixes': p_grp['is_six'].sum(),
        'p_dots': p_grp['bowler_runs_conceded'].apply(lambda x: (x == 0).sum()),
        'p_matches': p_grp['match_id'].nunique(),
        'p_comp_weight': p_grp['comp_weight'].mean()
    })
    
    b_grp = base_df.groupby(['bowler_id', 'bowler_name'])
    b_stats = pd.DataFrame({
        'b_legal_balls': b_grp['is_legal'].sum(),
        'b_runs': b_grp['bowler_runs_conceded'].sum(),
        'b_exp_runs': b_grp['exp_bowl_runs'].sum(),
        'b_wickets': b_grp['is_bowler_wicket'].sum(),
        'b_exp_wickets': b_grp['exp_bowl_wkt'].sum(),
    })
    
    all_grp = df.groupby(['bowler_id', 'bowler_name'])
    all_stats = pd.DataFrame({
        'all_legal_balls': all_grp['is_legal'].sum(),
        'all_runs': all_grp['bowler_runs_conceded'].sum(),
        'all_wickets': all_grp['is_bowler_wicket'].sum(),
    })
    
    merged = p_stats.join(b_stats, how='inner').join(all_stats, how='inner')
    merged = merged[merged['p_legal_balls'] >= min_pressure_balls].copy()
    
    # Runs Saved Above Expectation per 6 balls (RSPO)
    merged['p_rspo'] = ((merged['p_exp_runs'] - merged['p_runs']) / merged['p_legal_balls']) * 6.0
    merged['b_rspo'] = ((merged['b_exp_runs'] - merged['b_runs']) / merged['b_legal_balls'].replace(0, 1)) * 6.0
    merged['delta_rspo'] = merged['p_rspo'] - merged['b_rspo'] # positive means saved even more under pressure
    
    # Wickets Above Expectation per 24 balls (WAPO)
    merged['p_wapo'] = ((merged['p_wickets'] - merged['p_exp_wickets']) / merged['p_legal_balls']) * 24.0
    
    # Traditional stats
    merged['p_econ'] = (merged['p_runs'] / merged['p_legal_balls']) * 6.0
    merged['p_sr'] = merged['p_legal_balls'] / merged['p_wickets'].replace(0, np.nan)
    merged['p_avg'] = merged['p_runs'] / merged['p_wickets'].replace(0, np.nan)
    merged['all_econ'] = (merged['all_runs'] / merged['all_legal_balls']) * 6.0
    merged['all_sr'] = merged['all_legal_balls'] / merged['all_wickets'].replace(0, np.nan)
    merged['delta_econ'] = merged['p_econ'] - merged['all_econ']
    
    merged['p_dot_pct'] = (merged['p_dots'] / merged['p_total_balls']) * 100
    merged['p_bdry_pct'] = ((merged['p_fours'] + merged['p_sixes']) / merged['p_total_balls']) * 100
    
    # Composite Score:
    # 1. Execution: RSPO (45%) + WAPO (35%) + Dot% (10%) + Low Bdry% (10%)
    z_rspo = (merged['p_rspo'] - merged['p_rspo'].mean()) / merged['p_rspo'].std()
    z_wapo = (merged['p_wapo'] - merged['p_wapo'].mean()) / merged['p_wapo'].std()
    z_dot = (merged['p_dot_pct'] - merged['p_dot_pct'].mean()) / merged['p_dot_pct'].std()
    z_bdry = (merged['p_bdry_pct'].mean() - merged['p_bdry_pct']) / merged['p_bdry_pct'].std()
    
    execution_score = (0.45 * z_rspo) + (0.35 * z_wapo) + (0.10 * z_dot) + (0.10 * z_bdry)
    
    # 2. Elevation: Delta RSPO (60%) + (all_econ - p_econ) (40%)
    z_d_rspo = (merged['delta_rspo'] - merged['delta_rspo'].mean()) / merged['delta_rspo'].std()
    z_d_econ = (merged['delta_econ'].mean() - merged['delta_econ']) / merged['delta_econ'].std()
    elevation_score = (0.60 * z_d_rspo) + (0.40 * z_d_econ)
    
    # Volume and comp factor
    vol_factor = np.log1p(merged['p_legal_balls']) / np.log1p(merged['p_legal_balls'].max())
    
    total_raw = (0.65 * execution_score) + (0.35 * elevation_score)
    final_score = total_raw * merged['p_comp_weight'] * (0.85 + 0.15 * vol_factor)
    
    merged['composite_rating'] = 50.0 + (final_score - final_score.mean()) / final_score.std() * 15.0
    return merged.sort_values('composite_rating', ascending=False).reset_index()

bowlers_eval = evaluate_bowlers(min_pressure_balls=150)
print("\n=== TOP 20 BOWLERS RANKING ===")
cols_bowl_show = ['bowler_id', 'bowler_name', 'p_legal_balls', 'p_wickets', 'p_econ', 'p_sr', 'p_rspo', 'p_wapo', 'delta_rspo', 'composite_rating']
print(bowlers_eval[cols_bowl_show].head(20).to_string())
