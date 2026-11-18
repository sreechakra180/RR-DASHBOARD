import pandas as pd
import numpy as np

df = pd.read_parquet('enriched_bbb.parquet')

# Filter for pressure deliveries
p_df = df[df['is_high_pressure']].copy()
b_df = df[~df['is_high_pressure']].copy()

print("Analyzing Batters...")
bat_p = p_df[p_df['is_batter_ball']]
bat_b = b_df[b_df['is_batter_ball']]
bat_all = df[df['is_batter_ball']]

# Aggregations for batters
p_grp = bat_p.groupby(['striker_id', 'striker_name'])
p_stats = pd.DataFrame({
    'p_balls': p_grp.size(),
    'p_runs': p_grp['runs'].sum(),
    'p_exp_runs': p_grp['exp_bat_runs'].sum(),
    'p_outs': p_grp['is_wicket'].sum(),
    'p_fours': p_grp['is_four'].sum(),
    'p_sixes': p_grp['is_six'].sum(),
    'p_dots': p_grp['runs'].apply(lambda x: (x == 0).sum()),
    'p_matches': p_grp['match_id'].nunique(),
    'p_ipl_balls': p_grp['comp_name'].apply(lambda x: (x.str.contains('IPL', na=False)).sum()),
    'p_comp_weight': p_grp['comp_weight'].mean()
})

all_grp = bat_all.groupby(['striker_id', 'striker_name'])
all_stats = pd.DataFrame({
    'all_balls': all_grp.size(),
    'all_runs': all_grp['runs'].sum(),
    'all_exp_runs': all_grp['exp_bat_runs'].sum(),
    'all_outs': all_grp['is_wicket'].sum(),
    'all_matches': all_grp['match_id'].nunique(),
})

bat_merged = p_stats.join(all_stats, how='inner')
# Qualification: min 175 pressure balls faced
bat_merged = bat_merged[bat_merged['p_balls'] >= 175].copy()

# Metrics
bat_merged['p_sr'] = (bat_merged['p_runs'] / bat_merged['p_balls']) * 100
bat_merged['p_avg'] = bat_merged['p_runs'] / bat_merged['p_outs'].replace(0, 1)
bat_merged['all_sr'] = (bat_merged['all_runs'] / bat_merged['all_balls']) * 100
bat_merged['all_avg'] = bat_merged['all_runs'] / bat_merged['all_outs'].replace(0, 1)

bat_merged['delta_sr'] = bat_merged['p_sr'] - bat_merged['all_sr']
bat_merged['delta_avg'] = bat_merged['p_avg'] - bat_merged['all_avg']

bat_merged['p_bdry_pct'] = ((bat_merged['p_fours'] + bat_merged['p_sixes']) / bat_merged['p_balls']) * 100
bat_merged['p_dot_pct'] = (bat_merged['p_dots'] / bat_merged['p_balls']) * 100

bat_merged['p_rae_100'] = ((bat_merged['p_runs'] - bat_merged['p_exp_runs']) / bat_merged['p_balls']) * 100
bat_merged['all_rae_100'] = ((bat_merged['all_runs'] - bat_merged['all_exp_runs']) / bat_merged['all_balls']) * 100
bat_merged['delta_rae_100'] = bat_merged['p_rae_100'] - bat_merged['all_rae_100']

bat_merged['ipl_pct'] = (bat_merged['p_ipl_balls'] / bat_merged['p_balls']) * 100

# Bayesian shrinkage
prior_sr = (bat_p['runs'].sum() / len(bat_p)) * 100
prior_avg = bat_p['runs'].sum() / max(1, bat_p['is_wicket'].sum())
K_b = 40.0
K_o = 3.0
bat_merged['bayes_sr'] = (bat_merged['p_balls'] * bat_merged['p_sr'] + K_b * prior_sr) / (bat_merged['p_balls'] + K_b)
bat_merged['bayes_avg'] = (bat_merged['p_outs'] * bat_merged['p_avg'] + K_o * prior_avg) / (bat_merged['p_outs'] + K_o)

# Standardization
z_sr = (bat_merged['bayes_sr'] - bat_merged['bayes_sr'].mean()) / bat_merged['bayes_sr'].std()
z_avg = (bat_merged['bayes_avg'] - bat_merged['bayes_avg'].mean()) / bat_merged['bayes_avg'].std()
z_rae = (bat_merged['p_rae_100'] - bat_merged['p_rae_100'].mean()) / bat_merged['p_rae_100'].std()
z_bdry = (bat_merged['p_bdry_pct'] - bat_merged['p_bdry_pct'].mean()) / bat_merged['p_bdry_pct'].std()
z_dot = (bat_merged['p_dot_pct'].mean() - bat_merged['p_dot_pct']) / bat_merged['p_dot_pct'].std()

score_exec = 0.35 * z_sr + 0.30 * z_avg + 0.20 * z_rae + 0.10 * z_bdry + 0.05 * z_dot

z_dsr = (bat_merged['delta_sr'] - bat_merged['delta_sr'].mean()) / bat_merged['delta_sr'].std()
z_davg = (bat_merged['delta_avg'] - bat_merged['delta_avg'].mean()) / bat_merged['delta_avg'].std()
z_drae = (bat_merged['delta_rae_100'] - bat_merged['delta_rae_100'].mean()) / bat_merged['delta_rae_100'].std()

score_clutch = 0.45 * z_dsr + 0.35 * z_davg + 0.20 * z_drae

vol_weight = np.log1p(bat_merged['p_balls']) / np.log1p(bat_merged['p_balls'].max())
comp_adj = bat_merged['p_comp_weight']

raw_bat = (0.60 * score_exec + 0.40 * score_clutch) * comp_adj * (0.85 + 0.15 * vol_weight)
bat_merged['final_rating'] = 50.0 + (raw_bat - raw_bat.mean()) / raw_bat.std() * 15.0

print("\n=== TOP 20 BATTERS UNDER PRESSURE (Min 175 Pressure Balls) ===")
cols_b = ['striker_id', 'striker_name', 'p_balls', 'p_runs', 'p_sr', 'delta_sr', 'p_avg', 'delta_avg', 'p_rae_100', 'p_bdry_pct', 'p_dot_pct', 'ipl_pct', 'final_rating']
top_batters = bat_merged.sort_values('final_rating', ascending=False).reset_index()
print(top_batters[cols_b].head(20).to_string())

# Bowlers
print("\nAnalyzing Bowlers...")
bowl_p = p_df[p_df['is_legal']]
bowl_all = df[df['is_legal']]

p_bgrp = p_df.groupby(['bowler_id', 'bowler_name'])
p_bstats = pd.DataFrame({
    'p_total_balls': p_bgrp.size(),
    'p_legal_balls': p_bgrp['is_legal'].sum(),
    'p_runs': p_bgrp['bowler_runs_conceded'].sum(),
    'p_exp_runs': p_bgrp['exp_bowl_runs'].sum(),
    'p_wickets': p_bgrp['is_bowler_wicket'].sum(),
    'p_exp_wickets': p_bgrp['exp_bowl_wkt'].sum(),
    'p_fours': p_bgrp['is_four'].sum(),
    'p_sixes': p_bgrp['is_six'].sum(),
    'p_dots': p_bgrp['bowler_runs_conceded'].apply(lambda x: (x == 0).sum()),
    'p_matches': p_bgrp['match_id'].nunique(),
    'p_ipl_balls': p_bgrp['comp_name'].apply(lambda x: (x.str.contains('IPL', na=False)).sum()),
    'p_comp_weight': p_bgrp['comp_weight'].mean()
})

all_bgrp = df.groupby(['bowler_id', 'bowler_name'])
all_bstats = pd.DataFrame({
    'all_total_balls': all_bgrp.size(),
    'all_legal_balls': all_bgrp['is_legal'].sum(),
    'all_runs': all_bgrp['bowler_runs_conceded'].sum(),
    'all_exp_runs': all_bgrp['exp_bowl_runs'].sum(),
    'all_wickets': all_bgrp['is_bowler_wicket'].sum(),
    'all_exp_wickets': all_bgrp['exp_bowl_wkt'].sum(),
    'all_matches': all_bgrp['match_id'].nunique(),
})

bowl_merged = p_bstats.join(all_bstats, how='inner')
bowl_merged = bowl_merged[bowl_merged['p_legal_balls'] >= 175].copy()

bowl_merged['p_econ'] = (bowl_merged['p_runs'] / bowl_merged['p_legal_balls']) * 6.0
bowl_merged['p_sr'] = bowl_merged['p_legal_balls'] / bowl_merged['p_wickets'].replace(0, np.nan)
bowl_merged['p_avg'] = bowl_merged['p_runs'] / bowl_merged['p_wickets'].replace(0, np.nan)

bowl_merged['all_econ'] = (bowl_merged['all_runs'] / bowl_merged['all_legal_balls']) * 6.0
bowl_merged['all_sr'] = bowl_merged['all_legal_balls'] / bowl_merged['all_wickets'].replace(0, np.nan)
bowl_merged['all_avg'] = bowl_merged['all_runs'] / bowl_merged['all_wickets'].replace(0, np.nan)

bowl_merged['delta_econ'] = bowl_merged['p_econ'] - bowl_merged['all_econ']
bowl_merged['delta_sr'] = bowl_merged['p_sr'] - bowl_merged['all_sr']

bowl_merged['p_dot_pct'] = (bowl_merged['p_dots'] / bowl_merged['p_total_balls']) * 100
bowl_merged['p_bdry_pct'] = ((bowl_merged['p_fours'] + bowl_merged['p_sixes']) / bowl_merged['p_total_balls']) * 100

bowl_merged['p_rspo'] = ((bowl_merged['p_exp_runs'] - bowl_merged['p_runs']) / bowl_merged['p_legal_balls']) * 6.0
bowl_merged['all_rspo'] = ((bowl_merged['all_exp_runs'] - bowl_merged['all_runs']) / bowl_merged['all_legal_balls']) * 6.0
bowl_merged['delta_rspo'] = bowl_merged['p_rspo'] - bowl_merged['all_rspo']

bowl_merged['p_wapo'] = ((bowl_merged['p_wickets'] - bowl_merged['p_exp_wickets']) / bowl_merged['p_legal_balls']) * 24.0
bowl_merged['ipl_pct'] = (bowl_merged['p_ipl_balls'] / bowl_merged['p_total_balls']) * 100

# Bayesian shrinkage
prior_econ = (p_df['bowler_runs_conceded'].sum() / max(1, p_df['is_legal'].sum())) * 6.0
prior_bsr = p_df['is_legal'].sum() / max(1, p_df['is_bowler_wicket'].sum())
bowl_merged['bayes_econ'] = (bowl_merged['p_legal_balls'] * bowl_merged['p_econ'] + K_b * prior_econ) / (bowl_merged['p_legal_balls'] + K_b)
bowl_merged['bayes_sr'] = (bowl_merged['p_wickets'] * bowl_merged['p_sr'] + K_o * prior_bsr) / (bowl_merged['p_wickets'] + K_o)

# Standardization
z_becon = (bowl_merged['bayes_econ'].mean() - bowl_merged['bayes_econ']) / bowl_merged['bayes_econ'].std()
z_bsr = (bowl_merged['bayes_sr'].mean() - bowl_merged['bayes_sr']) / bowl_merged['bayes_sr'].std()
z_rspo = (bowl_merged['p_rspo'] - bowl_merged['p_rspo'].mean()) / bowl_merged['p_rspo'].std()
z_wapo = (bowl_merged['p_wapo'] - bowl_merged['p_wapo'].mean()) / bowl_merged['p_wapo'].std()
z_bdot = (bowl_merged['p_dot_pct'] - bowl_merged['p_dot_pct'].mean()) / bowl_merged['p_dot_pct'].std()
z_bbdry = (bowl_merged['p_bdry_pct'].mean() - bowl_merged['p_bdry_pct']) / bowl_merged['p_bdry_pct'].std()

score_bexec = 0.30 * z_becon + 0.25 * z_bsr + 0.25 * z_rspo + 0.10 * z_wapo + 0.05 * z_bdot + 0.05 * z_bbdry

z_decon = (bowl_merged['delta_econ'].mean() - bowl_merged['delta_econ']) / bowl_merged['delta_econ'].std()
z_dsr = (bowl_merged['delta_sr'].mean() - bowl_merged['delta_sr']) / bowl_merged['delta_sr'].std()
z_drspo = (bowl_merged['delta_rspo'] - bowl_merged['delta_rspo'].mean()) / bowl_merged['delta_rspo'].std()

score_bclutch = 0.40 * z_decon + 0.30 * z_dsr + 0.30 * z_drspo

vol_bweight = np.log1p(bowl_merged['p_legal_balls']) / np.log1p(bowl_merged['p_legal_balls'].max())
comp_badj = bowl_merged['p_comp_weight']

raw_bowl = (0.60 * score_bexec + 0.40 * score_bclutch) * comp_badj * (0.85 + 0.15 * vol_bweight)
bowl_merged['final_rating'] = 50.0 + (raw_bowl - raw_bowl.mean()) / raw_bowl.std() * 15.0

print("\n=== TOP 20 BOWLERS UNDER PRESSURE (Min 175 Legal Pressure Balls) ===")
cols_bo = ['bowler_id', 'bowler_name', 'p_legal_balls', 'p_wickets', 'p_econ', 'delta_econ', 'p_sr', 'delta_sr', 'p_rspo', 'p_wapo', 'ipl_pct', 'final_rating']
top_bowlers = bowl_merged.sort_values('final_rating', ascending=False).reset_index()
print(top_bowlers[cols_bo].head(20).to_string())
