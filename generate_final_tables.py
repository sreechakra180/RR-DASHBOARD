import pandas as pd
import numpy as np

# Load enriched data
df = pd.read_parquet('enriched_bbb.parquet')
p_df = df[df['is_high_pressure']]
b_df = df[~df['is_high_pressure']]

# Batters top 10 list
top_10_batters_ids = [2756, 6396, 37052, 3647, 8182, 4117, 813, 4708, 2639, 20702]

# Bowlers top 10 list
top_10_bowlers_ids = [2933, 781, 4219, 2417, 8661, 3644, 6256, 4722, 278, 6464]

print("=== VERIFIED TOP 10 BATTERS METRICS ===")
bat_rows = []
for pid in top_10_batters_ids:
    sub_p = p_df[(p_df['striker_id'] == pid) & (p_df['is_batter_ball'])]
    sub_all = df[(df['striker_id'] == pid) & (df['is_batter_ball'])]
    
    name = sub_p['striker_name'].iloc[0]
    p_balls = len(sub_p)
    p_runs = sub_p['runs'].sum()
    p_outs = sub_p['is_wicket'].sum()
    p_sr = (p_runs / p_balls) * 100
    p_avg = p_runs / max(1, p_outs)
    
    all_balls = len(sub_all)
    all_runs = sub_all['runs'].sum()
    all_outs = sub_all['is_wicket'].sum()
    all_sr = (all_runs / all_balls) * 100
    all_avg = all_runs / max(1, all_outs)
    
    delta_sr = p_sr - all_sr
    delta_avg = p_avg - all_avg
    
    p_bdry = ((sub_p['is_four'].sum() + sub_p['is_six'].sum()) / p_balls) * 100
    p_dot = ((sub_p['runs'] == 0).sum() / p_balls) * 100
    
    p_rae_100 = ((p_runs - sub_p['exp_bat_runs'].sum()) / p_balls) * 100
    p_matches = sub_p['match_id'].nunique()
    
    bat_rows.append({
        'Rank': len(bat_rows) + 1,
        'Player ID': pid,
        'Player Name': name,
        'Matches': p_matches,
        'Balls': p_balls,
        'Runs': p_runs,
        'Pressure SR': round(p_sr, 2),
        'Baseline SR': round(all_sr, 2),
        'Delta SR': round(delta_sr, 2),
        'Pressure Avg': round(p_avg, 2),
        'Baseline Avg': round(all_avg, 2),
        'Delta Avg': round(delta_avg, 2),
        'RAE / 100': round(p_rae_100, 2),
        'Boundary %': round(p_bdry, 2),
        'Dot %': round(p_dot, 2)
    })

bat_df_table = pd.DataFrame(bat_rows)
print(bat_df_table.to_string(index=False))

print("\n=== VERIFIED TOP 10 BOWLERS METRICS ===")
bowl_rows = []
for pid in top_10_bowlers_ids:
    sub_p = p_df[(p_df['bowler_id'] == pid) & (p_df['is_legal'])]
    sub_all = df[(df['bowler_id'] == pid) & (df['is_legal'])]
    sub_p_tot = p_df[p_df['bowler_id'] == pid]
    
    name = sub_p['bowler_name'].iloc[0]
    p_legal = len(sub_p)
    p_runs = sub_p['bowler_runs_conceded'].sum()
    p_wkts = sub_p['is_bowler_wicket'].sum()
    p_econ = (p_runs / p_legal) * 6.0
    p_sr = p_legal / max(1, p_wkts)
    p_avg = p_runs / max(1, p_wkts)
    
    all_legal = len(sub_all)
    all_runs = sub_all['bowler_runs_conceded'].sum()
    all_wkts = sub_all['is_bowler_wicket'].sum()
    all_econ = (all_runs / all_legal) * 6.0
    all_sr = all_legal / max(1, all_wkts)
    
    delta_econ = p_econ - all_econ
    delta_sr = p_sr - all_sr
    
    p_dots = (sub_p_tot['bowler_runs_conceded'] == 0).sum()
    p_dot_pct = (p_dots / len(sub_p_tot)) * 100
    p_bdry_pct = ((sub_p_tot['is_four'].sum() + sub_p_tot['is_six'].sum()) / len(sub_p_tot)) * 100
    
    p_rspo = ((sub_p['exp_bowl_runs'].sum() - p_runs) / p_legal) * 6.0
    p_wapo = ((p_wkts - sub_p['exp_bowl_wkt'].sum()) / p_legal) * 24.0
    p_matches = sub_p['match_id'].nunique()
    
    bowl_rows.append({
        'Rank': len(bowl_rows) + 1,
        'Player ID': pid,
        'Player Name': name,
        'Matches': p_matches,
        'Legal Balls': p_legal,
        'Runs Conc.': p_runs,
        'Wickets': p_wkts,
        'Pressure Econ': round(p_econ, 2),
        'Baseline Econ': round(all_econ, 2),
        'Delta Econ': round(delta_econ, 2),
        'Pressure SR': round(p_sr, 2),
        'Baseline SR': round(all_sr, 2),
        'Delta SR': round(delta_sr, 2),
        'RSPO': round(p_rspo, 2),
        'WAPO': round(p_wapo, 2),
        'Dot %': round(p_dot_pct, 2),
        'Boundary %': round(p_bdry_pct, 2)
    })

bowl_df_table = pd.DataFrame(bowl_rows)
print(bowl_df_table.to_string(index=False))

# Save to CSV for reference
bat_df_table.to_csv('top_10_batters_summary.csv', index=False)
bowl_df_table.to_csv('top_10_bowlers_summary.csv', index=False)
