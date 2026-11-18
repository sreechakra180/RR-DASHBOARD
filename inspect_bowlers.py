import pandas as pd
import numpy as np

# Load processed data
df = pd.read_parquet('processed_bbb.parquet')

# Apply high pressure definition
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

def get_bowling_stats(min_balls=150):
    p_df = df[df['is_high_pressure']].copy()
    all_df = df.copy()
    
    def agg_bowl(sub_df, prefix=''):
        grouped = sub_df.groupby(['bowler_id', 'bowler_name'])
        stats = pd.DataFrame({
            f'{prefix}total_balls': grouped.size(),
            f'{prefix}legal_balls': grouped['is_legal'].sum(),
            f'{prefix}runs': grouped['bowler_runs_conceded'].sum(),
            f'{prefix}wickets': grouped['is_bowler_wicket'].sum(),
            f'{prefix}dots': (grouped['bowler_runs_conceded'].apply(lambda x: (x == 0).sum())),
            f'{prefix}fours': grouped['is_four'].sum(),
            f'{prefix}sixes': grouped['is_six'].sum(),
            f'{prefix}matches': grouped['match_id'].nunique(),
        })
        stats[f'{prefix}econ'] = (stats[f'{prefix}runs'] / stats[f'{prefix}legal_balls'].replace(0, np.nan)) * 6.0
        stats[f'{prefix}avg'] = stats[f'{prefix}runs'] / stats[f'{prefix}wickets'].replace(0, np.nan)
        stats[f'{prefix}sr'] = stats[f'{prefix}legal_balls'] / stats[f'{prefix}wickets'].replace(0, np.nan)
        stats[f'{prefix}dot_pct'] = (stats[f'{prefix}dots'] / stats[f'{prefix}total_balls']) * 100
        stats[f'{prefix}boundary_pct'] = ((stats[f'{prefix}fours'] + stats[f'{prefix}sixes']) / stats[f'{prefix}total_balls']) * 100
        return stats

    p_stats = agg_bowl(p_df, prefix='p_')
    all_stats = agg_bowl(all_df, prefix='all_')
    
    merged = p_stats.join(all_stats, how='inner')
    merged = merged[merged['p_legal_balls'] >= min_balls].copy()
    
    merged['delta_econ'] = merged['p_econ'] - merged['all_econ']
    merged['delta_sr'] = merged['p_sr'] - merged['all_sr']
    merged['delta_dot'] = merged['p_dot_pct'] - merged['all_dot_pct']
    
    return merged.reset_index()

bowlers = get_bowling_stats(min_balls=150)
print(f"Qualifying bowlers (>=150 legal pressure balls): {len(bowlers)}")

cols_show = ['bowler_id', 'bowler_name', 'p_legal_balls', 'p_runs', 'p_wickets', 'p_econ', 'all_econ', 'delta_econ', 'p_avg', 'p_sr', 'p_dot_pct', 'p_boundary_pct']
print("\nTop 15 bowlers by Pressure Economy Rate (min 150 legal balls):")
print(bowlers.sort_values('p_econ', ascending=True)[cols_show].head(15).to_string())

print("\nTop 15 bowlers by Pressure Wickets (min 150 legal balls):")
print(bowlers.sort_values('p_wickets', ascending=False)[cols_show].head(15).to_string())

print("\nTop 15 bowlers by Pressure Dot Ball % (min 150 legal balls):")
print(bowlers.sort_values('p_dot_pct', ascending=False)[cols_show].head(15).to_string())
