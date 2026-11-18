import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

df = pd.read_parquet('processed_bbb.parquet')

# Train 2nd innings WP model
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

# Batter stats calculation
def get_batting_stats(min_balls=150):
    # Filter to batter balls
    bat_df = df[df['is_batter_ball']].copy()
    
    # Pressure
    p_df = bat_df[bat_df['is_high_pressure']]
    b_df = bat_df[~bat_df['is_high_pressure']]
    
    def agg_bat(sub_df, prefix=''):
        grouped = sub_df.groupby(['striker_id', 'striker_name'])
        stats = pd.DataFrame({
            f'{prefix}balls': grouped.size(),
            f'{prefix}runs': grouped['runs'].sum(),
            f'{prefix}outs': grouped['is_wicket'].sum(),
            f'{prefix}fours': grouped['is_four'].sum(),
            f'{prefix}sixes': grouped['is_six'].sum(),
            f'{prefix}dots': (grouped['runs'].apply(lambda x: (x == 0).sum())),
            f'{prefix}matches': grouped['match_id'].nunique(),
        })
        stats[f'{prefix}sr'] = (stats[f'{prefix}runs'] / stats[f'{prefix}balls']) * 100
        stats[f'{prefix}avg'] = stats[f'{prefix}runs'] / stats[f'{prefix}outs'].replace(0, 1)
        stats[f'{prefix}boundary_pct'] = ((stats[f'{prefix}fours'] + stats[f'{prefix}sixes']) / stats[f'{prefix}balls']) * 100
        stats[f'{prefix}dot_pct'] = (stats[f'{prefix}dots'] / stats[f'{prefix}balls']) * 100
        return stats

    p_stats = agg_bat(p_df, prefix='p_')
    all_stats = agg_bat(bat_df, prefix='all_')
    
    merged = p_stats.join(all_stats, how='inner')
    merged = merged[merged['p_balls'] >= min_balls].copy()
    
    # Deltas
    merged['delta_sr'] = merged['p_sr'] - merged['all_sr']
    merged['delta_avg'] = merged['p_avg'] - merged['all_avg']
    merged['delta_dot'] = merged['p_dot_pct'] - merged['all_dot_pct']
    
    return merged.reset_index()

batters = get_batting_stats(min_balls=150)
print(f"Qualifying batters (>=150 pressure balls): {len(batters)}")
print("\nTop 15 batters by Pressure Strike Rate (min 150 balls):")
cols_show = ['striker_id', 'striker_name', 'p_balls', 'p_runs', 'p_sr', 'all_sr', 'delta_sr', 'p_avg', 'all_avg', 'p_boundary_pct', 'p_dot_pct']
print(batters.sort_values('p_sr', ascending=False)[cols_show].head(15).to_string())

print("\nTop 15 batters by Pressure Average (min 150 balls):")
print(batters.sort_values('p_avg', ascending=False)[cols_show].head(15).to_string())
