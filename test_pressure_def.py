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

# Criteria
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

print('Total balls:', len(df))
print(f'High pressure balls: {df["is_high_pressure"].sum()} ({df["is_high_pressure"].mean()*100:.1f}%)')
print('High pressure by innings:')
print(df.groupby('innings')['is_high_pressure'].agg(['count', 'sum', 'mean']))
