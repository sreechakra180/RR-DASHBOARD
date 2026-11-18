import pandas as pd
import numpy as np

def build_dataset():
    print("Loading raw files...")
    bbb = pd.read_csv('Data Pack/ball_by_ball.csv')
    matches = pd.read_csv('Data Pack/matches.csv')
    comps = pd.read_csv('Data Pack/competitions.csv')
    teams = pd.read_csv('Data Pack/teams.csv')
    
    # Precompute match info
    # Innings 1 total runs
    bbb['ball_runs'] = bbb['runs'] + bbb['extras1_runs'] + bbb['extras_2_runs']
    bbb['is_legal'] = ~bbb['extras_type'].isin(['wide', 'Overstep'])
    
    inn1_totals = bbb[bbb['innings'] == 1].groupby('match_id')['ball_runs'].sum().to_dict()
    
    # Target and total balls for each match
    match_dict = {}
    for _, row in matches.iterrows():
        mid = row['match_id']
        rev_target = row['revised_target']
        if pd.notna(rev_target) and rev_target > 0:
            target = float(rev_target)
        elif mid in inn1_totals:
            target = float(inn1_totals[mid] + 1)
        else:
            target = np.nan
            
        inn1_max_balls = float(row['first_inn_revised_overs']) if pd.notna(row['first_inn_revised_overs']) else 120.0
        inn2_max_balls = float(row['second_inn_revised_overs']) if pd.notna(row['second_inn_revised_overs']) else 120.0
        
        is_knockout = int(row['knock_out']) if pd.notna(row['knock_out']) else 0
        stage = str(row['stage']) if pd.notna(row['stage']) else ''
        if stage in ['Final', 'Qualifier', 'Eliminator', 'Semi Final', 'Super 4']:
            is_knockout = 1
            
        win_team_id = row['win_team_id']
        comp_id = row['comp_id']
        
        match_dict[mid] = {
            'target': target,
            'inn1_max_balls': inn1_max_balls,
            'inn2_max_balls': inn2_max_balls,
            'is_knockout': is_knockout,
            'stage': stage,
            'win_team_id': win_team_id,
            'comp_id': comp_id
        }
        
    comp_info = comps.set_index('comp_id').to_dict('index')
    
    # Sort BBB properly
    bbb = bbb.sort_values(['match_id', 'innings', 'overid', 'over_ball_num']).reset_index(drop=True)
    
    # Bowler conceded runs: runs + wide + noball extras
    # Wide or Overstep extras go to bowler
    def calc_bowler_runs(r):
        ext_type = r['extras_type']
        e1 = r['extras1_runs']
        runs = r['runs']
        if ext_type in ['wide', 'Overstep']:
            return runs + e1
        elif pd.isna(ext_type):
            return runs
        else: # legbyes, byes, penalty, other
            return runs
    bbb['bowler_runs_conceded'] = bbb.apply(calc_bowler_runs, axis=1)
    
    # Bowler wicket: dismissal_type in caught, bowled, lbw, stump, hit wicket
    bowler_dismissals = {'Caught', 'Bowled', 'LBW', 'Stump', 'Hit Wicket'}
    bbb['is_bowler_wicket'] = bbb['is_wicket'] & bbb['dismissal_type'].isin(bowler_dismissals)
    
    # Batter ball faced: extras_type != 'wide'
    bbb['is_batter_ball'] = bbb['extras_type'] != 'wide'
    
    print("Vectorizing cumulative match progression...")
    # Group by match_id and innings to compute cum_runs_before, cum_wickets_before, legal_balls_before
    # Note: cumulative sum shifted by 1 gives state *before* current delivery
    grouped = bbb.groupby(['match_id', 'innings'])
    
    bbb['cum_runs_before'] = grouped['ball_runs'].shift(1).fillna(0).astype(int)
    bbb['cum_wickets_before'] = grouped['is_wicket'].shift(1).fillna(0).astype(int)
    # Cumulative legal balls
    bbb['legal_balls_bowled_before'] = grouped['is_legal'].shift(1).fillna(0).astype(int)
    
    # Now cumulate within groups properly
    bbb['cum_runs_before'] = grouped['ball_runs'].apply(lambda x: x.cumsum().shift(1).fillna(0)).reset_index(drop=True).astype(int)
    bbb['cum_wickets_before'] = grouped['is_wicket'].apply(lambda x: x.cumsum().shift(1).fillna(0)).reset_index(drop=True).astype(int)
    bbb['legal_balls_bowled_before'] = grouped['is_legal'].apply(lambda x: x.cumsum().shift(1).fillna(0)).reset_index(drop=True).astype(int)
    
    print("Adding match metadata and situation indicators...")
    # Add match info
    targets = []
    max_balls_list = []
    is_ko_list = []
    stage_list = []
    win_team_list = []
    comp_format_list = []
    comp_name_list = []
    season_list = []
    
    for idx, row in bbb.iterrows():
        mid = row['match_id']
        inn = row['innings']
        minfo = match_dict.get(mid, {})
        tgt = minfo.get('target', np.nan)
        if inn == 1:
            max_b = minfo.get('inn1_max_balls', 120.0)
        elif inn == 2:
            max_b = minfo.get('inn2_max_balls', 120.0)
        else: # Super over
            max_b = 6.0
            tgt = minfo.get('target', np.nan)
            
        targets.append(tgt)
        max_balls_list.append(max_b)
        is_ko_list.append(minfo.get('is_knockout', 0))
        stage_list.append(minfo.get('stage', ''))
        win_team_list.append(minfo.get('win_team_id', np.nan))
        
        cid = minfo.get('comp_id', np.nan)
        cinfo = comp_info.get(cid, {})
        comp_format_list.append(cinfo.get('match_format', ''))
        comp_name_list.append(cinfo.get('comp_name', ''))
        season_list.append(cinfo.get('season', np.nan))
        
    bbb['target'] = targets
    bbb['max_balls'] = max_balls_list
    bbb['is_knockout'] = is_ko_list
    bbb['stage'] = stage_list
    bbb['win_team_id'] = win_team_list
    bbb['match_format'] = comp_format_list
    bbb['comp_name'] = comp_name_list
    bbb['season'] = season_list
    
    # Derived situation variables
    bbb['balls_remaining'] = np.maximum(0, bbb['max_balls'] - bbb['legal_balls_bowled_before'])
    bbb['wickets_in_hand'] = 10 - bbb['cum_wickets_before']
    
    # Runs needed (for innings >= 2)
    bbb['runs_needed'] = np.where(bbb['innings'] >= 2, np.maximum(0, bbb['target'] - bbb['cum_runs_before']), np.nan)
    
    # Required run rate
    bbb['rrr'] = np.where(
        (bbb['innings'] >= 2) & (bbb['balls_remaining'] > 0),
        (bbb['runs_needed'] / bbb['balls_remaining']) * 6.0,
        np.where(bbb['innings'] >= 2, 36.0, np.nan)
    )
    
    # Current run rate
    bbb['crr'] = np.where(
        bbb['legal_balls_bowled_before'] > 0,
        (bbb['cum_runs_before'] / bbb['legal_balls_bowled_before']) * 6.0,
        0.0
    )
    
    # Chase won flag for 2nd innings
    bbb['chase_won'] = np.where(
        bbb['innings'] == 2,
        (bbb['batting_team_id'] == bbb['win_team_id']).astype(int),
        np.nan
    )
    
    print("Saving processed ball-by-ball parquet...")
    bbb.to_parquet('processed_bbb.parquet', index=False)
    print("Done. Saved processed_bbb.parquet. Total rows:", len(bbb))

if __name__ == '__main__':
    build_dataset()
