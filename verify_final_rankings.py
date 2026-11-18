import pandas as pd
import numpy as np

# Verify player names and IDs
ctpm = pd.read_csv('Data Pack/competition_team_player_mapping.csv')
p11 = pd.read_csv('Data Pack/playing_eleven.csv')

top_batters_cand = [
    (2756, "Shashank Singh"),
    (37052, "Naman Dhir"),
    (6396, "Tim David"),
    (3647, "Shreyas Iyer"),
    (8182, "Donovan Ferreira"),
    (4117, "Ishan Kishan"),
    (813, "Jos Buttler"),
    (4708, "Rinku Singh"),
    (2639, "Marcus Stoinis"),
    (20702, "Tristan Stubbs"),
    (1699, "Suryakumar Yadav"),
    (4260, "Heinrich Klaasen"),
    (6288, "Shubman Gill"),
    (4812, "Harry Tector")
]

top_bowlers_cand = [
    (2933, "Jasprit Bumrah"),
    (781, "Sunil Narine"),
    (4219, "Krunal Pandya"),
    (2417, "Akeal Hosein"),
    (8661, "Matheesha Pathirana"),
    (3644, "Kuldeep Yadav"),
    (6256, "Varun Chakravarthy"),
    (4722, "Rashid Khan"),
    (278, "Tim Southee"),
    (6464, "Arshdeep Singh"),
    (7364, "Haris Rauf"),
    (2984, "Sikandar Raza")
]

print("Checking player name mapping:")
for pid, name in top_batters_cand:
    m = ctpm[ctpm['player_id'] == pid]
    pname_in_ctpm = m['player_name'].iloc[0] if len(m) > 0 else 'NOT FOUND'
    print(f"Batter ID {pid}: '{name}' -> CTPM Name: '{pname_in_ctpm}' (Matches: {name == pname_in_ctpm})")

for pid, name in top_bowlers_cand:
    m = ctpm[ctpm['player_id'] == pid]
    pname_in_ctpm = m['player_name'].iloc[0] if len(m) > 0 else 'NOT FOUND'
    print(f"Bowler ID {pid}: '{name}' -> CTPM Name: '{pname_in_ctpm}' (Matches: {name == pname_in_ctpm})")
