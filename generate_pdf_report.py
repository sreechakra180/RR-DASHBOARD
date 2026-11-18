import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether

def create_pdf_report():
    pdf_filename = "METHODOLOGY_REPORT.pdf"
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=letter,
        leftMargin=0.55 * inch,
        rightMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Monochrome Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=19,
        textColor=colors.black,
        alignment=1, # Center
        spaceAfter=4
    )
    
    super_title = ParagraphStyle(
        'SuperTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor('#555555'),
        alignment=1,
        spaceAfter=2
    )
    
    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#444444'),
        alignment=1,
        spaceAfter=10
    )
    
    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11.5,
        leading=15,
        textColor=colors.black,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'H2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor('#222222'),
        spaceBefore=7,
        spaceAfter=3,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor('#111111'),
        spaceAfter=4
    )
    
    bullet_style = ParagraphStyle(
        'Bullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#111111'),
        leftIndent=12,
        spaceAfter=3
    )
    
    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        alignment=1 # Center
    )
    
    table_cell_left = ParagraphStyle(
        'TableCellLeft',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        alignment=0 # Left
    )
    
    table_hdr_style = ParagraphStyle(
        'TableHdr',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white,
        alignment=1
    )

    story = []
    
    # Header Banner
    story.append(Paragraph("RAJASTHAN ROYALS | SUPERR SELECTOR SEASON 2", super_title))
    story.append(Paragraph("QUANTIFYING CRICKET LEVERAGE: IDENTIFYING THE MOST RELIABLE PLAYERS UNDER PRESSURE", title_style))
    story.append(Paragraph("Round 1 Technical Methodology Report & Official Submission Roster | Dataset: IPL & T20I (2022–2026)", meta_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.black, spaceAfter=8, spaceBefore=0))
    
    # 1. Executive Summary
    story.append(Paragraph("1. EXECUTIVE SUMMARY & PROBLEM CONTEXT", h1_style))
    story.append(Paragraph(
        "T20 matches are decided at high-leverage inflexion points. While aggregate career metrics (such as overall strike rate or bowling average) capture a baseline level of ability across relaxed match states, they often mask severe degradation under intense pressure. The core mandate of this challenge is to construct an objective, data-driven framework that isolates high-pressure situations from ordinary ones, benchmarks each player against their usual performance level, and ranks the Top 10 Batters and Top 10 Bowlers who deliver most reliably when it matters most.",
        body_style
    ))
    story.append(Paragraph(
        "Using ball-by-ball records from 867 competitive T20 matches (200,999 deliveries across IPL and T20Is from 2022 to 2026), we developed the <b>Rajasthan Royals Pressure Rating Framework (RR-PRF)</b>. This methodology integrates: (1) A dynamic Win Probability (WP) and Leverage Index (LI) model; (2) Situational collapse and death-phase indicators; (3) Gradient-boosted Expected Value baselines (Expected Runs and Wickets); (4) Context-adjusted metrics measuring absolute execution and clutch elevation; and (5) Empirical Bayes shrinkage and competition tier normalization.",
        body_style
    ))
    
    # 2. Pressure Definition
    story.append(Paragraph("2. PRESSURE DEFINITION: MATHEMATICAL CRITERIA & VALIDATION", h1_style))
    story.append(Paragraph("<b>2.1 Second Innings (The Chase): Win Probability & Leverage Index (LI)</b>", h2_style))
    story.append(Paragraph(
        "In the second innings, chase pressure is governed by required run rate relative to remaining resources. We trained a Logistic Win Probability model on all chase deliveries using: runs needed (R_req), balls remaining (B_rem), wickets in hand (W_hand), required run rate (RRR), and target. The model achieved an outstanding AUC of 0.8956 (Log Loss: 0.403).",
        body_style
    ))
    story.append(Paragraph(
        "Leverage Index (LI) is formulated as: <b>LI = 4 · WP · (1 - WP) · √[120 / (B_rem + 6)]</b>. The term 4·WP·(1-WP) captures contest uncertainty (peaking at 1.0 when WP=0.5 and dropping to 0 when the game is decided), while √[120 / (B_rem + 6)] scales the impact per ball as remaining balls decay. A delivery is flagged as High Pressure if: (a) LI ≥ 1.2; OR (b) Late-game crunch: B_rem ≤ 30 with 7.5 ≤ RRR ≤ 24.0 and W_hand ≥ 2; OR (c) Top-order crisis: losing ≥ 3 wickets in powerplay or ≥ 4 wickets before over 10.",
        body_style
    ))
    
    story.append(Paragraph("<b>2.2 First Innings & Super Overs</b>", h2_style))
    story.append(Paragraph(
        "In the first innings, pressure stems from phase demands and innings fragility: (a) <i>Collapse Recovery</i>: losing ≥ 3 wickets in powerplay (overs 1–6) or ≥ 4 wickets before over 10; (b) <i>Death Overs (Overs 17–20)</i>: where expected scoring variance and boundary leverage are highest; (c) <i>Knockout Stakes Multiplier</i>: overs 15–20 in playoff fixtures (Finals, Qualifiers, Eliminators); (d) <i>Super Overs (Innings 3 & 4)</i>: 100% categorized as Extreme High Pressure.",
        body_style
    ))
    
    story.append(Paragraph("<b>2.3 Empirical Validation: Distinguishing Pressure from Ordinary Match States</b>", h2_style))
    story.append(Paragraph(
        "The framework isolates exactly <b>51,730 deliveries (25.74% of the dataset)</b> as High Pressure. Empirical evidence validates the separation: (1) <i>Dismissal Rate Spike</i>: Wicket probability surges by +47.6% (from 4.82% in ordinary play to 7.12% under pressure—1 wicket every 14.0 balls); (2) <i>Scoring Escalation</i>: Run rate climbs from 8.34 rpo to 8.91 rpo, reflecting high boundary demand under extreme hazard.",
        body_style
    ))

    # 3. Performance Measurement Framework
    story.append(Paragraph("3. PERFORMANCE MEASUREMENT: CONTEXT-ADJUSTED METRICS & RATING LOGIC", h1_style))
    story.append(Paragraph(
        "Evaluating raw rates in isolation creates severe bias (e.g. death bowlers face structurally higher run rates than middle-overs bowlers). To establish fair baselines, we trained machine-learned Expected Value models (HistGradientBoosting) on situational features (over, innings, wickets in hand, cumulative score, competition quality) to predict expected runs (E[Runs]) and expected wickets (E[Wkts]) on every delivery. We derive:",
        body_style
    ))
    story.append(Paragraph("• <b>Runs Above Expectation per 100 Balls (RAE_100)</b>: Net runs scored by a batter above what an average player would score in the exact same match situations.", bullet_style))
    story.append(Paragraph("• <b>Runs Saved per 6 Balls (RSPO)</b>: Net runs saved per over by a bowler compared to situation expectation.", bullet_style))
    story.append(Paragraph("• <b>Wickets Above Expectation per 24 Balls (WAPO)</b>: Excess wicket-taking potency relative to expectation.", bullet_style))
    story.append(Paragraph("• <b>Clutch Elevation (Delta vs Baseline)</b>: ΔSR = SR_Pressure - SR_Baseline; ΔAvg = Avg_Pressure - Avg_Baseline; ΔEcon = Econ_Pressure - Econ_Baseline; ΔSR_bowl = SR_Pressure - SR_Baseline.", bullet_style))
    story.append(Paragraph("• <b>Fair Comparisons & Shrinkage</b>: Minimum 175 legal pressure balls qualification; Empirical Bayes shrinkage (K=40 balls) regressing small samples towards population priors; Competition tier weights: IPL = 1.00, Full Member T20I = 0.95, Others = 0.85.", bullet_style))

    # 4. Top 10 Batters Table
    story.append(Spacer(1, 4))
    story.append(Paragraph("4. OFFICIAL TOP 10 BATTERS UNDER PRESSURE", h1_style))
    
    bat_df = pd.read_csv('top_10_batters_summary.csv')
    bat_table_data = [[
        Paragraph('<b>Rank</b>', table_hdr_style),
        Paragraph('<b>Player Name</b>', table_hdr_style),
        Paragraph('<b>ID</b>', table_hdr_style),
        Paragraph('<b>P.Balls</b>', table_hdr_style),
        Paragraph('<b>P.Runs</b>', table_hdr_style),
        Paragraph('<b>P.SR</b>', table_hdr_style),
        Paragraph('<b>Base.SR</b>', table_hdr_style),
        Paragraph('<b>Δ SR</b>', table_hdr_style),
        Paragraph('<b>P.Avg</b>', table_hdr_style),
        Paragraph('<b>Δ Avg</b>', table_hdr_style),
        Paragraph('<b>RAE/100</b>', table_hdr_style),
        Paragraph('<b>Dot%</b>', table_hdr_style)
    ]]
    
    for _, r in bat_df.iterrows():
        bat_table_data.append([
            Paragraph(str(r['Rank']), table_cell_style),
            Paragraph(f"<b>{r['Player Name']}</b>", table_cell_left),
            Paragraph(str(r['Player ID']), table_cell_style),
            Paragraph(str(r['Balls']), table_cell_style),
            Paragraph(str(r['Runs']), table_cell_style),
            Paragraph(f"{r['Pressure SR']:.1f}", table_cell_style),
            Paragraph(f"{r['Baseline SR']:.1f}", table_cell_style),
            Paragraph(f"{r['Delta SR']:+.1f}", table_cell_style),
            Paragraph(f"{r['Pressure Avg']:.1f}", table_cell_style),
            Paragraph(f"{r['Delta Avg']:+.1f}", table_cell_style),
            Paragraph(f"{r['RAE / 100']:+.1f}", table_cell_style),
            Paragraph(f"{r['Dot %']:.1f}%", table_cell_style)
        ])
        
    t_bat = Table(bat_table_data, colWidths=[0.35*inch, 1.4*inch, 0.45*inch, 0.45*inch, 0.45*inch, 0.52*inch, 0.52*inch, 0.5*inch, 0.5*inch, 0.5*inch, 0.65*inch, 0.45*inch])
    t_bat.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1A1A1A')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F5F5F5')]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CCCCCC')),
        ('BOX', (0,0), (-1,-1), 1, colors.black)
    ]))
    story.append(t_bat)
    
    story.append(Paragraph("<b>Analytical Highlights for Top Batters:</b>", h2_style))
    story.append(Paragraph("• <b>Rank 1: Shashank Singh (ID: 2756)</b> — Phenomenal pressure elevation in 100% IPL games. Batting average doubles from 37.71 to 76.00 (+38.29 elevation) while striking at 184.62 (+23.01 ΔSR). Dot ball percentage is just 25.10%, demonstrating elite ice-cold nerve in tight finishes.", bullet_style))
    story.append(Paragraph("• <b>Rank 2: Tim David (ID: 6396)</b> — Unrivaled volume and power. 1,254 pressure runs across 645 balls at 194.42 SR (+42.90 RAE/100) and 28.53% boundary rate across 75 matches.", bullet_style))
    story.append(Paragraph("• <b>Rank 3: Naman Dhir (ID: 37052)</b> — Highest scoring rate under pressure: 206.04 SR (+37.62 ΔSR) and +50.53 RAE/100 with a 32.97% boundary rate.", bullet_style))
    story.append(Paragraph("• <b>Rank 4: Shreyas Iyer (ID: 3647)</b> — Master of knockout pressure. Elevated SR by +34.26 (to 187.40) with 699 runs and an elite 24.93% dot rate.", bullet_style))
    story.append(Paragraph("• <b>Ranks 5–10</b>: Donovan Ferreira (Rank 5, SR 183.94, +44.72 RAE); Ishan Kishan (Rank 6, Avg 36.10, SR 173.56, +27.19 ΔSR); Jos Buttler (Rank 7, SR 189.04, +33.65 ΔSR, 23.59% dot rate); Rinku Singh (Rank 8, 1,082 pressure runs, Avg 34.90, SR 172.29); Marcus Stoinis (Rank 9, 725 runs, SR 175.54); Tristan Stubbs (Rank 10, Avg 38.43, SR 164.93).", bullet_style))

    # 5. Top 10 Bowlers Table
    story.append(Spacer(1, 4))
    story.append(Paragraph("5. OFFICIAL TOP 10 BOWLERS UNDER PRESSURE", h1_style))
    
    bowl_df = pd.read_csv('top_10_bowlers_summary.csv')
    bowl_table_data = [[
        Paragraph('<b>Rank</b>', table_hdr_style),
        Paragraph('<b>Player Name</b>', table_hdr_style),
        Paragraph('<b>ID</b>', table_hdr_style),
        Paragraph('<b>Balls</b>', table_hdr_style),
        Paragraph('<b>Wkts</b>', table_hdr_style),
        Paragraph('<b>P.Econ</b>', table_hdr_style),
        Paragraph('<b>Base.Ec</b>', table_hdr_style),
        Paragraph('<b>Δ Econ</b>', table_hdr_style),
        Paragraph('<b>P.SR</b>', table_hdr_style),
        Paragraph('<b>Δ SR</b>', table_hdr_style),
        Paragraph('<b>RSPO</b>', table_hdr_style),
        Paragraph('<b>Dot%</b>', table_hdr_style)
    ]]
    
    for _, r in bowl_df.iterrows():
        bowl_table_data.append([
            Paragraph(str(r['Rank']), table_cell_style),
            Paragraph(f"<b>{r['Player Name']}</b>", table_cell_left),
            Paragraph(str(r['Player ID']), table_cell_style),
            Paragraph(str(r['Legal Balls']), table_cell_style),
            Paragraph(str(r['Wickets']), table_cell_style),
            Paragraph(f"{r['Pressure Econ']:.2f}", table_cell_style),
            Paragraph(f"{r['Baseline Econ']:.2f}", table_cell_style),
            Paragraph(f"{r['Delta Econ']:+.2f}", table_cell_style),
            Paragraph(f"{r['Pressure SR']:.1f}", table_cell_style),
            Paragraph(f"{r['Delta SR']:+.1f}", table_cell_style),
            Paragraph(f"{r['RSPO']:+.2f}", table_cell_style),
            Paragraph(f"{r['Dot %']:.1f}%", table_cell_style)
        ])
        
    t_bowl = Table(bowl_table_data, colWidths=[0.35*inch, 1.4*inch, 0.45*inch, 0.45*inch, 0.45*inch, 0.52*inch, 0.52*inch, 0.5*inch, 0.5*inch, 0.5*inch, 0.65*inch, 0.45*inch])
    t_bowl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1A1A1A')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F5F5F5')]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CCCCCC')),
        ('BOX', (0,0), (-1,-1), 1, colors.black)
    ]))
    story.append(t_bowl)
    
    story.append(Paragraph("<b>Analytical Highlights for Top Bowlers:</b>", h2_style))
    story.append(Paragraph("• <b>Rank 1: Jasprit Bumrah (ID: 2933)</b> — Generational outlier. Bowling overs 17–20 where average economy exceeds 10.5 rpo, Bumrah concedes an absurd 7.06 rpo across 555 pressure balls while taking 44 wickets at 12.61 strike rate (-7.49 ΔSR). Saves +2.12 runs per over above expectation (No. 1 in world cricket) with 12.81% boundary rate.", bullet_style))
    story.append(Paragraph("• <b>Rank 2: Sunil Narine (ID: 781)</b> — 100% IPL competition against elite lineups. Economy 6.43 under pressure (-0.33 ΔEcon), taking 23 wickets at 14.48 strike rate (-8.75 ΔSR) with an extraordinary 9.73% boundary concession rate and +1.80 RSPO.", bullet_style))
    story.append(Paragraph("• <b>Rank 3: Krunal Pandya (ID: 4219)</b> — Choking middle overs and collapses. Economy 7.03 (-0.60 ΔEcon), strike rate 14.00 (-8.45 ΔSR), +1.36 RSPO.", bullet_style))
    story.append(Paragraph("• <b>Rank 4: Akeal Hosein (ID: 2417)</b> — Economy 6.77 (-0.75 ΔEcon), strike rate 11.41 (-10.02 ΔSR elevation!), +1.29 RSPO.", bullet_style))
    story.append(Paragraph("• <b>Rank 5: Matheesha Pathirana (ID: 8661)</b> — Death overs strike weapon. 46 wickets in 469 balls at a lethal 10.20 strike rate (a wicket every 1.7 overs under pressure) and +1.31 RSPO.", bullet_style))
    story.append(Paragraph("• <b>Ranks 6–10</b>: Kuldeep Yadav (Rank 6, 38 wkts, Econ 7.77, SR 12.05); Varun Chakravarthy (Rank 7, Econ 7.31, 41.84% dot rate); Rashid Khan (Rank 8, 54 pressure wickets, Econ 7.41, SR 13.13); Tim Southee (Rank 9, SR 10.48, RSPO +1.10); Arshdeep Singh (Rank 10, 82 pressure wickets across 1,035 balls, highest workload).", bullet_style))

    # 6. Reliability & Dropdown Reference
    story.append(Spacer(1, 4))
    story.append(Paragraph("6. RELIABILITY, LIMITATIONS & OFFICIAL UNSTOP SELECTION SUMMARY", h1_style))
    story.append(Paragraph(
        "<b>Consistency Checks:</b> Player ratings exhibited stable performance across multiple seasons (2022 to 2026) and playoff games. <b>Limitations:</b> Ball-by-ball records do not track dropped catches or field placements, and dew factor during day-night matches was modeled via second innings baseline adjustments rather than direct humidity sensors.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Official Unstop Dropdown Selection Roster:</b><br/>"
        "• <b>Batters (1 to 10):</b> 1. Shashank Singh | 2. Tim David | 3. Naman Dhir | 4. Shreyas Iyer | 5. Donovan Ferreira | 6. Ishan Kishan | 7. Jos Buttler | 8. Rinku Singh | 9. Marcus Stoinis | 10. Tristan Stubbs.<br/>"
        "• <b>Bowlers (1 to 10):</b> 1. Jasprit Bumrah | 2. Sunil Narine | 3. Krunal Pandya | 4. Akeal Hosein | 5. Matheesha Pathirana | 6. Kuldeep Yadav | 7. Varun Chakravarthy | 8. Rashid Khan | 9. Tim Southee | 10. Arshdeep Singh.",
        body_style
    ))

    doc.build(story)
    print("Saved METHODOLOGY_REPORT.pdf successfully!")

create_pdf_report()
