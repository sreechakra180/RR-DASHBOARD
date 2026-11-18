import pandas as pd
import json

# Load tables
bat_df = pd.read_csv('top_10_batters_summary.csv')
bowl_df = pd.read_csv('top_10_bowlers_summary.csv')

bat_json = bat_df.to_dict(orient='records')
bowl_json = bowl_df.to_dict(orient='records')

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rajasthan Royals | SupeRR Selector S2 - Pressure Analytics Dashboard</title>
    <!-- Tailwind CSS CDN -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Chart.js CDN -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        body {{
            font-family: 'Inter', sans-serif;
            background-color: #F8F9FA;
            color: #1A1A1A;
        }}
        .rr-pink-bg {{
            background-color: #E6007E;
        }}
        .rr-pink-text {{
            color: #E6007E;
        }}
        .rr-pink-border {{
            border-color: #E6007E;
        }}
        .rr-navy-bg {{
            background-color: #0A192F;
        }}
        .rr-navy-text {{
            color: #0A192F;
        }}
        .tab-active {{
            border-bottom: 3px solid #E6007E;
            color: #E6007E;
            font-weight: 700;
        }}
        .card-shadow {{
            box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.06), 0 2px 6px -1px rgba(0, 0, 0, 0.04);
        }}
    </style>
</head>
<body class="min-h-screen flex flex-col">

    <!-- Top Navigation Header -->
    <header class="bg-white border-b border-gray-200 sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div class="flex justify-between items-center h-16">
                <div class="flex items-center space-x-3">
                    <div class="w-10 h-10 rounded-full rr-pink-bg flex items-center justify-center text-white font-black text-xl shadow-md">
                        RR
                    </div>
                    <div>
                        <div class="flex items-center space-x-2">
                            <span class="font-extrabold text-lg tracking-tight text-gray-900">SupeRR Selector</span>
                            <span class="bg-pink-100 rr-pink-text text-xs px-2.5 py-0.5 rounded-full font-bold uppercase tracking-wider">Season 2</span>
                        </div>
                        <p class="text-xs text-gray-500 font-medium">Clutch Analytics Platform • 2022–2026 Dataset</p>
                    </div>
                </div>
                <div class="flex items-center space-x-3">
                    <span class="inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold bg-gray-100 text-gray-800 border border-gray-200">
                        <span class="w-2 h-2 mr-1.5 rounded-full rr-pink-bg animate-pulse"></span> 200,999 Balls Analysed
                    </span>
                    <a href="METHODOLOGY_REPORT.pdf" download class="hidden sm:inline-flex items-center px-4 py-2 border border-transparent text-xs font-bold rounded-lg shadow-sm text-white rr-pink-bg hover:opacity-95 transition">
                        Download PDF Report
                    </a>
                </div>
            </div>
        </div>
    </header>

    <!-- Subheader / KPI Banner -->
    <section class="bg-white border-b border-gray-100 py-6">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div class="md:flex md:items-center md:justify-between">
                <div>
                    <h1 class="text-2xl sm:text-3xl font-extrabold text-gray-900 tracking-tight">
                        Pressure Performance & Leverage Ratings
                    </h1>
                    <p class="mt-1 text-sm text-gray-600 max-w-3xl">
                        A dynamic Win Probability (AUC 0.896) and Machine-Learned Expected Value model evaluating player elevation when matches hang in the balance.
                    </p>
                </div>
                <div class="mt-4 md:mt-0 flex gap-2">
                    <button onclick="switchTab('batters')" id="btn-tab-batters" class="px-5 py-2.5 rounded-lg text-sm font-bold shadow-sm transition rr-pink-bg text-white">
                        Top 10 Batters
                    </button>
                    <button onclick="switchTab('bowlers')" id="btn-tab-bowlers" class="px-5 py-2.5 rounded-lg text-sm font-bold shadow-sm transition bg-gray-100 text-gray-700 hover:bg-gray-200">
                        Top 10 Bowlers
                    </button>
                    <button onclick="switchTab('methodology')" id="btn-tab-methodology" class="px-5 py-2.5 rounded-lg text-sm font-bold shadow-sm transition bg-gray-100 text-gray-700 hover:bg-gray-200">
                        Methodology
                    </button>
                </div>
            </div>

            <!-- Metric Cards -->
            <div class="grid grid-cols-2 sm:grid-cols-4 gap-4 mt-6">
                <div class="bg-pink-50/50 border border-pink-100 rounded-xl p-4">
                    <span class="text-xs font-semibold uppercase tracking-wider text-gray-500">Pressure Threshold</span>
                    <div class="text-2xl font-extrabold rr-pink-text mt-1">LI ≥ 1.20</div>
                    <span class="text-xs text-gray-500">Top 25.7% high-leverage deliveries</span>
                </div>
                <div class="bg-gray-50 border border-gray-200 rounded-xl p-4">
                    <span class="text-xs font-semibold uppercase tracking-wider text-gray-500">Hazard Surge</span>
                    <div class="text-2xl font-extrabold text-gray-900 mt-1">+47.6%</div>
                    <span class="text-xs text-gray-500">Wicket rate jumps: 4.82% → 7.12%</span>
                </div>
                <div class="bg-gray-50 border border-gray-200 rounded-xl p-4">
                    <span class="text-xs font-semibold uppercase tracking-wider text-gray-500">Batter #1 Clutch Peak</span>
                    <div class="text-2xl font-extrabold text-gray-900 mt-1">76.00 Avg</div>
                    <span class="text-xs text-gray-500">Shashank Singh (+38.3 ΔAvg, 184.6 SR)</span>
                </div>
                <div class="bg-gray-50 border border-gray-200 rounded-xl p-4">
                    <span class="text-xs font-semibold uppercase tracking-wider text-gray-500">Bowler #1 Death Defense</span>
                    <div class="text-2xl font-extrabold text-gray-900 mt-1">7.06 Econ</div>
                    <span class="text-xs text-gray-500">Jasprit Bumrah (+2.12 Runs Saved/Over)</span>
                </div>
            </div>
        </div>
    </section>

    <!-- Main Content Container -->
    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex-1 w-full">

        <!-- TAB 1: BATTERS -->
        <div id="tab-batters" class="space-y-8">
            <!-- Chart Section -->
            <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div class="lg:col-span-2 bg-white p-6 rounded-2xl border border-gray-200 card-shadow">
                    <div class="flex justify-between items-center mb-4">
                        <div>
                            <h3 class="text-lg font-bold text-gray-900">Strike Rate Elevation Under Pressure (vs Usual Baseline)</h3>
                            <p class="text-xs text-gray-500">Pink: Pressure Strike Rate | Gray: Baseline Career Strike Rate</p>
                        </div>
                        <span class="text-xs font-bold rr-pink-text bg-pink-50 px-2.5 py-1 rounded-md">Top 10 Batters</span>
                    </div>
                    <div class="h-64">
                        <canvas id="battersChart"></canvas>
                    </div>
                </div>

                <div class="bg-white p-6 rounded-2xl border border-gray-200 card-shadow flex flex-col justify-between">
                    <div>
                        <div class="flex items-center space-x-2">
                            <span class="w-3 h-3 rounded-full rr-pink-bg"></span>
                            <h3 class="text-lg font-bold text-gray-900">Key Finding: Shashank Singh</h3>
                        </div>
                        <p class="mt-3 text-sm text-gray-600 leading-relaxed">
                            While many power hitters suffer a decline in wicket-preservation when forced to accelerate, <b>Shashank Singh</b> achieves the single most extreme clutch surge in the dataset:
                        </p>
                        <div class="mt-4 space-y-3">
                            <div class="flex justify-between items-center text-sm border-b border-gray-100 pb-2">
                                <span class="text-gray-500">Pressure Average:</span>
                                <span class="font-extrabold text-gray-900 text-base">76.00 <span class="text-xs font-semibold text-emerald-600">(+38.3 vs Base)</span></span>
                            </div>
                            <div class="flex justify-between items-center text-sm border-b border-gray-100 pb-2">
                                <span class="text-gray-500">Pressure Strike Rate:</span>
                                <span class="font-extrabold text-gray-900 text-base">184.62 <span class="text-xs font-semibold text-emerald-600">(+23.0 vs Base)</span></span>
                            </div>
                            <div class="flex justify-between items-center text-sm">
                                <span class="text-gray-500">Dot Ball Percentage:</span>
                                <span class="font-extrabold text-gray-900 text-base">25.10% <span class="text-xs font-semibold text-gray-500">(Elite rotation)</span></span>
                            </div>
                        </div>
                    </div>
                    <div class="mt-4 pt-3 border-t border-gray-100 text-xs text-gray-500 italic">
                        Calculated across 247 high-pressure deliveries in 100% IPL games.
                    </div>
                </div>
            </div>

            <!-- Table Section -->
            <div class="bg-white rounded-2xl border border-gray-200 card-shadow overflow-hidden">
                <div class="p-5 border-b border-gray-100 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                    <div>
                        <h3 class="text-lg font-bold text-gray-900">Official Top 10 Batters Ranking</h3>
                        <p class="text-xs text-gray-500">Ordered by Composite Pressure Rating (Minimum 175 legal pressure balls)</p>
                    </div>
                    <div class="relative w-full sm:w-64">
                        <input type="text" id="batterSearch" placeholder="Search batter name..." class="w-full text-xs px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-pink-500">
                    </div>
                </div>
                <div class="overflow-x-auto">
                    <table class="w-full text-left text-xs" id="battersTable">
                        <thead class="bg-gray-50 text-gray-600 font-bold uppercase tracking-wider border-b border-gray-200">
                            <tr>
                                <th class="py-3.5 px-4 text-center">Rank</th>
                                <th class="py-3.5 px-4">Player</th>
                                <th class="py-3.5 px-3 text-center">ID</th>
                                <th class="py-3.5 px-3 text-right">P. Balls</th>
                                <th class="py-3.5 px-3 text-right">P. Runs</th>
                                <th class="py-3.5 px-3 text-right">Pressure SR</th>
                                <th class="py-3.5 px-3 text-right">Baseline SR</th>
                                <th class="py-3.5 px-3 text-right text-pink-600">Δ SR</th>
                                <th class="py-3.5 px-3 text-right">Pressure Avg</th>
                                <th class="py-3.5 px-3 text-right">RAE / 100</th>
                                <th class="py-3.5 px-3 text-right">Dot %</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-gray-100" id="battersBody">
                            <!-- Populated via JS -->
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- TAB 2: BOWLERS -->
        <div id="tab-bowlers" class="hidden space-y-8">
            <!-- Chart Section -->
            <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div class="lg:col-span-2 bg-white p-6 rounded-2xl border border-gray-200 card-shadow">
                    <div class="flex justify-between items-center mb-4">
                        <div>
                            <h3 class="text-lg font-bold text-gray-900">Runs Saved Per Over Above Phase Expectation (RSPO)</h3>
                            <p class="text-xs text-gray-500">Higher value = more runs prevented per over relative to average bowler</p>
                        </div>
                        <span class="text-xs font-bold rr-pink-text bg-pink-50 px-2.5 py-1 rounded-md">Top 10 Bowlers</span>
                    </div>
                    <div class="h-64">
                        <canvas id="bowlersChart"></canvas>
                    </div>
                </div>

                <div class="bg-white p-6 rounded-2xl border border-gray-200 card-shadow flex flex-col justify-between">
                    <div>
                        <div class="flex items-center space-x-2">
                            <span class="w-3 h-3 rounded-full rr-pink-bg"></span>
                            <h3 class="text-lg font-bold text-gray-900">Key Finding: Jasprit Bumrah</h3>
                        </div>
                        <p class="mt-3 text-sm text-gray-600 leading-relaxed">
                            In death overs (overs 17–20) where baseline T20 economy exceeds 10.5 rpo, <b>Jasprit Bumrah</b> operates in an analytical league of his own:
                        </p>
                        <div class="mt-4 space-y-3">
                            <div class="flex justify-between items-center text-sm border-b border-gray-100 pb-2">
                                <span class="text-gray-500">Pressure Economy:</span>
                                <span class="font-extrabold text-gray-900 text-base">7.06 <span class="text-xs font-semibold text-emerald-600">(-3.5 vs phase exp)</span></span>
                            </div>
                            <div class="flex justify-between items-center text-sm border-b border-gray-100 pb-2">
                                <span class="text-gray-500">Runs Saved / Over (RSPO):</span>
                                <span class="font-extrabold rr-pink-text text-base">+2.12 <span class="text-xs font-semibold text-gray-500">(#1 in world)</span></span>
                            </div>
                            <div class="flex justify-between items-center text-sm">
                                <span class="text-gray-500">Boundary Concession %:</span>
                                <span class="font-extrabold text-gray-900 text-base">12.81% <span class="text-xs font-semibold text-gray-500">(Lockdown)</span></span>
                            </div>
                        </div>
                    </div>
                    <div class="mt-4 pt-3 border-t border-gray-100 text-xs text-gray-500 italic">
                        Calculated across 555 legal pressure deliveries with 44 wickets.
                    </div>
                </div>
            </div>

            <!-- Table Section -->
            <div class="bg-white rounded-2xl border border-gray-200 card-shadow overflow-hidden">
                <div class="p-5 border-b border-gray-100 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                    <div>
                        <h3 class="text-lg font-bold text-gray-900">Official Top 10 Bowlers Ranking</h3>
                        <p class="text-xs text-gray-500">Ordered by Composite Pressure Containment & Wicket Taking Rating</p>
                    </div>
                    <div class="relative w-full sm:w-64">
                        <input type="text" id="bowlerSearch" placeholder="Search bowler name..." class="w-full text-xs px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-pink-500">
                    </div>
                </div>
                <div class="overflow-x-auto">
                    <table class="w-full text-left text-xs" id="bowlersTable">
                        <thead class="bg-gray-50 text-gray-600 font-bold uppercase tracking-wider border-b border-gray-200">
                            <tr>
                                <th class="py-3.5 px-4 text-center">Rank</th>
                                <th class="py-3.5 px-4">Player</th>
                                <th class="py-3.5 px-3 text-center">ID</th>
                                <th class="py-3.5 px-3 text-right">P. Balls</th>
                                <th class="py-3.5 px-3 text-right">Wickets</th>
                                <th class="py-3.5 px-3 text-right">Pressure Econ</th>
                                <th class="py-3.5 px-3 text-right">Base Econ</th>
                                <th class="py-3.5 px-3 text-right text-pink-600">Δ Econ</th>
                                <th class="py-3.5 px-3 text-right">Pressure SR</th>
                                <th class="py-3.5 px-3 text-right">RSPO</th>
                                <th class="py-3.5 px-3 text-right">Dot %</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-gray-100" id="bowlersBody">
                            <!-- Populated via JS -->
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- TAB 3: METHODOLOGY -->
        <div id="tab-methodology" class="hidden bg-white p-8 rounded-2xl border border-gray-200 card-shadow space-y-6">
            <div class="border-b border-gray-100 pb-4">
                <span class="text-xs font-bold rr-pink-text uppercase tracking-widest">Mathematical Architecture</span>
                <h2 class="text-2xl font-extrabold text-gray-900 mt-1">Rajasthan Royals Pressure Rating Framework (RR-PRF)</h2>
                <p class="text-sm text-gray-500 mt-1">Full documentation of the data pipeline, formulas, and baseline calibrations.</p>
            </div>

            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div class="p-5 bg-gray-50 rounded-xl border border-gray-200 space-y-3">
                    <h4 class="font-bold text-gray-900 flex items-center">
                        <span class="w-2 h-2 rounded-full rr-pink-bg mr-2"></span> 1. In-Game Leverage Index (LI)
                    </h4>
                    <p class="text-xs text-gray-600 leading-relaxed">
                        Calibrated on 95,234 chase deliveries via Logistic Win Probability (AUC: 0.896):
                    </p>
                    <div class="bg-white p-3 rounded border border-gray-200 font-mono text-xs text-gray-800">
                        LI = 4 · WP · (1 - WP) · √[120 / (B_rem + 6)]
                    </div>
                    <p class="text-xs text-gray-500">
                        Triggered when LI ≥ 1.20, late crunch (B_rem ≤ 30 with RRR 7.5–24.0), or chase collapse (W ≥ 3 in powerplay).
                    </p>
                </div>

                <div class="p-5 bg-gray-50 rounded-xl border border-gray-200 space-y-3">
                    <h4 class="font-bold text-gray-900 flex items-center">
                        <span class="w-2 h-2 rounded-full rr-pink-bg mr-2"></span> 2. Context Expected Values (RAE & RSPO)
                    </h4>
                    <p class="text-xs text-gray-600 leading-relaxed">
                        Gradient-boosted machine learning predicting expected runs E[R] and wickets E[W] per delivery:
                    </p>
                    <div class="bg-white p-3 rounded border border-gray-200 font-mono text-xs text-gray-800">
                        RAE_100 = [Σ(Runs - E[Runs]) / Balls] · 100<br/>
                        RSPO = [Σ(E[Runs] - Conceded) / Legal Balls] · 6
                    </div>
                    <p class="text-xs text-gray-500">
                        Measures performance relative to what an average T20 player would produce in identical conditions.
                    </p>
                </div>
            </div>

            <div class="p-5 bg-pink-50/50 rounded-xl border border-pink-100">
                <h4 class="font-bold text-gray-900 mb-2">Fairness & Normalisation Controls</h4>
                <ul class="text-xs text-gray-600 space-y-2 list-disc list-inside">
                    <li><b>Eligibility Filter:</b> Minimum 175 legal pressure deliveries to eliminate small-sample noise.</li>
                    <li><b>Empirical Bayes Shrinkage:</b> Regressing noisy rates toward population priors (K = 40 balls).</li>
                    <li><b>Tier Weights:</b> IPL matches weighted at 1.00; Full-Member Elite T20Is at 0.95; Associate matches at 0.85.</li>
                </ul>
            </div>
        </div>

    </main>

    <!-- Footer -->
    <footer class="bg-white border-t border-gray-200 py-6 mt-12">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center text-xs text-gray-500">
            <p>Developed for <b>Rajasthan Royals SupeRR Selector Season 2</b> • Round 1 Technical Submission</p>
            <p class="mt-1 text-gray-400">All rankings derived strictly from historical data across 867 matches (2022–2026).</p>
        </div>
    </footer>

    <!-- JavaScript Data Binding & Charts -->
    <script>
        const battersData = {json.dumps(bat_json)};
        const bowlersData = {json.dumps(bowl_json)};

        function renderBattersTable(data) {{
            const tbody = document.getElementById('battersBody');
            tbody.innerHTML = '';
            data.forEach((p, idx) => {{
                const row = document.createElement('tr');
                row.className = idx % 2 === 0 ? 'bg-white hover:bg-pink-50/30' : 'bg-gray-50/50 hover:bg-pink-50/30';
                row.innerHTML = `
                    <td class="py-3 px-4 text-center font-bold ${{p.Rank === 1 ? 'rr-pink-text' : 'text-gray-700'}}">${{p.Rank}}</td>
                    <td class="py-3 px-4 font-bold text-gray-900">${{p['Player Name']}}</td>
                    <td class="py-3 px-3 text-center text-gray-400 font-mono">${{p['Player ID']}}</td>
                    <td class="py-3 px-3 text-right font-medium">${{p.Balls}}</td>
                    <td class="py-3 px-3 text-right font-medium">${{p.Runs}}</td>
                    <td class="py-3 px-3 text-right font-extrabold text-gray-900">${{p['Pressure SR'].toFixed(1)}}</td>
                    <td class="py-3 px-3 text-right text-gray-500">${{p['Baseline SR'].toFixed(1)}}</td>
                    <td class="py-3 px-3 text-right font-bold text-pink-600">+${{p['Delta SR'].toFixed(1)}}</td>
                    <td class="py-3 px-3 text-right font-extrabold text-gray-900">${{p['Pressure Avg'].toFixed(1)}}</td>
                    <td class="py-3 px-3 text-right font-bold ${{p['RAE / 100'] > 0 ? 'text-emerald-600' : 'text-gray-700'}}">+${{p['RAE / 100'].toFixed(1)}}</td>
                    <td class="py-3 px-3 text-right text-gray-500">${{p['Dot %'].toFixed(1)}}%</td>
                `;
                tbody.appendChild(row);
            }});
        }}

        function renderBowlersTable(data) {{
            const tbody = document.getElementById('bowlersBody');
            tbody.innerHTML = '';
            data.forEach((p, idx) => {{
                const row = document.createElement('tr');
                row.className = idx % 2 === 0 ? 'bg-white hover:bg-pink-50/30' : 'bg-gray-50/50 hover:bg-pink-50/30';
                row.innerHTML = `
                    <td class="py-3 px-4 text-center font-bold ${{p.Rank === 1 ? 'rr-pink-text' : 'text-gray-700'}}">${{p.Rank}}</td>
                    <td class="py-3 px-4 font-bold text-gray-900">${{p['Player Name']}}</td>
                    <td class="py-3 px-3 text-center text-gray-400 font-mono">${{p['Player ID']}}</td>
                    <td class="py-3 px-3 text-right font-medium">${{p['Legal Balls']}}</td>
                    <td class="py-3 px-3 text-right font-extrabold text-gray-900">${{p.Wickets}}</td>
                    <td class="py-3 px-3 text-right font-extrabold text-gray-900">${{p['Pressure Econ'].toFixed(2)}}</td>
                    <td class="py-3 px-3 text-right text-gray-500">${{p['Baseline Econ'].toFixed(2)}}</td>
                    <td class="py-3 px-3 text-right font-bold ${{p['Delta Econ'] <= 0 ? 'text-emerald-600' : 'text-pink-600'}}">${{p['Delta Econ'] > 0 ? '+' : ''}}${{p['Delta Econ'].toFixed(2)}}</td>
                    <td class="py-3 px-3 text-right font-medium">${{p['Pressure SR'].toFixed(1)}}</td>
                    <td class="py-3 px-3 text-right font-bold text-emerald-600">+${{p.RSPO.toFixed(2)}}</td>
                    <td class="py-3 px-3 text-right text-gray-500">${{p['Dot %'].toFixed(1)}}%</td>
                `;
                tbody.appendChild(row);
            }});
        }}

        // Tab Switching
        function switchTab(tab) {{
            const tabs = ['batters', 'bowlers', 'methodology'];
            tabs.forEach(t => {{
                document.getElementById('tab-' + t).classList.add('hidden');
                document.getElementById('btn-tab-' + t).className = 'px-5 py-2.5 rounded-lg text-sm font-bold shadow-sm transition bg-gray-100 text-gray-700 hover:bg-gray-200';
            }});
            document.getElementById('tab-' + tab).classList.remove('hidden');
            document.getElementById('btn-tab-' + tab).className = 'px-5 py-2.5 rounded-lg text-sm font-bold shadow-sm transition rr-pink-bg text-white';
        }}

        // Live Search
        document.getElementById('batterSearch').addEventListener('input', function(e) {{
            const query = e.target.value.toLowerCase();
            const filtered = battersData.filter(b => b['Player Name'].toLowerCase().includes(query));
            renderBattersTable(filtered);
        }});

        document.getElementById('bowlerSearch').addEventListener('input', function(e) {{
            const query = e.target.value.toLowerCase();
            const filtered = bowlersData.filter(b => b['Player Name'].toLowerCase().includes(query));
            renderBowlersTable(filtered);
        }});

        // Charts Initialization
        window.addEventListener('DOMContentLoaded', () => {{
            renderBattersTable(battersData);
            renderBowlersTable(bowlersData);

            // Batters Chart
            const ctxB = document.getElementById('battersChart').getContext('2d');
            new Chart(ctxB, {{
                type: 'bar',
                data: {{
                    labels: battersData.map(b => b['Player Name'].split(' ').slice(-1)[0]),
                    datasets: [
                        {{
                            label: 'Pressure SR',
                            data: battersData.map(b => b['Pressure SR']),
                            backgroundColor: '#E6007E',
                            borderRadius: 4
                        }},
                        {{
                            label: 'Baseline SR',
                            data: battersData.map(b => b['Baseline SR']),
                            backgroundColor: '#D1D5DB',
                            borderRadius: 4
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ position: 'top', labels: {{ font: {{ size: 10, weight: 'bold' }} }} }}
                    }},
                    scales: {{
                        y: {{ min: 120, max: 220, ticks: {{ font: {{ size: 10 }} }} }},
                        x: {{ ticks: {{ font: {{ size: 10, weight: 'bold' }} }} }}
                    }}
                }}
            }});

            // Bowlers Chart
            const ctxBo = document.getElementById('bowlersChart').getContext('2d');
            new Chart(ctxBo, {{
                type: 'bar',
                data: {{
                    labels: bowlersData.map(b => b['Player Name'].split(' ').slice(-1)[0]),
                    datasets: [
                        {{
                            label: 'Runs Saved / Over (RSPO)',
                            data: bowlersData.map(b => b.RSPO),
                            backgroundColor: '#E6007E',
                            borderRadius: 4
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ display: false }}
                    }},
                    scales: {{
                        y: {{ min: 0, max: 2.5, ticks: {{ font: {{ size: 10 }} }} }},
                        x: {{ ticks: {{ font: {{ size: 10, weight: 'bold' }} }} }}
                    }}
                }}
            }});
        }});
    </script>
</body>
</html>
"""

with open('dashboard.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

with open('index.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("Generated dashboard.html and index.html successfully!")
