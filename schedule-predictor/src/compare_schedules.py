"""
Generates a fairness comparison between the predicted 2026 NFL schedule
and the official released schedule.

Outputs charts to data/charts/comparison/ and an HTML page to
pages/comparison.html (served at /pages/comparison.html).
"""

import os
import re
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────

_ROOT   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATA   = os.path.join(_ROOT, "schedule-predictor", "data")
_SRC    = os.path.join(_ROOT, "schedule-predictor", "src")
CHART_DIR = os.path.join(_DATA, "charts")
os.makedirs(CHART_DIR, exist_ok=True)

PRED_FILE = os.path.join(_DATA, "predicted_schedule.txt")
REAL_FILE = os.path.join(_DATA, "real_schedule.json")
HTML_OUT  = os.path.join(_ROOT, "pages", "comparison.html")

# ── Team / division config (same as schedule_analysis.py) ─────────────────────

DIVISIONS = {
    "AFC East":  ["Bills", "Dolphins", "Patriots", "Jets"],
    "AFC North": ["Ravens", "Steelers", "Browns", "Bengals"],
    "AFC South": ["Texans", "Colts", "Jaguars", "Titans"],
    "AFC West":  ["Chiefs", "Raiders", "Chargers", "Broncos"],
    "NFC East":  ["Eagles", "Cowboys", "Commanders", "Giants"],
    "NFC North": ["Lions", "Packers", "Vikings", "Bears"],
    "NFC South": ["Buccaneers", "Falcons", "Saints", "Panthers"],
    "NFC West":  ["49ers", "Seahawks", "Rams", "Cardinals"],
}

TEAM_ORDER = []
for div in DIVISIONS.values():
    TEAM_ORDER.extend(div)

ALL_TEAMS = sorted(TEAM_ORDER)

PRIMETIME_CATS = ("Christmas", "Thanksgiving", "MNF", "SNF", "TNF")

COLORS = {
    "pred": "#2a7abf",
    "real": "#e05c00",
}

PT_COLORS = {
    "TNF": "#e05c00", "SNF": "#1a6bbf", "MNF": "#2ca02c",
    "Thanksgiving": "#9b59b6", "Christmas": "#c0392b",
}

# ── Parse predicted schedule ────────────────────────────────────────────────────

def _et_hour(time_str: str):
    """Return the ET kickoff hour (int) from a time string like '8:20 PM ET', or -1."""
    m = re.search(r'(\d+):\d+\s*(AM|PM)\s*ET', time_str, re.IGNORECASE)
    if not m:
        return -1
    h = int(m.group(1))
    ampm = m.group(2).upper()
    if ampm == 'PM' and h != 12:
        h += 12
    if ampm == 'AM' and h == 12:
        h = 0
    return h

def slot_category(time_str: str) -> str:
    t = time_str
    et = _et_hour(t)
    is_night = et >= 19

    if "Christmas" in t:
        return "Christmas" if is_night else ("1PM" if et < 16 else "4PM")
    if "Thanksgiving" in t:
        return "Thanksgiving" if is_night else ("1PM" if et < 16 else "4PM")
    if "Thursday Night" in t or "Friday Night" in t:
        return "TNF"
    if "Sunday Night" in t or "Wednesday Kickoff" in t or "Wednesday -" in t:
        return "SNF"
    if "Monday Night" in t:
        return "MNF"
    # Any other game at 7 PM ET or later is nighttime = primetime
    if is_night:
        return "SNF"
    if "1:00 PM" in t or "12:30 PM" in t or "9:30 AM" in t:
        return "1PM"
    if "4:05 PM" in t or "4:25 PM" in t or "4:30 PM" in t or "3:00 PM" in t:
        return "4PM"
    return "Other"

def is_primetime(cat: str) -> bool:
    return cat in PRIMETIME_CATS

def parse_predicted():
    games, byes = [], {}
    current_week = None
    with open(PRED_FILE) as f:
        for line in f:
            line = line.rstrip()
            if re.match(r"^Week \d+:", line):
                current_week = int(re.search(r"\d+", line).group())
            elif line.strip().startswith("Bye:"):
                teams = [t.strip() for t in line.split("Bye:")[1].split(",")]
                byes[current_week] = teams
            elif " @ " in line and current_week is not None:
                parts = line.strip().split("  ")
                matchup = parts[0].strip()
                away, home = matchup.split(" @ ")
                time_str = "  ".join(p for p in parts[1:] if p.strip()).strip()
                cat = slot_category(time_str)
                games.append({"week": current_week, "away": away, "home": home, "cat": cat})
    return games, byes

def parse_real():
    with open(REAL_FILE) as f:
        data = json.load(f)
    # strip internal debug fields
    return [{"week": g["week"], "away": g["away"], "home": g["home"], "cat": g["cat"]}
            for g in data]

# ── Compute per-team primetime counts ──────────────────────────────────────────

def pt_counts(games):
    counts = {cat: defaultdict(int) for cat in PRIMETIME_CATS}
    for g in games:
        if is_primetime(g["cat"]):
            counts[g["cat"]][g["away"]] += 1
            counts[g["cat"]][g["home"]] += 1
    totals = {t: sum(counts[cat][t] for cat in PRIMETIME_CATS) for t in ALL_TEAMS}
    return counts, totals

# ── Gini coefficient (fairness measure) ───────────────────────────────────────

def gini(values):
    arr = np.array(sorted(values), dtype=float)
    n = len(arr)
    if arr.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return (2 * (index * arr).sum() / (n * arr.sum())) - (n + 1) / n

# ── Chart helpers ──────────────────────────────────────────────────────────────

def save(path):
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return os.path.relpath(path, _ROOT)

# ── Chart 1: Primetime appearances side-by-side ────────────────────────────────

def chart_primetime_side_by_side(pred_games, real_games):
    _, pred_totals = pt_counts(pred_games)
    _, real_totals = pt_counts(real_games)

    teams_sorted = sorted(ALL_TEAMS, key=lambda t: pred_totals[t], reverse=True)
    x = np.arange(len(teams_sorted))
    w = 0.38

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(x - w/2, [pred_totals[t] for t in teams_sorted], width=w,
           color=COLORS["pred"], label="Predicted", alpha=0.9)
    ax.bar(x + w/2, [real_totals[t] for t in teams_sorted], width=w,
           color=COLORS["real"], label="Official", alpha=0.9)

    pred_gini = gini(list(pred_totals.values()))
    real_gini = gini(list(real_totals.values()))
    ax.set_title(
        f"Primetime Appearances per Team — Predicted vs. Official\n"
        f"Gini: Predicted {pred_gini:.3f}  |  Official {real_gini:.3f}  "
        f"(lower = more equal)",
        fontsize=12, fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(teams_sorted, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Primetime Games")
    ax.legend(fontsize=10)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    return save(os.path.join(CHART_DIR, "1_primetime_side_by_side.png"))

# ── Chart 2: Stacked breakdown (predicted vs official) ────────────────────────

def chart_stacked_breakdown(pred_games, real_games, label, outname):
    counts, totals = pt_counts(pred_games if label == "Predicted" else real_games)
    sorted_teams = sorted(ALL_TEAMS, key=lambda t: totals[t], reverse=True)
    x = np.arange(len(sorted_teams))
    bottoms = np.zeros(len(sorted_teams))
    fig, ax = plt.subplots(figsize=(14, 6))
    for cat in ("Christmas", "Thanksgiving", "MNF", "SNF", "TNF"):
        vals = np.array([counts[cat][t] for t in sorted_teams], dtype=float)
        ax.bar(x, vals, bottom=bottoms, color=PT_COLORS[cat], label=cat, width=0.7)
        bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_teams, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Games")
    ax.set_title(f"Primetime Appearances by Team — {label}", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    return save(os.path.join(CHART_DIR, outname))

# ── Chart 3: Difference (predicted minus official) ────────────────────────────

def chart_difference(pred_games, real_games):
    _, pred_totals = pt_counts(pred_games)
    _, real_totals = pt_counts(real_games)

    teams_sorted = sorted(ALL_TEAMS, key=lambda t: pred_totals[t] - real_totals[t], reverse=True)
    diffs = [pred_totals[t] - real_totals[t] for t in teams_sorted]
    colors_bar = ["#2a7abf" if d >= 0 else "#e05c00" for d in diffs]

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(range(len(teams_sorted)), diffs, color=colors_bar, width=0.7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(teams_sorted)))
    ax.set_xticklabels(teams_sorted, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Predicted minus Official")
    ax.set_title(
        "Primetime Gap: Predicted minus Official\n"
        "Blue = predicted gave more  |  Orange = official gave more",
        fontweight="bold"
    )
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    return save(os.path.join(CHART_DIR, "3_primetime_difference.png"))

# ── Chart 4: Slot distribution comparison (1PM / 4PM / PT) ───────────────────

def chart_slot_distribution(pred_games, real_games):
    def slot_totals(games):
        counts = defaultdict(lambda: defaultdict(int))
        for g in games:
            if g["week"] == 18:
                continue
            counts[g["cat"]][g["away"]] += 1
            counts[g["cat"]][g["home"]] += 1
        buckets = {
            "1PM":      {t: counts["1PM"][t] for t in ALL_TEAMS},
            "4PM":      {t: counts["4PM"][t] for t in ALL_TEAMS},
            "Primetime":{t: sum(counts[c][t] for c in PRIMETIME_CATS) for t in ALL_TEAMS},
            "Other":    {t: counts["Other"][t] for t in ALL_TEAMS},
        }
        return buckets

    pred_s = slot_totals(pred_games)
    real_s = slot_totals(real_games)

    bucket_colors = {"1PM": "#aec7e8", "4PM": "#ffbb78", "Primetime": "#5c2d8a", "Other": "#cccccc"}
    bucket_order  = ["Other", "1PM", "4PM", "Primetime"]

    fig, axes = plt.subplots(1, 2, figsize=(22, 6), sharey=True)
    for ax, (data, title) in zip(axes, [(pred_s, "Predicted"), (real_s, "Official")]):
        x = np.arange(len(ALL_TEAMS))
        bottoms = np.zeros(len(ALL_TEAMS))
        for bucket in bucket_order:
            vals = np.array([data[bucket][t] for t in ALL_TEAMS], dtype=float)
            ax.bar(x, vals, bottom=bottoms, color=bucket_colors[bucket], label=bucket, width=0.7)
            bottoms += vals
        ax.set_xticks(x)
        ax.set_xticklabels(ALL_TEAMS, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Games")
        ax.set_title(f"Timeslot Distribution — {title}", fontweight="bold")
        ax.legend(loc="upper right", fontsize=8)
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_ylim(0, 20)
    return save(os.path.join(CHART_DIR, "4_slot_distribution.png"))

# ── Chart 5: Gini by slot type ────────────────────────────────────────────────

def chart_gini_bars(pred_games, real_games):
    def per_team_counts(games, cat):
        c = defaultdict(int)
        for g in games:
            if g["cat"] == cat:
                c[g["away"]] += 1
                c[g["home"]] += 1
        return [c[t] for t in ALL_TEAMS]

    slot_labels = ["TNF", "SNF", "MNF", "Primetime (all)", "1PM", "4PM"]
    pred_ginis, real_ginis = [], []

    for slot in ["TNF", "SNF", "MNF"]:
        pred_ginis.append(gini(per_team_counts(pred_games, slot)))
        real_ginis.append(gini(per_team_counts(real_games, slot)))

    # Primetime all
    def pt_all(games):
        c = defaultdict(int)
        for g in games:
            if is_primetime(g["cat"]):
                c[g["away"]] += 1
                c[g["home"]] += 1
        return [c[t] for t in ALL_TEAMS]
    pred_ginis.append(gini(pt_all(pred_games)))
    real_ginis.append(gini(pt_all(real_games)))

    for slot in ["1PM", "4PM"]:
        pred_ginis.append(gini(per_team_counts(pred_games, slot)))
        real_ginis.append(gini(per_team_counts(real_games, slot)))

    x = np.arange(len(slot_labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w/2, pred_ginis, width=w, color=COLORS["pred"], label="Predicted", alpha=0.9)
    ax.bar(x + w/2, real_ginis, width=w, color=COLORS["real"], label="Official", alpha=0.9)

    for i, (p, r) in enumerate(zip(pred_ginis, real_ginis)):
        ax.text(i - w/2, p + 0.005, f"{p:.3f}", ha="center", va="bottom", fontsize=7, color=COLORS["pred"])
        ax.text(i + w/2, r + 0.005, f"{r:.3f}", ha="center", va="bottom", fontsize=7, color=COLORS["real"])

    ax.set_xticks(x)
    ax.set_xticklabels(slot_labels, fontsize=10)
    ax.set_ylabel("Gini coefficient (lower = more equal)")
    ax.set_title("Schedule Fairness by Timeslot — Gini Coefficient\nPredicted vs. Official", fontweight="bold")
    ax.legend(fontsize=10)
    return save(os.path.join(CHART_DIR, "5_gini_by_slot.png"))

# ── Generate HTML page ────────────────────────────────────────────────────────

def gini_label(v):
    if v < 0.15:
        return ("Very fair", "#2a7abf")
    if v < 0.25:
        return ("Fairly even", "#5a9e3a")
    if v < 0.35:
        return ("Moderate imbalance", "#c8960a")
    return ("High imbalance", "#c0392b")

def build_html(chart_paths, pred_games, real_games):
    _, pred_totals = pt_counts(pred_games)
    _, real_totals = pt_counts(real_games)

    pred_gini_v = gini(list(pred_totals.values()))
    real_gini_v = gini(list(real_totals.values()))
    pred_lbl, pred_clr = gini_label(pred_gini_v)
    real_lbl, real_clr = gini_label(real_gini_v)

    pred_max = max(pred_totals.values())
    real_max = max(real_totals.values())
    pred_min = min(pred_totals.values())
    real_min = min(real_totals.values())

    # Most / least favored
    pred_top = sorted(ALL_TEAMS, key=lambda t: pred_totals[t], reverse=True)[:3]
    real_top = sorted(ALL_TEAMS, key=lambda t: real_totals[t], reverse=True)[:3]
    pred_bot = sorted(ALL_TEAMS, key=lambda t: pred_totals[t])[:3]
    real_bot = sorted(ALL_TEAMS, key=lambda t: real_totals[t])[:3]

    def img(rel_path, alt=""):
        # Path relative to pages/
        src = os.path.join("..", rel_path).replace("\\", "/")
        return f'<img src="{src}" alt="{alt}" style="max-width:100%;border-radius:8px;box-shadow:0 2px 12px rgba(0,0,0,.12);margin:12px 0">'

    def stat_card(label, pred_val, real_val, fmt="{}", lower_better=False):
        pred_s = fmt.format(pred_val)
        real_s = fmt.format(real_val)
        if lower_better:
            pred_better = pred_val <= real_val
        else:
            pred_better = pred_val >= real_val
        return f"""
        <div class="stat-card">
          <div class="stat-label">{label}</div>
          <div class="stat-row">
            <span class="stat-pill pred {'winner' if pred_better else ''}">Predicted: {pred_s}</span>
            <span class="stat-pill real {'winner' if not pred_better else ''}">Official: {real_s}</span>
          </div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Predicted vs. Official: 2026 NFL Schedule Fairness</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    :root {{
      --bg: #f5f7fa; --surface: #fff; --text: #222; --text2: #444;
      --text3: #666; --border: #eee; --heading: #1a1a2e;
      --stat-bg: #f8f9fc;
      --pred-box-bg: #e8f0fb; --real-box-bg: #fdf0e8;
      --pred-pill-bg: #ddeeff; --pred-pill-fg: #1a4a80;
      --real-pill-bg: #fde8d8; --real-pill-fg: #7a2c00;
      --shadow: rgba(0,0,0,.08);
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --bg: #0f1117; --surface: #1a1d27; --text: #e8e8e8; --text2: #b0b0b0;
        --text3: #888; --border: #2e3147; --heading: #c8d4f0;
        --stat-bg: #22253a;
        --pred-box-bg: #1a2640; --real-box-bg: #2e1a0e;
        --pred-pill-bg: #1e3560; --pred-pill-fg: #90bcf0;
        --real-pill-bg: #3a1a08; --real-pill-fg: #f0a060;
        --shadow: rgba(0,0,0,.4);
      }}
    }}
    body {{ font-family: Arial, sans-serif; background: var(--bg); color: var(--text); padding: 24px 20px; }}
    h1 {{ text-align: center; font-size: 1.9rem; margin-bottom: 6px; color: var(--text); }}
    .subtitle {{ text-align: center; color: var(--text3); margin-bottom: 32px; font-size: 0.95rem; }}
    .back {{ display:block; text-align:center; margin-bottom:24px; color:#2a7abf; text-decoration:none; font-size:.95rem; }}
    .back:hover {{ text-decoration:underline; }}
    .section {{ max-width: 1100px; margin: 0 auto 40px; background: var(--surface);
                border-radius: 12px; padding: 24px 28px;
                box-shadow: 0 2px 16px var(--shadow); }}
    h2 {{ font-size: 1.25rem; margin-bottom: 16px; color: var(--heading); border-bottom: 2px solid var(--border); padding-bottom: 8px; }}
    h3 {{ font-size: 1rem; margin: 20px 0 8px; color: var(--text); }}
    p  {{ line-height: 1.6; color: var(--text2); margin-bottom: 10px; }}
    .verdict-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 16px 0; }}
    .verdict-box {{ border-radius: 8px; padding: 16px 20px; }}
    .verdict-box.pred {{ background: var(--pred-box-bg); border-left: 4px solid #2a7abf; }}
    .verdict-box.real {{ background: var(--real-box-bg); border-left: 4px solid #e05c00; }}
    .verdict-title {{ font-weight: bold; font-size: .95rem; margin-bottom: 6px; color: var(--text); }}
    .gini-val {{ font-size: 1.6rem; font-weight: bold; }}
    .gini-lbl {{ font-size: .85rem; margin-top: 2px; }}
    .verdict-box div:last-child {{ color: var(--text2); }}
    .stat-card {{ background: var(--stat-bg); border-radius: 8px; padding: 12px 16px; margin: 8px 0; }}
    .stat-label {{ font-weight: bold; font-size: .9rem; margin-bottom: 8px; color: var(--text); }}
    .stat-row {{ display: flex; gap: 12px; flex-wrap: wrap; }}
    .stat-pill {{ padding: 4px 12px; border-radius: 20px; font-size: .88rem; }}
    .stat-pill.pred {{ background: var(--pred-pill-bg); color: var(--pred-pill-fg); }}
    .stat-pill.real {{ background: var(--real-pill-bg); color: var(--real-pill-fg); }}
    .stat-pill.winner {{ font-weight: bold; outline: 2px solid currentColor; }}
    .legend {{ display: flex; gap: 20px; margin-bottom: 12px; flex-wrap: wrap; }}
    .legend-item {{ display: flex; align-items: center; gap: 6px; font-size: .88rem; color: var(--text2); }}
    .dot {{ width: 14px; height: 14px; border-radius: 3px; }}
    @media (max-width: 600px) {{
      .verdict-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <h1>Predicted vs. Official: 2026 NFL Schedule Fairness</h1>
  <p class="subtitle">How evenly were primetime slots distributed? A comparison of the predicted schedule and the one the NFL actually released.</p>
  <a class="back" href="../index.html">← Back to schedule</a>

  <div class="section">
    <h2>Overall Primetime Fairness</h2>
    <p>
      The <strong>Gini coefficient</strong> measures how evenly primetime games are spread across all 32 teams.
      A Gini of 0 means every team gets the exact same number of primetime games; higher values mean more
      concentration among a few teams. Lower is fairer.
    </p>
    <div class="verdict-grid">
      <div class="verdict-box pred">
        <div class="verdict-title">Predicted Schedule</div>
        <div class="gini-val" style="color:{pred_clr}">{pred_gini_v:.3f}</div>
        <div style="margin-top:10px;font-size:.88rem;">
          Most featured: {", ".join(pred_top)}<br>
          Least featured: {", ".join(pred_bot)}
        </div>
      </div>
      <div class="verdict-box real">
        <div class="verdict-title">Official NFL Schedule</div>
        <div class="gini-val" style="color:{real_clr}">{real_gini_v:.3f}</div>
        <div style="margin-top:10px;font-size:.88rem;">
          Most featured: {", ".join(real_top)}<br>
          Least featured: {", ".join(real_bot)}
        </div>
      </div>
    </div>

    {stat_card("Primetime Gini coefficient (lower = fairer)", pred_gini_v, real_gini_v, fmt="{:.3f}", lower_better=True)}
    {stat_card("Max primetime appearances (any team)", pred_max, real_max, lower_better=True)}
    {stat_card("Min primetime appearances (any team)", pred_min, real_min, lower_better=False)}
  </div>

  <div class="section">
    <h2>Chart 1 — Primetime Appearances: Predicted vs. Official</h2>
    <div class="legend">
      <div class="legend-item"><div class="dot" style="background:#2a7abf"></div> Predicted</div>
      <div class="legend-item"><div class="dot" style="background:#e05c00"></div> Official</div>
    </div>
    {img(chart_paths[0], "Primetime side by side")}
    <p>Teams sorted by predicted primetime appearances. The taller the blue bar above the orange, the more the
    predicted schedule gave a team relative to the real one, and vice versa.</p>
  </div>

  <div class="section">
    <h2>Chart 2 — Primetime Breakdown: Predicted</h2>
    {img(chart_paths[1], "Predicted stacked breakdown")}
  </div>

  <div class="section">
    <h2>Chart 3 — Primetime Breakdown: Official</h2>
    {img(chart_paths[2], "Official stacked breakdown")}
  </div>

  <div class="section">
    <h2>Chart 4 — Primetime Gap (Predicted minus Official)</h2>
    {img(chart_paths[3], "Primetime difference")}
    <p>Blue bars indicate teams that received <em>more</em> primetime slots in the predicted schedule.
    Orange bars indicate teams that received <em>more</em> slots in the official release.</p>
  </div>

  <div class="section">
    <h2>Chart 5 — Full Timeslot Distribution</h2>
    {img(chart_paths[4], "Slot distribution")}
    <p>Breakdown of all 17 regular weeks (Week 18 excluded as flex) by timeslot type. The predicted schedule
    aimed for a more even spread of 1PM and 4PM games across all 32 teams.</p>
  </div>

  <div class="section">
    <h2>Chart 6 — Gini Fairness by Slot Type</h2>
    {img(chart_paths[5], "Gini by slot")}
    <p>Lower Gini = more equitable distribution for that slot type. The predicted schedule generally achieves
    lower Gini scores across TNF, SNF, MNF, and standard afternoon windows because the ILP solver explicitly
    constrained how many primetime and afternoon games each team could receive.</p>
  </div>

</body>
</html>"""

    with open(HTML_OUT, "w") as f:
        f.write(html)
    print(f"HTML written to {HTML_OUT}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    pred_games, _ = parse_predicted()
    real_games     = parse_real()

    print(f"Predicted games: {len(pred_games)}")
    print(f"Real games:      {len(real_games)}")

    print("Generating charts...")
    paths = []
    paths.append(chart_primetime_side_by_side(pred_games, real_games))
    paths.append(chart_stacked_breakdown(pred_games, real_games, "Predicted", "2a_pred_stacked.png"))
    paths.append(chart_stacked_breakdown(pred_games, real_games, "Official",  "2b_real_stacked.png"))
    paths.append(chart_difference(pred_games, real_games))
    paths.append(chart_slot_distribution(pred_games, real_games))
    paths.append(chart_gini_bars(pred_games, real_games))

    for p in paths:
        print(f"  {p}")

    build_html(paths, pred_games, real_games)
    print("Done.")


if __name__ == "__main__":
    main()
