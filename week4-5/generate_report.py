"""
generate_report.py  —  Weeks 4-5 Deliverable
Generates a comprehensive HTML data cleaning report per IDX Exchange Handbook.

Covers all handbook-specified checks:
  - Column pruning (>90% null drop)
  - Date parsing & 3 consistency flags
  - Numeric quality flags
  - Geographic data quality (null coords, sentinel 0, lon > 0 error)
  - Before/after row and column counts

Author: Ruoyu Wang
Date:   May 2026
"""

import os
import json
import pandas as pd
from datetime import date

WEEK3_DIR  = r"D:\IDXexchange\week4"
DATA_DIR   = os.path.join(WEEK3_DIR, "data")
REPORT_DIR = os.path.join(WEEK3_DIR, "report")
os.makedirs(REPORT_DIR, exist_ok=True)

# ── Load summary data ──────────────────────────────────────────────────────

with open(os.path.join(DATA_DIR, "cleaning_summary.json")) as f:
    summary = json.load(f)

flag_log = pd.read_csv(os.path.join(DATA_DIR, "numeric_flag_log.csv"))

s   = summary["sold"]
ls  = summary["listed"]
geo = summary["geo_stats"]

today = date.today().strftime("%B %Y")

# ── Helper functions ───────────────────────────────────────────────────────

def progress_bar(pct, color="#0d9488", height=8):
    return (
        f'<div style="background:#e5e7eb;border-radius:4px;height:{height}px;overflow:hidden;">'
        f'<div style="width:{min(pct,100):.2f}%;background:{color};height:100%;border-radius:4px;"></div>'
        f'</div>'
    )

def stat_bar_row(label, value, total, color="#0d9488"):
    pct = value / total * 100 if total else 0
    return f"""
<div style="margin-bottom:12px;">
  <div style="display:flex;justify-content:space-between;font-size:0.85em;margin-bottom:3px;">
    <span>{label}</span>
    <span style="font-weight:600;color:#0f766e;">{value:,}
      <span style="color:#6b7280;font-weight:400;">/ {total:,} ({pct:.3f}%)</span>
    </span>
  </div>
  {progress_bar(pct, color)}
</div>"""

def before_after_card(title, before_val, after_val, unit="", note="", icon=""):
    delta = after_val - before_val
    delta_str = f"{delta:+,}"
    delta_color = "#dc2626" if delta > 0 else "#16a34a"
    return f"""
<div class="ba-card">
  <div class="ba-title">{icon} {title}</div>
  <div class="ba-row">
    <div class="ba-box before">
      <div class="ba-label">Before</div>
      <div class="ba-num">{before_val:,}{unit}</div>
    </div>
    <div class="ba-arrow">&#8594;</div>
    <div class="ba-box after">
      <div class="ba-label">After</div>
      <div class="ba-num">{after_val:,}{unit}</div>
    </div>
    <div class="ba-delta" style="color:{delta_color};">{delta_str}</div>
  </div>
  {f'<div class="ba-note">{note}</div>' if note else ''}
</div>"""

def flag_table_html(df, dataset):
    rows = df[df['dataset'] == dataset]
    total = s['clean_rows'] if dataset == 'SOLD' else ls['clean_rows']
    html_rows = ""
    for _, r in rows.iterrows():
        pct = r['flagged'] / total * 100 if total else 0
        bar = progress_bar(pct, "#f59e0b", 6)
        html_rows += f"""
<tr>
  <td><code>{r['column']}</code></td>
  <td>{r['rule']}</td>
  <td style="text-align:right;font-weight:600;">{int(r['flagged']):,}</td>
  <td style="text-align:right;color:#6b7280;">{pct:.3f}%</td>
  <td style="min-width:100px;">{bar}</td>
</tr>"""
    return f"""
<table>
  <thead><tr><th>Column</th><th>Rule</th><th>Flagged</th><th>%</th><th>Rate</th></tr></thead>
  <tbody>{html_rows}</tbody>
</table>"""

def geo_null_row(label, key, total, pct_key=None):
    val = geo.get(key, None)
    if val is None:
        return ""
    if isinstance(val, float) and val <= 1.0:
        pct = val * 100
        n = round(pct / 100 * total)
    else:
        n = int(val)
        pct = n / total * 100 if total else 0
    bar = progress_bar(pct, "#6366f1", 6)
    status_color = "#16a34a" if pct < 5 else ("#ea580c" if pct < 20 else "#dc2626")
    status = "Good" if pct < 5 else ("Partial" if pct < 20 else "Sparse")
    return f"""
<tr>
  <td>{label}</td>
  <td style="text-align:right;">{n:,}</td>
  <td style="text-align:right;">{pct:.1f}%</td>
  <td style="min-width:120px;">{bar}</td>
  <td style="color:{status_color};font-weight:600;">{status}</td>
</tr>"""

def geo_count_row(label, key, total):
    n = geo.get(key, 0)
    if n is None:
        n = 0
    n = int(n)
    pct = n / total * 100 if total else 0
    bar = progress_bar(pct, "#dc2626" if n > 0 else "#16a34a", 6)
    status_color = "#16a34a" if n == 0 else "#dc2626"
    status = "None" if n == 0 else "Flagged"
    return f"""
<tr>
  <td>{label}</td>
  <td style="text-align:right;">{n:,}</td>
  <td style="text-align:right;">{pct:.3f}%</td>
  <td style="min-width:120px;">{bar}</td>
  <td style="color:{status_color};font-weight:600;">{status}</td>
</tr>"""

def dropped_cols_html(col_list, label):
    items = "".join(f'<span class="col-badge">{c}</span>' for c in col_list)
    return (
        f'<div style="margin:10px 0;"><strong>{label} &mdash; {len(col_list)} dropped:</strong>'
        f'<div style="margin-top:8px;">{items}</div></div>'
    )

# ── Compute summary stats ──────────────────────────────────────────────────

sold_flag_rate   = s["any_invalid_flag"]  / s["clean_rows"]  * 100
listed_flag_rate = ls["any_invalid_flag"] / ls["clean_rows"] * 100

date_flags_total = (
    s.get("date_flags", 0)
    + s.get("purchase_after_close_flags", 0)
    + s.get("negative_timeline_flags", 0)
)

# ── HTML ──────────────────────────────────────────────────────────────────

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>IDX Exchange &#8212; Weeks 4&#8211;5 Data Cleaning Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: #f0fdf4;
    color: #1f2937;
    line-height: 1.65;
  }}

  .header {{
    background: linear-gradient(135deg, #064e3b 0%, #0d9488 60%, #14b8a6 100%);
    color: #fff;
    padding: 48px 60px 40px;
  }}
  .header-top {{
    display: flex; align-items: flex-start; justify-content: space-between;
  }}
  .header h1 {{ font-size: 2em; font-weight: 800; letter-spacing: -0.5px; }}
  .header .sub {{ font-size: 1em; opacity: 0.85; margin-top: 6px; }}
  .header .meta {{ font-size: 0.82em; opacity: 0.65; margin-top: 16px; }}
  .week-badge {{
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.35);
    border-radius: 24px;
    padding: 6px 18px;
    font-size: 0.85em; font-weight: 700;
    white-space: nowrap; margin-top: 4px;
  }}

  .pipeline {{
    background: #fff;
    border-bottom: 1px solid #d1fae5;
    padding: 14px 60px;
    display: flex; align-items: center; gap: 0;
    overflow-x: auto;
  }}
  .pipe-step {{
    display: flex; align-items: center; gap: 8px;
    font-size: 0.82em; font-weight: 600; color: #9ca3af;
    white-space: nowrap;
  }}
  .pipe-step.done {{ color: #0f766e; }}
  .pipe-step.active {{ color: #0d9488; }}
  .pipe-step .dot {{
    width: 28px; height: 28px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.75em; font-weight: 700;
    background: #e5e7eb; color: #9ca3af;
  }}
  .pipe-step.done .dot {{ background: #0f766e; color: #fff; }}
  .pipe-step.active .dot {{ background: #14b8a6; color: #fff; box-shadow: 0 0 0 3px #ccfbf1; }}
  .pipe-sep {{ color: #d1fae5; margin: 0 10px; font-size: 1.2em; }}

  .toc {{
    background: #fff;
    border-left: 4px solid #0d9488;
    margin: 28px 48px;
    padding: 18px 28px;
    border-radius: 6px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.05);
  }}
  .toc h2 {{ color: #0f766e; font-size: 1em; font-weight: 700; margin-bottom: 10px; }}
  .toc ol {{ padding-left: 20px; }}
  .toc li {{ margin: 4px 0; font-size: 0.9em; }}
  .toc a {{ color: #0d9488; text-decoration: none; }}
  .toc a:hover {{ text-decoration: underline; }}

  .section {{
    background: #fff;
    margin: 22px 48px;
    padding: 28px 34px;
    border-radius: 10px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.05);
    border-top: 3px solid #0d9488;
  }}
  h2.section-title {{
    font-size: 1.15em;
    color: #064e3b;
    font-weight: 800;
    padding-bottom: 10px;
    margin-bottom: 20px;
    border-bottom: 1px solid #d1fae5;
    display: flex; align-items: center; gap: 10px;
  }}
  h2.section-title .step-num {{
    background: #0d9488; color: #fff;
    border-radius: 50%; width: 28px; height: 28px;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 0.85em; font-weight: 700; flex-shrink: 0;
  }}
  h3 {{ color: #0f766e; margin: 20px 0 10px; font-size: 0.98em; font-weight: 700; }}

  .score-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(165px, 1fr));
    gap: 14px;
    margin: 16px 0;
  }}
  .score-kpi {{
    border: 1px solid #d1fae5;
    border-radius: 8px;
    padding: 16px 18px;
    background: #f0fdf4;
    position: relative; overflow: hidden;
  }}
  .score-kpi::before {{
    content: ''; position: absolute; left: 0; top: 0; bottom: 0;
    width: 4px; background: #0d9488;
  }}
  .score-kpi.warn::before {{ background: #f59e0b; }}
  .score-kpi.danger::before {{ background: #dc2626; }}
  .score-kpi.neutral::before {{ background: #6366f1; }}
  .score-kpi .val {{ font-size: 1.55em; font-weight: 800; color: #064e3b; line-height: 1; }}
  .score-kpi.warn .val {{ color: #b45309; }}
  .score-kpi.danger .val {{ color: #b91c1c; }}
  .score-kpi.neutral .val {{ color: #4338ca; }}
  .score-kpi .lbl {{ font-size: 0.73em; color: #6b7280; margin-top: 5px; }}

  .ba-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 16px; margin: 16px 0;
  }}
  .ba-card {{
    border: 1px solid #d1fae5; border-radius: 8px;
    padding: 16px 18px; background: #f9fafb;
  }}
  .ba-title {{ font-size: 0.85em; font-weight: 700; color: #374151; margin-bottom: 12px; }}
  .ba-row {{ display: flex; align-items: center; gap: 10px; }}
  .ba-box {{ flex: 1; border-radius: 6px; padding: 10px 12px; text-align: center; }}
  .ba-box.before {{ background: #fee2e2; border: 1px solid #fca5a5; }}
  .ba-box.after  {{ background: #dcfce7; border: 1px solid #86efac; }}
  .ba-label {{ font-size: 0.7em; color: #6b7280; }}
  .ba-num {{ font-size: 1.2em; font-weight: 700; color: #1f2937; }}
  .ba-arrow {{ font-size: 1.3em; color: #9ca3af; }}
  .ba-delta {{ font-size: 1.05em; font-weight: 700; min-width: 50px; text-align: right; }}
  .ba-note {{ margin-top: 8px; font-size: 0.78em; color: #6b7280;
    border-top: 1px dashed #e5e7eb; padding-top: 8px; }}

  table {{
    width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 0.86em;
  }}
  th {{
    background: #064e3b; color: #fff;
    padding: 8px 12px; text-align: left; font-weight: 600;
  }}
  td {{ padding: 7px 12px; border-bottom: 1px solid #f0fdf4; }}
  tr:hover td {{ background: #f0fdf4; }}

  .col-badge {{
    display: inline-block;
    background: #fee2e2; color: #991b1b;
    border-radius: 4px; padding: 2px 7px;
    font-size: 0.75em; font-family: monospace;
    margin: 2px 3px;
  }}

  .note {{
    background: #f0fdf4; border-left: 3px solid #0d9488;
    padding: 10px 14px; border-radius: 4px;
    font-size: 0.85em; color: #374151; margin: 12px 0;
  }}
  .note.warn {{ background: #fffbeb; border-left-color: #f59e0b; }}
  .note.danger {{ background: #fef2f2; border-left-color: #dc2626; }}

  .two-col {{
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 24px; margin: 16px 0;
  }}
  @media (max-width: 800px) {{
    .two-col {{ grid-template-columns: 1fr; }}
    .header {{ padding: 30px 24px; }}
    .section {{ margin: 16px 16px; padding: 20px 18px; }}
    .pipeline {{ padding: 12px 16px; }}
  }}

  footer {{
    text-align: center; padding: 28px; color: #9ca3af;
    font-size: 0.8em; border-top: 1px solid #d1fae5; margin-top: 12px;
  }}
</style>
</head>
<body>

<!-- HEADER -->
<div class="header">
  <div class="header-top">
    <div>
      <h1>IDX Exchange &#8212; MLS Data Cleaning</h1>
      <div class="sub">Weeks 4&#8211;5 Deliverable: Column Pruning, Date Standardisation, Numeric &amp; Geographic Quality Flags</div>
      <div class="meta">
        Author: Ruoyu Wang &nbsp;&#183;&nbsp; {today} &nbsp;&#183;&nbsp;
        Source: CRMLS Residential | Jan 2024 &#8211; Apr 2026
      </div>
    </div>
    <div class="week-badge">Weeks 4&#8211;5 Report</div>
  </div>
</div>

<!-- PIPELINE STRIP -->
<div class="pipeline">
  <div class="pipe-step done"><div class="dot">&#10003;</div> Data Extraction</div>
  <span class="pipe-sep">&#8250;</span>
  <div class="pipe-step done"><div class="dot">&#10003;</div> Aggregation</div>
  <span class="pipe-sep">&#8250;</span>
  <div class="pipe-step done"><div class="dot">&#10003;</div> EDA &amp; Enrichment</div>
  <span class="pipe-sep">&#8250;</span>
  <div class="pipe-step active"><div class="dot">4</div> Data Cleaning &#8592; you are here</div>
  <span class="pipe-sep">&#8250;</span>
  <div class="pipe-step"><div class="dot">5</div> Feature Engineering</div>
  <span class="pipe-sep">&#8250;</span>
  <div class="pipe-step"><div class="dot">6</div> Outlier Detection</div>
  <span class="pipe-sep">&#8250;</span>
  <div class="pipe-step"><div class="dot">7</div> Tableau Dashboards</div>
</div>

<!-- TOC -->
<div class="toc">
  <h2>Table of Contents</h2>
  <ol>
    <li><a href="#s1">Cleaning Scorecard</a></li>
    <li><a href="#s2">Column Pruning &#8212; Dropping High-Null Columns</a></li>
    <li><a href="#s3">Date Field Standardisation &amp; Consistency Flags</a></li>
    <li><a href="#s4">Numeric Quality Flags</a></li>
    <li><a href="#s5">Geographic Data Quality</a></li>
    <li><a href="#s6">Output Files &amp; Next Steps</a></li>
  </ol>
</div>

<!-- SECTION 1: SCORECARD -->
<div class="section" id="s1">
  <h2 class="section-title">
    <span class="step-num">1</span>
    Cleaning Scorecard
  </h2>
  <p style="margin-bottom:16px;color:#6b7280;font-size:0.9em;">
    High-level results of all cleaning operations applied to both datasets.
    No rows were deleted &#8212; quality flags are added as columns for downstream filtering in Week 7.
  </p>
  <div class="score-grid">
    <div class="score-kpi">
      <div class="val">{s['raw_rows']:,}</div>
      <div class="lbl">Sold records (total rows)</div>
    </div>
    <div class="score-kpi">
      <div class="val">{ls['raw_rows']:,}</div>
      <div class="lbl">Listed records (total rows)</div>
    </div>
    <div class="score-kpi danger">
      <div class="val">{s['cols_dropped']}</div>
      <div class="lbl">Sold columns dropped (&gt;90% null)</div>
    </div>
    <div class="score-kpi danger">
      <div class="val">{ls['cols_dropped']}</div>
      <div class="lbl">Listed columns dropped (&gt;90% null)</div>
    </div>
    <div class="score-kpi warn">
      <div class="val">{s.get('date_flags', 0):,}</div>
      <div class="lbl">Sold: listing date after close date</div>
    </div>
    <div class="score-kpi warn">
      <div class="val">{s.get('purchase_after_close_flags', 0):,}</div>
      <div class="lbl">Sold: purchase date after close date</div>
    </div>
    <div class="score-kpi warn">
      <div class="val">{s.get('negative_timeline_flags', 0):,}</div>
      <div class="lbl">Sold: listing date after purchase date</div>
    </div>
    <div class="score-kpi warn">
      <div class="val">{s['any_invalid_flag']:,}</div>
      <div class="lbl">Sold rows with any quality flag</div>
    </div>
    <div class="score-kpi warn">
      <div class="val">{ls['any_invalid_flag']:,}</div>
      <div class="lbl">Listed rows with any quality flag</div>
    </div>
    <div class="score-kpi neutral">
      <div class="val">{s['clean_cols']}</div>
      <div class="lbl">Sold columns after pruning</div>
    </div>
    <div class="score-kpi neutral">
      <div class="val">{ls['clean_cols']}</div>
      <div class="lbl">Listed columns after pruning</div>
    </div>
  </div>

  <h3>Flag Rate Overview</h3>
  {stat_bar_row("Sold &#8212; rows with any quality flag", s['any_invalid_flag'], s['clean_rows'], "#f59e0b")}
  {stat_bar_row("Listed &#8212; rows with any quality flag", ls['any_invalid_flag'], ls['clean_rows'], "#f59e0b")}
  {stat_bar_row("Sold &#8212; listing_after_close_flag", s.get('date_flags', 0), s['clean_rows'], "#ef4444")}
  {stat_bar_row("Sold &#8212; purchase_after_close_flag", s.get('purchase_after_close_flags', 0), s['clean_rows'], "#ef4444")}
  {stat_bar_row("Sold &#8212; negative_timeline_flag", s.get('negative_timeline_flags', 0), s['clean_rows'], "#ef4444")}

  <div class="note" style="margin-top:14px;">
    <strong>Strategy:</strong> Flagged records are <em>retained</em> with boolean indicator columns.
    Hard removal will happen in Week 7 (outlier detection) after final thresholds are agreed with the team.
  </div>
</div>

<!-- SECTION 2: COLUMN PRUNING -->
<div class="section" id="s2">
  <h2 class="section-title">
    <span class="step-num">2</span>
    Column Pruning &#8212; Dropping High-Null Columns
  </h2>
  <p style="color:#6b7280;font-size:0.9em;margin-bottom:16px;">
    Any column with more than 90% missing values was removed.
    These columns carry negligible analytical value and inflate file size and memory usage.
  </p>

  <div class="ba-grid">
    {before_after_card("Sold &#8212; Column Count", s['raw_cols'], s['clean_cols'],
      note=f"{s['cols_dropped']} columns removed")}
    {before_after_card("Listed &#8212; Column Count", ls['raw_cols'], ls['clean_cols'],
      note=f"{ls['cols_dropped']} columns removed")}
  </div>

  <div class="two-col" style="margin-top:20px;">
    <div>{dropped_cols_html(s['dropped_col_names'], "Sold")}</div>
    <div>{dropped_cols_html(ls['dropped_col_names'], "Listed")}</div>
  </div>

  <div class="note">
    Eight columns are <strong>100% null</strong> in both datasets
    (TaxYear, FireplacesTotal, TaxAnnualAmount, AboveGradeFinishedArea,
    ElementarySchoolDistrict, BusinessType, CoveredSpaces, MiddleOrJuniorSchoolDistrict)
    &#8212; flagged in Week 2&#8211;3 EDA and now dropped.
  </div>
</div>

<!-- SECTION 3: DATE FLAGS -->
<div class="section" id="s3">
  <h2 class="section-title">
    <span class="step-num">3</span>
    Date Field Standardisation &amp; Consistency Flags
  </h2>

  <div class="two-col">
    <div>
      <h3>Sold &#8212; Date Columns Parsed</h3>
      <table>
        <thead><tr><th>Column</th><th>New NaT Values</th><th>Status</th></tr></thead>
        <tbody>
          {''.join(
              f'<tr><td><code>{col}</code></td>'
              f'<td style="text-align:right;">{v["after_null"] - v["before_null"]:,}</td>'
              f'<td style="color:#16a34a;font-weight:600;">&#10003; Parsed</td></tr>'
              for col, v in s['date_parse_results'].items()
          )}
        </tbody>
      </table>
    </div>
    <div>
      <h3>Listed &#8212; Date Columns Parsed</h3>
      <table>
        <thead><tr><th>Column</th><th>New NaT Values</th><th>Status</th></tr></thead>
        <tbody>
          {''.join(
              f'<tr><td><code>{col}</code></td>'
              f'<td style="text-align:right;">{v["after_null"] - v["before_null"]:,}</td>'
              f'<td style="color:#16a34a;font-weight:600;">&#10003; Parsed</td></tr>'
              for col, v in ls['date_parse_results'].items()
          )}
        </tbody>
      </table>
    </div>
  </div>

  <h3>Date Consistency Checks &#8212; Sold Dataset</h3>
  <p style="color:#6b7280;font-size:0.88em;margin-bottom:12px;">
    Valid sequence: <strong>ListingContractDate</strong> &le; <strong>PurchaseContractDate</strong> &le; <strong>CloseDate</strong>
  </p>
  <table>
    <thead>
      <tr><th>Flag Column</th><th>Rule Violated</th><th>Count</th><th>% of Sold</th><th>Rate</th></tr>
    </thead>
    <tbody>
      <tr>
        <td><code>listing_after_close_flag</code></td>
        <td>ListingContractDate &gt; CloseDate</td>
        <td style="text-align:right;font-weight:600;">{s.get('date_flags', 0):,}</td>
        <td style="text-align:right;">{s.get('date_flags', 0)/s['clean_rows']*100:.3f}%</td>
        <td style="min-width:120px;">{progress_bar(s.get('date_flags',0)/s['clean_rows']*100, '#ef4444', 6)}</td>
      </tr>
      <tr>
        <td><code>purchase_after_close_flag</code></td>
        <td>PurchaseContractDate &gt; CloseDate</td>
        <td style="text-align:right;font-weight:600;">{s.get('purchase_after_close_flags', 0):,}</td>
        <td style="text-align:right;">{s.get('purchase_after_close_flags', 0)/s['clean_rows']*100:.3f}%</td>
        <td style="min-width:120px;">{progress_bar(s.get('purchase_after_close_flags',0)/s['clean_rows']*100, '#ef4444', 6)}</td>
      </tr>
      <tr>
        <td><code>negative_timeline_flag</code></td>
        <td>ListingContractDate &gt; PurchaseContractDate</td>
        <td style="text-align:right;font-weight:600;">{s.get('negative_timeline_flags', 0):,}</td>
        <td style="text-align:right;">{s.get('negative_timeline_flags', 0)/s['clean_rows']*100:.3f}%</td>
        <td style="min-width:120px;">{progress_bar(s.get('negative_timeline_flags',0)/s['clean_rows']*100, '#ef4444', 6)}</td>
      </tr>
    </tbody>
  </table>

  <div class="score-grid" style="grid-template-columns:repeat(4,1fr);margin-top:16px;">
    <div class="score-kpi warn">
      <div class="val">{date_flags_total:,}</div>
      <div class="lbl">Total date-flag records (any)</div>
    </div>
    <div class="score-kpi">
      <div class="val">{s['clean_rows'] - date_flags_total:,}</div>
      <div class="lbl">Records: all dates consistent</div>
    </div>
    <div class="score-kpi">
      <div class="val">{(s['clean_rows'] - date_flags_total) / s['clean_rows'] * 100:.2f}%</div>
      <div class="lbl">Date-consistent rows</div>
    </div>
    <div class="score-kpi neutral">
      <div class="val">3</div>
      <div class="lbl">Date consistency rules applied</div>
    </div>
  </div>

  <div class="note warn" style="margin-top:12px;">
    Records with date inconsistencies will be reviewed for data-entry errors or escrow
    re-listings in Week 7. All are retained with flag columns for now.
  </div>
</div>

<!-- SECTION 4: NUMERIC FLAGS -->
<div class="section" id="s4">
  <h2 class="section-title">
    <span class="step-num">4</span>
    Numeric Quality Flags
  </h2>
  <p style="color:#6b7280;font-size:0.9em;margin-bottom:16px;">
    Rules applied per column. Each violation adds a boolean <code>*_invalid_flag</code> column.
    A composite <code>any_invalid_flag</code> is also added (1 if any rule fires).
  </p>

  <div class="two-col">
    <div>
      <h3>Sold Dataset</h3>
      {flag_table_html(flag_log, 'SOLD')}
      <div style="margin-top:10px;">
        {stat_bar_row("any_invalid_flag (SOLD)", s['any_invalid_flag'], s['clean_rows'], "#f59e0b")}
      </div>
    </div>
    <div>
      <h3>Listed Dataset</h3>
      {flag_table_html(flag_log, 'LISTED')}
      <div style="margin-top:10px;">
        {stat_bar_row("any_invalid_flag (LISTED)", ls['any_invalid_flag'], ls['clean_rows'], "#f59e0b")}
      </div>
    </div>
  </div>

  <div class="note" style="margin-top:16px;">
    <strong>Notable:</strong> OriginalListPrice has records &gt;$50M including a maximum of ~$1.39B
    &#8212; almost certainly a data-entry error in the source MLS system.
    LivingArea records &le; 0 are likely sentinel values (&#8722;1, 0) from the API.
    Both will be hard-removed in Week 7 after review with the team.
  </div>
</div>

<!-- SECTION 5: GEOGRAPHIC -->
<div class="section" id="s5">
  <h2 class="section-title">
    <span class="step-num">5</span>
    Geographic Data Quality
  </h2>
  <p style="color:#6b7280;font-size:0.9em;margin-bottom:16px;">
    Four coordinate checks per the handbook: null lat/lon, sentinel zero values,
    and incorrect-sign longitude (California coordinates must be negative).
    Three boolean flag columns added: <code>geo_null_flag</code>,
    <code>geo_sentinel_flag</code>, <code>geo_invalid_flag</code>.
  </p>

  <div class="two-col">
    <div>
      <h3>Sold &#8212; Null Coverage</h3>
      <table>
        <thead><tr><th>Field</th><th>Null Count</th><th>Null %</th><th>Rate</th><th>Status</th></tr></thead>
        <tbody>
          {geo_null_row("Latitude",       "sold_Latitude",       s['clean_rows'])}
          {geo_null_row("Longitude",      "sold_Longitude",      s['clean_rows'])}
          {geo_null_row("PostalCode",     "sold_PostalCode",     s['clean_rows'])}
          {geo_null_row("City",           "sold_City",           s['clean_rows'])}
          {geo_null_row("CountyOrParish", "sold_CountyOrParish", s['clean_rows'])}
          {geo_null_row("StateOrProvince","sold_StateOrProvince",s['clean_rows'])}
        </tbody>
      </table>
    </div>
    <div>
      <h3>Listed &#8212; Null Coverage</h3>
      <table>
        <thead><tr><th>Field</th><th>Null Count</th><th>Null %</th><th>Rate</th><th>Status</th></tr></thead>
        <tbody>
          {geo_null_row("Latitude",       "listed_Latitude",       ls['clean_rows'])}
          {geo_null_row("Longitude",      "listed_Longitude",      ls['clean_rows'])}
          {geo_null_row("PostalCode",     "listed_PostalCode",     ls['clean_rows'])}
          {geo_null_row("City",           "listed_City",           ls['clean_rows'])}
          {geo_null_row("CountyOrParish", "listed_CountyOrParish", ls['clean_rows'])}
          {geo_null_row("StateOrProvince","listed_StateOrProvince",ls['clean_rows'])}
        </tbody>
      </table>
    </div>
  </div>

  <h3 style="margin-top:20px;">Coordinate Validity Checks (CA: lat &#8712; [32,42], lon &#8712; [&#8722;125,&#8722;114])</h3>
  <div class="two-col">
    <div>
      <h3>Sold Dataset</h3>
      <table>
        <thead><tr><th>Check</th><th>Count</th><th>%</th><th>Rate</th><th>Status</th></tr></thead>
        <tbody>
          {geo_count_row("Null lat or lon", "sold_null_geo", s['clean_rows'])}
          {geo_count_row("Latitude == 0 (sentinel)", "sold_zero_lat", s['clean_rows'])}
          {geo_count_row("Longitude == 0 (sentinel)", "sold_zero_lon", s['clean_rows'])}
          {geo_count_row("Longitude &gt; 0 (wrong sign)", "sold_pos_lon", s['clean_rows'])}
        </tbody>
      </table>
    </div>
    <div>
      <h3>Listed Dataset</h3>
      <table>
        <thead><tr><th>Check</th><th>Count</th><th>%</th><th>Rate</th><th>Status</th></tr></thead>
        <tbody>
          {geo_count_row("Null lat or lon", "listed_null_geo", ls['clean_rows'])}
          {geo_count_row("Latitude == 0 (sentinel)", "listed_zero_lat", ls['clean_rows'])}
          {geo_count_row("Longitude == 0 (sentinel)", "listed_zero_lon", ls['clean_rows'])}
          {geo_count_row("Longitude &gt; 0 (wrong sign)", "listed_pos_lon", ls['clean_rows'])}
        </tbody>
      </table>
    </div>
  </div>

  <div class="note" style="margin-top:12px;">
    Postal code and county coverage is effectively 100%, sufficient for planned Tableau
    choropleth maps. Lat/Lon null rates (3.9% Sold, 14.2% Listed) are acceptable for regional
    mapping but may affect granular spatial analysis. The {geo.get('sold_pos_lon',0)} sold and
    {geo.get('listed_pos_lon',0)} listed records with Longitude &gt; 0 are definite data-entry
    errors and will be removed in Week 7.
  </div>
</div>

<!-- SECTION 6: OUTPUTS -->
<div class="section" id="s6">
  <h2 class="section-title">
    <span class="step-num">6</span>
    Output Files &amp; Next Steps
  </h2>

  <h3>Files Produced (week4/data/)</h3>
  <table>
    <thead><tr><th>File</th><th>Description</th><th>Rows</th><th>Cols</th></tr></thead>
    <tbody>
      <tr>
        <td><code>sold_clean_flagged.csv</code></td>
        <td>Sold &#8212; columns pruned, dates parsed, numeric &amp; geo flags added</td>
        <td style="text-align:right;">{s['clean_rows']:,}</td>
        <td style="text-align:right;">{s['clean_cols']}</td>
      </tr>
      <tr>
        <td><code>listed_clean_flagged.csv</code></td>
        <td>Listed &#8212; columns pruned, dates parsed, numeric &amp; geo flags added</td>
        <td style="text-align:right;">{ls['clean_rows']:,}</td>
        <td style="text-align:right;">{ls['clean_cols']}</td>
      </tr>
      <tr>
        <td><code>cleaning_summary.json</code></td>
        <td>Machine-readable summary of all cleaning steps and counts</td>
        <td style="text-align:right;">&#8212;</td>
        <td style="text-align:right;">&#8212;</td>
      </tr>
      <tr>
        <td><code>numeric_flag_log.csv</code></td>
        <td>Per-rule flag counts for every numeric cleaning rule applied</td>
        <td style="text-align:right;">&#8212;</td>
        <td style="text-align:right;">&#8212;</td>
      </tr>
    </tbody>
  </table>

  <h3 style="margin-top:20px;">Flag Columns Added</h3>
  <table style="width:auto;">
    <thead><tr><th>Column</th><th>Dataset</th><th>Meaning</th></tr></thead>
    <tbody>
      <tr><td><code>listing_after_close_flag</code></td><td>Sold</td><td>1 if ListingContractDate &gt; CloseDate</td></tr>
      <tr><td><code>purchase_after_close_flag</code></td><td>Sold</td><td>1 if PurchaseContractDate &gt; CloseDate</td></tr>
      <tr><td><code>negative_timeline_flag</code></td><td>Sold</td><td>1 if ListingContractDate &gt; PurchaseContractDate</td></tr>
      <tr><td><code>ClosePrice_invalid_flag</code></td><td>Sold</td><td>1 if ClosePrice &le; 0 OR &gt; $50M</td></tr>
      <tr><td><code>OriginalListPrice_invalid_flag</code></td><td>Sold</td><td>1 if OLP &le; 0 OR &gt; $50M</td></tr>
      <tr><td><code>LivingArea_invalid_flag</code></td><td>Sold + Listed</td><td>1 if LivingArea &le; 0</td></tr>
      <tr><td><code>DaysOnMarket_invalid_flag</code></td><td>Sold</td><td>1 if DaysOnMarket &lt; 0</td></tr>
      <tr><td><code>ListPrice_invalid_flag</code></td><td>Listed</td><td>1 if ListPrice &le; 0 OR &gt; $50M</td></tr>
      <tr><td><code>geo_null_flag</code></td><td>Both</td><td>1 if Latitude or Longitude is null</td></tr>
      <tr><td><code>geo_sentinel_flag</code></td><td>Both</td><td>1 if Latitude == 0 or Longitude == 0</td></tr>
      <tr><td><code>geo_invalid_flag</code></td><td>Both</td><td>1 if Longitude &gt; 0 (wrong sign for CA)</td></tr>
      <tr><td><code>any_invalid_flag</code></td><td>Both</td><td>1 if any numeric or date flag fires</td></tr>
    </tbody>
  </table>

  <h3 style="margin-top:20px;">Upcoming &#8212; Week 6: Feature Engineering</h3>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:10px;">
    <div class="note">
      <strong>Price features:</strong><br>
      &#183; <code>price_per_sqft</code> = ClosePrice / LivingArea<br>
      &#183; <code>price_ratio</code> = ClosePrice / OriginalListPrice<br>
      &#183; <code>list_price_change</code> = (ListPrice &#8722; OriginalListPrice) / OriginalListPrice
    </div>
    <div class="note">
      <strong>Time features:</strong><br>
      &#183; <code>days_to_contract</code> = PurchaseContractDate &#8722; ListingContractDate<br>
      &#183; <code>contract_to_close</code> = CloseDate &#8722; PurchaseContractDate<br>
      &#183; <code>listing_month</code>, <code>listing_quarter</code>, <code>listing_year</code>
    </div>
  </div>
</div>

<footer>
  IDX Exchange Data Analyst Internship &nbsp;&#183;&nbsp;
  Weeks 4&#8211;5 Report &nbsp;&#183;&nbsp;
  Generated {today} &nbsp;&#183;&nbsp; Ruoyu Wang
</footer>

</body>
</html>"""

out_path = os.path.join(REPORT_DIR, "week4_5_report.html")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Report saved: {out_path}")
