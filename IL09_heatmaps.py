"""
IL-09 Democratic Primary — Performance Heat Maps
=================================================
Two candidate-level heat map layers toggled by dropdown:
  1. vs Model      — actual vote share minus model prediction (pp)
  2. vs District   — actual vote share minus district-wide average (pp)

Both use the same diverging red→white→green colorscale (±20 pp).
Select a candidate from the dropdown to switch between them.

Run independently; writes IL09_heatmaps.html.
Requires the simplified CSV (run win_probability_precinct.py first).
"""

import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

SHAPEFILE_PATH               = 'data/shapefile/IL24/IL24.shp'
CONGRESSIONAL_DISTRICTS_PATH = 'data/shapefile/congressional_districts.shp'
CHICAGO_BOUNDARY_PATH        = 'data/shapefile/Chicago/Chicago.shp'
EVANSTON_BOUNDARY_PATH       = 'data/shapefile/Evanston/Evanston.shp'
PRECINCT_CSV                 = 'data/csv_data/expectations/IL_09_precinct_probabilities.csv'
SIMPLIFIED_CSV               = 'data/csv_data/expectations/IL_09_precinct_probabilities_simplified.csv'
OUTPUT_HTML                  = 'IL09_heatmaps.html'

CANDIDATES = ['Fine', 'Biss', 'Abughazaleh', 'Simmons', 'Amiwala', 'Andrew', 'Huynh']

COLORS = {
    'Fine': 'green', 'Abughazaleh': 'orange', 'Biss': 'purple',
    'Amiwala': 'turquoise', 'Simmons': 'deeppink', 'Andrew': 'red',
    'Huynh': 'gray', 'Other': '#999999',
}

MODELED_RESULT_COLS = {
    'Daniel Biss': 'Biss', 'Mike Simmons': 'Simmons', 'Bushra Amiwala': 'Amiwala',
    'Laura Fine': 'Fine', 'Phil Andrew': 'Andrew', 'Kat Abughazaleh': 'Abughazaleh',
    'Hoan Huynh': 'Huynh',
}

OTHER_RESULT_COLS = [
    'Justin Ford', 'Patricia A. Brown', 'Jeff Cohen', 'Nick Pyati',
    'Sam Polan', 'Bethany Johnson', 'Howard Rosenblum', 'Mark Arnold Fredrickson',
]

TOTAL_VOTES_COL = 'Total Votes'
DIVERGING_CS = [
    [0.0,  'rgb(180,0,0)'],   [0.35, 'rgb(240,100,100)'],
    [0.5,  'rgb(220,220,220)'],[0.65, 'rgb(80,200,100)'],
    [1.0,  'rgb(0,140,50)'],
]

# ============================================================================
# STEP 1: LOAD CSV
# ============================================================================

print("Loading precinct CSV...")
df = pd.read_csv(PRECINCT_CSV)
print(f"  {len(df)} precincts")

def build_display_name(row):
    jf  = str(row.get('JoinField', '') or '').strip()
    alt = str(row.get('JoinFieldAlt', '') or '').strip()
    if jf.upper().startswith('COOK:') and jf.split(':')[1].strip().isdigit():
        return alt.split(':', 1)[1].strip().title() if (alt and ':' in alt) else jf
    return jf.split(':', 1)[1].strip().title() if ':' in jf else jf.title()

df['display_name']   = df.apply(build_display_name, axis=1)
df['JoinField_norm'] = df['JoinField'].str.upper()

# ============================================================================
# STEP 2: ACTUAL VOTES
# ============================================================================

for col in list(MODELED_RESULT_COLS.keys()) + OTHER_RESULT_COLS:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

for csv_col, cand in MODELED_RESULT_COLS.items():
    df[f'actual_votes_{cand}'] = df[csv_col] if csv_col in df.columns else 0

present_other = [c for c in OTHER_RESULT_COLS if c in df.columns]
df['actual_votes_Other'] = df[present_other].sum(axis=1) if present_other else 0

if TOTAL_VOTES_COL in df.columns:
    df['actual_total_votes'] = pd.to_numeric(df[TOTAL_VOTES_COL], errors='coerce').fillna(0)
else:
    df['actual_total_votes'] = df[[f'actual_votes_{c}' for c in CANDIDATES] + ['actual_votes_Other']].sum(axis=1)

for cand in CANDIDATES:
    df[f'actual_pct_{cand}'] = np.where(
        df['actual_total_votes'] > 0,
        df[f'actual_votes_{cand}'] / df['actual_total_votes'] * 100, 0.0)

# District-wide pct per candidate (denominator for vs-district layer)
total_votes_dist = int(df['actual_total_votes'].sum())
district_pcts = {
    cand: df[f'actual_votes_{cand}'].sum() / total_votes_dist * 100
    for cand in CANDIDATES
} if total_votes_dist > 0 else {c: 0.0 for c in CANDIDATES}

# ============================================================================
# STEP 3: PREDICTED PCT (from simplified CSV)
# ============================================================================

has_predictions = False
try:
    df_pred = pd.read_csv(SIMPLIFIED_CSV,
                          usecols=['JoinField'] + [f'median_pct_{c}' for c in CANDIDATES])
    sample_max = df_pred[f'median_pct_{CANDIDATES[0]}'].max()
    if sample_max < 2.0:
        for c in CANDIDATES:
            df_pred[f'median_pct_{c}'] *= 100
    df_pred['JoinField_norm'] = df_pred['JoinField'].str.upper()
    rename_map = {f'median_pct_{c}': f'predicted_pct_{c}' for c in CANDIDATES}
    df_pred = df_pred.rename(columns=rename_map)
    merge_cols = ['JoinField_norm'] + [f'predicted_pct_{c}' for c in CANDIDATES]
    df = df.merge(df_pred[merge_cols], on='JoinField_norm', how='left')
    has_predictions = True
    print(f"  Loaded predicted_pct columns for {len(CANDIDATES)} candidates")
except FileNotFoundError:
    print(f"  WARNING: {SIMPLIFIED_CSV} not found — vs Model layer will be empty.")
    print(f"  Run win_probability_precinct.py to generate it.")
    for c in CANDIDATES:
        df[f'predicted_pct_{c}'] = np.nan
except KeyError as e:
    print(f"  WARNING: missing column {e} in simplified CSV.")
    for c in CANDIDATES:
        df[f'predicted_pct_{c}'] = np.nan

# ============================================================================
# STEP 4: LOAD SHAPEFILE AND JOIN
# ============================================================================

print("\nLoading shapefiles...")
gdf          = gpd.read_file(SHAPEFILE_PATH)
gdf_congress = gpd.read_file(CONGRESSIONAL_DISTRICTS_PATH)

for g in [gdf, gdf_congress]:
    if g.crs is None:
        g.set_crs(epsg=4326, inplace=True)
if gdf_congress.crs != gdf.crs:
    gdf_congress = gdf_congress.to_crs(gdf.crs)

district_col = next(
    (c for c in ['DISTRICT', 'CD', 'CONG_DIST', 'DIST_NUM', 'NAME', 'NAMELSAD']
     if c in gdf_congress.columns), None)
il09_mask = (
    (gdf_congress[district_col] == '09') |
    (gdf_congress[district_col] == '9')  |
    (gdf_congress[district_col] == 9)    |
    (gdf_congress[district_col].astype(str).str.contains('09', na=False)) |
    (gdf_congress[district_col].astype(str).str.contains('9', na=False))
)
il09_geom = (gdf_congress[il09_mask].geometry.union_all()
             if hasattr(gdf_congress[il09_mask].geometry, 'union_all')
             else gdf_congress[il09_mask].geometry.unary_union)
gdf['geometry'] = gdf.geometry.intersection(il09_geom)
gdf = gdf[~gdf.geometry.is_empty].copy()
print(f"  Clipped to {len(gdf)} precincts in IL-09")

gdf['JoinField_norm'] = gdf['JoinField'].str.upper()
df['JoinField_norm']  = df['JoinField'].str.upper()
geom_df    = gdf[['JoinField_norm', 'geometry']].copy()
gdf_merged = df.merge(geom_df, on='JoinField_norm', how='inner')

if 'JoinField2' in df.columns:
    df['JoinField2_norm'] = df['JoinField2'].str.upper()
    matched_norms  = set(gdf_merged['JoinField_norm'])
    unmatched_csv  = df[~df['JoinField_norm'].isin(matched_norms)].copy()
    unmatched_geom = geom_df[~geom_df['JoinField_norm'].isin(matched_norms)].copy()
    if len(unmatched_csv) > 0 and len(unmatched_geom) > 0:
        extra = unmatched_csv.merge(
            unmatched_geom.rename(columns={'JoinField_norm': 'JoinField2_norm'}),
            on='JoinField2_norm', how='inner')
        if len(extra) > 0:
            print(f"  Secondary join matched {len(extra)} additional precincts")
            gdf_merged = pd.concat([gdf_merged, extra], ignore_index=True)

gdf_merged = gpd.GeoDataFrame(gdf_merged, geometry='geometry')
if gdf_merged.crs is None:
    gdf_merged = gdf_merged.set_crs(epsg=4326)
elif gdf_merged.crs.to_epsg() != 4326:
    gdf_merged = gdf_merged.to_crs(epsg=4326)
invalid = ~gdf_merged.geometry.is_valid
if invalid.sum():
    gdf_merged.loc[invalid, 'geometry'] = gdf_merged.loc[invalid, 'geometry'].buffer(0)
gdf_merged = gdf_merged[~gdf_merged.geometry.is_empty].reset_index(drop=True)
print(f"  Joined: {len(gdf_merged)} precincts")

# Region
def assign_region(jf):
    jf = str(jf).strip().upper()
    if jf.startswith('CITY OF CHICAGO:'): return 'Chicago'
    if jf.startswith('COOK:'):
        suffix = jf.split(':', 1)[1].strip()
        if suffix.isdigit() and 7501001 <= int(suffix) <= 7509999: return 'Evanston'
        return 'Suburban Cook'
    if jf.startswith('LAKE:'):    return 'Lake County'
    if jf.startswith('MCHENRY:'): return 'McHenry County'
    return 'Other'

gdf_merged['region'] = gdf_merged['JoinField'].apply(assign_region)

gdf_merged['_id'] = range(len(gdf_merged))

# ============================================================================
# STEP 5: BUILD FIGURE
# Two sets of traces: vs_model_traces and vs_district_traces
# Each set has one trace per candidate.
# Dropdown switches both which set is visible AND which candidate.
# ============================================================================

print("\nBuilding heat map figure...")
fig = go.Figure()

# ── Hover builders ────────────────────────────────────────────────────────

def hover_vs_model(row, cand):
    name   = str(row.get('display_name', '') or '').strip() or 'Unknown'
    region = row.get('region', '')
    actual = row.get(f'actual_pct_{cand}', 0)
    pred   = row.get(f'predicted_pct_{cand}', None)
    if pred is None or pd.isna(pred):
        return f"<b>{name}</b><br><i>{region}</i><br>No prediction data"
    diff  = actual - pred
    arrow = '▲' if diff > 0 else '▼' if diff < 0 else '●'
    clr   = '#4caf50' if diff > 0 else 'tomato' if diff < 0 else '#aaa'
    return (
        f"<b>{name}</b><br><i>{region}</i><br>"
        f"<span style='font-family:monospace'>"
        f"{cand}<br>"
        f"Actual:    {actual:>6.1f}%<br>"
        f"Predicted: {pred:>6.1f}%<br>"
        f"<span style='color:{clr};font-weight:bold;'>{arrow} {diff:+.1f} pts</span>"
        f"</span>"
    )

def hover_vs_district(row, cand):
    name     = str(row.get('display_name', '') or '').strip() or 'Unknown'
    region   = row.get('region', '')
    actual   = row.get(f'actual_pct_{cand}', 0)
    dist_avg = district_pcts[cand]
    diff     = actual - dist_avg
    has_v    = row.get('actual_total_votes', 0) > 0
    if not has_v:
        return f"<b>{name}</b><br><i>{region}</i><br>No results"
    arrow = '▲' if diff > 0 else '▼' if diff < 0 else '●'
    clr   = '#4caf50' if diff > 0 else 'tomato' if diff < 0 else '#aaa'
    return (
        f"<b>{name}</b><br><i>{region}</i><br>"
        f"<span style='font-family:monospace'>"
        f"{cand}<br>"
        f"Precinct:  {actual:>6.1f}%<br>"
        f"District:  {dist_avg:>6.1f}%<br>"
        f"<span style='color:{clr};font-weight:bold;'>{arrow} {diff:+.1f} pts</span>"
        f"</span>"
    )

# ── Add traces ────────────────────────────────────────────────────────────
# Order: [vs_model_Fine, vs_model_Biss, ..., vs_district_Fine, vs_district_Biss, ...]
# Default visible: vs_model_Fine only.

vs_model_start    = 0
vs_district_start = len(CANDIDATES)

for i, cand in enumerate(CANDIDATES):
    actual_col = f'actual_pct_{cand}'
    pred_col   = f'predicted_pct_{cand}'
    has_data   = gdf_merged[pred_col].notna() & (gdf_merged[actual_col] > 0)

    z_model = np.where(
        has_data,
        np.clip(gdf_merged[actual_col] - gdf_merged[pred_col], -20, 20),
        np.nan
    )
    hover_m = gdf_merged.apply(lambda r: hover_vs_model(r, cand), axis=1)

    fig.add_trace(go.Choroplethmapbox(
        geojson=gdf_merged.__geo_interface__,
        locations=gdf_merged['_id'],
        z=z_model, zmin=-20, zmax=20,
        colorscale=DIVERGING_CS,
        showscale=(i == 0),
        colorbar=dict(
            title=dict(text='pp vs model', font=dict(color='white', size=11)),
            tickfont=dict(color='white', size=10),
            tickvals=[-20, -10, 0, 10, 20],
            ticktext=['-20+', '-10', '0', '+10', '+20'],
            len=0.5, thickness=14, x=1.01, bgcolor='rgba(0,0,0,0.4)',
        ),
        marker_line_width=0.4,
        marker_line_color='rgba(255,255,255,0.3)',
        marker_opacity=0.85,
        text=hover_m,
        hovertemplate='%{text}<extra></extra>',
        name=f'{cand} vs model',
        featureidkey='properties._id',
        visible=(i == 0),
    ))

for i, cand in enumerate(CANDIDATES):
    actual_col = f'actual_pct_{cand}'
    has_data   = gdf_merged['actual_total_votes'] > 0

    z_dist = np.where(
        has_data,
        np.clip(gdf_merged[actual_col] - district_pcts[cand], -20, 20),
        np.nan
    )
    hover_d = gdf_merged.apply(lambda r: hover_vs_district(r, cand), axis=1)

    fig.add_trace(go.Choroplethmapbox(
        geojson=gdf_merged.__geo_interface__,
        locations=gdf_merged['_id'],
        z=z_dist, zmin=-20, zmax=20,
        colorscale=DIVERGING_CS,
        showscale=(i == 0),
        colorbar=dict(
            title=dict(text='pp vs district', font=dict(color='white', size=11)),
            tickfont=dict(color='white', size=10),
            tickvals=[-20, -10, 0, 10, 20],
            ticktext=['-20+', '-10', '0', '+10', '+20'],
            len=0.5, thickness=14, x=1.01, bgcolor='rgba(0,0,0,0.4)',
        ),
        marker_line_width=0.4,
        marker_line_color='rgba(255,255,255,0.3)',
        marker_opacity=0.85,
        text=hover_d,
        hovertemplate='%{text}<extra></extra>',
        name=f'{cand} vs district',
        featureidkey='properties._id',
        visible=False,
    ))

n_heatmap_traces = len(CANDIDATES) * 2

# ── Boundaries ────────────────────────────────────────────────────────────

def geom_to_lonlat(geom):
    lons, lats = [], []
    polys = list(geom.geoms) if geom.geom_type == 'MultiPolygon' else [geom]
    for poly in polys:
        xs, ys = poly.exterior.xy
        lons += list(xs) + [None]; lats += list(ys) + [None]
        for interior in poly.interiors:
            xs, ys = interior.xy
            lons += list(xs) + [None]; lats += list(ys) + [None]
    return lons, lats

def add_boundary(shp_path, color='black', width=3, filter_col=None,
                 filter_val=None, label='boundary'):
    try:
        gdf_b = gpd.read_file(shp_path)
        if gdf_b.crs is None: gdf_b = gdf_b.set_crs(epsg=4326)
        else: gdf_b = gdf_b.to_crs(epsg=4326)
        if filter_col and filter_val is not None:
            gdf_b = gdf_b[gdf_b[filter_col].astype(str).str.contains(
                str(filter_val), case=False, na=False)]
        if len(gdf_b) == 0:
            print(f"  WARNING: no features in {shp_path}"); return
        geom = (gdf_b.geometry.union_all()
                if hasattr(gdf_b.geometry, 'union_all')
                else gdf_b.geometry.unary_union)
        lons, lats = geom_to_lonlat(geom)
        fig.add_trace(go.Scattermapbox(
            lon=lons, lat=lats, mode='lines',
            line=dict(width=width, color=color),
            hoverinfo='skip', showlegend=False, visible=True,
        ))
        print(f"  ✓ Boundary: {label}")
    except Exception as e:
        print(f"  ⚠ Could not load {shp_path}: {e}")

print("\nLoading boundary shapefiles...")
add_boundary(CONGRESSIONAL_DISTRICTS_PATH, color='black', width=3.5,
             filter_col=district_col, filter_val='9', label='IL-09 district')
add_boundary(CHICAGO_BOUNDARY_PATH,  color='black', width=2, label='Chicago')
add_boundary(EVANSTON_BOUNDARY_PATH, color='black', width=2, label='Evanston')

n_boundary_traces = len(fig.data) - n_heatmap_traces

# ── Dropdown: one button per candidate × layer ────────────────────────────
# Each button shows exactly one heatmap trace + all boundaries.

def make_vis(show_idx):
    """show_idx = index into fig.data to make visible (one heatmap trace)."""
    vis = [False] * n_heatmap_traces + [True] * n_boundary_traces
    vis[show_idx] = True
    return vis

buttons = []
for i, cand in enumerate(CANDIDATES):
    buttons.append(dict(
        label=f'{cand} — vs Model',
        method='update',
        args=[
            {'visible': make_vis(vs_model_start + i)},
            {'title': f'IL-09 Heat Map | {cand} — Actual vs Model Prediction'},
        ],
    ))
for i, cand in enumerate(CANDIDATES):
    buttons.append(dict(
        label=f'{cand} — vs District',
        method='update',
        args=[
            {'visible': make_vis(vs_district_start + i)},
            {'title': f'IL-09 Heat Map | {cand} — Actual vs District Average'},
        ],
    ))

# Map center
gdf_proj   = gdf_merged.to_crs(epsg=3857)
union_geom = (gdf_proj.geometry.union_all()
              if hasattr(gdf_proj.geometry, 'union_all')
              else gdf_proj.geometry.unary_union)
center = gpd.GeoSeries([union_geom.centroid], crs=3857).to_crs(4326)[0]

fig.update_layout(
    title=dict(
        text=f'IL-09 Heat Map | {CANDIDATES[0]} — Actual vs Model Prediction',
        x=0.5, xanchor='center', font=dict(size=14),
    ),
    mapbox=dict(style='carto-positron', zoom=9.5,
                center=dict(lat=center.y, lon=center.x)),
    margin={'r': 0, 't': 50, 'l': 0, 'b': 0},
    height=800,
    showlegend=False,
    modebar=dict(orientation='v', bgcolor='rgba(255,255,255,0.7)',
                 color='#333', activecolor='#667eea'),
    updatemenus=[dict(
        type='dropdown', direction='down',
        x=0.01, xanchor='left', y=0.99, yanchor='top',
        buttons=buttons,
        bgcolor='white', bordercolor='#ccc',
        font=dict(size=12), showactive=True, active=0,
    )],
    annotations=[dict(
        text='<b>Layer:</b>',
        x=0.01, y=1.055, xref='paper', yref='paper',
        xanchor='left', yanchor='bottom',
        showarrow=False, font=dict(size=12),
    )],
)

# ============================================================================
# STEP 6: WRITE HTML
# ============================================================================

print(f"\nWriting {OUTPUT_HTML}...")
plotly_html = fig.to_html(
    include_plotlyjs='cdn', div_id='heatmap-div',
    config={'displaylogo': False,
            'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
            'toImageButtonOptions': {'format': 'png', 'filename': 'IL09_heatmap'}})

# Key stats for the legend note below the map
note_rows = []
for cand in CANDIDATES:
    dist_pct = district_pcts[cand]
    note_rows.append(
        f"<span style='color:{COLORS[cand]};font-weight:700'>{cand}</span> "
        f"<span style='color:rgba(255,255,255,0.6)'>{dist_pct:.1f}% district</span>"
    )
district_note = '&nbsp;&nbsp;·&nbsp;&nbsp;'.join(note_rows)

full_html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=0.6">
  <title>IL-09 Democratic Primary — Performance Heat Maps</title>
  <style>
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    body {{
      font-family: Arial, sans-serif;
      background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
      background-attachment: fixed; min-height: 100vh;
    }}
    nav {{
      background-color: rgba(20,20,40,0.97); padding: 15px 0;
      box-shadow: 0 4px 6px rgba(0,0,0,0.4); position: relative; z-index: 100;
    }}
    .nav-container {{
      max-width: 1400px; margin: 0 auto; padding: 0 20px;
      display: flex; justify-content: space-between; align-items: center;
      flex-wrap: wrap; gap: 8px;
    }}
    .nav-title {{ color: white; font-size: 1.4rem; font-weight: bold; }}
    .nav-button {{
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white; padding: 9px 18px; text-decoration: none;
      border-radius: 20px; font-weight: bold; font-size: 0.9rem;
    }}
    .container {{ max-width: 1400px; margin: 36px auto; padding: 0 20px; position: relative; z-index: 1; }}
    .hero {{
      background: rgba(255,255,255,0.97); padding: 36px; border-radius: 14px;
      box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }}
    .hero h1 {{
      text-align: center; margin-bottom: 18px;
      border-bottom: 3px solid #333; padding-bottom: 10px; font-size: 1.7rem;
    }}
    #heatmap-div {{ height: 800px; }}
    .legend-note {{
      max-width: 1400px; margin: 16px auto; padding: 12px 20px;
      background: rgba(255,255,255,0.06); border-radius: 8px;
      font-size: 0.82rem; text-align: center; color: rgba(255,255,255,0.8);
      font-family: monospace; line-height: 1.8;
    }}
    footer {{
      background-color: rgba(20,20,40,0.97); color: #ccc; text-align: center;
      padding: 18px; margin-top: 30px; font-size: 0.9rem;
    }}
  </style>
</head>
<body>
<nav>
  <div class="nav-container">
    <div class="nav-title">Cole's Election Models</div>
    <a href="index.html" class="nav-button">Home</a>
    <a href="IL09_precinct_map.html" class="nav-button">Prediction Map</a>
    <a href="IL09_actual_results_map.html" class="nav-button">Actual Results</a>
    <a href="IL09_turnout_map.html" class="nav-button">Turnout Map</a>
  </div>
</nav>
<div class="container">
  <div class="hero">
    <h1>IL-09 Democratic Primary — Performance Heat Maps</h1>
    <p style="text-align:center;color:#555;margin-bottom:16px;font-size:0.95rem;">
      Red = underperformed &nbsp;·&nbsp; White = on target &nbsp;·&nbsp;
      Green = overperformed &nbsp;·&nbsp; Scale: ±20 percentage points
    </p>
    <div id="map-container">{plotly_html}</div>
  </div>
</div>
<div class="legend-note">
  District-wide vote shares (baseline for "vs District" layer): &nbsp;&nbsp;
  {district_note}
</div>
<footer>Cole's Election Models &nbsp;·&nbsp; IL-09 Democratic Primary 2026</footer>
</body>
</html>
"""

with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
    f.write(full_html)

print(f"\n{'='*60}\nHEAT MAPS COMPLETE\n{'='*60}")
print(f"\nOpen {OUTPUT_HTML} in your browser.")
print(f"\nDropdown has {len(buttons)} options:")
print(f"  {len(CANDIDATES)} × vs Model  (actual − prediction)")
print(f"  {len(CANDIDATES)} × vs District (actual − district avg)")
print(f"\nDistrict-wide vote shares:")
for cand in CANDIDATES:
    print(f"  {cand:<16}: {district_pcts[cand]:.1f}%")
