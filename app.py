import io

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Crime Hotspot Analysis", layout="wide")

COORDS = {
    'Delhi': (28.61, 77.21), 'Mumbai': (19.08, 72.88), 'Bangalore': (12.97, 77.59),
    'Hyderabad': (17.39, 78.49), 'Kolkata': (22.57, 88.36), 'Chennai': (13.08, 80.27),
    'Pune': (18.52, 73.86), 'Ahmedabad': (23.02, 72.57), 'Jaipur': (26.91, 75.79),
    'Lucknow': (26.85, 80.95), 'Kanpur': (26.45, 80.35), 'Surat': (21.17, 72.83),
    'Nagpur': (21.15, 79.09), 'Agra': (27.18, 78.01), 'Ludhiana': (30.90, 75.86),
    'Visakhapatnam': (17.69, 83.22), 'Thane': (19.22, 72.98), 'Ghaziabad': (28.67, 77.42),
    'Indore': (22.72, 75.86), 'Patna': (25.59, 85.14), 'Bhopal': (23.26, 77.41),
    'Meerut': (28.98, 77.71), 'Srinagar': (34.08, 74.80), 'Nashik': (19.99, 73.79),
    'Vasai': (19.39, 72.84), 'Varanasi': (25.32, 83.01), 'Kalyan': (19.24, 73.13),
    'Faridabad': (28.41, 77.31), 'Rajkot': (22.30, 70.80),
}
PERIOD_HOURS = {'Morning': 6, 'Afternoon': 5, 'Evening': 4, 'Night': 9}
DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
WOMEN_CRIMES = ['Sexual Assault', 'Domestic Violence', 'Kidnapping',
                'Assault', 'Cybercrime', 'Extortion']


def period(h):
    if 5 <= h < 11: return 'Morning'
    if 11 <= h < 16: return 'Afternoon'
    if 16 <= h < 20: return 'Evening'
    return 'Night'


REQUIRED = ['city', 'crime_description', 'victim_gender', 'date_of_occurrence']


def parse_dt(series, fmt):
    """Parse with the expected format; fall back to flexible parsing for other datasets."""
    out = pd.to_datetime(series, format=fmt, errors='coerce')
    if out.isna().mean() > 0.5:
        out = pd.to_datetime(series, errors='coerce')
    return out


@st.cache_data
def load(src):
    df = pd.read_csv(io.BytesIO(src) if isinstance(src, bytes) else src)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError("Missing required column(s): " + ", ".join(missing))
    if 'report_number' not in df.columns:
        df['report_number'] = np.arange(1, len(df) + 1)
    df['city'] = df['city'].astype(str).str.strip().str.title()
    df['crime_description'] = df['crime_description'].astype(str).str.strip().str.title()
    df['victim_gender'] = df['victim_gender'].astype(str).str.strip()
    df = df.drop_duplicates()
    df['date'] = parse_dt(df['date_of_occurrence'], '%m-%d-%Y %H:%M')
    if 'time_of_occurrence' in df.columns:
        df['hour'] = parse_dt(df['time_of_occurrence'], '%d-%m-%Y %H:%M').dt.hour
    else:
        df['hour'] = df['date'].dt.hour
    before = len(df)
    df = df.dropna(subset=['date', 'hour']).copy()
    df.attrs['dropped'] = before - len(df)
    df['hour'] = df['hour'].astype(int)
    df['year'] = df.date.dt.year
    df['day'] = df.date.dt.day_name()
    df['day_type'] = np.where(df.date.dt.dayofweek >= 5, 'Weekend', 'Weekday')
    df['time_period'] = df['hour'].apply(period)
    # coordinates: use latitude/longitude columns if the dataset has them, else city centres
    lat_col = next((c for c in ('latitude', 'lat') if c in df.columns), None)
    lon_col = next((c for c in ('longitude', 'lon', 'lng') if c in df.columns), None)
    if lat_col and lon_col:
        df['lat'] = pd.to_numeric(df[lat_col], errors='coerce')
        df['lon'] = pd.to_numeric(df[lon_col], errors='coerce')
    else:
        df['lat'] = df.city.map(lambda c: COORDS.get(c, (np.nan, np.nan))[0])
        df['lon'] = df.city.map(lambda c: COORDS.get(c, (np.nan, np.nan))[1])
    return df


def style_geo(fig, lats, lons):
    """India view for Indian data; otherwise zoom to wherever the points are."""
    if lats.between(6, 38).all() and lons.between(66, 99).all():
        fig.update_geos(scope='asia', center=dict(lat=22.5, lon=80), projection_scale=4.2,
                        showcountries=True)
    else:
        fig.update_geos(fitbounds='locations', showcountries=True, showland=True)


def score_levels(s):
    score = s / s.max() * 100
    level = pd.cut(score, [-1, 30, 60, 80, 100],
                   labels=['Low', 'Medium', 'High', 'Very High'])
    return score, level


# ---------- data ----------
st.title("Crime Hotspot Analysis: Crimes Against Women")
st.caption("Historical pattern analysis on the Indian Crimes Dataset (city level). "
           "Shows where incidents were recorded, not that any area is inherently unsafe.")

# ---------- data source ----------
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload your own CSV (optional)", type="csv")
with st.sidebar.expander("Required columns"):
    st.markdown(
        "- **city**\n- **crime_description**\n- **victim_gender**\n"
        "- **date_of_occurrence**\n\nOptional: time_of_occurrence, report_number, "
        "latitude + longitude. Column names are not case-sensitive and spaces become underscores."
    )

try:
    if uploaded is not None:
        raw = load(uploaded.getvalue())
        source_name = uploaded.name
    else:
        raw = load("crime_dataset_india.csv")
        source_name = "crime_dataset_india.csv (default)"
except FileNotFoundError:
    st.info("Default dataset not found. Upload a CSV in the sidebar to begin.")
    st.stop()
except ValueError as e:
    st.error(f"Could not use this file. {e}")
    st.stop()
except Exception as e:
    st.error(f"Could not read this file: {e}")
    st.stop()

if raw.empty:
    st.error("The file has no usable rows after cleaning.")
    st.stop()

st.sidebar.success(f"Using: {source_name} ({len(raw):,} rows)")
if raw.attrs.get('dropped'):
    st.sidebar.warning(f"{raw.attrs['dropped']:,} rows dropped (unreadable date/time).")

# ---------- sidebar filters ----------
st.sidebar.header("Filters")
all_types = sorted(raw.crime_description.unique())
default_types = [t for t in WOMEN_CRIMES if t in all_types] or all_types
types = st.sidebar.multiselect("Crime types", all_types, default=default_types)
genders = sorted(raw.victim_gender.unique())
gender_opts = genders + ['All']
gender = st.sidebar.radio("Victim gender", gender_opts,
                          index=gender_opts.index('F') if 'F' in gender_opts else len(gender_opts) - 1)
years = st.sidebar.multiselect("Year", sorted(raw.year.unique()),
                               default=sorted(raw.year.unique()))
cities = st.sidebar.multiselect("City", sorted(raw.city.unique()))
periods = st.sidebar.multiselect("Time period", list(PERIOD_HOURS),
                                 default=list(PERIOD_HOURS))

d = raw[raw.crime_description.isin(types) & raw.year.isin(years)
        & raw.time_period.isin(periods)]
if gender != 'All':
    d = d[d.victim_gender == gender]
if cities:
    d = d[d.city.isin(cities)]

if d.empty:
    st.warning("No records match the filters.")
    st.stop()

city_tbl = d.groupby('city').size().rename('incidents').reset_index()
city_tbl['score'], city_tbl['level'] = score_levels(city_tbl.incidents)
city_tbl = city_tbl.sort_values('incidents', ascending=False)
rate = d.groupby('time_period').size().reindex(list(PERIOD_HOURS)).fillna(0) / pd.Series(PERIOD_HOURS)

# ---------- tabs ----------
t1, t2, t3, t4 = st.tabs(["Overview", "Hotspots", "Time analysis", "Recommendations"])

with t1:
    c = st.columns(5)
    c[0].metric("Total incidents", f"{len(d):,}")
    c[1].metric("Top city", city_tbl.city.iloc[0])
    c[2].metric("Most common crime", d.crime_description.mode().iloc[0])
    c[3].metric("Peak hour", f"{int(d.hour.mode().iloc[0]):02d}:00")
    c[4].metric("Weekend share", f"{(d.day_type == 'Weekend').mean():.0%}")
    a, b = st.columns(2)
    yr = d.groupby('year').size().reset_index(name='incidents')
    a.plotly_chart(px.line(yr, x='year', y='incidents', markers=True,
                           title="Yearly trend (2024 is partial)"), width='stretch')
    ct = d.crime_description.value_counts().reset_index()
    b.plotly_chart(px.bar(ct, x='count', y='crime_description', orientation='h',
                          title="Crime types"), width='stretch')
    st.plotly_chart(px.bar(city_tbl.head(10), x='city', y='incidents',
                           title="Top 10 cities"), width='stretch')

with t2:
    mode = st.radio("Map view", ["City bubbles", "Geographic grid"], horizontal=True)
    if mode == "City bubbles":
        m = city_tbl.merge(d.groupby('city')[['lat', 'lon']].mean().reset_index(), on='city')
        no_coord = m[m.lat.isna() | m.lon.isna()].city.tolist()
        m = m.dropna(subset=['lat', 'lon'])
        if no_coord:
            st.warning("No coordinates for: " + ", ".join(no_coord) +
                       ". Add latitude and longitude columns to your CSV to map them.")
        if m.empty:
            st.info("No coordinates available, so the map cannot be drawn. See the table below.")
        if not m.empty:
            fig = px.scatter_geo(m, lat='lat', lon='lon', size='incidents', color='level',
                                 hover_name='city', hover_data=['incidents', 'score'],
                                 color_discrete_map={'Low': 'green', 'Medium': 'orange',
                                                     'High': 'red', 'Very High': 'darkred'},
                                 category_orders={'level': ['Low', 'Medium', 'High', 'Very High']})
            style_geo(fig, m.lat, m.lon)
            fig.update_layout(height=600, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, width='stretch')
    else:
        cell = st.slider("Grid cell size (degrees)", 1.0, 4.0, 2.0, 0.5)
        dg = d.dropna(subset=['lat', 'lon'])
        if dg.empty:
            st.info("No coordinates available, so the grid cannot be drawn.")
        else:
            g = dg.assign(gx=np.floor(dg.lon / cell) * cell, gy=np.floor(dg.lat / cell) * cell)
            grid = (g.groupby(['gy', 'gx'])
                      .agg(incidents=('report_number', 'count'),
                           cities=('city', lambda s: ', '.join(sorted(s.unique()))))
                      .reset_index())
            grid['score'], grid['level'] = score_levels(grid.incidents)
            grid['cx'] = grid.gx + cell / 2
            grid['cy'] = grid.gy + cell / 2
            fig = px.scatter_geo(grid, lat='cy', lon='cx', size='incidents', color='score',
                                 color_continuous_scale='YlOrRd', hover_name='cities',
                                 hover_data=['incidents', 'level'])
            fig.update_traces(marker=dict(symbol='square', sizemin=8))
            style_geo(fig, grid.cy, grid.cx)
            fig.update_layout(height=600, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, width='stretch')
            st.dataframe(grid.sort_values('score', ascending=False)
                         .drop(columns=['cx', 'cy']), width='stretch')
    st.subheader("Hotspot score by city")
    st.caption("Score = city incidents / max city incidents x 100. "
               "Low 0-30, Medium 31-60, High 61-80, Very High 81-100.")
    st.dataframe(city_tbl, width='stretch')

with t3:
    pv = d.pivot_table(index='day', columns='hour', values='report_number',
                       aggfunc='count', fill_value=0).reindex(DAYS)
    st.plotly_chart(px.imshow(pv, aspect='auto', color_continuous_scale='YlOrRd',
                              title="Day x Hour heatmap", labels=dict(color="Incidents")),
                    width='stretch')
    a, b = st.columns(2)
    a.plotly_chart(px.bar(rate.reset_index(name='per_hour').rename(columns={'index': 'period'}),
                          x='time_period', y='per_hour',
                          title="Incidents per hour of each period"), width='stretch')
    ct = pd.crosstab(d.city, d.time_period).reindex(columns=list(PERIOD_HOURS), fill_value=0)
    top = city_tbl.city.head(10)
    b.plotly_chart(px.imshow((ct / pd.Series(PERIOD_HOURS)).loc[top], text_auto='.1f',
                             color_continuous_scale='Reds',
                             title="City x period (incidents per hour)"),
                   width='stretch')
    st.caption("Night covers 9 hours and evening 4, so raw counts would overstate night. "
               "Rates per hour give a fair comparison.")

with t4:
    st.subheader("Evidence-based planning considerations")
    for _, r in city_tbl.head(5).iterrows():
        sub = d[d.city == r.city]
        r_p = (sub.groupby('time_period').size().reindex(list(PERIOD_HOURS)).fillna(0)
               / pd.Series(PERIOD_HOURS)).idxmax()
        top_crime = sub.crime_description.mode().iloc[0]
        st.markdown(
            f"**{r.city}** ({r.level}, score {r.score:.0f}): {r.incidents} incidents, "
            f"highest hourly rate in the **{r_p.lower()}**, most common type **{top_crime}**.\n"
            f"- Review lighting and visibility in the {r_p.lower()} hours\n"
            f"- Evaluate CCTV and patrol allocation in high-incidence areas\n"
            f"- Review transport access and emergency assistance points"
        )
    st.info("These are patterns in historical records. They are not proof that a place "
            "is inherently unsafe. The data is city level and appears synthetic, so treat "
            "results as a demonstration of the method.")
