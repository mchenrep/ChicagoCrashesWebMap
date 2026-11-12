"""Chicago Traffic Crash Explorer (v2, performance rewrite)
- GPU-rendered pydeck map; click a point to see its details
- Filters applied server-side (SoQL) behind an "Apply" button
- Trends aggregated on the server over ALL matching crashes
- Injury-risk model trained on pre-aggregated counts (small download)
"""
import datetime as dt

import pandas as pd
import pydeck as pdk
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from urllib3.util.retry import Retry

API = "https://data.cityofchicago.org/resource/85ca-t3if.json"
try:
    APP_TOKEN = st.secrets["SOCRATA_APP_TOKEN"]
except Exception:
    APP_TOKEN = None

WEATHER = ["CLEAR", "RAIN", "SNOW", "CLOUDY/OVERCAST", "FOG/SMOKE/HAZE",
           "FREEZING RAIN/DRIZZLE", "SLEET/HAIL", "UNKNOWN"]
LIGHTING = ["DAYLIGHT", "DARKNESS, LIGHTED ROAD", "DARKNESS", "DUSK", "DAWN"]
SURFACE = ["DRY", "WET", "SNOW OR SLUSH", "ICE"]
SEVERITY = ["NO INDICATION OF INJURY", "REPORTED, NOT EVIDENT", "NONINCAPACITATING INJURY",
            "INCAPACITATING INJURY", "FATAL"]
COLORS = {"NO INDICATION OF INJURY": [46, 160, 67], "REPORTED, NOT EVIDENT": [52, 120, 246],
          "NONINCAPACITATING INJURY": [245, 158, 11], "INCAPACITATING INJURY": [220, 38, 38],
          "FATAL": [20, 20, 20]}
DAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]  # 1 = Sunday

FULL_FIELDS = ("crash_record_id,crash_date,posted_speed_limit,weather_condition,lighting_condition,"
               "roadway_surface_cond,first_crash_type,prim_contributory_cause,most_severe_injury,"
               "injuries_total,injuries_fatal,num_units,hit_and_run_i,street_no,street_direction,"
               "street_name,latitude,longitude")


# ----------------------------- data access ----------------------------------
@st.cache_resource
def _session():
    s = requests.Session()
    retry = Retry(total=4, backoff_factor=1.5, status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["GET"])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    if APP_TOKEN:
        s.headers["X-App-Token"] = APP_TOKEN
    return s


def _get(params, timeout=60):
    r = _session().get(API, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _in(col, values):
    return f"{col} in(" + ",".join(f"'{v}'" for v in values) + ")"


def build_where(d0, d1, hours, weather, severity, hit_run):
    c = ["latitude IS NOT NULL", "latitude > 41", "longitude IS NOT NULL",
         f"crash_date between '{d0}T00:00:00' and '{d1}T23:59:59'",
         f"crash_hour between {hours[0]} and {hours[1]}"]
    if weather:
        c.append(_in("weather_condition", weather))
    if severity:
        c.append(_in("most_severe_injury", severity))
    if hit_run:
        c.append("hit_and_run_i = 'Y'")
    return " AND ".join(c)


@st.cache_data(ttl=3600, show_spinner=False)
def count_crashes(where):
    try:
        return int(_get({"$select": "count(*)", "$where": where})[0]["count"])
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def load_points(where, limit, light):
    """Fetch in pages of 2,500 so no single request is huge."""
    select = "latitude,longitude" if light else FULL_FIELDS
    rows, offset, page = [], 0, 2500
    while offset < limit:
        chunk = _get({"$select": select, "$where": where, "$order": "crash_date DESC",
                      "$limit": min(page, limit - offset), "$offset": offset})
        rows += chunk
        if len(chunk) < page:
            break
        offset += page
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if not light:
        for c in FULL_FIELDS.split(","):
            if c not in df:
                df[c] = None
    for c in ["latitude", "longitude", "injuries_total", "injuries_fatal", "num_units",
              "posted_speed_limit"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if not light:
        df["when"] = pd.to_datetime(df["crash_date"]).dt.strftime("%a %b %d, %Y %I:%M %p")
        df["address"] = (df[["street_no", "street_direction", "street_name"]].fillna("")
                         .astype(str).agg(" ".join, axis=1).str.strip())
        df["color"] = df["most_severe_injury"].map(COLORS).apply(
            lambda c: c if isinstance(c, list) else [128, 128, 128])
        df = df.drop(columns=["crash_date", "street_no", "street_direction", "street_name"])
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def trend(where, col, top):
    rows = _get({"$select": f"{col}, count(*) AS n", "$where": where + f" AND {col} IS NOT NULL",
                 "$group": col, "$order": "n DESC", "$limit": top})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["n"] = pd.to_numeric(out["n"])
    return out


@st.cache_resource(show_spinner="Training model (one-time, ~10-20 s)...")
def train_model():
    since = (dt.date.today() - dt.timedelta(days=3 * 365)).isoformat()
    grp = ["crash_hour", "crash_day_of_week", "posted_speed_limit",
           "weather_condition", "lighting_condition", "roadway_surface_cond"]
    base = f"crash_date > '{since}T00:00:00'"
    sel = ",".join(grp)

    def grouped(extra=""):
        return pd.DataFrame(_get({"$select": sel + ",count(*) AS n", "$where": base + extra,
                                  "$group": sel, "$limit": 50000}, timeout=120))

    tot, inj = grouped(), grouped(" AND injuries_total > 0")
    tot = tot.dropna(subset=grp)
    df = tot.merge(inj.rename(columns={"n": "inj"}), on=grp, how="left")
    df["inj"] = df["inj"].fillna(0)
    for c in ["crash_hour", "crash_day_of_week", "posted_speed_limit", "n", "inj"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()
    cat = ["weather_condition", "lighting_condition", "roadway_surface_cond"]
    num = ["crash_hour", "crash_day_of_week", "posted_speed_limit"]

    def expand(d):
        pos = d[grp].assign(y=1, w=d["inj"])
        neg = d[grp].assign(y=0, w=d["n"] - d["inj"])
        e = pd.concat([pos, neg])
        return e[e["w"] > 0]

    tr, te = train_test_split(df, test_size=0.2, random_state=42)
    tr, te = expand(tr), expand(te)
    pre = ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)],
                            remainder="passthrough")
    model = Pipeline([("pre", pre), ("gb", HistGradientBoostingClassifier(max_depth=4, max_iter=150))])
    model.fit(tr[num + cat], tr["y"], gb__sample_weight=tr["w"])
    auc = roc_auc_score(te["y"], model.predict_proba(te[num + cat])[:, 1], sample_weight=te["w"])
    base_rate = df["inj"].sum() / df["n"].sum()
    return model, auc, base_rate, int(df["n"].sum())


def explain_error(e):
    """Translate an exception into (title, message, tips)."""
    if isinstance(e, requests.exceptions.HTTPError) and e.response is not None:
        code = e.response.status_code
        if code == 429:
            return ("Too many requests",
                    "The Chicago Data Portal is rate-limiting this app.",
                    ["Wait about 30 seconds, then press Retry",
                     "Lower **Max points** in the sidebar",
                     "Add a free Socrata app token (see README) for higher limits"])
        if code in (400, 414):
            return ("That filter combination isn't supported",
                    "The portal couldn't run this query as written.",
                    ["Remove one or more filters (weather, severity)",
                     "Use a shorter date range"])
        if code in (401, 403):
            return ("Access denied", "The portal rejected the request, possibly a bad app token.",
                    ["Remove or replace SOCRATA_APP_TOKEN in secrets.toml"])
        if code >= 500:
            return ("The data portal had a problem",
                    "This often happens when a query is too expensive to run without an index.",
                    ["Narrow the date range", "Remove a filter or lower **Max points**",
                     "Wait a minute and press Retry"])
    if isinstance(e, requests.exceptions.Timeout):
        return ("The query timed out",
                "This filter combination is too heavy for the portal to answer quickly.",
                ["Narrow the date range (e.g. 30 days)", "Remove a filter", "Lower **Max points**"])
    if isinstance(e, requests.exceptions.ConnectionError):
        return ("Can't reach the data portal", "Your connection to data.cityofchicago.org failed.",
                ["Check your internet connection", "Press Retry"])
    if isinstance(e, (KeyError, ValueError, IndexError)):
        return ("Unexpected data", "The portal returned data in a shape the app didn't expect.",
                ["Change the filters and press Apply", "Press Retry"])
    return ("Something went wrong", "An unexpected error occurred.", ["Press Retry", "Change the filters"])


def error_screen(e, context):
    title, msg, tips = explain_error(e)
    with st.container():
        st.error(f"{title}. {msg}")
        st.markdown("Suggestions:\n" + "\n".join(f"- {t}" for t in tips))
        if st.button("Retry", key=f"retry_{context}"):
            st.cache_data.clear()
            st.rerun()
        with st.expander("Technical details"):
            st.code(f"{type(e).__name__}: {e}")


def empty_screen():
    with st.container():
        st.info("No crashes match these filters.")
        st.markdown("Suggestions:\n- Widen the date range\n- Clear the weather or severity filters\n"
                    "- Expand the hour-of-day range")


# --------------------------------- UI ---------------------------------------
st.set_page_config(page_title="Chicago Crash Explorer", layout="wide")
st.title("Chicago Traffic Crash Explorer")
st.caption("Live data from the City of Chicago Data Portal. Click a point on the map for crash details.")

today = dt.date.today()
with st.sidebar:
    with st.form("filters"):
        st.header("Filters")
        d_range = st.date_input("Date range", (today - dt.timedelta(days=60), today),
                                min_value=dt.date(2015, 9, 1), max_value=today)
        hours = st.slider("Hour of day", 0, 23, (0, 23))
        weather = st.multiselect("Weather", WEATHER)
        severity = st.multiselect("Most severe injury", SEVERITY)
        hit_run = st.checkbox("Hit & run only")
        mode = st.radio("Map style", ["Points (clickable)", "Heatmap"], horizontal=True)
        limit = st.select_slider("Max points", [1000, 2500, 5000, 10000, 20000], value=2500,
                                 help="Most recent crashes first. Heatmap mode can use more points.")
        st.form_submit_button("Apply filters", use_container_width=True)

if len(d_range) != 2:
    st.warning("Pick both a start and end date, then press Apply.")
    st.stop()

where = build_where(d_range[0], d_range[1], hours, weather, severity, hit_run)
heat = mode == "Heatmap"
try:
    with st.spinner("Loading crashes..."):
        df = load_points(where, limit, light=heat)
except Exception as e:
    error_screen(e, "map")
    st.stop()
total = count_crashes(where)

tab_map, tab_trends, tab_model = st.tabs(["Map", "Trends", "Injury risk predictor"])

with tab_map:
    if df.empty:
        empty_screen()
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Matching crashes", f"{total:,}" if total is not None else "n/a")
        c2.metric("Shown on map", f"{len(df):,}")
        if not heat:
            c3.metric("Shown with injuries", f"{int((df['injuries_total'] > 0).sum()):,}")
        if total and total > len(df):
            st.info(f"Showing the {len(df):,} most recent of {total:,} matches.")

        view = pdk.ViewState(latitude=41.8781, longitude=-87.6298, zoom=10.3)
        if heat:
            layer = pdk.Layer("HeatmapLayer", data=df[["latitude", "longitude"]],
                              get_position="[longitude, latitude]", radius_pixels=28, opacity=0.8)
            tooltip = None
        else:
            keep = ["longitude", "latitude", "color", "first_crash_type", "most_severe_injury", "when",
                    "address", "injuries_total", "injuries_fatal", "num_units", "prim_contributory_cause",
                    "weather_condition", "lighting_condition", "roadway_surface_cond",
                    "posted_speed_limit", "hit_and_run_i"]
            layer = pdk.Layer("ScatterplotLayer", id="crashes", data=df[keep],
                              get_position="[longitude, latitude]", get_fill_color="color",
                              get_radius=35, radius_min_pixels=3, radius_max_pixels=10,
                              pickable=True, auto_highlight=True, stroked=False)
            tooltip = {"text": "{address}\n{first_crash_type}\n{most_severe_injury}"}
        deck = pdk.Deck(layers=[layer], initial_view_state=view, map_style="light", tooltip=tooltip)
        event = st.pydeck_chart(deck, height=560, on_select="ignore" if heat else "rerun",
                                selection_mode="single-object", key="crash_map")

        if not heat:
            st.markdown(" &nbsp; ".join(
                f"<span style='display:inline-block;width:10px;height:10px;border-radius:50%;"
                f"background:rgb({c[0]},{c[1]},{c[2]});margin-right:4px'></span>{k.capitalize()}"
                for k, c in COLORS.items()), unsafe_allow_html=True)
            sel = getattr(event, "selection", None) or {}
            picked = (sel.get("objects", {}) or {}).get("crashes", [])
            if picked:
                r = picked[0]
                with st.container(border=True):
                    st.subheader(r.get("address") or "Unknown location")
                    st.caption(r.get("when", ""))
                    a, b, c = st.columns(3)
                    a.markdown(f"**Crash type**  \n{r.get('first_crash_type', '')}")
                    a.markdown(f"**Severity**  \n{r.get('most_severe_injury', '')}")
                    b.markdown(f"**Injuries**  \n{r.get('injuries_total', 0)} "
                               f"(fatal: {r.get('injuries_fatal', 0)})")
                    b.markdown(f"**Vehicles**  \n{r.get('num_units', '')}")
                    c.markdown(f"**Weather / light**  \n{r.get('weather_condition', '')} / "
                               f"{r.get('lighting_condition', '')}")
                    c.markdown(f"**Road**  \n{r.get('roadway_surface_cond', '')}, "
                               f"limit {r.get('posted_speed_limit', '')} mph")
                    st.markdown(f"**Primary cause:** {r.get('prim_contributory_cause', '')}  "
                                f"| **Hit & run:** {r.get('hit_and_run_i') or 'N'}")
            else:
                st.caption("Click a point to see its details here.")
            with st.expander("View / download loaded data"):
                show = df.drop(columns=["color", "crash_record_id"], errors="ignore")
                st.dataframe(show, use_container_width=True)
                st.download_button("Download CSV", show.to_csv(index=False), "chicago_crashes.csv")

with tab_trends:
    st.caption("Computed on the server across ALL matching crashes, not just the points on the map.")

    def bar(container, title, frame, label, order=None):
        container.subheader(title)
        if frame.empty:
            container.caption("No data for these filters.")
            return
        s = frame.set_index(label)["n"]
        if order is not None:
            s = s.reindex(order).fillna(0)
        container.bar_chart(s)

    if df.empty:
        empty_screen()
    else:
        try:
            a, b = st.columns(2)
            h = trend(where, "crash_hour", 24)
            if not h.empty:
                h["crash_hour"] = pd.to_numeric(h["crash_hour"]).astype(int)
                h = h.sort_values("crash_hour")
            bar(a, "By hour of day", h, "crash_hour")
            d = trend(where, "crash_day_of_week", 7)
            if not d.empty:
                d["day"] = d["crash_day_of_week"].astype(int).map(lambda i: DAYS[i - 1])
            bar(b, "By day of week", d, "day", order=DAYS)
            bar(a, "Top contributory causes", trend(where, "prim_contributory_cause", 8),
                "prim_contributory_cause")
            bar(b, "Crash types", trend(where, "first_crash_type", 8), "first_crash_type")
        except Exception as e:
            error_screen(e, "trends")

with tab_model:
    st.subheader("Probability a crash involves an injury")
    st.caption("Given that a crash happens, how likely is someone to be hurt under these conditions? "
               "Gradient-boosted model trained on ~3 years of Chicago crashes (aggregated on the server).")
    if st.button("Load / train model") or st.session_state.get("model_ready"):
        st.session_state["model_ready"] = True
        try:
            model, auc, base_rate, n = train_model()
        except Exception as e:
            st.session_state["model_ready"] = False
            error_screen(e, "model")
            st.stop()
        c1, c2, c3 = st.columns(3)
        hour = c1.slider("Hour", 0, 23, 17)
        day = c1.selectbox("Day of week", DAYS, index=5)
        wx = c2.selectbox("Weather", WEATHER[:-1])
        light = c2.selectbox("Lighting", LIGHTING)
        surf = c3.selectbox("Road surface", SURFACE)
        speed = c3.slider("Posted speed limit (mph)", 5, 70, 30, step=5)
        X = pd.DataFrame([{"crash_hour": hour, "crash_day_of_week": DAYS.index(day) + 1,
                           "posted_speed_limit": speed, "weather_condition": wx,
                           "lighting_condition": light, "roadway_surface_cond": surf}])
        p = float(model.predict_proba(X)[0, 1])
        st.metric("Predicted injury probability", f"{p:.1%}",
                  delta=f"{p - base_rate:+.1%} vs. average ({base_rate:.1%})", delta_color="inverse")
        st.progress(min(p, 1.0))
        st.caption(f"Held-out ROC-AUC: {auc:.2f} on {n:,} crashes. Conditions alone have modest "
                   "predictive power; treat this as a risk indicator, not a forecast.")
    else:
        st.info("Click **Load / train model** to start (takes a few seconds the first time).")
