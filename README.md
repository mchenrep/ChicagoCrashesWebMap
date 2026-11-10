# Chicago Traffic Crash Explorer

Interactive Streamlit app built on the City of Chicago
[Traffic Crashes – Crashes](https://data.cityofchicago.org/Transportation/Traffic-Crashes-Crashes/85ca-t3if) dataset.

## Features
- **Map**: clickable, severity-colored markers (popup shows address, time, crash type, injuries, cause, weather, road, hit-and-run) or a heatmap.
- **Filters**: date range, hour of day, weather, injury severity, hit-and-run.
- **Trends**: crashes by hour, weekday, cause, and type for the current filter.
- **Injury-risk predictor**: gradient-boosting model (trained on ~3 years of crashes) returns the probability a crash involves an injury for the conditions you choose.
- **Export**: download the filtered data as CSV.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
Optional: add `SOCRATA_APP_TOKEN = "..."` to `.streamlit/secrets.toml` for higher API rate limits.

## Scalability notes
- Filters are translated to SoQL and executed server-side; only a capped number of rows (default 3,000) is transferred.
- A separate count query reports the total matches so users know when results are truncated.
- `st.cache_data` (1 hr TTL) caches queries; `st.cache_resource` trains the model once per process.
- Marker clustering keeps the browser responsive; the heatmap handles up to 20k points.

## Deploy
Push to GitHub and deploy free on Streamlit Community Cloud (main file: `app.py`).
