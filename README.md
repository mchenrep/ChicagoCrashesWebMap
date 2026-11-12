# Chicago Traffic Crash Explorer

An interactive web application for exploring traffic crashes in Chicago. Users can filter crashes on a map, click individual crashes to see their details, view citywide trends, and get a model-based estimate of how likely a crash is to involve an injury under chosen conditions.

**Live app:** [Click Here](https://chicagocrasheswebmap-fdgsjcxsyejizerzhrekyz.streamlit.app/)

## Overview

Traffic crash data is public but hard to use in raw form: the full table has several hundred thousand records and dozens of columns. This app queries the data live, lets users narrow it with a few filters, and presents it as a map, a set of summary charts, and a simple predictive tool.

## Data

- **Source:** City of Chicago Data Portal, [Traffic Crashes - Crashes](https://data.cityofchicago.org/Transportation/Traffic-Crashes-Crashes/85ca-t3if) (dataset id `85ca-t3if`), reported by the Chicago Police Department.
- **Access:** Socrata Open Data API (SoQL). No data files are stored in the repository; every view is built from live queries.
- **Fields used:** crash date and hour, location, crash type, most severe injury, injury counts, number of vehicles, primary contributory cause, weather, lighting, road surface, posted speed limit, and hit-and-run flag.
- **Limitations:** Records are police-reported, so minor crashes may be under-reported. Locations are entered by officers and can contain errors. Rows without coordinates are excluded from the map.

## Features

**Map**
- Points are colored by most severe injury (no indication of injury through fatal).
- Click a point to see its address, time, crash type, injuries, vehicles, primary cause, weather, lighting, road condition, speed limit, and hit-and-run status.
- A heatmap mode shows where crashes concentrate.
- Filters: date range, hour of day, weather, injury severity, hit-and-run only, and maximum number of points.
- Loaded data can be viewed in a table and downloaded as CSV.

**Trends**
- Crashes by hour of day, day of week, primary contributory cause, and crash type.
- Computed on the server across all crashes that match the filters, not only those drawn on the map.

**Injury risk predictor**
- The user chooses hour, day of week, weather, lighting, road surface, and posted speed limit.
- The app returns the estimated probability that a crash under those conditions involves at least one injury, compared with the overall average.

## Model

- **Task:** binary classification, injury vs. no injury, conditional on a crash having occurred. It does not predict whether a crash will happen.
- **Training data:** crashes from roughly the last three years, aggregated on the server by combination of the input features (counts of all crashes and of injury crashes per combination). This keeps the download to a few thousand rows instead of hundreds of thousands, and the counts are used as sample weights.
- **Model:** scikit-learn `HistGradientBoostingClassifier` with one-hot encoded categorical inputs.
- **Evaluation:** 80/20 split at the feature-combination level, scored with weighted ROC-AUC (shown in the app).
- **Interpretation:** weather, lighting, time, and speed limit carry only modest signal about injury outcomes, so the output should be read as a relative risk indicator rather than a forecast. Factors such as crash type, vehicle types, and occupant behavior are not included.

## Architecture and scalability

- **Server-side filtering.** All filters are translated to SoQL and run on the data portal, so the app never downloads the whole table.
- **Capped, paged loading.** Map points are fetched in pages of 2,500, newest first, up to a user-selected limit. A separate count query reports how many crashes match.
- **GPU-rendered map.** Points are drawn with pydeck (deck.gl), which handles thousands of points smoothly in the browser.
- **Deferred computation.** Filters are submitted together with an Apply button, so changing a control does not trigger a query. The model is trained only when requested.
- **Caching.** Query results are cached for one hour with `st.cache_data`; the model and HTTP session are cached with `st.cache_resource`.
- **Resilience.** Requests retry with backoff on rate limits and server errors. Failures show a descriptive error message with suggestions and a retry button, and filter combinations with no matches show an empty-results message.

## Running locally

Requires Python 3.10 or newer.

```bash
pip install -r requirements.txt
streamlit run app.py
```

If the `streamlit` command is not found, use `python -m streamlit run app.py`.

### Optional: API token

Unauthenticated requests to the data portal are throttled. A free app token raises the limit. Create `.streamlit/secrets.toml`:

```toml
SOCRATA_APP_TOKEN = "your-token"
```

Do not commit this file.

## Deployment

The app is deployed on Streamlit Community Cloud from this repository (main file `app.py`). The app token is stored in the platform's secrets settings.

## Project structure

```
app.py                  Streamlit application
requirements.txt        Python dependencies
.streamlit/config.toml  Theme and display settings
README.md               This file
```

## Possible extensions

- Community-area or ward summaries with a choropleth layer
- A location-level risk layer based on crash density
- Additional model inputs, such as crash type or vehicle count
- Comparison of two date ranges side by side
