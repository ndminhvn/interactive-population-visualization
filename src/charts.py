from typing import List, Dict, Any
import pandas as pd
import altair as alt
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# ----------------------------------------------------------------------
# Simple line chart for population over years
# ----------------------------------------------------------------------
def population_trend_chart(
    df: pd.DataFrame,
    csv_row: int,
    year_cols: List[str],
    county_name: str,
):
    """
    Generate an Altair chart for the chosen county's population trend
    (2010–2019). We return an Altair Chart object so trame-vega can call
    .to_dict() internally.
    """
    row = df.loc[csv_row, year_cols]

    # Build a small DataFrame for Altair
    data = pd.DataFrame(
        {
            "year": [int(y) for y in year_cols],
            "population": [int(row[y]) for y in year_cols],
        }
    )

    chart = (
        alt.Chart(data)
        .mark_line(point=True)
        .encode(
            x=alt.X("year:O", title="Year"),
            y=alt.Y(
                "population:Q",
                title="Population",
                scale=alt.Scale(nice=True),  # <— the magic line
            ),
        )
        .properties(
            width=400,
            height=300,
            title=f"Population trend for {county_name}",
        )
    )

    return chart


# ----------------------------------------------------------------------
# Future: Trend + Forecast overlay (keep placeholder ready)
# ----------------------------------------------------------------------
def population_forecast_chart(
    df: pd.DataFrame,
    csv_row: int,
    year_cols: List[str],
    county_name: str,
    forecast_values: Dict[int, float],
) -> Dict[str, Any]:
    """
    (For later) Line chart with both historical + forecasted years.
    forecast_values: {2020: val, 2021: val, ...}
    """
    row = df.loc[csv_row, year_cols]
    data_hist = [
        {"year": int(year), "population": int(row[year]), "type": "Historical"}
        for year in year_cols
    ]
    data_fore = [
        {"year": year, "population": val, "type": "Forecast"}
        for year, val in forecast_values.items()
    ]

    data = data_hist + data_fore

    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "description": f"Population + forecast for {county_name}",
        "data": {"values": data},
        "encoding": {
            "x": {"field": "year", "type": "ordinal"},
            "y": {
                "field": "population",
                "type": "quantitative",
            },
            "color": {
                "field": "type",
                "type": "nominal",
                "scale": {"range": ["#1f77b4", "#ff7f0e"]},
                "title": "",
            },
        },
        "mark": {"type": "line", "point": True},
    }

    return spec


def mini_population_bar_chart(
    df: pd.DataFrame,
    csv_row: int,
    year_cols: List[str],
):
    """Small Altair bar chart comparing 2010 vs 2019 population."""
    pops = df.loc[csv_row, year_cols].astype(int)
    pop_2010 = int(pops.iloc[0])
    pop_2019 = int(pops.iloc[-1])

    data = pd.DataFrame(
        {
            "year": ["2010", "2019"],
            "population": [pop_2010, pop_2019],
        }
    )

    chart = (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("population:Q", title="Population"),
            y=alt.Y("year:N", title="", sort=["2010", "2019"]),
            color=alt.Color(
                "year:N",
                legend=None,
                scale=alt.Scale(range=["#2196f3", "#4caf50"]),
            ),
        )
        .properties(
            width=350,
            height=150,
            title="2010 vs 2019 Population",
        )
    )

    return chart


def top_similar_counties(df, csv_row, year_cols, top_k=5):
    """
    Compute top 5 most similar counties based on population trend (2010-2019).
    Similarity = Euclidean distance on standardized population vectors.
    """

    # Extract population matrix (N x 10)
    pop_matrix = df[year_cols].astype(int).values

    # Normalize (important!)
    scaler = StandardScaler()
    pop_norm = scaler.fit_transform(pop_matrix)

    # Vector for selected county
    target_vec = pop_norm[csv_row]

    # Compute distances
    dists = np.linalg.norm(pop_norm - target_vec, axis=1)

    # Exclude the selected county itself
    dists[csv_row] = np.inf

    # Get indices of top 5 similar counties
    top_idx = np.argsort(dists)[:top_k]

    results = [
        {"county": df.iloc[i]["County"], "distance": float(dists[i])} for i in top_idx
    ]

    return results


def growth_gauge_chart(df, csv_row, year_cols):
    pops = df.loc[csv_row, year_cols].astype(int)
    pop_2010 = pops.iloc[0]
    pop_2019 = pops.iloc[-1]

    pct_change = (pop_2019 - pop_2010) / pop_2010 * 100
    angle = max(min(pct_change, 100), -100) * 1.8  # clamp & convert to degrees

    # Background semicircle 0–180 degrees
    bg = pd.DataFrame([{"start": 0, "end": 180}])

    # Needle
    needle = pd.DataFrame([{"angle": angle}])

    # Preformatted label (string)
    label = pd.DataFrame([{"growth_label": f"{pct_change:+.2f}%"}])

    chart_bg = (
        alt.Chart(bg)
        .mark_arc(innerRadius=60, outerRadius=100, opacity=0.2, color="#90caf9")
        .encode(theta="end:Q")
    )

    chart_needle = (
        alt.Chart(needle)
        .mark_arc(innerRadius=0, outerRadius=70, color="#e53935")
        .encode(theta=alt.Theta("angle:Q", stack=None))
    )

    chart_text = (
        alt.Chart(label)
        .mark_text(
            align="center",
            baseline="middle",
            fontSize=22,
            fontWeight="bold",
            color="black",
        )
        .encode(text="growth_label:N")
    )

    return (chart_bg + chart_needle + chart_text).properties(
        width=300,
        height=200,
        title="10-Year Population Growth Gauge",
    )


# ----------------------------------------------------------
# Similar Counties Bar Chart (based on Euclidean distance)
# ----------------------------------------------------------
# ---------------------------------------------
# NEW: Bar Chart for Similar Counties
# ---------------------------------------------
def similar_counties_bar_chart(similar_list):
    df_sim = pd.DataFrame(similar_list)

    chart = (
        alt.Chart(df_sim)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("similarity:Q", title="Similarity (0–100)"),
            y=alt.Y("county:N", sort="-x", title="County"),
            color=alt.Color(
                "similarity:Q",
                scale=alt.Scale(scheme="blues"),
                legend=alt.Legend(title="Similarity"),
            ),
            tooltip=[
                alt.Tooltip("county:N"),
                alt.Tooltip("similarity:Q", format=".1f"),
            ],
        )
        .properties(
            title="Top 5 Similar Counties",
            width=400,
            height=230,
        )
    )

    return chart


def population_share_pie(df, csv_row, year_cols, county_name):
    latest_year = year_cols[-1]
    total = df[latest_year].astype(int).sum()
    county_pop = int(df.loc[csv_row, latest_year])
    share = county_pop / total * 100

    data = pd.DataFrame(
        {
            "category": ["This County", "Rest of Virginia"],
            "value": [county_pop, total - county_pop],
        }
    )

    chart = (
        alt.Chart(data)
        .mark_arc(outerRadius=140, innerRadius=40)
        .encode(
            theta="value:Q",
            color=alt.Color("category:N", scale=alt.Scale(scheme="blues")),
            tooltip=["category", "value"],
        )
        .properties(
            title=f"{county_name}: Share of VA Population ({latest_year})",
            width=350,
            height=300,
        )
    )
    return chart


def yearly_population_bar_chart(df, csv_row, year_cols, county_name):
    row = df.loc[csv_row, year_cols].astype(int)

    data = pd.DataFrame(
        {
            "year": [int(y) for y in year_cols],
            "population": [int(row[y]) for y in year_cols],
        }
    )

    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X("year:O", title="Year"),
            y=alt.Y("population:Q", title="Population"),
            tooltip=["year", "population"],
        )
        .properties(width=400, height=350, title=f"{county_name}: Population by Year")
    )

    return chart


def top_growth_leaderboard_chart(
    df: pd.DataFrame,
    year_cols: List[str],
    top_k: int = 5,
):
    """
    Clean Altair chart for Top K fastest-growing counties.
    Fixes:
    - No duplicate title
    - Proper X/Y axes
    - No overlapping labels
    - Uses layered bars + right-aligned value labels
    """

    import altair as alt

    alt.data_transformers.disable_max_rows()

    # Get start and end years
    col_start = year_cols[0]  # "2010"
    col_end = year_cols[-1]  # "2019"

    # Compute growth
    pop_start = df[col_start].astype(int)
    pop_end = df[col_end].astype(int)

    growth_abs = pop_end - pop_start
    growth_pct = ((growth_abs / pop_start.replace({0: pd.NA})) * 100).fillna(0)

    # Prepare data
    tmp = (
        pd.DataFrame(
            {
                "County": df["County"],
                "GrowthPct": growth_pct,
                "GrowthAbs": growth_abs,
            }
        )
        .sort_values("GrowthPct", ascending=False)
        .head(top_k)
    )

    # Base bars
    bars = (
        alt.Chart(tmp)
        .mark_bar()
        .encode(
            x=alt.X(
                "GrowthPct:Q",
                title="Growth (2010–2019, %)",
                axis=alt.Axis(format=".0f"),
            ),
            y=alt.Y("County:N", sort="-x", title="County"),
            color=alt.Color(
                "GrowthPct:Q",
                scale=alt.Scale(scheme="blues"),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("County:N"),
                alt.Tooltip("GrowthPct:Q", title="Growth %", format=".2f"),
                alt.Tooltip("GrowthAbs:Q", title="Population Change", format=","),
            ],
        )
        .properties(width=380, height=250)
    )

    # Value labels at end of bars
    labels = (
        alt.Chart(tmp)
        .mark_text(align="left", baseline="middle", dx=4)
        .encode(
            x="GrowthPct:Q",
            y="County:N",
            text=alt.Text("GrowthPct:Q", format=".1f"),
        )
    )

    # Combine + set title ONCE
    chart = (bars + labels).properties(
        title=alt.TitleParams(text="", fontSize=18, anchor="start")
    )

    return chart


def forecast_population_chart(county_name, df, year_cols):
    """
    Creates a future population forecast using linear regression.
    Predicts population for next 10 years.
    """

    import numpy as np
    import altair as alt

    alt.data_transformers.disable_max_rows()

    # Extract county row
    row = df[df["County"] == county_name].iloc[0]

    # Convert years & population to arrays
    years = np.array([int(y) for y in year_cols])
    pops = row[year_cols].astype(int).values

    # Fit simple linear regression
    coef = np.polyfit(years, pops, 1)
    poly = np.poly1d(coef)

    # Predict next 10 years
    future_years = np.arange(2020, 2031)
    future_preds = poly(future_years)

    # Build dataframes for altair
    df_hist = pd.DataFrame({"Year": years, "Population": pops, "Type": "Historical"})

    df_future = pd.DataFrame(
        {"Year": future_years, "Population": future_preds, "Type": "Forecast"}
    )

    df_all = pd.concat([df_hist, df_future])

    # Altair chart
    chart = (
        alt.Chart(df_all)
        .mark_line(point=True)
        .encode(
            x=alt.X("Year:O", title="Year"),
            y=alt.Y("Population:Q", title="Population"),
            color=alt.Color("Type:N", scale=alt.Scale(scheme="category10")),
            tooltip=["Year", "Population", "Type"],
        )
        .properties(
            width=380, height=300, title=f"Population Forecast for {county_name}"
        )
    )

    return chart


def growth_correlation_heatmap(df: pd.DataFrame, year_cols):
    """
    Computes year-over-year GROWTH correlation matrix.
    Example:
        growth_2010 = pop_2011 - pop_2010
        growth_2011 = pop_2012 - pop_2011
        ...
    """

    # --- Compute year-over-year growth ---
    growth_data = {}

    for i in range(len(year_cols) - 1):
        y1 = year_cols[i]
        y2 = year_cols[i + 1]
        growth_label = f"{y1}->{y2}"

        growth_data[growth_label] = df[y2].astype(int) - df[y1].astype(int)

    growth_df = pd.DataFrame(growth_data)

    # --- Compute correlation matrix ---
    corr_matrix = growth_df.corr()

    # Melt it for Altair
    melted = (
        corr_matrix.reset_index()
        .melt(id_vars="index", var_name="Year2", value_name="Correlation")
        .rename(columns={"index": "Year1"})
    )

    # --- Build heatmap ---
    chart = (
        alt.Chart(melted)
        .mark_rect()
        .encode(
            x=alt.X("Year1:N", title="Growth Period", sort=list(growth_data.keys())),
            y=alt.Y("Year2:N", title="Growth Period", sort=list(growth_data.keys())),
            color=alt.Color(
                "Correlation:Q",
                scale=alt.Scale(scheme="redblue", domain=(-1, 1)),
                title="Correlation",
            ),
            tooltip=[
                alt.Tooltip("Year1:N", title="Growth Period (X)"),
                alt.Tooltip("Year2:N", title="Growth Period (Y)"),
                alt.Tooltip("Correlation:Q", format=".3f"),
            ],
        )
        .properties(width=380, height=380, title="Correlation of Year-over-Year Growth")
    )

    return chart


def kmeans_cluster_plot(df, year_cols, k=4):
    # Extract population matrix
    X = df[year_cols].astype(int).values

    # Normalize
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)

    # Run K-means
    model = KMeans(n_clusters=k, random_state=42)
    clusters = model.fit_predict(X_norm)

    df_plot = df.copy()
    df_plot["Cluster"] = clusters.astype(str)

    # PCA for 2D plot
    pca = PCA(n_components=2)
    comps = pca.fit_transform(X_norm)

    df_plot["PC1"] = comps[:, 0]
    df_plot["PC2"] = comps[:, 1]

    chart = (
        alt.Chart(df_plot)
        .mark_circle(size=120)
        .encode(x="PC1:Q", y="PC2:Q", color="Cluster:N", tooltip=["County", "Cluster"])
        .properties(width=380, height=260, title=f"K-Means Clusters (k={k})")
    )

    return chart, clusters
