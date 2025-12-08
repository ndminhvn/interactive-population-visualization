import argparse
import vtk
import pandas as pd
import altair as alt
import numpy as np
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vtk as vtk_widgets
from trame.widgets import vuetify3, vega, html
from sklearn.preprocessing import StandardScaler

from data_loader import load_all_data
from map_utils import (
    create_virginia_actor,
    create_virginia_renderer,
    highlight_county,
)
from normalize_name import normalize_name
from charts import (
    population_trend_chart,
    # mini_population_bar_chart,
    growth_gauge_chart,
    top_similar_counties,
    similar_counties_bar_chart,
    population_share_pie,
    yearly_population_bar_chart,
    top_growth_leaderboard_chart,
    forecast_population_chart,
    growth_correlation_heatmap,
)


# -----------------------------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------------------------
APP_DATA = load_all_data()
DF = APP_DATA["df"]
YEAR_COLS = APP_DATA["year_cols"]
POLY = APP_DATA["poly"]
COUNTY_INDEX = APP_DATA["county_index"]
COUNTY_LIST = APP_DATA["county_list"]


# -----------------------------------------------------------------------------
# VTK PIPELINE
# -----------------------------------------------------------------------------
actor, mapper, color_array = create_virginia_actor(POLY)
renderer = create_virginia_renderer(actor)

render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)
render_window.SetSize(900, 700)

# Attach interactor so trame-vtk can render
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(render_window)


# -----------------------------------------------------------------------------
# TRAME SERVER & STATE
# -----------------------------------------------------------------------------
server = get_server()
state, ctrl = server.state, server.controller

state.county_stats = ""
state.mini_chart = {}
state.selected_info_tab = "trend"

# Compare-mode state
state.compare_mode = False
state.compare_county_a = COUNTY_LIST[0]
state.compare_county_b = COUNTY_LIST[1]

# Main selection
state.selected_county = COUNTY_LIST[0]
state.county_items = [{"text": c, "value": c} for c in COUNTY_LIST]

# (Currently unused, but kept for compatibility)
state.main_panel_tab = "default"

state.cluster_k = 4
state.cluster_assignments = []


# -----------------------------------------------------------------------------
# COUNTY-LEVEL STATISTICS
# -----------------------------------------------------------------------------
def compute_county_stats(df, csv_row, year_cols):
    """Return a formatted multiline string of county stats."""
    pops = df.loc[csv_row, year_cols].astype(int)
    pop_2010 = pops.iloc[0]
    pop_2019 = pops.iloc[-1]

    abs_change = pop_2019 - pop_2010
    pct_change = (abs_change / pop_2010) * 100

    # Rank by final population
    df["Pop2019"] = df[year_cols[-1]].astype(int)
    df["GrowthPct"] = (df[year_cols[-1]] - df[year_cols[0]]) / df[year_cols[0]] * 100

    pop_rank = df["Pop2019"].rank(ascending=False).astype(int)[csv_row]
    growth_rank = df["GrowthPct"].rank(ascending=False).astype(int)[csv_row]

    text = (
        f"2010 Population: {pop_2010:,}\n"
        f"2019 Population: {pop_2019:,}\n"
        f"10-year Change: {abs_change:+,} ({pct_change:+.2f}%)\n"
        f"Population Rank: {pop_rank} / {len(df)}\n"
        f"Growth Rank: {growth_rank} / {len(df)}"
    )

    return text


# -----------------------------------------------------------------------------
# (Optional override) Improved Similar Counties (0–100 scaled)
# -----------------------------------------------------------------------------
def top_similar_counties(df, csv_row, year_cols, top_k=5):
    pop_matrix = df[year_cols].astype(int).values

    scaler = StandardScaler()
    pop_norm = scaler.fit_transform(pop_matrix)

    target_vec = pop_norm[csv_row]

    dists = np.linalg.norm(pop_norm - target_vec, axis=1)
    dists[csv_row] = np.inf  # Ignore itself

    max_dist = np.nanmax(dists[np.isfinite(dists)])
    similarity_raw = max_dist - dists

    top_idx = np.argsort(dists)[:top_k]
    sim_vals = similarity_raw[top_idx]

    sim_min, sim_max = sim_vals.min(), sim_vals.max()

    if sim_max > sim_min:
        sim_scaled = ((sim_vals - sim_min) / (sim_max - sim_min)) * 100
    else:
        sim_scaled = np.ones_like(sim_vals) * 50

    sim_scaled = np.clip(sim_scaled, 5, 100)

    results = [
        {
            "county": df.iloc[i]["County"],
            "similarity": float(sim_scaled[j]),
        }
        for j, i in enumerate(top_idx)
    ]

    return results


# -----------------------------------------------------------------------------
# TWO-COUNTY COMPARISON CHART
# -----------------------------------------------------------------------------
def compare_two_counties_chart(df, county_a, county_b, year_cols):
    row_a = df[df["County"] == county_a].index[0]
    row_b = df[df["County"] == county_b].index[0]

    pops_a = df.loc[row_a, year_cols].astype(int)
    pops_b = df.loc[row_b, year_cols].astype(int)

    chart_df = pd.DataFrame(
        {
            "Year": year_cols,
            county_a: pops_a.values,
            county_b: pops_b.values,
        }
    )

    melted = chart_df.melt("Year", var_name="County", value_name="Population")

    chart = (
        alt.Chart(melted)
        .mark_line(point=True)
        .encode(
            x=alt.X("Year:N", title="Year"),
            y=alt.Y("Population:Q", title="Population"),
            color="County:N",
            tooltip=["Year", "County", "Population"],
        )
        .properties(
            width=380,
            height=260,
            title=f"Population Comparison: {county_a} vs {county_b}",
        )
    )

    return chart


def apply_cluster_colors(clusters):
    """Color each county by its cluster."""
    rng = np.random.RandomState(42)
    unique_clusters = np.unique(clusters)
    cluster_colors = rng.rand(len(unique_clusters), 3)

    for county_norm, info in COUNTY_INDEX.items():
        cid = info["csv_row"]
        cluster_id = clusters[cid]
        r, g, b = cluster_colors[cluster_id]

        for cell_id in info["vtk_cell_ids"]:
            color_array.SetTuple(cell_id, (r * 255, g * 255, b * 255))

    color_array.Modified()
    ctrl.update_view()


def reset_map_to_selected_county():
    """Restore to normal map + selected highlight."""
    base_color = (180, 255, 240)

    for i in range(color_array.GetNumberOfTuples()):
        color_array.SetTuple(i, base_color)

    selected = state.selected_county
    if not selected:
        return

    norm = normalize_name(selected)
    info = COUNTY_INDEX.get(norm)
    if info:
        highlight_county(POLY, color_array, info["vtk_cell_ids"])

    color_array.Modified()
    ctrl.update_view()


# -----------------------------------------------------------------------------
# UPDATE VISUALS (MAP + CHARTS) FOR SELECTED COUNTY
# -----------------------------------------------------------------------------
def update_visuals_for_county(selected_name: str):
    """Highlight the county on the VTK map and update all county-dependent charts."""
    from charts import mini_population_bar_chart  # local import to avoid cycles

    if not selected_name:
        return

    norm = normalize_name(selected_name)
    info = COUNTY_INDEX.get(norm)
    if not info:
        print("No info for:", norm)
        return

    # Highlight polygons on map
    cell_ids = info["vtk_cell_ids"]
    highlight_county(POLY, color_array, cell_ids)

    # Refresh VTK rendering
    if hasattr(ctrl, "update_view"):
        ctrl.update_view()

    # Row index for selected county
    csv_row = info["csv_row"]

    # 1) Trend chart
    chart = population_trend_chart(DF, csv_row, YEAR_COLS, selected_name)
    if hasattr(ctrl, "update_chart"):
        ctrl.update_chart(chart)

    # 2) Stats text
    stats_text = compute_county_stats(DF, csv_row, YEAR_COLS)
    if hasattr(ctrl, "update_info_card"):
        ctrl.update_info_card(stats_text)

    # 3) Mini 2010 vs 2019 bar
    mini = mini_population_bar_chart(DF, csv_row, YEAR_COLS)
    if hasattr(ctrl, "update_mini_chart"):
        ctrl.update_mini_chart(mini)

    # 4) Growth gauge
    gauge = growth_gauge_chart(DF, csv_row, YEAR_COLS)
    if hasattr(ctrl, "update_gauge_chart"):
        ctrl.update_gauge_chart(gauge)

    # 5) Similar counties
    similar = top_similar_counties(DF, csv_row, YEAR_COLS)
    similar_chart = similar_counties_bar_chart(similar)
    if hasattr(ctrl, "update_similar_chart"):
        ctrl.update_similar_chart(similar_chart)

    # 6) Population share pie
    pie = population_share_pie(DF, csv_row, YEAR_COLS, selected_name)
    if hasattr(ctrl, "update_pie_chart"):
        ctrl.update_pie_chart(pie)

    # 7) Yearly bar chart (2010–2019)
    yearly_chart = yearly_population_bar_chart(DF, csv_row, YEAR_COLS, selected_name)
    if hasattr(ctrl, "update_yearly_bars"):
        ctrl.update_yearly_bars(yearly_chart)


# -----------------------------------------------------------------------------
# STATE REACTIONS
# -----------------------------------------------------------------------------
@state.change("selected_county")
def _on_selected_county_changed(selected_county, **kwargs):
    update_visuals_for_county(selected_county)


@state.change("selected_county")
def update_forecast(selected_county, **kwargs):
    """Update linear-regression forecast chart when selected county changes."""
    if not selected_county:
        return

    chart = forecast_population_chart(selected_county, DF, YEAR_COLS)
    ctrl.update_forecast_chart(chart)


@state.change("top_section")
def update_corr(top_section=None, **kwargs):
    """Compute correlation heatmap only when correlation tab is active."""
    if top_section == "corr":
        chart = growth_correlation_heatmap(DF, YEAR_COLS)
        ctrl.update_corr_chart(chart)


@state.change("compare_county_a", "compare_county_b")
def update_compare_chart(compare_county_a, compare_county_b, **kwargs):
    """Update the two-county comparison chart when A/B selections change."""
    if state.compare_mode and compare_county_a and compare_county_b:
        chart = compare_two_counties_chart(
            DF, compare_county_a, compare_county_b, YEAR_COLS
        )
        ctrl.update_compare_chart(chart)


@state.change("compare_mode")
def _switch_compare_mode(compare_mode, **kwargs):
    """
    When compare_mode is toggled:
    - If True: switch right panel to 'compare' and generate the comparison chart.
    - If False: go back to 'main'.
    """
    if compare_mode:
        # Show compare panel
        state.top_section = "compare"

        # Also immediately update comparison chart with current A/B selection
        if state.compare_county_a and state.compare_county_b:
            chart = compare_two_counties_chart(
                DF, state.compare_county_a, state.compare_county_b, YEAR_COLS
            )
            ctrl.update_compare_chart(chart)
    else:
        # Return to main analysis
        state.top_section = "main"


@state.change("cluster_k", "top_section")
def update_clusters(cluster_k=None, top_section=None, **kwargs):
    if top_section != "clusters":  # MUST MATCH YOUR TAB NAME EXACTLY
        return

    from charts import kmeans_cluster_plot

    chart, clusters = kmeans_cluster_plot(DF, YEAR_COLS, k=state.cluster_k)

    state.cluster_assignments = clusters.tolist()
    ctrl.update_cluster_chart(chart)
    apply_cluster_colors(clusters)


@state.change("top_section")
def _on_tab_change(top_section=None, **kwargs):
    # When leaving the cluster tab → restore normal map
    if top_section != "clusters":
        reset_map_to_selected_county()


# -----------------------------------------------------------------------------
# UI LAYOUT
# -----------------------------------------------------------------------------
with SinglePageLayout(server) as layout:
    layout.title = "Virginia Population Explorer"

    # -----------------------------
    # TOOLBAR
    # -----------------------------
    with layout.toolbar:
        html.Div(
            "Virginia Population Explorer",
            classes="text-h6 font-weight-medium",
        )
        vuetify3.VSpacer()

        # Main county selector
        vuetify3.VSelect(
            v_model=("selected_county", state.selected_county),
            items=("county_items", []),
            item_title="text",
            item_value="value",
            label="Select County",
            hide_details=True,
            density="compact",
            style="max-width: 300px",
        )

        # Compare mode toggle + selectors
        vuetify3.VSwitch(
            v_model=("compare_mode", state.compare_mode),
            label="Compare Counties",
            hide_details=True,
            inset=True,
            classes="mx-4",
        )

        vuetify3.VSelect(
            v_model=("compare_county_a", state.compare_county_a),
            items=("county_items", []),
            item_title="text",
            item_value="value",
            label="County A",
            dense=True,
            hide_details=True,
            style="max-width: 180px;",
            classes="mr-2",
        )

        vuetify3.VSelect(
            v_model=("compare_county_b", state.compare_county_b),
            items=("county_items", []),
            item_title="text",
            item_value="value",
            label="County B",
            dense=True,
            hide_details=True,
            style="max-width: 180px;",
        )

    # ---------------------------------------------------------------------
    # TOP-LEVEL TABS (controls ONLY the right column, NOT the entire page)
    # ---------------------------------------------------------------------
    with vuetify3.VContainer(classes="mt-4 mb-2 px-6"):
        with vuetify3.VTabs(
            v_model=("top_section", "main"),
            grow=True,
            density="comfortable",
            background_color="white",
            classes="elevation-2",
        ):
            vuetify3.VTab("MAIN ANALYSIS", value="main")
            vuetify3.VTab("POPULATION SHARE", value="share")
            vuetify3.VTab("YEARLY BARS", value="bars")
            vuetify3.VTab("TOP GROWTH", value="growth")
            vuetify3.VTab("PREDICTIVE FORECAST", value="forecast")
            vuetify3.VTab("GROWTH CORRELATION", value="corr")
            vuetify3.VTab("CLUSTERS", value="clusters")

    # -----------------------------
    # MAIN CONTENT
    # -----------------------------
    with layout.content:
        with vuetify3.VContainer(fluid=True, classes="pa-2"):
            with vuetify3.VRow(no_gutters=True):

                # -----------------------------------------
                # LEFT COLUMN — VTK MAP (ALWAYS VISIBLE)
                # -----------------------------------------
                with vuetify3.VCol(cols=7, classes="pr-2"):
                    with vuetify3.VCard(
                        elevation=2,
                        classes="pa-2",
                        style="height: 600px;",
                    ):
                        html.Div(
                            "Virginia Map",
                            classes="text-subtitle-1 mb-2",
                        )

                        vtk_view = vtk_widgets.VtkRemoteView(
                            render_window,
                            ref="vtk_view",
                        )
                        ctrl.update_view = vtk_view.update

                # -----------------------------------------
                # RIGHT COLUMN — SWITCHED BY TOP TABS
                # -----------------------------------------
                with vuetify3.VCol(cols=5, classes="pl-2"):

                    with vuetify3.VWindow(v_model=("top_section", "main")):

                        # ====================================================
                        # RIGHT COLUMN PANEL 1 — MAIN ANALYSIS
                        # ====================================================
                        with vuetify3.VWindowItem(value="main"):

                            # Right-side tabs (Trend / Stats / Comparison / Similar / Gauge)
                            with vuetify3.VTabs(
                                v_model=("selected_info_tab", "trend"),
                                background_color="white",
                                grow=True,
                                density="comfortable",
                                classes="mb-2",
                            ):
                                vuetify3.VTab("Trend", value="trend")
                                vuetify3.VTab("Statistics", value="stats")
                                vuetify3.VTab("Comparison", value="compare")
                                vuetify3.VTab("Similar Counties", value="similar")
                                vuetify3.VTab("Growth Gauge", value="gauge")

                            with vuetify3.VWindow(v_model=("selected_info_tab",)):

                                # --- Trend ---
                                with vuetify3.VWindowItem(value="trend"):
                                    with vuetify3.VCard(
                                        elevation=2,
                                        classes="pa-2",
                                        style="height: 600px;",
                                    ):
                                        html.Div(
                                            "Population Trend (2010–2019)",
                                            classes="text-subtitle-1 mb-2",
                                        )
                                        chart_view = vega.Figure(
                                            figure=None,
                                            style="width: 100%; height: 100%;",
                                        )
                                        ctrl.update_chart = chart_view.update

                                # --- Statistics ---
                                with vuetify3.VWindowItem(value="stats"):
                                    with vuetify3.VCard(
                                        elevation=2,
                                        classes="pa-3",
                                        style="height: 600px;",
                                    ):
                                        html.Div(
                                            "County Statistics",
                                            classes="text-subtitle-1 mb-2",
                                        )
                                        html.Pre(
                                            v_text=("county_stats", ""),
                                            style="white-space: pre-wrap; font-size: 14px;",
                                        )
                                        ctrl.update_info_card = lambda text: setattr(
                                            state, "county_stats", text
                                        )

                                # --- Comparison (2010 vs 2019 mini bar) ---
                                with vuetify3.VWindowItem(value="compare"):
                                    with vuetify3.VCard(
                                        elevation=2,
                                        classes="pa-3",
                                        style="height: 600px;",
                                    ):
                                        html.Div(
                                            "2010 vs 2019 Comparison",
                                            classes="text-subtitle-1 mb-2",
                                        )
                                        mini_chart_view = vega.Figure(
                                            figure=None,
                                            style="width: 100%; height: 100%;",
                                        )
                                        ctrl.update_mini_chart = mini_chart_view.update

                                # --- Similar Counties ---
                                with vuetify3.VWindowItem(value="similar"):
                                    with vuetify3.VCard(
                                        elevation=2,
                                        classes="pa-4",
                                        style="height: 600px;",
                                    ):
                                        html.Div(
                                            "Top 5 Demographically Similar Counties",
                                            classes="text-h6 mb-4",
                                        )
                                        similar_chart_view = vega.Figure(
                                            figure=None,
                                            style="width:100%; height:100%;",
                                        )
                                        ctrl.update_similar_chart = (
                                            similar_chart_view.update
                                        )

                                # --- Growth Gauge ---
                                with vuetify3.VWindowItem(value="gauge"):
                                    with vuetify3.VCard(
                                        elevation=2,
                                        classes="pa-4",
                                        style="height: 600px;",
                                    ):
                                        html.Div(
                                            "10-Year Growth Gauge",
                                            classes="text-h6 mb-4",
                                        )
                                        gauge_view = vega.Figure(
                                            figure=None,
                                            style="width:100%; height:100%;",
                                        )
                                        ctrl.update_gauge_chart = gauge_view.update

                        # ====================================================
                        # RIGHT COLUMN PANEL 2 — POPULATION SHARE PIE
                        # ====================================================
                        with vuetify3.VWindowItem(value="share"):
                            with vuetify3.VCard(
                                elevation=2,
                                classes="pa-4",
                                style="height: 600px;",
                            ):
                                html.Div(
                                    "Population Share Within Virginia",
                                    classes="text-h6 mb-4",
                                )
                                pie_chart_view = vega.Figure(
                                    figure=None,
                                    style="width:100%; height:100%;",
                                )
                                ctrl.update_pie_chart = pie_chart_view.update

                        # ====================================================
                        # RIGHT COLUMN PANEL 3 — YEARLY BAR CHART
                        # ====================================================
                        with vuetify3.VWindowItem(value="bars"):
                            with vuetify3.VCard(
                                elevation=2,
                                classes="pa-4",
                                style="height: 600px;",
                            ):
                                html.Div(
                                    "Yearly Population (2010–2019)",
                                    classes="text-h6 mb-4",
                                )

                                yearly_bar_view = vega.Figure(
                                    figure=None,
                                    style="width:100%; height:100%;",
                                )
                                ctrl.update_yearly_bars = yearly_bar_view.update

                        # ====================================================
                        # RIGHT COLUMN PANEL 4 — TOP GROWTH LEADERBOARD
                        # ====================================================
                        with vuetify3.VWindowItem(value="growth"):
                            with vuetify3.VCard(
                                elevation=2,
                                classes="pa-4",
                                style="height: 600px;",
                            ):
                                html.Div(
                                    "Top 5 Fastest Growing Counties (2010–2019)",
                                    classes="text-h6 mb-4",
                                )

                                growth_chart_view = vega.Figure(
                                    figure=None,
                                    style="width:100%; height:100%;",
                                )
                                ctrl.update_growth_chart = growth_chart_view.update

                        # ====================================================
                        # RIGHT COLUMN PANEL 5 — PREDICTIVE FORECAST
                        # ====================================================
                        with vuetify3.VWindowItem(value="forecast"):
                            with vuetify3.VCard(
                                elevation=2,
                                classes="pa-4",
                                style="height: 600px;",
                            ):
                                html.Div(
                                    "Future Population Forecast",
                                    classes="text-h6 mb-4",
                                )

                                # Short explanation
                                html.Div(
                                    "Forecasting population using simple linear regression (2010–2019).",
                                    classes="mb-3 text-body-2",
                                )

                                forecast_view = vega.Figure(
                                    figure=None,
                                    style="width:100%; height:100%;",
                                )
                                ctrl.update_forecast_chart = forecast_view.update

                        # ====================================================
                        # RIGHT COLUMN PANEL 6 — GROWTH CORRELATION
                        # ====================================================
                        with vuetify3.VWindowItem(value="corr"):
                            with vuetify3.VCard(
                                elevation=2,
                                classes="pa-4",
                                style="height: 600px;",
                            ):
                                html.Div(
                                    "Year-over-Year Growth Correlation",
                                    classes="text-h6 mb-4",
                                )

                                corr_view = vega.Figure(
                                    figure=None,
                                    style="width:100%; height:100%;",
                                )
                                ctrl.update_corr_chart = corr_view.update

                        # ====================================================
                        # RIGHT COLUMN PANEL 7 — COUNTY COMPARISON
                        # ====================================================
                        with vuetify3.VWindowItem(value="compare"):
                            with vuetify3.VCard(
                                elevation=2,
                                classes="pa-4",
                                style="height: 600px;",
                            ):
                                html.Div(
                                    "County Comparison (2010–2019)",
                                    classes="text-h6 mb-2",
                                )

                                comparison_view = vega.Figure(
                                    figure=None,
                                    style="width:100%; height:100%;",
                                )
                                ctrl.update_compare_chart = comparison_view.update
                        # ====================================================
                        # RIGHT COLUMN PANEL 8 — CLUSTERS
                        # ====================================================
                        with vuetify3.VWindowItem(value="clusters"):
                            with vuetify3.VCard(
                                elevation=2, classes="pa-4", style="height: 600px;"
                            ):
                                html.Div(
                                    "County Clustering (K-Means)",
                                    classes="text-h6 mb-2",
                                )

                                vuetify3.VSlider(
                                    v_model=("cluster_k", state.cluster_k),
                                    min=2,
                                    max=8,
                                    step=1,
                                    label="Number of Clusters (K)",
                                    thumb_label=True,
                                    hide_details=True,
                                    classes="mb-4",
                                )

                                cluster_view = vega.Figure(
                                    figure=None, style="width:100%; height:100%;"
                                )
                                ctrl.update_cluster_chart = cluster_view.update


# -----------------------------------------------------------------------------
# INITIALIZATION AFTER SERVER IS READY
# -----------------------------------------------------------------------------
@ctrl.add("on_server_ready")
def _on_ready(**_):
    # Initialize with first county highlighted + charts
    update_visuals_for_county(state.selected_county)

    # Initialize global Top Growth leaderboard (does not depend on selection)
    growth_chart = top_growth_leaderboard_chart(DF, YEAR_COLS)
    if hasattr(ctrl, "update_growth_chart"):
        ctrl.update_growth_chart(growth_chart)


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run Virginia Population Explorer")
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to serve on (default: 8080)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host to bind to (default: localhost, or 0.0.0.0 for Docker)",
    )

    args = parser.parse_args()

    server.start(host=args.host, port=args.port, open_browser=True)
