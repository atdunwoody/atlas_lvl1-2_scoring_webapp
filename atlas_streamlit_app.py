"""Interactive Atlas scoring viewer for Streamlit.

Required packages:
    pip install "streamlit>=1.42" "plotly>=5.24" pandas numpy geopandas pyogrio shapely

Launch command:
    streamlit run atlas_streamlit_app.py

The app reads the Level 1 and Level 2 CSV outputs created by
Atlas_Integrated_Scoring.ipynb. A polygon layer is also required because the
score outputs do not contain geometry. The polygon layer must contain the same
BSR identifiers used in bsr_scores.csv, such as CC1 or UGR11.
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


SCORE_DIR = Path("data/outputs")
BSR_GPKG_PATH = SCORE_DIR / "bsr_scores.gpkg"
BSR_LAYER_NAME = "bsr"
BSR_ID_FIELD = "BSR"
MAP_STYLE = "carto-positron"

CORE_SCORE_FILES = {
    "bsr": "bsr_scores.csv",
    "life_stage": "life_stage_scores.csv",
    "limiting_factor": "limiting_factor_scores_integrated.csv",
    "action": "action_scores.csv",
    "grid": "calculation_grid.csv",
}

SUPPORTING_SCORE_FILES = {
    "action_components": "action_score_components.csv",
}

SCORE_FILES = {**CORE_SCORE_FILES, **SUPPORTING_SCORE_FILES}

REQUIRED_COLUMNS = {
    "bsr": {
        "bsr",
        "basin",
        "detailed_fish_use_score",
        "source_fish_use_score_raw",
        "source_fish_use_score_normalized",
        "source_fish_use_score_100",
        "overall_impact_score",
        "overall_risk_score",
        "highest_priority_life_stage",
        "highest_priority_limiting_factor",
        "highest_priority_action",
        "highest_priority_action_benefit_score",
        "sum_action_benefit_provisional",
    },
    "life_stage": {
        "bsr",
        "basin",
        "species",
        "life_stage",
        "fish_use_rating",
        "population_priority",
        "impact_score",
        "risk_score",
    },
    "limiting_factor": {
        "bsr",
        "basin",
        "limiting_factor",
        "condition_score_raw_1_5",
        "condition_score",
        "impact_score",
        "risk_score",
    },
    "action": {
        "bsr",
        "basin",
        "action_id",
        "action_type",
        "condition_improvement_score",
        "limiting_factor_amelioration_score",
        "overall_benefit_score",
    },
    "grid": {
        "bsr",
        "basin",
        "species",
        "life_stage",
        "limiting_factor",
        "fish_use_rating",
        "population_priority",
        "condition_score",
        "vulnerability_score",
        "impact_component",
        "risk_component",
    },
    "action_components": {
        "bsr",
        "basin",
        "limiting_factor",
        "action_id",
        "action_type",
        "lfat_score",
        "condition_improvement_component",
        "amelioration_component",
        "benefit_component",
    },
}

DEFAULT_COLOR_SCALE = "Blues"
RISK_COLOR_SCALE = "Reds"
LIMITING_IMPACT_COLOR_SCALE = "Purples"
ACTION_BENEFIT_COLOR_SCALE = "Greens"

DISPLAY_LABELS = {
    "bsr": "BSR",
    "basin": "Basin",
    "detailed_fish_use_score": "Detailed Fish Use Score",
    "source_fish_use_score_raw": "Source Fish Use Score (raw)",
    "source_fish_use_score_normalized": "Source Fish Use Score (normalized)",
    "source_fish_use_score_100": "Fish Use Score /100",
    "overall_impact_score": "Overall Limiting-Factor Impact",
    "overall_risk_score": "Overall Risk Score",
    "lf_sum_impact_score": "Summed Limiting-Factor Impact",
    "lf_sum_risk_score": "Summed Limiting-Factor Risk",
    "top_life_stage_risk_score": "Top Life-Stage Risk Score",
    "top_limiting_factor_risk_score": "Top Limiting-Factor Risk Score",
    "condition_score_raw_1_5": "Raw Condition Score (1–5)",
    "condition_score": "Condition Score",
    "impact_score": "Impact Score",
    "risk_score": "Risk Score",
    "condition_improvement_score": "Condition Improvement Score",
    "limiting_factor_amelioration_score": "Limiting-Factor Amelioration Score",
    "overall_benefit_score": "Overall Benefit Score",
    "highest_priority_action_benefit_score": "Highest-Priority Action Benefit Score",
    "sum_action_benefit_provisional": "Sum of Action Benefit Scores",
    "benefit_rank_within_bsr": "Benefit Rank Within BSR",
}


def configure_page() -> None:
    """Set page-level options and light visual styling."""
    st.set_page_config(
        page_title="Atlas Integrated Scoring",
        page_icon="🗺️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1700px;}
        [data-testid="stMetric"] {border: 1px solid #d9e2ea; border-radius: 0.45rem; padding: 0.7rem;}
        [data-testid="stSidebar"] {min-width: 320px;}
        div[data-testid="stAlert"] {border-radius: 0.35rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def require_columns(table_name: str, table: pd.DataFrame) -> None:
    """Raise a readable error when an expected notebook output field is absent."""
    missing = sorted(REQUIRED_COLUMNS[table_name] - set(table.columns))
    if missing:
        raise ValueError(
            f"{SCORE_FILES[table_name]} is missing required columns: {missing}"
        )


@st.cache_data(show_spinner=False)
def load_score_tables(score_dir_text: str) -> dict[str, pd.DataFrame]:
    """Load core scoring tables and available action components."""
    score_dir = Path(score_dir_text).expanduser()
    missing_files = [
        filename
        for filename in CORE_SCORE_FILES.values()
        if not (score_dir / filename).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"Missing scoring outputs in {score_dir}: {missing_files}"
        )

    tables = {
        name: pd.read_csv(score_dir / filename)
        for name, filename in CORE_SCORE_FILES.items()
    }
    tables.update(
        {
            name: pd.read_csv(score_dir / filename)
            for name, filename in SUPPORTING_SCORE_FILES.items()
            if (score_dir / filename).is_file()
        }
    )

    for name, table in tables.items():
        require_columns(name, table)
        for identifier_column in ("bsr", "basin"):
            if identifier_column in table.columns:
                table[identifier_column] = (
                    table[identifier_column].astype(str).str.strip()
                )

    if tables["bsr"]["bsr"].duplicated().any():
        duplicates = tables["bsr"].loc[
            tables["bsr"]["bsr"].duplicated(keep=False), "bsr"
        ].unique()
        raise ValueError(f"bsr_scores.csv contains duplicate BSRs: {duplicates}")

    expected_bsrs = set(tables["bsr"]["bsr"])
    for name in ("life_stage", "limiting_factor", "action", "grid"):
        table_bsrs = set(tables[name]["bsr"])
        if table_bsrs != expected_bsrs:
            missing = sorted(expected_bsrs - table_bsrs)
            extra = sorted(table_bsrs - expected_bsrs)
            raise ValueError(
                f"{SCORE_FILES[name]} has inconsistent BSR coverage. "
                f"Missing: {missing}; extra: {extra}"
            )

    for name in ("action_components",):
        if name not in tables:
            continue
        table_bsrs = set(tables[name]["bsr"])
        if table_bsrs != expected_bsrs:
            missing = sorted(expected_bsrs - table_bsrs)
            extra = sorted(table_bsrs - expected_bsrs)
            raise ValueError(
                f"{SCORE_FILES[name]} has inconsistent BSR coverage. "
                f"Missing: {missing}; extra: {extra}"
            )

    if {
        "risk_balance_difference",
        "impact_balance_difference",
    }.issubset(tables["bsr"].columns):
        balance = tables["bsr"][[
            "risk_balance_difference",
            "impact_balance_difference",
        ]].abs().max()
        if (balance > 1e-9).any():
            raise ValueError(
                "The BSR summary does not reconcile between life-stage and "
                "limiting-factor aggregation paths."
            )

    return tables


@st.cache_data(show_spinner=False)
def load_geometry_path(path_text: str, layer_name: str) -> gpd.GeoDataFrame:
    """Read a configured GeoPackage or GeoJSON path."""
    path = Path(path_text).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Spatial layer not found: {path}")
    layer = layer_name.strip() or None
    return gpd.read_file(path, layer=layer)


def prepare_geometry(
    source: gpd.GeoDataFrame,
    spatial_id_field: str,
    bsr_scores: pd.DataFrame,
) -> tuple[gpd.GeoDataFrame, list[str]]:
    """Standardize geometry and match its identifier to the scoring BSR key."""
    if spatial_id_field not in source.columns:
        raise ValueError(f"Spatial ID field not found: {spatial_id_field}")
    if source.crs is None:
        raise ValueError("The BSR polygon layer has no coordinate reference system.")

    spatial = source[[spatial_id_field, "geometry"]].copy()
    spatial = spatial.loc[
        spatial.geometry.notna() & ~spatial.geometry.is_empty
    ].copy()
    if spatial.empty:
        raise ValueError("The spatial layer contains no non-empty geometry.")

    polygon_types = {"Polygon", "MultiPolygon"}
    invalid_types = sorted(set(spatial.geometry.geom_type) - polygon_types)
    if invalid_types:
        raise ValueError(
            "The map requires polygon geometry. Found geometry types: "
            f"{invalid_types}"
        )

    spatial["_spatial_id"] = (
        spatial[spatial_id_field].astype(str).str.strip()
    )

    score_bsrs = set(bsr_scores["bsr"])
    unmatched_ids = sorted(set(spatial["_spatial_id"]) - score_bsrs)
    spatial["bsr"] = spatial["_spatial_id"]
    spatial = spatial.loc[
        spatial["bsr"].isin(score_bsrs), ["bsr", "geometry"]
    ].copy()
    if spatial.empty:
        raise ValueError(
            "No values in the GeoPackage BSR field matched bsr_scores.csv."
        )

    spatial["bsr"] = spatial["bsr"].astype(str).str.strip()
    try:
        spatial.geometry = spatial.geometry.make_valid()
    except AttributeError:
        pass

    spatial = spatial.dissolve(by="bsr", as_index=False)
    spatial = spatial.to_crs(epsg=4326)
    return spatial, unmatched_ids


def filter_table(table: pd.DataFrame, basin: str) -> pd.DataFrame:
    """Apply the common basin filter."""
    if basin == "All basins":
        return table.copy()
    return table.loc[table["basin"].eq(basin)].copy()


def format_score(value: Any, digits: int = 2) -> str:
    """Format numeric KPI values without implying unnecessary precision."""
    if pd.isna(value):
        return "Not available"
    return f"{float(value):,.{digits}f}"


def round_float_columns(table: pd.DataFrame) -> pd.DataFrame:
    """Round floating-point values for display without changing source tables."""
    rounded = table.copy()
    float_columns = rounded.select_dtypes(include=["floating"]).columns
    rounded[float_columns] = rounded[float_columns].round(2)
    return rounded


def show_score_table(table: pd.DataFrame) -> None:
    """Display score tables with consistent two-decimal numeric formatting."""
    rounded = round_float_columns(table)
    float_columns = rounded.select_dtypes(include=["floating"]).columns
    column_config = {
        column: st.column_config.NumberColumn(format="%.2f")
        for column in float_columns
    }
    st.dataframe(
        rounded,
        use_container_width=True,
        hide_index=True,
        column_config=column_config,
    )


def map_center_zoom(geometry: gpd.GeoDataFrame) -> tuple[dict[str, float], float]:
    """Center the map and frame the BSR extent relatively tightly."""
    min_x, min_y, max_x, max_y = geometry.total_bounds
    center = {"lon": float((min_x + max_x) / 2), "lat": float((min_y + max_y) / 2)}
    span = max(float(max_x - min_x), float(max_y - min_y))
    if span <= 0.10:
        zoom = 10.0
    elif span <= 0.25:
        zoom = 9.0
    elif span <= 0.50:
        zoom = 8.0
    elif span <= 1.00:
        zoom = 7.0
    elif span <= 2.00:
        zoom = 6.0
    elif span <= 5.00:
        zoom = 5.0
    else:
        zoom = 4.0
    return center, min(zoom + 2.0, 12.0)


def selected_point_bsr(point: dict[str, Any]) -> str | None:
    """Extract a BSR identifier from a Plotly selection point."""
    custom = point.get("customdata")
    if isinstance(custom, (list, tuple)) and custom:
        return str(custom[0])
    if custom is not None:
        return str(custom)
    location = point.get("location")
    return None if location is None else str(location)


def capture_map_selection(chart_key: str) -> None:
    """Synchronize a clicked map polygon with the sidebar BSR selector."""
    event = st.session_state.get(chart_key)
    if not event:
        return
    try:
        points = event.selection.points
    except AttributeError:
        points = event.get("selection", {}).get("points", [])
    if not points:
        return
    clicked_bsr = selected_point_bsr(points[-1])
    available = set(st.session_state.get("available_bsrs", []))
    if clicked_bsr in available:
        st.session_state["selected_bsr"] = clicked_bsr


def render_choropleth(
    geometry: gpd.GeoDataFrame,
    values: pd.DataFrame,
    metric: str,
    metric_label: str,
    title: str,
    chart_key: str,
    map_style: str,
    *,
    categorical: bool = False,
    hover_columns: list[str] | None = None,
    color_scale: str = DEFAULT_COLOR_SCALE,
) -> None:
    """Render an interactive BSR choropleth and register click selection."""
    hover_columns = hover_columns or []
    requested = ["bsr", metric, *hover_columns]
    requested = list(dict.fromkeys(column for column in requested if column in values.columns))
    value_table = values[requested].drop_duplicates("bsr")
    mapped = geometry.merge(value_table, on="bsr", how="inner", validate="one_to_one")
    mapped = mapped.loc[mapped[metric].notna()].copy()

    if mapped.empty:
        st.warning(f"No mapped BSRs contain values for {metric_label}.")
        return

    geojson = json.loads(mapped[["bsr", "geometry"]].to_json())
    plot_data = round_float_columns(pd.DataFrame(mapped.drop(columns="geometry")))
    center, zoom = map_center_zoom(mapped)

    hover_fields = list(dict.fromkeys([metric, *hover_columns]))
    hover_data = {}
    for column in hover_fields:
        if column not in plot_data.columns or column == "bsr":
            continue
        if pd.api.types.is_float_dtype(plot_data[column]):
            hover_data[column] = ":.2f"
        else:
            hover_data[column] = True

    common = {
        "data_frame": plot_data,
        "geojson": geojson,
        "locations": "bsr",
        "featureidkey": "properties.bsr",
        "color": metric,
        "hover_name": "bsr",
        "hover_data": hover_data,
        "custom_data": ["bsr"],
        "opacity": 0.78,
        "zoom": zoom,
        "center": center,
        "title": title,
        "labels": {
            column: DISPLAY_LABELS.get(
                column,
                column.replace("_", " ").title(),
            )
            for column in requested
        }
        | {metric: metric_label},
    }

    if categorical:
        common["color_discrete_sequence"] = px.colors.qualitative.Safe
    else:
        common["color_continuous_scale"] = color_scale
        numeric = pd.to_numeric(plot_data[metric], errors="coerce")
        minimum = float(numeric.min())
        maximum = float(numeric.max())
        if np.isclose(minimum, maximum):
            pad = max(abs(minimum) * 0.05, 0.5)
            common["range_color"] = (minimum - pad, maximum + pad)
        else:
            common["range_color"] = (minimum, maximum)

    if hasattr(px, "choropleth_map"):
        figure = px.choropleth_map(map_style=map_style, **common)
        figure.update_layout(
            map={"bearing": 0, "pitch": 0},
            margin={"r": 0, "t": 55, "l": 0, "b": 0},
            height=600,
            legend_title_text=metric_label,
        )
    else:
        figure = px.choropleth_mapbox(mapbox_style=map_style, **common)
        figure.update_layout(
            margin={"r": 0, "t": 55, "l": 0, "b": 0},
            height=600,
            legend_title_text=metric_label,
        )

    figure.update_traces(marker_line_width=1.1, marker_line_color="#ffffff")
    st.plotly_chart(
        figure,
        use_container_width=True,
        key=chart_key,
        on_select=partial(capture_map_selection, chart_key),
        selection_mode="points",
        config={"displaylogo": False, "scrollZoom": True},
    )


def horizontal_bar(
    table: pd.DataFrame,
    value: str,
    category: str,
    title: str,
    *,
    color: str | None = None,
    value_label: str | None = None,
) -> None:
    """Render a consistently formatted horizontal comparison chart."""
    ordered = round_float_columns(table).sort_values(value, ascending=True)
    figure = px.bar(
        ordered,
        x=value,
        y=category,
        color=color,
        orientation="h",
        title=title,
        labels={value: value_label or value, category: ""},
        text_auto=".2f",
    )
    figure.update_layout(
        height=max(360, 30 * len(ordered) + 110),
        margin={"r": 10, "t": 55, "l": 10, "b": 10},
        legend_title_text="Species" if color == "species" else color,
    )
    figure.update_traces(
        textposition="outside",
        cliponaxis=False,
        hovertemplate=(
            f"%{{y}}<br>{value_label or value}: %{{x:.2f}}<extra></extra>"
        ),
    )
    st.plotly_chart(
        figure,
        use_container_width=True,
        config={"displaylogo": False},
    )


def render_overall_risk(
    tables: dict[str, pd.DataFrame],
    geometry: gpd.GeoDataFrame,
    basin: str,
    selected_bsr: str,
    map_style: str,
) -> None:
    """Render the overall risk map and its two principal drill-downs."""
    bsr = filter_table(tables["bsr"], basin)
    life = filter_table(tables["life_stage"], basin)
    limiting = filter_table(tables["limiting_factor"], basin)

    st.header("Level 1: Overall risk")
    st.caption(
        "Click a polygon or use the sidebar BSR selector. The charts below "
        "partition the selected BSR score by fish use and limiting factor."
    )
    render_choropleth(
        geometry,
        bsr,
        "overall_risk_score",
        "Overall risk score",
        "Overall Risk Score",
        "map_overall_risk",
        map_style,
        hover_columns=[
            "basin",
            "source_fish_use_score_100",
            "detailed_fish_use_score",
            "overall_impact_score",
            "lf_sum_impact_score",
            "lf_sum_risk_score",
            "highest_priority_life_stage",
            "highest_priority_limiting_factor",
        ],
        color_scale=RISK_COLOR_SCALE,
    )

    row = bsr.loc[bsr["bsr"].eq(selected_bsr)].iloc[0]
    metric_columns = st.columns(4)
    metric_columns[0].metric("Selected BSR", selected_bsr)
    metric_columns[1].metric("Overall risk", format_score(row["overall_risk_score"]))
    metric_columns[2].metric("Fish Use Score /100", format_score(row["source_fish_use_score_100"]))
    metric_columns[3].metric("Overall LF impact", format_score(row["overall_impact_score"]))

    st.markdown(
        f"**Highest-priority life stage:** {row['highest_priority_life_stage']}  \n"
        f"**Highest-priority limiting factor:** {row['highest_priority_limiting_factor']}"
    )

    left, right = st.columns(2)
    life_selected = life.loc[life["bsr"].eq(selected_bsr)].copy()
    life_selected["fish_use"] = (
        life_selected["species"] + " | " + life_selected["life_stage"]
    )
    factor_selected = limiting.loc[limiting["bsr"].eq(selected_bsr)].copy()

    with left:
        horizontal_bar(
            life_selected,
            "risk_score",
            "fish_use",
            f"{selected_bsr}: risk by species and life stage",
            color="species",
            value_label="Risk score",
        )
    with right:
        horizontal_bar(
            factor_selected,
            "risk_score",
            "limiting_factor",
            f"{selected_bsr}: risk by limiting factor",
            value_label="Risk score",
        )

    with st.expander(f"Show score tables for BSR: {selected_bsr}"):
        st.subheader("Species and life stages")
        show_score_table(
            life_selected[
                [
                    "species",
                    "life_stage",
                    "fish_use_rating",
                    "population_priority",
                    "impact_score",
                    "risk_score",
                ]
            ].sort_values("risk_score", ascending=False)
        )
        st.subheader("Limiting factors")
        show_score_table(
            factor_selected[
                [
                    "limiting_factor",
                    "condition_score",
                    "impact_score",
                    "risk_score",
                ]
            ].sort_values("risk_score", ascending=False)
        )


def render_fish_use(
    tables: dict[str, pd.DataFrame],
    geometry: gpd.GeoDataFrame,
    basin: str,
    selected_bsr: str,
    map_style: str,
) -> None:
    """Render overall fish-use mapping and species/life-stage drill-down."""
    bsr = filter_table(tables["bsr"], basin)
    life = filter_table(tables["life_stage"], basin)

    st.header("Level 1: Fish use")
    render_choropleth(
        geometry,
        bsr,
        "source_fish_use_score_100",
        "Fish Use Score /100",
        "Fish Use Score /100",
        "map_fish_use",
        map_style,
        hover_columns=[
            "basin",
            "source_fish_use_score_raw",
            "source_fish_use_score_normalized",
            "source_fish_use_score_100",
            "detailed_fish_use_score",
        ],
    )

    selected = life.loc[life["bsr"].eq(selected_bsr)].copy()
    selected["fish_use"] = selected["species"] + " | " + selected["life_stage"]
    horizontal_bar(
        selected,
        "fish_use_rating",
        "fish_use",
        f"{selected_bsr}: fish-use rating by species and life stage",
        color="species",
        value_label="Fish-use rating",
    )

    with st.expander("Optional map: highest-priority life stage"):
        st.caption(
            "This categorical map is based on the population-weighted Level 1 "
            "risk score, not fish use alone."
        )
        render_choropleth(
            geometry,
            bsr,
            "highest_priority_life_stage",
            "Highest-priority life stage",
            "Highest-Priority Life Stage",
            "map_top_life_stage",
            map_style,
            categorical=True,
            hover_columns=[
                "basin",
                "source_fish_use_score_100",
                "detailed_fish_use_score",
                "top_life_stage_risk_score",
                "overall_risk_score",
            ],
        )

    with st.expander(f"Show species and life-stage data for BSR: {selected_bsr}"):
        show_score_table(
            selected[
                [
                    "species",
                    "life_stage",
                    "fish_use_rating",
                    "population_priority",
                    "impact_score",
                    "risk_score",
                ]
            ].sort_values(["species", "life_stage"])
        )


def render_limiting_factors(
    tables: dict[str, pd.DataFrame],
    geometry: gpd.GeoDataFrame,
    basin: str,
    selected_bsr: str,
    map_style: str,
) -> None:
    """Render overall and factor-specific maps with biological drill-down."""
    bsr = filter_table(tables["bsr"], basin)
    limiting = filter_table(tables["limiting_factor"], basin)
    grid = filter_table(tables["grid"], basin)

    st.header("Level 1: Limiting factors")
    overall_labels = {
        "Overall limiting-factor impact": "overall_impact_score",
        "Population-weighted limiting-factor risk": "overall_risk_score",
    }
    overall_label = st.radio(
        "Overall limiting-factor map value",
        options=list(overall_labels),
        horizontal=True,
    )
    overall_metric = overall_labels[overall_label]
    overall_color_scale = (
        LIMITING_IMPACT_COLOR_SCALE
        if overall_metric == "overall_impact_score"
        else RISK_COLOR_SCALE
    )
    render_choropleth(
        geometry,
        bsr,
        overall_metric,
        overall_label,
        overall_label,
        "map_limiting_factor_overall",
        map_style,
        hover_columns=[
            "basin",
            "source_fish_use_score_100",
            "overall_impact_score",
            "overall_risk_score",
            "lf_sum_impact_score",
            "lf_sum_risk_score",
            "highest_priority_limiting_factor",
            "top_limiting_factor_risk_score",
        ],
        color_scale=overall_color_scale,
    )

    st.subheader("Specific limiting factor")
    factor_options = sorted(limiting["limiting_factor"].dropna().unique())
    selected_factor = st.selectbox("Limiting factor", factor_options)
    factor_score_labels = {
        "Impact score": "impact_score",
        "Population-weighted risk score": "risk_score",
        "Condition score": "condition_score",
    }
    factor_score_label = st.radio(
        "Factor-specific map value",
        options=list(factor_score_labels),
        horizontal=True,
    )
    factor_score = factor_score_labels[factor_score_label]
    factor_color_scale = {
        "impact_score": LIMITING_IMPACT_COLOR_SCALE,
        "risk_score": RISK_COLOR_SCALE,
        "condition_score": DEFAULT_COLOR_SCALE,
    }[factor_score]
    factor_map = limiting.loc[limiting["limiting_factor"].eq(selected_factor)].copy()
    render_choropleth(
        geometry,
        factor_map,
        factor_score,
        factor_score_label,
        f"{selected_factor}: {factor_score_label}",
        "map_specific_limiting_factor",
        map_style,
        hover_columns=[
            "basin",
            "condition_score",
            "impact_score",
            "risk_score",
        ],
        color_scale=factor_color_scale,
    )

    left, right = st.columns(2)
    bsr_factors = limiting.loc[limiting["bsr"].eq(selected_bsr)].copy()
    with left:
        horizontal_bar(
            bsr_factors,
            factor_score,
            "limiting_factor",
            f"{selected_bsr}: limiting-factor comparison",
            value_label=factor_score_label,
        )

    biological = grid.loc[
        grid["bsr"].eq(selected_bsr)
        & grid["limiting_factor"].eq(selected_factor)
    ].copy()
    biological["fish_use"] = biological["species"] + " | " + biological["life_stage"]
    component_labels = {
        "Impact component": "impact_component",
        "Population-weighted risk component": "risk_component",
    }
    with right:
        component_label = st.radio(
            "Species/life-stage contribution",
            options=list(component_labels),
            horizontal=False,
        )
        horizontal_bar(
            biological,
            component_labels[component_label],
            "fish_use",
            f"{selected_bsr}: {selected_factor} by fish use",
            color="species",
            value_label=component_label,
        )

    with st.expander("Highest-priority limiting factor map"):
        render_choropleth(
            geometry,
            bsr,
            "highest_priority_limiting_factor",
            "Highest-priority limiting factor",
            "Highest-Priority Limiting Factor",
            "map_top_limiting_factor",
            map_style,
            categorical=True,
            hover_columns=[
                "basin",
                "source_fish_use_score_100",
                "overall_impact_score",
                "overall_risk_score",
                "top_limiting_factor_risk_score",
            ],
        )

    with st.expander(f"Show limiting-factor data for: {selected_factor} in {selected_bsr}"):
        show_score_table(
            biological[
                [
                    "species",
                    "life_stage",
                    "fish_use_rating",
                    "population_priority",
                    "condition_score",
                    "vulnerability_score",
                    "impact_component",
                    "risk_component",
                ]
            ].sort_values("risk_component", ascending=False)
        )


def render_actions(
    tables: dict[str, pd.DataFrame],
    geometry: gpd.GeoDataFrame,
    basin: str,
    selected_bsr: str,
    map_style: str,
) -> None:
    """Render Level 2 action maps, rankings, and the provisional summed score."""
    bsr = filter_table(tables["bsr"], basin)
    actions = filter_table(tables["action"], basin)
    action_components = (
        filter_table(tables["action_components"], basin)
        if "action_components" in tables
        else None
    )

    st.header("Level 2: Action-specific benefit")
    action_options = (
        actions[["action_id", "action_type"]]
        .drop_duplicates()
        .sort_values("action_id")
    )
    action_lookup = dict(zip(action_options["action_type"], action_options["action_id"]))
    selected_action = st.selectbox("Action type", list(action_lookup))

    action_score_labels = {
        "Overall benefit score": "overall_benefit_score",
        "Limiting-factor amelioration score": "limiting_factor_amelioration_score",
        "Condition improvement score": "condition_improvement_score",
    }
    action_score_label = st.radio(
        "Action map value",
        options=list(action_score_labels),
        horizontal=True,
    )
    action_score = action_score_labels[action_score_label]
    action_map = actions.loc[actions["action_type"].eq(selected_action)].copy()
    render_choropleth(
        geometry,
        action_map,
        action_score,
        action_score_label,
        f"{selected_action}: {action_score_label}",
        "map_action_specific",
        map_style,
        hover_columns=[
            "basin",
            "condition_improvement_score",
            "limiting_factor_amelioration_score",
            "overall_benefit_score",
            "benefit_rank_within_bsr",
        ],
        color_scale=ACTION_BENEFIT_COLOR_SCALE,
    )

    selected = actions.loc[actions["bsr"].eq(selected_bsr)].copy()
    horizontal_bar(
        selected,
        action_score,
        "action_type",
        f"{selected_bsr}: action-specific scores",
        value_label=action_score_label,
    )

    if action_components is not None:
        component_rows = action_components.loc[
            action_components["bsr"].eq(selected_bsr)
            & action_components["action_type"].eq(selected_action)
        ].copy()
        with st.expander(f"Show limiting-factor contributions to the selected action: {selected_action}"):
            horizontal_bar(
                component_rows,
                "benefit_component",
                "limiting_factor",
                f"{selected_bsr}: benefit components for {selected_action}",
                value_label="Benefit component",
            )
            show_score_table(
                component_rows[
                    [
                        "limiting_factor",
                        "lfat_score",
                        "condition_improvement_component",
                        "amelioration_component",
                        "benefit_component",
                    ]
                ].sort_values("benefit_component", ascending=False)
            )

    st.subheader("Highest-priority action type")
    render_choropleth(
        geometry,
        bsr,
        "highest_priority_action",
        "Highest-priority action type",
        "Highest-Priority Action Type",
        "map_top_action",
        map_style,
        categorical=True,
        hover_columns=[
            "basin",
            "highest_priority_action_benefit_score",
            "sum_action_benefit_provisional",
            "overall_risk_score",
        ],
    )

    with st.expander(f"Show action table for BSR: {selected_bsr}"):
        show_score_table(
            selected[
                [
                    "action_id",
                    "action_type",
                    "condition_improvement_score",
                    "limiting_factor_amelioration_score",
                    "overall_benefit_score",
                    "benefit_rank_within_bsr",
                ]
            ].sort_values("benefit_rank_within_bsr")
        )


def load_bsr_geometry(bsr_scores: pd.DataFrame) -> gpd.GeoDataFrame:
    """Load and prepare the hardcoded BSR GeoPackage layer."""
    source = load_geometry_path(str(BSR_GPKG_PATH), BSR_LAYER_NAME)
    geometry, unmatched = prepare_geometry(source, BSR_ID_FIELD, bsr_scores)
    if unmatched:
        st.warning(
            f"{len(unmatched)} spatial identifiers did not match scoring BSRs: "
            + ", ".join(unmatched[:10])
            + (" ..." if len(unmatched) > 10 else "")
        )
    return geometry


def main() -> None:
    """Run the Streamlit application."""
    configure_page()
    st.title("Atlas Integrated Scoring")
    st.caption("Interactive Level 1 risk and Level 2 action-benefit maps")

    try:
        tables = load_score_tables(str(SCORE_DIR))
    except Exception as error:
        st.error(str(error))
        st.stop()

    try:
        geometry = load_bsr_geometry(tables["bsr"])
    except Exception as error:
        st.error(str(error))
        st.stop()

    st.sidebar.header("Map and drill-down")
    basin_options = ["All basins", *sorted(tables["bsr"]["basin"].unique())]
    basin = st.sidebar.selectbox("Basin", basin_options)
    available_bsrs = sorted(filter_table(tables["bsr"], basin)["bsr"].unique())
    st.session_state["available_bsrs"] = available_bsrs
    if st.session_state.get("selected_bsr") not in available_bsrs:
        st.session_state["selected_bsr"] = available_bsrs[0]
    selected_bsr = st.sidebar.selectbox(
        "Drill-down BSR",
        available_bsrs,
        key="selected_bsr",
        help="Map clicks update this selector.",
    )
    page = st.sidebar.radio(
        "View",
        [
            "Overall risk",
            "Fish use",
            "Limiting factors",
            "Action benefits",
        ],
    )

    mapped_bsrs = set(geometry["bsr"])
    missing_geometry = sorted(set(available_bsrs) - mapped_bsrs)
    if missing_geometry:
        st.warning(
            f"{len(missing_geometry)} filtered BSRs have no matching polygon and will "
            "not appear on maps: "
            + ", ".join(missing_geometry[:10])
            + (" ..." if len(missing_geometry) > 10 else "")
        )

    if page == "Overall risk":
        render_overall_risk(tables, geometry, basin, selected_bsr, MAP_STYLE)
    elif page == "Fish use":
        render_fish_use(tables, geometry, basin, selected_bsr, MAP_STYLE)
    elif page == "Limiting factors":
        render_limiting_factors(tables, geometry, basin, selected_bsr, MAP_STYLE)
    else:
        render_actions(tables, geometry, basin, selected_bsr, MAP_STYLE)

    st.divider()
    st.caption(
        "Scores are relative prioritization indicators derived from the provisional "
        "Atlas framework."
    )


if __name__ == "__main__":
    main()
