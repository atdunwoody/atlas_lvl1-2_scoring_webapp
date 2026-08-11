"""Interactive Atlas scoring viewer for Streamlit.

Required packages:
    pip install "streamlit>=1.42" "plotly>=5.24" pandas numpy geopandas pyogrio shapely

Launch command:
    streamlit run atlas_streamlit_app.py

The app reads the Level 1 and Level 2 CSV outputs created by
Atlas_Integrated_Scoring.ipynb. A polygon layer is also required because the
score outputs do not contain geometry. The polygon layer must contain either
the fish-use BSR identifier or the condition BSR identifier.
"""

from __future__ import annotations

import json
import os
import tempfile
from functools import partial
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


DEFAULT_SCORE_DIR = Path(
    os.getenv(
        "ATLAS_SCORE_DIR",
        r"data\outputs",
    )
)
DEFAULT_BSR_LAYER = os.getenv("ATLAS_BSR_LAYER", "")
DEFAULT_BSR_LAYER_NAME = os.getenv("ATLAS_BSR_LAYER_NAME", "")
DEFAULT_BSR_ID_FIELD = os.getenv("ATLAS_BSR_ID_FIELD", "BSR")

SCORE_FILES = {
    "bsr": "bsr_scores.csv",
    "life_stage": "life_stage_scores.csv",
    "limiting_factor": "limiting_factor_scores_integrated.csv",
    "action": "action_scores.csv",
    "grid": "calculation_grid.csv",
}

REQUIRED_COLUMNS = {
    "bsr": {
        "bsr",
        "basin",
        "condition_bsr",
        "source_fish_use_score",
        "source_fish_use_score_100",
        "workbook_fish_use_score",
        "overall_impact_score",
        "overall_risk_score",
        "highest_priority_life_stage",
        "highest_priority_limiting_factor",
        "highest_priority_action_type",
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
}

NUMERIC_COLOR_SCALE = [
    [0.00, "#f7fbff"],
    [0.25, "#c6dbef"],
    [0.50, "#6baed6"],
    [0.75, "#2171b5"],
    [1.00, "#08306b"],
]


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
    """Load and validate the five scoring tables used by the app."""
    score_dir = Path(score_dir_text).expanduser()
    missing_files = [
        filename
        for filename in SCORE_FILES.values()
        if not (score_dir / filename).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"Missing scoring outputs in {score_dir}: {missing_files}"
        )

    tables = {
        name: pd.read_csv(score_dir / filename)
        for name, filename in SCORE_FILES.items()
    }

    for name, table in tables.items():
        require_columns(name, table)
        table["bsr"] = table["bsr"].astype(str).str.strip()
        table["basin"] = table["basin"].astype(str).str.strip()

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


@st.cache_data(show_spinner=False)
def load_geometry_upload(
    file_name: str,
    file_bytes: bytes,
    layer_name: str,
) -> gpd.GeoDataFrame:
    """Read an uploaded GeoPackage or GeoJSON through a temporary file."""
    suffix = Path(file_name).suffix.lower()
    if suffix not in {".gpkg", ".geojson", ".json"}:
        raise ValueError("Upload a .gpkg, .geojson, or .json polygon file.")

    with tempfile.TemporaryDirectory(prefix="atlas_bsr_") as temp_dir:
        temp_path = Path(temp_dir) / f"bsr{suffix}"
        temp_path.write_bytes(file_bytes)
        layer = layer_name.strip() or None
        return gpd.read_file(temp_path, layer=layer)


def prepare_geometry(
    source: gpd.GeoDataFrame,
    spatial_id_field: str,
    id_convention: str,
    bsr_scores: pd.DataFrame,
) -> tuple[gpd.GeoDataFrame, list[str]]:
    """Standardize geometry and translate its identifier to the app's BSR key."""
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

    if id_convention == "Condition BSR identifier":
        crosswalk = bsr_scores[["bsr", "condition_bsr"]].copy()
        crosswalk["condition_bsr"] = (
            crosswalk["condition_bsr"].astype(str).str.strip()
        )
        if crosswalk["condition_bsr"].duplicated().any():
            raise ValueError(
                "condition_bsr is not unique in bsr_scores.csv and cannot be "
                "used as a spatial crosswalk."
            )
        spatial = spatial.merge(
            crosswalk,
            left_on="_spatial_id",
            right_on="condition_bsr",
            how="left",
            validate="many_to_one",
        )
    else:
        spatial["bsr"] = spatial["_spatial_id"]

    unmatched_ids = sorted(spatial.loc[spatial["bsr"].isna(), "_spatial_id"].unique())
    spatial = spatial.loc[spatial["bsr"].notna(), ["bsr", "geometry"]].copy()
    if spatial.empty:
        raise ValueError(
            "No spatial identifiers matched the scoring outputs. Check the ID "
            "field and identifier convention."
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


def format_score(value: Any, digits: int = 3) -> str:
    """Format numeric KPI values without implying unnecessary precision."""
    if pd.isna(value):
        return "Not available"
    return f"{float(value):,.{digits}f}"


def map_center_zoom(geometry: gpd.GeoDataFrame) -> tuple[dict[str, float], float]:
    """Estimate a reasonable initial center and zoom from polygon bounds."""
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
    return center, zoom


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
    plot_data = pd.DataFrame(mapped.drop(columns="geometry"))
    center, zoom = map_center_zoom(mapped)

    common = {
        "data_frame": plot_data,
        "geojson": geojson,
        "locations": "bsr",
        "featureidkey": "properties.bsr",
        "color": metric,
        "hover_name": "bsr",
        "hover_data": {
            column: True
            for column in hover_columns
            if column in plot_data.columns and column != metric
        },
        "custom_data": ["bsr"],
        "opacity": 0.78,
        "zoom": zoom,
        "center": center,
        "title": title,
        "labels": {metric: metric_label},
    }

    if categorical:
        common["color_discrete_sequence"] = px.colors.qualitative.Safe
    else:
        common["color_continuous_scale"] = NUMERIC_COLOR_SCALE
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
    ordered = table.sort_values(value, ascending=True)
    figure = px.bar(
        ordered,
        x=value,
        y=category,
        color=color,
        orientation="h",
        title=title,
        labels={value: value_label or value, category: ""},
        text_auto=".3f",
    )
    figure.update_layout(
        height=max(360, 30 * len(ordered) + 110),
        margin={"r": 10, "t": 55, "l": 10, "b": 10},
        legend_title_text="Species" if color == "species" else color,
    )
    figure.update_traces(textposition="outside", cliponaxis=False)
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
            "highest_priority_life_stage",
            "highest_priority_limiting_factor",
        ],
    )

    row = bsr.loc[bsr["bsr"].eq(selected_bsr)].iloc[0]
    metric_columns = st.columns(4)
    metric_columns[0].metric("Selected BSR", selected_bsr)
    metric_columns[1].metric("Overall risk", format_score(row["overall_risk_score"]))
    metric_columns[2].metric("Fish Use Score /100", format_score(row["source_fish_use_score_100"], 1))
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

    with st.expander("Show selected BSR score tables"):
        st.subheader("Species and life stages")
        st.dataframe(
            life_selected[
                [
                    "species",
                    "life_stage",
                    "fish_use_rating",
                    "population_priority",
                    "impact_score",
                    "risk_score",
                ]
            ].sort_values("risk_score", ascending=False),
            use_container_width=True,
            hide_index=True,
        )
        st.subheader("Limiting factors")
        st.dataframe(
            factor_selected[
                [
                    "limiting_factor",
                    "condition_score_raw_1_5",
                    "condition_score",
                    "impact_score",
                    "risk_score",
                ]
            ].sort_values("risk_score", ascending=False),
            use_container_width=True,
            hide_index=True,
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
    fish_metric_labels = {
        "Fish Use Score /100": "source_fish_use_score_100",
        "Source Fish Use Score": "source_fish_use_score",
        "Sum of detailed life-stage Score": "workbook_fish_use_score",
    }
    metric_label = st.radio(
        "Overall fish-use map value",
        options=list(fish_metric_labels),
        horizontal=True,
    )
    metric = fish_metric_labels[metric_label]
    render_choropleth(
        geometry,
        bsr,
        metric,
        metric_label,
        metric_label,
        "map_fish_use",
        map_style,
        hover_columns=["basin", "source_fish_use_score", "source_fish_use_score_100"],
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
            hover_columns=["basin", "top_life_stage_risk_score"],
        )

    with st.expander("Show species and life-stage data"):
        st.dataframe(
            selected[
                [
                    "species",
                    "life_stage",
                    "fish_use_rating",
                    "population_priority",
                    "impact_score",
                    "risk_score",
                ]
            ].sort_values(["species", "life_stage"]),
            use_container_width=True,
            hide_index=True,
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
    render_choropleth(
        geometry,
        bsr,
        overall_labels[overall_label],
        overall_label,
        overall_label,
        "map_limiting_factor_overall",
        map_style,
        hover_columns=["basin", "highest_priority_limiting_factor"],
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
    factor_map = limiting.loc[limiting["limiting_factor"].eq(selected_factor)].copy()
    render_choropleth(
        geometry,
        factor_map,
        factor_score,
        factor_score_label,
        f"{selected_factor}: {factor_score_label}",
        "map_specific_limiting_factor",
        map_style,
        hover_columns=["basin", "condition_score_raw_1_5", "condition_score"],
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

    with st.expander("Optional map: highest-priority limiting factor"):
        render_choropleth(
            geometry,
            bsr,
            "highest_priority_limiting_factor",
            "Highest-priority limiting factor",
            "Highest-Priority Limiting Factor",
            "map_top_limiting_factor",
            map_style,
            categorical=True,
            hover_columns=["basin", "top_limiting_factor_risk_score"],
        )

    with st.expander("Show selected limiting-factor data"):
        st.dataframe(
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
            ].sort_values("risk_component", ascending=False),
            use_container_width=True,
            hide_index=True,
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
        hover_columns=["basin", "benefit_rank_within_bsr"],
    )

    selected = actions.loc[actions["bsr"].eq(selected_bsr)].copy()
    horizontal_bar(
        selected,
        action_score,
        "action_type",
        f"{selected_bsr}: action-specific scores",
        value_label=action_score_label,
    )

    st.subheader("Highest-priority action type")
    render_choropleth(
        geometry,
        bsr,
        "highest_priority_action_type",
        "Highest-priority action type",
        "Highest-Priority Action Type",
        "map_top_action",
        map_style,
        categorical=True,
        hover_columns=["basin", "highest_priority_action_benefit_score"],
    )

    with st.expander("Provisional map: sum of action-specific benefit scores"):
        st.warning(
            "This score is exploratory. Summing action scores may double count "
            "overlapping benefits and may reward BSRs with more applicable action "
            "pathways. Do not treat it as a finalized overall restoration-potential "
            "score without further review."
        )
        render_choropleth(
            geometry,
            bsr,
            "sum_action_benefit_provisional",
            "Sum of action-specific benefit scores",
            "Provisional Overall Benefit Score",
            "map_action_sum",
            map_style,
            hover_columns=["basin", "highest_priority_action_type"],
        )

    with st.expander("Show selected BSR action table"):
        st.dataframe(
            selected[
                [
                    "action_id",
                    "action_type",
                    "condition_improvement_score",
                    "limiting_factor_amelioration_score",
                    "overall_benefit_score",
                    "benefit_rank_within_bsr",
                ]
            ].sort_values("benefit_rank_within_bsr"),
            use_container_width=True,
            hide_index=True,
        )


def geometry_controls(bsr_scores: pd.DataFrame) -> gpd.GeoDataFrame:
    """Render sidebar spatial controls and return app-ready BSR polygons."""
    st.sidebar.subheader("BSR polygons")
    source_mode = st.sidebar.radio(
        "Spatial source",
        ["File path", "Upload file"],
        horizontal=True,
    )
    layer_name = st.sidebar.text_input(
        "GeoPackage layer name (optional)",
        value=DEFAULT_BSR_LAYER_NAME,
        help="Leave blank to read the default or only layer.",
    )

    if source_mode == "File path":
        path_text = st.sidebar.text_input(
            "GeoPackage or GeoJSON path",
            value=DEFAULT_BSR_LAYER,
        ).strip()
        if not path_text:
            st.info(
                "Provide a BSR polygon path in the sidebar, or switch to Upload file. "
                "The scoring CSVs do not contain geometry."
            )
            st.stop()
        source = load_geometry_path(path_text, layer_name)
    else:
        uploaded = st.sidebar.file_uploader(
            "Upload BSR polygons",
            type=["gpkg", "geojson", "json"],
        )
        if uploaded is None:
            st.info("Upload a BSR GeoPackage or GeoJSON file to draw the maps.")
            st.stop()
        source = load_geometry_upload(uploaded.name, uploaded.getvalue(), layer_name)

    attribute_columns = [column for column in source.columns if column != "geometry"]
    if not attribute_columns:
        raise ValueError("The spatial layer contains no attribute fields.")
    default_index = 0
    for index, column in enumerate(attribute_columns):
        if column.lower() == DEFAULT_BSR_ID_FIELD.lower():
            default_index = index
            break
    spatial_id_field = st.sidebar.selectbox(
        "Spatial BSR ID field",
        attribute_columns,
        index=default_index,
    )
    id_convention = st.sidebar.radio(
        "Spatial identifier matches",
        ["Fish-use BSR identifier", "Condition BSR identifier"],
        help=(
            "Choose Condition BSR when the polygons use CC1 through CC9. "
            "Choose Fish-use BSR when they use identifiers such as CC2B or CC3B1."
        ),
    )

    geometry, unmatched = prepare_geometry(
        source,
        spatial_id_field,
        id_convention,
        bsr_scores,
    )
    if unmatched:
        st.sidebar.warning(
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

    st.sidebar.header("Atlas controls")
    score_dir = st.sidebar.text_input(
        "Scoring output directory",
        value=str(DEFAULT_SCORE_DIR),
        help="Directory created by Atlas_Integrated_Scoring.ipynb.",
    )

    try:
        tables = load_score_tables(score_dir)
    except Exception as error:
        st.error(str(error))
        st.stop()

    try:
        geometry = geometry_controls(tables["bsr"])
    except Exception as error:
        st.error(str(error))
        st.stop()

    st.sidebar.subheader("Map and drill-down")
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
    map_style = st.sidebar.selectbox(
        "Base map",
        ["carto-positron", "open-street-map", "white-bg"],
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
        render_overall_risk(tables, geometry, basin, selected_bsr, map_style)
    elif page == "Fish use":
        render_fish_use(tables, geometry, basin, selected_bsr, map_style)
    elif page == "Limiting factors":
        render_limiting_factors(tables, geometry, basin, selected_bsr, map_style)
    else:
        render_actions(tables, geometry, basin, selected_bsr, map_style)

    st.divider()
    st.caption(
        "Scores are relative prioritization indicators derived from the provisional "
        "Atlas framework. Review the assumptions and source review flags before "
        "publication or decision-making."
    )


if __name__ == "__main__":
    main()
