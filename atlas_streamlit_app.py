"""Interactive Atlas scoring viewer for Streamlit.

Required packages:
    pip install "streamlit>=1.42" "plotly>=5.24" pandas numpy geopandas pyogrio shapely

Launch command:
    streamlit run atlas_streamlit_app.py

The app reads the Level 1 and Level 2 outputs created by
Atlas_Integrated_Scoring.ipynb, including the scored BSR GeoPackage. The BSR
layer must contain the same identifiers used in bsr_scores.csv, such as CC1 or
UGR11. Fish-use views distinguish the BSR-level fish_use_score, species-level
species_aggregate_score, and life-stage LS_corrected_score fields.
"""

from __future__ import annotations

import json
import re
from functools import partial
from html import escape
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
    "action_components": "QC/action_score_components.csv",
}

SCORE_FILES = {**CORE_SCORE_FILES, **SUPPORTING_SCORE_FILES}
SCORE_SCHEMA_VERSION = "2026-08-11-fish-use-levels-v2"

REQUIRED_COLUMNS = {
    "bsr": {
        "bsr",
        "basin",
        "fish_use_score",
        "overall_impact_score",
        "overall_risk_score",
        "highest_risk_species_life_stage",
        "top_species_life_stage_risk_score",
        "top_species_life_stage_risk_tie_count",
        "highest_risk_limiting_factor",
        "top_limiting_factor_risk_score",
        "top_limiting_factor_risk_tie_count",
        "highest_risk_aligned_action_type",
        "highest_action_benefit_score",
        "top_action_benefit_tie_count",
    },
    "life_stage": {
        "bsr",
        "basin",
        "species",
        "life_stage",
        "LS_corrected_score",
        "species_aggregate_score",
        "population_priority",
        "impact_score",
        "risk_score",
        "species_life_stage_label",
    },
    "limiting_factor": {
        "bsr",
        "basin",
        "limiting_factor",
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
        "LS_corrected_score",
        "species_aggregate_score",
        "fish_use_score",
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

LEGACY_COLUMN_REPLACEMENTS = {
    "highest_risk_species_life_stage": (
        "highest_priority_life_stage",
    ),
    "top_species_life_stage_risk_score": (
        "top_life_stage_risk_score",
    ),
    "top_species_life_stage_risk_tie_count": (
        "highest_priority_life_stage_tie_count",
    ),
    "species_life_stage_label": ("priority_label",),
    "highest_risk_limiting_factor": (
        "highest_priority_limiting_factor",
    ),
    "top_limiting_factor_risk_tie_count": (
        "highest_priority_limiting_factor_tie_count",
    ),
    "highest_risk_aligned_action_type": ("highest_priority_action",),
    "highest_action_benefit_score": (
        "highest_priority_action_benefit_score",
    ),
    "top_action_benefit_tie_count": (
        "highest_priority_action_tie_count",
    ),
}

DEFAULT_COLOR_SCALE = "Blues"
RISK_COLOR_SCALE = "Reds"
LIMITING_IMPACT_COLOR_SCALE = "Purples"
ACTION_BENEFIT_COLOR_SCALE = "Greens"

OVERALL_LIMITING_FACTOR_HELP = (
    "Overall limiting-factor impact sums Overall Fish Use Score × condition "
    "score × vulnerability score across species, life stages, and limiting "
    "factors. Population-weighted limiting-factor risk uses the same "
    "components after multiplying each species and life-stage pathway by its "
    "population priority. Both are relative aggregate scores, not "
    "probabilities, and may exceed 1."
)
LIMITING_FACTOR_SELECTION_HELP = (
    "Select the limiting factor used in the comparison chart, biological "
    "drill-down, data table, and specific limiting-factor map."
)
FACTOR_SPECIFIC_MAP_HELP = (
    "Impact score sums Overall Fish Use Score × condition score × "
    "vulnerability score for the selected limiting factor. Population-weighted "
    "risk score additionally weights each species and life-stage contribution "
    "by population priority. Condition score is the selected factor's "
    "0.1-to-1 input score and does not include fish use, vulnerability, or "
    "population priority."
)
ACTION_MAP_HELP = (
    "Condition improvement score sums condition score × the action-to-factor "
    "weight. Limiting-factor amelioration score applies that weight to factor "
    "impact. Overall benefit score applies it to population-weighted factor "
    "risk. These scores indicate relative alignment, not expected project "
    "effectiveness, feasibility, or cost."
)

DISPLAY_LABELS = {
    "bsr": "BSR",
    "basin": "Basin",
    "fish_use_score": "Overall Fish Use Score",
    "species_aggregate_score": "Species Fish Use Score",
    "LS_corrected_score": "Life-Stage Fish Use Score",
    "overall_impact_score": "Overall Limiting-Factor Impact",
    "overall_risk_score": "Overall Risk Score",
    "highest_risk_species_life_stage": "Highest Priority Life Stage",
    "top_species_life_stage_risk_score": (
        "Highest Priority Life Stage Score"
    ),
    "top_species_life_stage_risk_tie_count": (
        "Highest Priority Life Stage Tie Count"
    ),
    "species_life_stage_label": "Species | Life Stage",
    "highest_risk_limiting_factor": "Highest Priority Limiting Factor",
    "top_limiting_factor_risk_score": (
        "Highest Priority Limiting Factor Score"
    ),
    "top_limiting_factor_risk_tie_count": (
        "Highest Priority Limiting Factor Tie Count"
    ),
    "condition_score": "Condition Score",
    "impact_score": "Impact Score",
    "risk_score": "Risk Score",
    "condition_improvement_score": "Condition Improvement Score",
    "limiting_factor_amelioration_score": "Limiting-Factor Amelioration Score",
    "overall_benefit_score": "Overall Benefit Score",
    "highest_risk_aligned_action_type": "Highest Risk-Aligned Action Type",
    "highest_action_benefit_score": "Highest Action Benefit Score",
    "top_action_benefit_tie_count": "Top Action Benefit Tie Count",
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


def render_scoring_methodology() -> None:
    """Explain the score inputs, equations, aggregation, and interpretation."""
    with st.expander("How scores are calculated"):
        st.markdown(
            """
            Each BSR is evaluated for every species, life stage, and limiting
            factor combination. Higher input values increase the calculated
            score.

            **Fish-use inputs**

            - **Overall Fish Use Score** is the BSR-level 0-to-1 fish-use
              score. It is applied as the common fish-use multiplier in the
              impact and risk calculations.
            - **Species Fish Use Score** and **Life-Stage Fish Use Score** are
              carried directly from the fish-use source data for reporting.
              They are not substituted for the Overall Fish Use Score in the
              current impact or risk equations.

            **Level 1 calculations**

            | Score component | Calculation |
            |---|---|
            | Impact component | Overall Fish Use Score x condition score x vulnerability score |
            | Risk component | Impact component x population priority |
            | Overall Limiting-Factor Impact | Sum of impact components across species, life stages, and limiting factors |
            | Overall Risk Score | Sum of risk components across species, life stages, and limiting factors |

            Overall Limiting-Factor Impact does not use population priority.
            Population-Weighted Limiting-Factor Risk uses the same impact
            components after multiplying each species and life-stage pathway
            by its population-priority value. It therefore gives more weight
            to pathways associated with higher-priority populations. On the
            Overall Risk page, this population-weighted total is labeled
            Overall Risk Score.

            The condition score maps the source 1-to-5 rating linearly to
            0.1-to-1.0. The vulnerability score maps rank 1 to 1.0 and rank 15
            to 0.1. For the combined Migration life stage, the calculation
            uses the higher of the adult and juvenile vulnerability scores.
            Species/life-stage and limiting-factor summaries are different
            groupings of the same components and reconcile to the BSR totals.

            **Level 2 action calculations**

            The action-to-limiting-factor weight equals relationship
            directness x frequency. For each action type, the app sums:

            - condition score x action weight for the **Condition Improvement Score**;
            - limiting-factor impact x action weight for the **Limiting-Factor Amelioration Score**; and
            - limiting-factor risk x action weight for the **Overall Benefit Score**.

            Aggregate scores are relative prioritization indicators and may
            exceed 1 because components are summed. They are not probabilities.
            Action scores indicate alignment with calculated risk, not expected
            project effectiveness, feasibility, cost, or realized benefit.
            The Highest Priority Life Stage and Highest Priority Limiting Factor
            identify the largest risk contributions within each BSR and are
            placeholders for future consideration.
            """
        )


def require_columns(table_name: str, table: pd.DataFrame) -> None:
    """Raise a readable error when an expected notebook output field is absent."""
    missing = sorted(REQUIRED_COLUMNS[table_name] - set(table.columns))
    if missing:
        legacy_matches = []
        for current_name in missing:
            for legacy_name in LEGACY_COLUMN_REPLACEMENTS.get(
                current_name, ()
            ):
                if legacy_name in table.columns:
                    legacy_matches.append(
                        f"{legacy_name} -> {current_name}"
                    )
        legacy_detail = (
            " Legacy fields found: " + ", ".join(legacy_matches) + "."
            if legacy_matches
            else ""
        )
        raise ValueError(
            f"{SCORE_FILES[table_name]} is missing current scoring columns: "
            f"{missing}.{legacy_detail} Re-run the revised integrated-scoring "
            "notebook and deploy its updated data/outputs files. The app does "
            "not substitute retired score fields."
        )


def validate_fish_use_detail_scores(
    life_stage: pd.DataFrame,
    calculation_grid: pd.DataFrame,
) -> None:
    """Verify that species and life-stage fish-use fields remain aligned."""
    keys = ["bsr", "basin", "species", "life_stage"]
    score_fields = ["LS_corrected_score", "species_aggregate_score"]

    for table_name, table in (
        ("life_stage_scores.csv", life_stage),
        ("calculation_grid.csv", calculation_grid),
    ):
        for field in score_fields:
            table[field] = pd.to_numeric(table[field], errors="coerce")
        if table[score_fields].isna().any().any():
            raise ValueError(
                f"{table_name} contains missing or nonnumeric species or "
                "life-stage fish-use scores."
            )
        if table[score_fields].lt(0).any().any():
            raise ValueError(
                f"{table_name} contains negative species or life-stage "
                "fish-use scores."
            )

    if life_stage.duplicated(keys).any():
        raise ValueError(
            "life_stage_scores.csv contains duplicate BSR, species, and "
            "life-stage records."
        )

    species_variants = life_stage.groupby(
        ["bsr", "species"], dropna=False
    )["species_aggregate_score"].nunique(dropna=False)
    if species_variants.gt(1).any():
        raise ValueError(
            "life_stage_scores.csv contains inconsistent "
            "species_aggregate_score values within a BSR and species."
        )

    grid_variants = calculation_grid.groupby(keys, dropna=False)[
        score_fields
    ].nunique(dropna=False)
    if grid_variants.gt(1).any().any():
        raise ValueError(
            "calculation_grid.csv contains inconsistent fish-use scores "
            "across limiting factors for the same life-stage record."
        )

    life_values = life_stage.set_index(keys)[score_fields].sort_index()
    grid_values = calculation_grid.groupby(keys, dropna=False)[
        score_fields
    ].first().sort_index()
    if not life_values.index.equals(grid_values.index) or not np.allclose(
        life_values.to_numpy(), grid_values.to_numpy()
    ):
        raise ValueError(
            "Species or life-stage fish-use scores do not match between "
            "life_stage_scores.csv and calculation_grid.csv."
        )


def score_file_signature(
    score_dir_text: str,
) -> tuple[tuple[str, int, int], ...]:
    """Return file metadata used to invalidate cached score tables."""
    score_dir = Path(score_dir_text).expanduser()
    signature = []
    for filename in SCORE_FILES.values():
        path = score_dir / filename
        if path.is_file():
            status = path.stat()
            signature.append(
                (filename, status.st_mtime_ns, status.st_size)
            )
        else:
            signature.append((filename, -1, -1))
    return tuple(signature)


@st.cache_data(show_spinner=False)
def load_score_tables(
    score_dir_text: str,
    schema_version: str,
    file_signature: tuple[tuple[str, int, int], ...],
) -> dict[str, pd.DataFrame]:
    """Load core scoring tables and available action components."""
    if schema_version != SCORE_SCHEMA_VERSION:
        raise ValueError("The requested score schema version is not supported.")
    if file_signature != score_file_signature(score_dir_text):
        raise ValueError(
            "One or more scoring files changed while they were being loaded. "
            "Refresh the app to reload a consistent set of outputs."
        )
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

    validate_fish_use_detail_scores(
        tables["life_stage"], tables["grid"]
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


def natural_sort_key(value: Any) -> tuple[Any, ...]:
    """Sort identifiers by text and embedded integer components."""
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", str(value))
        if part
    )


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
        width="stretch",
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
    hover_columns = [
        column
        for column in (hover_columns or [])
        if column not in {"bsr", "basin"}
    ]
    if metric not in values.columns:
        st.error(
            f"Cannot render {metric_label}: required field {metric!r} is "
            "missing from the current score table. Re-run the revised "
            "integrated-scoring notebook and deploy the updated outputs."
        )
        return
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

    hover_fields = list(dict.fromkeys(["bsr", metric, *hover_columns]))
    hover_fields = [
        column for column in hover_fields if column in plot_data.columns
    ]
    hover_labels = {
        column: DISPLAY_LABELS.get(
            column,
            column.replace("_", " ").title(),
        )
        for column in hover_fields
    } | {metric: metric_label}

    def hover_text(row: pd.Series) -> str:
        lines = []
        for column in hover_fields:
            value = row[column]
            if pd.isna(value):
                displayed = "Not available"
            elif pd.api.types.is_numeric_dtype(plot_data[column]):
                displayed = f"{float(value):,.2f}"
            else:
                displayed = str(value)
            label = escape(str(hover_labels[column]))
            lines.append(f"<b>{label}:</b> {escape(displayed)}")
        return "<br>".join(lines)

    plot_data["_hover_text"] = plot_data.apply(hover_text, axis=1)

    common = {
        "data_frame": plot_data,
        "geojson": geojson,
        "locations": "bsr",
        "featureidkey": "properties.bsr",
        "color": metric,
        "hover_name": None,
        "hover_data": {"bsr": False, "_hover_text": False},
        "custom_data": ["bsr", "_hover_text"],
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

    figure.update_traces(
        marker_line_width=1.1,
        marker_line_color="#ffffff",
        hovertemplate="%{customdata[1]}<extra></extra>",
    )
    st.plotly_chart(
        figure,
        width="stretch",
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
        width="stretch",
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
        "partition the selected BSR score by species and life stage and by "
        "limiting factor."
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
            "fish_use_score",
            "overall_impact_score",
            "highest_risk_species_life_stage",
            "highest_risk_limiting_factor",
        ],
        color_scale=RISK_COLOR_SCALE,
    )

    row = bsr.loc[bsr["bsr"].eq(selected_bsr)].iloc[0]
    metric_columns = st.columns(4)
    metric_columns[0].metric("Selected BSR", selected_bsr)
    metric_columns[1].metric("Overall risk", format_score(row["overall_risk_score"]))
    metric_columns[2].metric(
        "Overall fish use score", format_score(row["fish_use_score"])
    )
    metric_columns[3].metric("Overall LF impact", format_score(row["overall_impact_score"]))

    st.markdown(
        "**Highest Priority Life Stage:** "
        f"{row['highest_risk_species_life_stage']}  \n"
        "**Highest Priority Limiting Factor:** "
        f"{row['highest_risk_limiting_factor']}"
    )
    st.caption(
        "The Highest Priority Life Stage and Highest Priority Limiting "
        "Factor indicators are placeholders for future consideration."
    )

    left, right = st.columns(2)
    life_selected = life.loc[life["bsr"].eq(selected_bsr)].copy()
    factor_selected = limiting.loc[limiting["bsr"].eq(selected_bsr)].copy()

    with left:
        horizontal_bar(
            life_selected,
            "risk_score",
            "species_life_stage_label",
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
                    "species_aggregate_score",
                    "life_stage",
                    "LS_corrected_score",
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
    """Render overall, species, and life-stage fish-use scores."""
    bsr = filter_table(tables["bsr"], basin)
    life = filter_table(tables["life_stage"], basin)

    species_hover_scores = (
        life[["bsr", "species", "species_aggregate_score"]]
        .drop_duplicates()
        .pivot(
            index="bsr",
            columns="species",
            values="species_aggregate_score",
        )
        .sort_index(axis=1)
    )
    species_hover_scores.columns = [
        f"{species} Fish Use Score"
        for species in species_hover_scores.columns
    ]
    species_hover_scores.columns.name = None
    species_hover_columns = species_hover_scores.columns.tolist()
    fish_hover_values = bsr.merge(
        species_hover_scores.reset_index(),
        on="bsr",
        how="left",
        validate="one_to_one",
    )

    st.header("Level 1: Fish use")
    st.caption(
        "Overall fish use is reported as fish_use_score. Species scores use "
        "species_aggregate_score, and life-stage scores use "
        "LS_corrected_score."
    )

    map_level = st.radio(
        "Fish-use map level",
        options=["Overall", "Species", "Life stage"],
        horizontal=True,
    )
    species_options = sorted(life["species"].dropna().unique())
    map_species = None
    map_life_stage = None

    if map_level == "Overall":
        map_values = fish_hover_values
        map_metric = "fish_use_score"
        map_metric_label = "Overall fish use score"
        map_title = "Overall Fish Use Score"
        hover_columns = ["fish_use_score", *species_hover_columns]
    else:
        map_species = st.selectbox(
            "Species",
            species_options,
            key="fish_use_map_species",
        )
        species_rows = life.loc[life["species"].eq(map_species)].copy()
        if map_level == "Species":
            map_values = species_rows[
                [
                    "bsr",
                    "species",
                    "species_aggregate_score",
                ]
            ].drop_duplicates("bsr").merge(
                fish_hover_values[
                    ["bsr", "fish_use_score", *species_hover_columns]
                ],
                on="bsr",
                how="left",
                validate="one_to_one",
            )
            map_metric = "species_aggregate_score"
            map_metric_label = f"{map_species} fish use score"
            map_title = f"{map_species}: Species Fish Use Score"
            selected_species_hover = f"{map_species} Fish Use Score"
            hover_columns = [
                "species",
                "fish_use_score",
                *[
                    column
                    for column in species_hover_columns
                    if column != selected_species_hover
                ],
            ]
        else:
            life_stage_options = sorted(
                species_rows["life_stage"].dropna().unique()
            )
            map_life_stage = st.selectbox(
                "Life stage",
                life_stage_options,
                key="fish_use_map_life_stage",
            )
            map_values = species_rows.loc[
                species_rows["life_stage"].eq(map_life_stage),
                [
                    "bsr",
                    "species",
                    "life_stage",
                    "LS_corrected_score",
                    "species_aggregate_score",
                ],
            ].merge(
                fish_hover_values[
                    ["bsr", "fish_use_score", *species_hover_columns]
                ],
                on="bsr",
                how="left",
                validate="one_to_one",
            )
            map_metric = "LS_corrected_score"
            map_metric_label = "Life-stage fish use score"
            map_title = (
                f"{map_species} | {map_life_stage}: "
                "Life-Stage Fish Use Score"
            )
            hover_columns = [
                "species",
                "life_stage",
                "fish_use_score",
                *species_hover_columns,
            ]

    render_choropleth(
        geometry,
        map_values,
        map_metric,
        map_metric_label,
        map_title,
        "map_fish_use",
        map_style,
        hover_columns=hover_columns,
    )

    selected = life.loc[life["bsr"].eq(selected_bsr)].copy()
    selected_bsr_row = bsr.loc[bsr["bsr"].eq(selected_bsr)].iloc[0]
    selected_map_row = map_values.loc[map_values["bsr"].eq(selected_bsr)]
    mapped_score = (
        selected_map_row[map_metric].iloc[0]
        if not selected_map_row.empty
        else np.nan
    )
    summary_columns = st.columns(3)
    summary_columns[0].metric("Selected BSR", selected_bsr)
    summary_columns[1].metric(
        "Overall fish use score",
        format_score(selected_bsr_row["fish_use_score"]),
    )
    summary_columns[2].metric(
        map_metric_label,
        format_score(mapped_score),
    )

    species_summary = (
        selected[["species", "species_aggregate_score"]]
        .drop_duplicates()
        .sort_values("species")
    )
    left, right = st.columns(2)
    with left:
        horizontal_bar(
            species_summary,
            "species_aggregate_score",
            "species",
            f"{selected_bsr}: fish use by species",
            value_label="Species fish use score",
        )
    with right:
        selected_species_options = sorted(selected["species"].unique())
        default_species = (
            map_species
            if map_species in selected_species_options
            else selected_species_options[0]
        )
        chart_species = st.selectbox(
            "Species for life-stage chart",
            selected_species_options,
            index=selected_species_options.index(default_species),
            key="fish_use_chart_species",
        )
        life_stage_summary = selected.loc[
            selected["species"].eq(chart_species),
            ["life_stage", "LS_corrected_score"],
        ]
        horizontal_bar(
            life_stage_summary,
            "LS_corrected_score",
            "life_stage",
            f"{selected_bsr}: {chart_species} fish use by life stage",
            value_label="Life-stage fish use score",
        )

    with st.expander("Highest Priority Life Stage"):
        st.caption(
            "This placeholder indicator is based on the population-weighted "
            "Level 1 risk score, not fish use alone, and is retained for "
            "future consideration."
        )
        render_choropleth(
            geometry,
            fish_hover_values,
            "highest_risk_species_life_stage",
            "Highest Priority Life Stage",
            "Highest Priority Life Stage",
            "map_top_life_stage",
            map_style,
            categorical=True,
            hover_columns=[
                "fish_use_score",
                *species_hover_columns,
                "top_species_life_stage_risk_score",
                "overall_risk_score",
            ],
        )

    with st.expander(f"Show species and life-stage data for BSR: {selected_bsr}"):
        show_score_table(
            selected[
                [
                    "species",
                    "species_aggregate_score",
                    "life_stage",
                    "LS_corrected_score",
                ]
            ].sort_values(
                ["species_aggregate_score", "species", "LS_corrected_score"],
                ascending=[False, True, False],
            )
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
        help=OVERALL_LIMITING_FACTOR_HELP,
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
            "fish_use_score",
            "overall_impact_score",
            "overall_risk_score",
            "highest_risk_limiting_factor",
            "top_limiting_factor_risk_score",
        ],
        color_scale=overall_color_scale,
    )

    st.subheader("Specific limiting-factor drill-down")
    factor_options = sorted(limiting["limiting_factor"].dropna().unique())
    selected_factor = st.selectbox(
        "Limiting factor",
        factor_options,
        help=LIMITING_FACTOR_SELECTION_HELP,
    )
    factor_score_labels = {
        "Impact score": "impact_score",
        "Population-weighted risk score": "risk_score",
        "Condition score": "condition_score",
    }
    factor_score_label = st.radio(
        "Factor-specific map value",
        options=list(factor_score_labels),
        horizontal=True,
        help=FACTOR_SPECIFIC_MAP_HELP,
    )
    factor_score = factor_score_labels[factor_score_label]
    factor_color_scale = {
        "impact_score": LIMITING_IMPACT_COLOR_SCALE,
        "risk_score": RISK_COLOR_SCALE,
        "condition_score": DEFAULT_COLOR_SCALE,
    }[factor_score]
    factor_map = limiting.loc[limiting["limiting_factor"].eq(selected_factor)].copy()

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
    biological["species_life_stage_label"] = (
        biological["species"] + " | " + biological["life_stage"]
    )
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
            "species_life_stage_label",
            f"{selected_bsr}: {selected_factor} by species and life stage",
            color="species",
            value_label=component_label,
        )

    with st.expander("Highest Priority Limiting Factor map"):
        st.caption(
            "This priority indicator is a placeholder retained for future "
            "consideration."
        )
        render_choropleth(
            geometry,
            bsr,
            "highest_risk_limiting_factor",
            "Highest Priority Limiting Factor",
            "Highest Priority Limiting Factor",
            "map_top_limiting_factor",
            map_style,
            categorical=True,
            hover_columns=[
                "fish_use_score",
                "overall_impact_score",
                "overall_risk_score",
                "top_limiting_factor_risk_score",
                "top_limiting_factor_risk_tie_count",
            ],
        )

    with st.expander(f"Show limiting-factor data for: {selected_factor} in {selected_bsr}"):
        show_score_table(
            biological[
                [
                    "species",
                    "species_aggregate_score",
                    "life_stage",
                    "LS_corrected_score",
                    "fish_use_score",
                    "population_priority",
                    "condition_score",
                    "vulnerability_score",
                    "impact_component",
                    "risk_component",
                ]
            ].sort_values("risk_component", ascending=False)
        )

    st.subheader("Specific limiting factor map")
    render_choropleth(
        geometry,
        factor_map,
        factor_score,
        factor_score_label,
        f"{selected_factor}: {factor_score_label}",
        "map_specific_limiting_factor",
        map_style,
        hover_columns=[
            "condition_score",
            "impact_score",
            "risk_score",
        ],
        color_scale=factor_color_scale,
    )


def render_actions(
    tables: dict[str, pd.DataFrame],
    geometry: gpd.GeoDataFrame,
    basin: str,
    selected_bsr: str,
    map_style: str,
) -> None:
    """Render Level 2 action maps, rankings, and score components."""
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
        help=ACTION_MAP_HELP,
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

    st.subheader("Highest risk-aligned action type")
    render_choropleth(
        geometry,
        bsr,
        "highest_risk_aligned_action_type",
        "Highest risk-aligned action type",
        "Highest Risk-Aligned Action Type",
        "map_top_action",
        map_style,
        categorical=True,
        hover_columns=[
            "highest_action_benefit_score",
            "top_action_benefit_tie_count",
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
    render_scoring_methodology()

    try:
        score_signature = score_file_signature(str(SCORE_DIR))
        tables = load_score_tables(
            str(SCORE_DIR), SCORE_SCHEMA_VERSION, score_signature
        )
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
    available_bsrs = sorted(
        filter_table(tables["bsr"], basin)["bsr"].unique(),
        key=natural_sort_key,
    )
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
