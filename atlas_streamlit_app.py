"""Interactive Atlas scoring viewer for Streamlit.

Required packages:
    pip install "streamlit>=1.42" "plotly>=5.24" pandas numpy geopandas pyogrio shapely

Launch command:
    streamlit run atlas_streamlit_app.py

The app reads the Level 1 and Level 2 outputs created by
Atlas_Integrated_Scoring.ipynb, including the scored BSR GeoPackage. The BSR
feature layer is detected from the score_bsr field written by the notebook.
Fish-use views distinguish the BSR-level fish_use_score, species-level
species_aggregate_score, and life-stage LS_corrected_score fields. Level 2
views use action_benefit_score for individual actions and
overall_benefit_score for the BSR-wide sum across actions.
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
import plotly.graph_objects as go
import pyogrio
import streamlit as st


SCORE_DIR = Path("data/outputs")
BSR_GPKG_PATH = SCORE_DIR / "bsr_scores.gpkg"
BSR_ID_FIELD_CANDIDATES = ("score_bsr", "BSR", "bsr")
MAP_STYLE = "carto-positron"
MAP_FILL_OPACITY = 0.78
MAP_SELECTED_OPACITY = 0.90
MAP_UNSELECTED_OPACITY = 0.6
MAP_BOUNDARY_COLOR = "#334155"
MAP_BOUNDARY_WIDTH = 0.4
MAP_SELECTED_OUTLINE_COLOR = "rgba(107, 114, 128, 0.85)"
MAP_SELECTED_OUTLINE_WIDTH = 2.4

CORE_SCORE_FILES = {
    "bsr": "bsr_scores.csv",
    "life_stage": "life_stage_scores.csv",
    "limiting_factor": "limiting_factor_scores_integrated.csv",
    "action": "action_scores.csv",
    "grid": "calculation_grid.csv",
    "action_components": "QC/action_score_components.csv",
}

SUPPORTING_SCORE_FILES: dict[str, str] = {}

SCORE_FILES = {**CORE_SCORE_FILES, **SUPPORTING_SCORE_FILES}
SCORE_SCHEMA_VERSION = "2026-08-12-notebook-174505-v5"

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
        "overall_benefit_score",
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
        "action_benefit_score",
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
    "Overall Limiting-Factor Impact sums Overall Fish Use Score × limiting-factor condition "
    "score × vulnerability score across species, life stages, and limiting "
    "factors. \nOverall Risk Score instead uses the Life-Stage Fish Use Score × "
    "limiting-factor condition score × vulnerability score × population "
    "priority for each species and life-stage pathway. \nA pathway with a zero "
    "Life-Stage Fish Use Score therefore has zero risk. Both are "
    "relative aggregate scores, not probabilities, and may exceed 1."
)
LIMITING_FACTOR_SELECTION_HELP = (
    "Select the limiting factor used in the comparison chart, biological "
    "drill-down, data table, and specific limiting-factor map."
)
FACTOR_SPECIFIC_MAP_HELP = (
    "Limiting-Factor Impact sums Overall Fish Use Score × limiting-factor "
    "condition score × vulnerability score for the selected limiting factor. "
    "\nLimiting-Factor Risk instead sums Life-Stage Fish Use Score × "
    "limiting-factor condition score × vulnerability score × population "
    "priority. \nLimiting-Factor Condition Score is the selected factor's "
    "0.1-to-1 input score and does not include "
    "fish use, vulnerability, or population priority."
)
ACTION_MAP_HELP = (
    "Condition Improvement Score sums limiting-factor condition score × Action "
    "Weight. \nLimiting-Factor Amelioration Score applies Action Weight to "
    "Limiting-Factor Impact. \nAction-Specific Benefit Score applies Action "
    "Weight to Limiting-Factor Risk. Overall Benefit Score sums the "
    "Action-Specific Benefit Scores for all "
    "actions within a BSR. These scores indicate relative alignment, not "
    "expected project effectiveness, feasibility, or cost."
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
        "Risk Score for Highest Priority Life Stage"
    ),
    "top_species_life_stage_risk_tie_count": (
        "Highest Priority Life Stage Tie Count"
    ),
    "species_life_stage_label": "Species | Life Stage",
    "highest_risk_limiting_factor": "Highest Priority Limiting Factor",
    "top_limiting_factor_risk_score": (
        "Limiting-Factor Risk for Highest Priority Limiting Factor"
    ),
    "top_limiting_factor_risk_tie_count": (
        "Highest Priority Limiting Factor Tie Count"
    ),
    "species": "Species",
    "life_stage": "Life Stage",
    "limiting_factor": "Limiting Factor",
    "population_priority": "Population Priority",
    "condition_score": "Limiting-Factor Condition Score",
    "vulnerability_score": "Vulnerability Score",
    "impact_component": "Impact Component",
    "risk_component": "Risk Component",
    "impact_score": "Impact Score",
    "risk_score": "Risk Score",
    "action_id": "Action ID",
    "action_type": "Action Type",
    "lfat_score": "Action Weight",
    "condition_improvement_score": "Condition Improvement Score",
    "limiting_factor_amelioration_score": "Limiting-Factor Amelioration Score",
    "condition_improvement_component": "Condition Improvement Component",
    "amelioration_component": "Limiting-Factor Amelioration Component",
    "benefit_component": "Action-Specific Benefit Component",
    "action_benefit_score": "Action-Specific Benefit Score",
    "overall_benefit_score": "Overall Benefit Score",
    "action_count": "Number of Actions",
    "highest_risk_aligned_action_type": "Highest Risk-Aligned Action Type",
    "highest_action_benefit_score": "Highest Action-Specific Benefit Score",
    "top_action_benefit_tie_count": "Top Action Benefit Tie Count",
    "benefit_rank_within_bsr": "Action-Specific Benefit Rank Within BSR",
    "highest_benefit_limiting_factor": (
        "Limiting Factor for Highest Benefit Component"
    ),
    "highest_benefit_action_type": "Action Type for Highest Benefit Component",
    "highest_benefit_limiting_factor_risk": (
        "Limiting-Factor Risk for Highest Benefit Component"
    ),
    "highest_benefit_action_weight": (
        "Action Weight for Highest Benefit Component"
    ),
    "highest_benefit_component_score": (
        "Highest Action-Specific Benefit Component"
    ),
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
              score used in overall limiting-factor impact.
            - **Life-Stage Fish Use Score** is the source
              **LS_corrected_score** used in population-weighted risk.
            - **Species Fish Use Score** is the source
              **species_aggregate_score**. It is reported for context but is
              not a direct multiplier in the impact or risk equations.

            **Level 1 calculations**
            """
        )

        st.markdown("**Pathway components**")
        st.latex(
            r"""
            \text{impact component}
            =
            \text{overall fish use}
            \times \text{condition}
            \times \text{vulnerability}
            """
        )
        st.latex(
            r"""
            \begin{aligned}
            \text{risk component}
            ={}& \text{life-stage fish use}
            \times \text{condition} \\
            &{}\times \text{vulnerability}
            \times \text{population priority}
            \end{aligned}
            """
        )

        st.markdown(
            """
            These are distinct calculations:
             
            - **Impact** applies the same overall BSR fish-use score to every species and life-stage pathway.
            - **Risk** uses the pathway-specific life-stage fish-use score and population priority.

            A species and life-stage pathway with a Life-Stage Fish Use Score of 0 therefore has zero risk.

            **Species and life-stage score**
            """
        )
        st.latex(
            r"""
            \begin{aligned}
            \text{life-stage risk score}
            ={}&
            \sum_{\substack{\text{15 limiting}\\\text{factors}}}
            \Bigl(
            \text{life-stage fish use}
            \times \text{condition} \\
            &{}\qquad\times \text{vulnerability}
            \times \text{population priority}
            \Bigr)
            \end{aligned}
            """
        )
        st.latex(
            r"""
            \begin{aligned}
            \text{risk score for highest priority life stage}
            ={}&
            \max_{\substack{\text{all species and}\\\text{life stages}}}
            \left(\text{life-stage risk score}\right)
            \end{aligned}
            """
        )
        st.markdown(
            """
            The **Highest Priority Life Stage** is the species and life-stage
            combination with the largest life-stage risk score within the BSR.
            Tied maximum scores retain all tied combinations.

            **Limiting-factor scores**
            """
        )
        st.latex(
            r"""
            \begin{aligned}
            \text{limiting-factor impact}
            ={}&
            \sum_{\substack{\text{all species and}\\\text{life stages}}}
            \Bigl(
            \text{overall fish use}
            \times \text{condition} \\
            &{}\qquad\times \text{vulnerability}
            \Bigr)
            \end{aligned}
            """
        )
        st.latex(
            r"""
            \begin{aligned}
            \text{limiting-factor risk}
            ={}&
            \sum_{\substack{\text{all species and}\\\text{life stages}}}
            \Bigl(
            \text{life-stage fish use}
            \times \text{condition} \\
            &{}\qquad\times \text{vulnerability}
            \times \text{population priority}
            \Bigr)
            \end{aligned}
            """
        )
        st.markdown("**Overall BSR scores**")
        st.latex(
            r"""
            \begin{aligned}
            \text{overall limiting-factor impact}
            ={}&
            \sum_{\substack{\text{all species, life stages,}\\
                            \text{and 15 limiting factors}}}
            \Bigl(
            \text{overall fish use}
            \times \text{condition} \\
            &{}\qquad\times \text{vulnerability}
            \Bigr)
            \end{aligned}
            """
        )
        st.latex(
            r"""
            \begin{aligned}
            \text{overall risk score}
            ={}&
            \sum_{\substack{\text{all species, life stages,}\\
                            \text{and 15 limiting factors}}}
            \Bigl(
            \text{life-stage fish use}
            \times \text{condition} \\
            &{}\qquad\times \text{vulnerability}
            \times \text{population priority}
            \Bigr)
            \end{aligned}
            """
        )
        st.markdown(
            """
            The limiting-factor condition score maps the source 1-to-5 rating
            linearly to 0.1-to-1.0. The vulnerability score maps rank 1 to 1.0
            and rank 15 to 0.1. For the combined Migration life stage, the
            calculation uses the higher of the adult and juvenile vulnerability
            scores.

            **Level 2 action calculations**
            """
        )

        st.latex(
            r"""
            \text{action weight}
            =
            \text{relationship directness}
            \times \text{frequency}
            """
        )
        st.latex(
            r"""
            \begin{aligned}
            \text{condition improvement score}
            ={}&
            \sum_{\substack{\text{15 limiting}\\\text{factors}}}
            \left(
            \text{condition}
            \times \text{action weight}
            \right)
            \end{aligned}
            """
        )
        st.latex(
            r"""
            \begin{aligned}
            \text{limiting-factor amelioration score}
            ={}&
            \sum_{\substack{\text{15 limiting}\\\text{factors}}}
            \left(
            \text{limiting-factor impact}
            \times \text{action weight}
            \right)
            \end{aligned}
            """
        )
        st.latex(
            r"""
            \text{action-specific benefit component}
            =
            \text{limiting-factor risk}
            \times \text{action weight}
            """
        )
        st.latex(
            r"""
            \begin{aligned}
            \text{action-specific benefit score}
            ={}&
            \sum_{\substack{\text{15 limiting}\\\text{factors}}}
            \left(
            \text{limiting-factor risk}
            \times \text{action weight}
            \right)
            \end{aligned}
            """
        )
        st.latex(
            r"""
            \begin{aligned}
            \text{overall benefit score}
            ={}&
            \sum_{\text{all actions}}
            \left(
            \text{action-specific benefit score}
            \right)
            \end{aligned}
            """
        )
        st.markdown(
            """
            The **Highest Risk-Aligned Action Type** has the largest
            Action-Specific Benefit Score within the BSR. Its map hover details
            identify the largest Action-Specific Benefit Component and report
            the corresponding Limiting-Factor Risk and Action Weight.

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
    """Load the core scoring tables and required action components."""
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
def load_scored_bsr_layer(
    path_text: str,
) -> tuple[gpd.GeoDataFrame, str, str]:
    """Find the notebook's scored feature layer and its BSR identifier."""
    path = Path(path_text).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Spatial layer not found: {path}")

    matches: list[tuple[str, str]] = []
    for layer_name, geometry_type in pyogrio.list_layers(path):
        if geometry_type is None:
            continue
        fields = [str(field) for field in pyogrio.read_info(
            path, layer=str(layer_name)
        )["fields"]]
        field_lookup = {field.casefold(): field for field in fields}
        identifier = next(
            (
                field_lookup[candidate.casefold()]
                for candidate in BSR_ID_FIELD_CANDIDATES
                if candidate.casefold() in field_lookup
            ),
            None,
        )
        if identifier is not None:
            matches.append((str(layer_name), identifier))

    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one scored polygon layer containing score_bsr "
            "or BSR in bsr_scores.gpkg; found "
            f"{len(matches)}."
        )

    layer_name, identifier = matches[0]
    source = gpd.read_file(path, layer=layer_name, engine="pyogrio")
    return source, identifier, layer_name


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


def show_score_table(
    table: pd.DataFrame,
    column_labels: dict[str, str] | None = None,
) -> None:
    """Display score tables with calculation-consistent, readable labels."""
    labels = {
        column: DISPLAY_LABELS.get(
            column,
            column.replace("_", " ").title(),
        )
        for column in table.columns
    }
    labels.update(column_labels or {})
    rounded = round_float_columns(table).rename(columns=labels)
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


def apply_bsr_selection_style(
    figure: Any,
    selected_bsr: str | None,
    chart_key: str,
) -> None:
    """Select the sidebar BSR on the map without obscuring other polygons."""
    selected_value = None if selected_bsr is None else str(selected_bsr)
    trace_locations: list[tuple[Any, list[str]]] = []

    for trace in figure.data:
        locations = getattr(trace, "locations", None)
        if locations is None:
            continue
        trace_locations.append((trace, [str(location) for location in locations]))

    selection_is_mapped = selected_value is not None and any(
        selected_value in locations for _, locations in trace_locations
    )

    for trace, locations in trace_locations:
        selected_indices = (
            [
                index
                for index, location in enumerate(locations)
                if location == selected_value
            ]
            if selection_is_mapped
            else None
        )
        trace.update(
            selectedpoints=selected_indices,
            selected={"marker": {"opacity": MAP_SELECTED_OPACITY}},
            unselected={"marker": {"opacity": MAP_UNSELECTED_OPACITY}},
        )

    # Changing this value tells Plotly to accept the dropdown-controlled
    # selection instead of retaining a previous map click.
    figure.update_layout(
        selectionrevision=f"{chart_key}:{selected_value or 'none'}"
    )


def add_selected_bsr_outline(
    figure: Any,
    mapped: gpd.GeoDataFrame,
    plot_data: pd.DataFrame,
    selected_bsr: str | None,
) -> None:
    """Overlay a crisp outline on the selected BSR without covering the map."""
    if selected_bsr is None or not figure.data:
        return
    selected_value = str(selected_bsr)
    selected_geometry = mapped.loc[
        mapped["bsr"].astype(str).eq(selected_value),
        ["bsr", "geometry"],
    ]
    if selected_geometry.empty:
        return

    selected_hover = plot_data.loc[
        plot_data["bsr"].astype(str).eq(selected_value),
        "_hover_text",
    ]
    hover_text = (
        selected_hover.iloc[0]
        if not selected_hover.empty
        else f"<b>BSR:</b> {escape(selected_value)}"
    )
    outline_arguments = {
        "geojson": json.loads(selected_geometry.to_json()),
        "locations": [selected_value],
        "featureidkey": "properties.bsr",
        "z": [0],
        "zmin": 0,
        "zmax": 1,
        "colorscale": [
            [0, "rgba(15, 23, 42, 0)"],
            [1, "rgba(15, 23, 42, 0)"],
        ],
        "showscale": False,
        "showlegend": False,
        "customdata": [[selected_value, hover_text]],
        "hovertemplate": "%{customdata[1]}<extra></extra>",
        "marker": {
            "line": {
                "color": MAP_SELECTED_OUTLINE_COLOR,
                "width": MAP_SELECTED_OUTLINE_WIDTH,
            },
            "opacity": 1,
        },
        "name": "Selected BSR outline",
    }
    trace_type = getattr(figure.data[0], "type", "")
    if trace_type == "choroplethmap":
        figure.add_trace(go.Choroplethmap(**outline_arguments))
    else:
        figure.add_trace(go.Choroplethmapbox(**outline_arguments))


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
        "opacity": MAP_FILL_OPACITY,
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
        marker_line_width=MAP_BOUNDARY_WIDTH,
        marker_line_color=MAP_BOUNDARY_COLOR,
        hovertemplate="%{customdata[1]}<extra></extra>",
    )
    apply_bsr_selection_style(
        figure,
        st.session_state.get("selected_bsr"),
        chart_key,
    )
    add_selected_bsr_outline(
        figure,
        mapped,
        plot_data,
        st.session_state.get("selected_bsr"),
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

    st.header("Level 1: Overall Risk")
    st.caption(
        "Click a polygon or use the sidebar BSR selector. The charts below "
        "partition the selected BSR score by species and life stage and by "
        "limiting factor."
    )
    render_choropleth(
        geometry,
        bsr,
        "overall_risk_score",
        "Overall Risk Score",
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
    metric_columns[1].metric(
        "Overall Risk Score", format_score(row["overall_risk_score"])
    )
    metric_columns[2].metric(
        "Overall Fish Use Score", format_score(row["fish_use_score"])
    )
    metric_columns[3].metric(
        "Overall Limiting-Factor Impact",
        format_score(row["overall_impact_score"]),
    )

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
            value_label="Life-Stage Risk Score",
        )
    with right:
        horizontal_bar(
            factor_selected,
            "risk_score",
            "limiting_factor",
            f"{selected_bsr}: risk by limiting factor",
            value_label="Limiting-Factor Risk",
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
            ].sort_values("risk_score", ascending=False),
            column_labels={
                "impact_score": "Life-Stage Impact Score",
                "risk_score": "Life-Stage Risk Score",
            },
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
            ].sort_values("risk_score", ascending=False),
            column_labels={
                "impact_score": "Limiting-Factor Impact",
                "risk_score": "Limiting-Factor Risk",
            },
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
        f"{species} Species Fish Use Score"
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

    st.header("Level 1: Fish Use")
    st.caption(
        "Fish-use results are reported as the Overall Fish Use Score, Species "
        "Fish Use Score, and Life-Stage Fish Use Score defined above."
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
        map_metric_label = "Overall Fish Use Score"
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
            map_metric_label = f"{map_species} Species Fish Use Score"
            map_title = f"{map_species}: Species Fish Use Score"
            selected_species_hover = f"{map_species} Species Fish Use Score"
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
            map_metric_label = "Life-Stage Fish Use Score"
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
        "Overall Fish Use Score",
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
            value_label="Species Fish Use Score",
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
            value_label="Life-Stage Fish Use Score",
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

    st.header("Level 1: Limiting Factors")
    overall_labels = {
        "Overall Limiting-Factor Impact": "overall_impact_score",
        "Overall Risk Score": "overall_risk_score",
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

    st.subheader("Specific Limiting-Factor Drill-Down")
    factor_options = sorted(limiting["limiting_factor"].dropna().unique())
    selected_factor = st.selectbox(
        "Limiting factor",
        factor_options,
        help=LIMITING_FACTOR_SELECTION_HELP,
    )
    factor_score_labels = {
        "Limiting-Factor Impact": "impact_score",
        "Limiting-Factor Risk": "risk_score",
        "Limiting-Factor Condition Score": "condition_score",
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
        "Impact Component": "impact_component",
        "Risk Component": "risk_component",
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
            ].sort_values("risk_component", ascending=False),
            column_labels={
                "condition_score": "Limiting-Factor Condition Score",
            },
        )

    st.subheader("Specific Limiting-Factor Map")
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


def summarize_action_benefits(
    bsr: pd.DataFrame,
    actions: pd.DataFrame,
    limiting: pd.DataFrame,
    action_components: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build highest-component hover fields and BSR-wide benefit totals."""
    bsr_values = bsr.copy()
    for column in ("highest_action_benefit_score", "overall_benefit_score"):
        bsr_values[column] = pd.to_numeric(
            bsr_values[column], errors="coerce"
        )
    if bsr_values[
        ["highest_action_benefit_score", "overall_benefit_score"]
    ].isna().any().any():
        raise ValueError(
            "bsr_scores.csv contains missing or nonnumeric action-benefit "
            "summary scores."
        )

    action_values = actions.copy()
    action_values["action_benefit_score"] = pd.to_numeric(
        action_values["action_benefit_score"], errors="coerce"
    )
    if action_values["action_benefit_score"].isna().any():
        raise ValueError(
            "action_scores.csv contains missing or nonnumeric "
            "Action-Specific Benefit Scores."
        )
    action_values["_highest_action_score"] = action_values.groupby("bsr")[
        "action_benefit_score"
    ].transform("max")
    top_actions = action_values.loc[
        action_values["action_benefit_score"].eq(
            action_values["_highest_action_score"]
        ),
        ["bsr", "action_id", "action_type"],
    ].drop_duplicates()
    calculated_top_scores = (
        action_values.groupby("bsr", as_index=False)["action_benefit_score"]
        .max()
        .rename(columns={"action_benefit_score": "_calculated_top_score"})
    )
    top_score_check = bsr_values[
        ["bsr", "highest_action_benefit_score"]
    ].merge(
        calculated_top_scores,
        on="bsr",
        how="left",
        validate="one_to_one",
    )
    if top_score_check["_calculated_top_score"].isna().any() or not np.allclose(
        top_score_check["highest_action_benefit_score"],
        top_score_check["_calculated_top_score"],
        rtol=1e-9,
        atol=1e-12,
    ):
        raise ValueError(
            "Highest Action-Specific Benefit Scores in bsr_scores.csv do not "
            "match action_scores.csv."
        )

    components = action_components.copy()
    for column in ("lfat_score", "benefit_component"):
        components[column] = pd.to_numeric(components[column], errors="coerce")
    if components[["lfat_score", "benefit_component"]].isna().any().any():
        raise ValueError(
            "action_score_components.csv contains missing or nonnumeric "
            "Action Weights or Action-Specific Benefit Components."
        )
    component_totals = (
        components.groupby(
            ["bsr", "action_id", "action_type"],
            as_index=False,
        )["benefit_component"]
        .sum()
        .rename(columns={"benefit_component": "_component_total"})
    )
    action_reconciliation = action_values.merge(
        component_totals,
        on=["bsr", "action_id", "action_type"],
        how="left",
        validate="one_to_one",
    )
    if action_reconciliation["_component_total"].isna().any() or not np.allclose(
        action_reconciliation["action_benefit_score"],
        action_reconciliation["_component_total"],
        rtol=1e-9,
        atol=1e-12,
    ):
        raise ValueError(
            "Action-Specific Benefit Scores do not reconcile to the sum of "
            "their limiting-factor benefit components."
        )
    limiting_risk = limiting[
        ["bsr", "limiting_factor", "risk_score"]
    ].rename(columns={"risk_score": "highest_benefit_limiting_factor_risk"})
    limiting_risk["highest_benefit_limiting_factor_risk"] = pd.to_numeric(
        limiting_risk["highest_benefit_limiting_factor_risk"],
        errors="coerce",
    )
    if limiting_risk["highest_benefit_limiting_factor_risk"].isna().any():
        raise ValueError(
            "limiting_factor_scores_integrated.csv contains missing or "
            "nonnumeric Limiting-Factor Risk values."
        )

    candidates = components.merge(
        top_actions,
        on=["bsr", "action_id", "action_type"],
        how="inner",
        validate="many_to_one",
    ).merge(
        limiting_risk,
        on=["bsr", "limiting_factor"],
        how="left",
        validate="many_to_one",
    )
    if candidates["highest_benefit_limiting_factor_risk"].isna().any():
        raise ValueError(
            "One or more highest-action benefit components have no matching "
            "Limiting-Factor Risk value."
        )
    expected_benefit = (
        candidates["highest_benefit_limiting_factor_risk"]
        * candidates["lfat_score"]
    )
    if not np.allclose(
        candidates["benefit_component"],
        expected_benefit,
        rtol=1e-9,
        atol=1e-12,
    ):
        raise ValueError(
            "Action-Specific Benefit Components do not equal "
            "Limiting-Factor Risk × Action Weight."
        )
    candidates["_highest_component_score"] = candidates.groupby("bsr")[
        "benefit_component"
    ].transform("max")
    highest_components = candidates.loc[
        candidates["benefit_component"].eq(
            candidates["_highest_component_score"]
        )
    ].sort_values(["bsr", "action_type", "limiting_factor"])

    def join_text(values: pd.Series) -> str:
        return " | ".join(dict.fromkeys(values.dropna().astype(str)))

    def join_scores(values: pd.Series) -> str:
        return " | ".join(
            dict.fromkeys(f"{float(value):,.2f}" for value in values.dropna())
        )

    component_summary = (
        highest_components.groupby("bsr", as_index=False)
        .agg(
            highest_benefit_action_type=("action_type", join_text),
            highest_benefit_limiting_factor=("limiting_factor", join_text),
            highest_benefit_limiting_factor_risk=(
                "highest_benefit_limiting_factor_risk",
                join_scores,
            ),
            highest_benefit_action_weight=("lfat_score", join_scores),
            highest_benefit_component_score=("benefit_component", "max"),
        )
    )
    missing_component_bsrs = sorted(
        set(bsr_values["bsr"]) - set(component_summary["bsr"])
    )
    if missing_component_bsrs:
        raise ValueError(
            "No highest-benefit component could be identified for BSRs: "
            + ", ".join(missing_component_bsrs)
        )
    highest_action_map = bsr_values.merge(
        component_summary,
        on="bsr",
        how="left",
        validate="one_to_one",
    )

    action_totals = (
        action_values.groupby("bsr", as_index=False)
        .agg(
            _calculated_overall_benefit_score=(
                "action_benefit_score",
                "sum",
            ),
            action_count=("action_type", "nunique"),
        )
    )
    overall_benefit_map = bsr_values[
        ["bsr", "basin", "overall_benefit_score"]
    ].merge(
        action_totals,
        on="bsr",
        how="left",
        validate="one_to_one",
    )
    if overall_benefit_map[
        "_calculated_overall_benefit_score"
    ].isna().any() or not np.allclose(
        overall_benefit_map["overall_benefit_score"],
        overall_benefit_map["_calculated_overall_benefit_score"],
        rtol=1e-9,
        atol=1e-12,
    ):
        raise ValueError(
            "Overall Benefit Scores in bsr_scores.csv do not equal the sum "
            "of Action-Specific Benefit Scores in action_scores.csv."
        )
    overall_benefit_map = overall_benefit_map.drop(
        columns="_calculated_overall_benefit_score"
    )
    return highest_action_map, overall_benefit_map


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
    limiting = filter_table(tables["limiting_factor"], basin)
    action_components = filter_table(tables["action_components"], basin)
    highest_action_map, overall_benefit_map = summarize_action_benefits(
        bsr,
        actions,
        limiting,
        action_components,
    )

    st.header("Level 2: Action Benefits")
    action_options = (
        actions[["action_id", "action_type"]]
        .drop_duplicates()
        .sort_values("action_id")
    )
    action_lookup = dict(zip(action_options["action_type"], action_options["action_id"]))
    selected_action = st.selectbox("Action type", list(action_lookup))

    action_score_labels = {
        "Action-Specific Benefit Score": "action_benefit_score",
        "Limiting-Factor Amelioration Score": "limiting_factor_amelioration_score",
        "Condition Improvement Score": "condition_improvement_score",
    }
    action_score_label = st.radio(
        "Action map value",
        options=list(action_score_labels),
        horizontal=True,
        help=ACTION_MAP_HELP,
    )
    action_score = action_score_labels[action_score_label]
    action_map = actions.loc[actions["action_type"].eq(selected_action)].copy()
    action_map_heading = (
        "Action-Specific Benefit Scores Map"
        if action_score == "action_benefit_score"
        else f"{action_score_label} Map"
    )
    action_map_title = (
        f"Action-Specific Benefit Scores Map: {selected_action}"
        if action_score == "action_benefit_score"
        else f"{selected_action}: {action_score_label}"
    )
    st.subheader(action_map_heading)
    render_choropleth(
        geometry,
        action_map,
        action_score,
        action_score_label,
        action_map_title,
        "map_action_specific",
        map_style,
        hover_columns=[
            "condition_improvement_score",
            "limiting_factor_amelioration_score",
            "action_benefit_score",
            "benefit_rank_within_bsr",
        ],
        color_scale=ACTION_BENEFIT_COLOR_SCALE,
    )

    selected = actions.loc[actions["bsr"].eq(selected_bsr)].copy()
    horizontal_bar(
        selected,
        action_score,
        "action_type",
        f"{selected_bsr}: {action_score_label} by action",
        value_label=action_score_label,
    )

    component_rows = action_components.loc[
        action_components["bsr"].eq(selected_bsr)
        & action_components["action_type"].eq(selected_action),
        [
            "bsr",
            "limiting_factor",
            "lfat_score",
            "condition_improvement_component",
            "amelioration_component",
            "benefit_component",
        ],
    ].merge(
        limiting[
            [
                "bsr",
                "limiting_factor",
                "condition_score",
                "impact_score",
                "risk_score",
            ]
        ],
        on=["bsr", "limiting_factor"],
        how="left",
        validate="many_to_one",
    )
    with st.expander(f"Show limiting-factor contributions to the selected action: {selected_action}"):
        horizontal_bar(
            component_rows,
            "benefit_component",
            "limiting_factor",
            f"{selected_bsr}: benefit components for {selected_action}",
            value_label="Action-Specific Benefit Component",
        )
        show_score_table(
            component_rows[
                [
                    "limiting_factor",
                    "condition_score",
                    "impact_score",
                    "risk_score",
                    "lfat_score",
                    "condition_improvement_component",
                    "amelioration_component",
                    "benefit_component",
                ]
            ].sort_values("benefit_component", ascending=False),
            column_labels={
                "impact_score": "Limiting-Factor Impact",
                "risk_score": "Limiting-Factor Risk",
            },
        )

    st.subheader("Highest Risk-Aligned Action Type")
    render_choropleth(
        geometry,
        highest_action_map,
        "highest_risk_aligned_action_type",
        "Highest Risk-Aligned Action Type",
        "Highest Risk-Aligned Action Type",
        "map_top_action",
        map_style,
        categorical=True,
        hover_columns=[
            "highest_benefit_action_type",
            "highest_benefit_limiting_factor",
            "highest_benefit_limiting_factor_risk",
            "highest_benefit_action_weight",
            "highest_benefit_component_score",
            "highest_action_benefit_score",
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
                    "action_benefit_score",
                    "benefit_rank_within_bsr",
                ]
            ].sort_values("benefit_rank_within_bsr")
        )

    st.subheader("Overall Benefit Score Map")
    st.caption(
        "Overall Benefit Score is the sum of Action-Specific Benefit Scores "
        "for all actions within each BSR."
    )
    render_choropleth(
        geometry,
        overall_benefit_map,
        "overall_benefit_score",
        "Overall Benefit Score",
        "Overall Benefit Score Map",
        "map_overall_benefit",
        map_style,
        hover_columns=["action_count"],
        color_scale=ACTION_BENEFIT_COLOR_SCALE,
    )


def load_bsr_geometry(bsr_scores: pd.DataFrame) -> gpd.GeoDataFrame:
    """Load the scored BSR feature layer written by the notebook."""
    source, identifier, _ = load_scored_bsr_layer(str(BSR_GPKG_PATH))
    geometry, unmatched = prepare_geometry(source, identifier, bsr_scores)
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
        help="Map clicks and this selector stay synchronized.",
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
