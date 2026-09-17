"""Interactive Atlas scoring viewer for Streamlit.

The app reads the Level 1 and Level 2 outputs created by
Atlas_Integrated_Scoring.ipynb, including the scored BSR GeoPackage. The BSR
feature layer is detected from the score_bsr field written by the notebook.
Fish-use views distinguish the BSR-level fish_use_score, species-level
species_aggregate_score, and life-stage LS_corrected_score fields. Level 2
views use action_benefit_score for individual actions. Displayed risk ceilings
hold life-stage fish use and population priority fixed and set vulnerability
and condition to 1. The action-specific benefit ceiling is 15.
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
from shapely.geometry import LineString


SCORE_DIR = Path("data/outputs")
BSR_GPKG_PATH = SCORE_DIR / "bsr_scores.gpkg"
BSR_ID_FIELD_CANDIDATES = ("score_bsr", "BSR", "bsr")
MAP_STYLE = "carto-positron"
MAP_FILL_OPACITY = 0.78
MAP_BOUNDARY_COLOR = "#334155"
MAP_BOUNDARY_WIDTH = 0.4
MAP_SELECTED_OUTLINE_COLOR = "#f3b63f"
MAP_SELECTED_OUTLINE_WIDTH = 3.0
EXCLUDED_BSR_FILL_COLOR = "#f1efeb"
EXCLUDED_BSR_HATCH_COLOR = "#54616b"
EXCLUDED_BSR_HATCH_WIDTH = 1.5

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
SCORE_SCHEMA_VERSION = "2026-09-16-observed-species-and-action-scales-v10"

# The contextual species_aggregate_score has a maximum possible value of 6.
SPECIES_FISH_USE_MAXIMUM = 6.0
ACTION_BENEFIT_MAXIMUM = 15.0
MAXIMUM_SUFFIX = "__maximum"
OBSERVED_MAXIMUM_SUFFIX = "__observed_maximum"
UNIT_SCORE_FIELDS = {
    "fish_use_score", "LS_corrected_score", "condition_score",
    "overall_limiting_factor_score", "population_priority",
    "vulnerability_score",
}
OVERALL_FISH_USE_HELP = (
    "Overall Fish Use Score ranges from 0 to 1 and is shown for context. "
    "It is not a direct multiplier in the risk equations."
)
OVERALL_RISK_MAXIMUM_HELP = (
    "The displayed upper bound is the highest observed Overall Risk Score "
    "among BSRs included in the analysis."
)
SPECIES_RISK_MAXIMUM_HELP = (
    "The displayed upper bound is the highest observed risk score for the "
    "selected species among BSRs included in the analysis."
)
ACTION_BENEFIT_MAXIMUM_HELP = (
    "The theoretical maximum is 15 for every BSR and action. Set the "
    "limiting-factor risk score and action weight to 1 for each of the 15 "
    "limiting factors, then sum their benefit components."
)

REQUIRED_COLUMNS = {
    "bsr": {
        "bsr",
        "basin",
        "fish_use_score",
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
        "risk_score",
        "species_life_stage_label",
    },
    "limiting_factor": {
        "bsr",
        "basin",
        "limiting_factor",
        "condition_score",
        "risk_score",
    },
    "action": {
        "bsr",
        "basin",
        "action_id",
        "action_type",
        "action_benefit_score",
        "benefit_rank_within_bsr",
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
        "risk_component",
    },
    "action_components": {
        "bsr",
        "basin",
        "limiting_factor",
        "action_id",
        "action_type",
        "lfat_score",
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
RISK_COLOR_SCALE = [
    [0.00, "#2166ac"],
    [0.50, "#f7f7f7"],
    [1.00, "#7f0000"],
]
LIMITING_FACTOR_COLOR_SCALE = [
    [0.00, "#ffffff"],
    [0.25, "#ffd27a"],
    [0.50, "#f48731"],
    [0.75, "#ce3f28"],
    [1.00, "#751625"],
]
# These sequential ramps darken as the score increases. Distinct hues help
# separate fish presence, limiting conditions, and action benefits on the maps.
FISH_USE_COLOR_SCALE = [
    [0.00, "#ffffff"],
    [0.25, "#b6e4ed"],
    [0.50, "#489fc2"],
    [0.75, "#24639a"],
    [1.00, "#102e61"],
]
ACTION_BENEFIT_COLOR_SCALE = [
    [0.00, "#ffffff"],
    [0.25, "#c1e39f"],
    [0.50, "#5ab77d"],
    [0.75, "#1c8064"],
    [1.00, "#06493e"],
]
TIER_ORDER = ["Tier 1", "Tier 2", "Tier 3", "Not included"]
TIER_COLORS = {
    "Tier 1": "#d06777",
    "Tier 2": "#e8be7b",
    "Tier 3": "#b2b5b0",
    "Not included": EXCLUDED_BSR_FILL_COLOR,
}
EXCLUDED_TIER_BSRS = {"UGR8", "UGR08"}
REFERENCE_CATEGORY_COLORS = [
    "#b3dac2",
    "#f4cdb1",
    "#c1ceea",
    "#e8bdce",
    "#cfdbb0",
    "#ccbee2",
    "#f2d9a8",
    "#acd1d9",
    "#dcc0af",
    "#9fc5e8",
    "#d5a6bd",
    "#b6d7a8",
    "#ffe599",
    "#a2c4c9",
    "#d9d2e9",
]
LIFE_STAGE_COLOR_MAP = {
    "Bull Trout | FMO": "#b3dac2",
    "Bull Trout | Spawning & Resident": "#f4cdb1",
    "Chinook | Migration": "#c1ceea",
    "Chinook | Winter Rearing": "#e8bdce",
    "Steelhead | Spawning": "#cfdbb0",
    "Steelhead | Summer Rearing": "#ccbee2",
    "Steelhead | Winter Rearing": "#f2d9a8",
}
LIMITING_FACTOR_COLOR_MAP = {
    "Altered Flow Timing": "#b3dac2",
    "Channel and Habitat Structure": "#f4cdb1",
    "Decreased Sediment Quantity": "#c1ceea",
    "Decreased Water Quantity": "#e8bdce",
    "Non-Native Species Interactions and Competition": "#cfdbb0",
    "Predation": "#ccbee2",
    "Riparian Condition": "#f2d9a8",
    "Side Channel and Wetland Habitat": "#acd1d9",
    "Summer Water Temperature": "#dcc0af",
}

LIMITING_FACTOR_SELECTION_HELP = (
    "Select the limiting factor shown in the comparison chart, biological "
    "risk drill-down, data table, and condition map."
)
ACTION_MAP_HELP = (
    "**Action-Specific Benefit Score:** applies Action Weight to "
    "Limiting-Factor Risk and sums the components for the selected action. "
    "Both scores and action-specific tiers update when Action type changes.\n\n"
    "These scores indicate relative alignment, not expected project "
    "effectiveness, feasibility, or cost."
)

DISPLAY_LABELS = {
    "bsr": "BSR",
    "basin": "Basin",
    "fish_use_score": "Overall Fish Use Score",
    "species_aggregate_score": "Species Fish Use Score",
    "LS_corrected_score": "Life-Stage Fish Use Score",
    "overall_risk_score": "Overall Risk Score",
    "species_risk_score": "Species Risk Score",
    "overall_limiting_factor_score": "Overall Limiting-Factor Condition Score",
    "preliminary_tier": "Preliminary Tiers",
    "preliminary_tier_basis": "Tier Assignment Basis",
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
    "risk_component": "Risk Component",
    "risk_score": "Risk Score",
    "action_id": "Action ID",
    "action_type": "Action Type",
    "lfat_score": "Action Weight",
    "benefit_component": "Action-Specific Benefit Component",
    "action_benefit_score": "Action-Specific Benefit Score",
    "overall_benefit_score": "Overall Benefit Score",
    "action_count": "Number of Actions",
    "highest_risk_aligned_action_type": "Highest Risk-Aligned Action Type",
    "highest_action_benefit_score": "Highest Action-Specific Benefit Score",
    "top_action_benefit_tie_count": "Top Action Benefit Tie Count",
    "benefit_rank_within_bsr": "Action-Specific Benefit Rank Within BSR",
    "priority_life_stage_1_risk_score": "First Priority Life-Stage Risk Score",
    "priority_life_stage_2_risk_score": "Second Priority Life-Stage Risk Score",
    "priority_life_stage_3_risk_score": "Third Priority Life-Stage Risk Score",
    "second_highest_risk_limiting_factor": (
        "Second Highest Priority Limiting Factor"
    ),
    "third_highest_risk_limiting_factor": (
        "Third Highest Priority Limiting Factor"
    ),
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
        page_icon="ðŸ—ºï¸",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1700px;}
        [data-testid="stMetric"] {border: 1px solid #d9e2ea; border-radius: 0.45rem; padding: 0.7rem;}
        [data-testid="stSidebar"] {min-width: 320px;}
        [data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
            background-color: #64748b !important;
            border-color: #64748b !important;
            color: #ffffff !important;
        }
        [data-testid="stSidebar"] [data-testid="stBaseButton-primary"]:hover {
            background-color: #526173 !important;
            border-color: #526173 !important;
        }
        div[data-testid="stAlert"] {border-radius: 0.35rem;}
        [data-testid="stTooltipIcon"], button[aria-label="Help"] {
            color: #334155 !important;
            margin-left: 0.25rem;
        }
        [data-testid="stTooltipIcon"] svg, button[aria-label="Help"] svg {
            width: 1.15rem !important;
            height: 1.15rem !important;
            stroke-width: 2.25 !important;
        }
        button[aria-label="Help"] {
            min-width: 1.55rem !important;
            min-height: 1.55rem !important;
            padding: 0.15rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


METHODOLOGY_MARKDOWN = '# How risk scores are calculated\n\nEach BSR is evaluated for every species, life stage, and limiting factor combination. Higher input values increase the calculated risk score.\n\n## Inputs\n\n### Fish-use inputs\n\n- **Life-Stage Fish Use Score** is specific to the BSR, species, and life stage. It is the fish-use multiplier used in the risk calculations.\n- The Life-Stage Fish Use Score is the **Corrected\\_{life stage}** score in the `Normalized Score Calculator` tab and the `LS_corrected_score` field in the fish-use input table.\n- Life-Stage Fish Use Scores are normalized from 0 to 1.\n- **Scores reported for context**\n  - **Species Fish Use Score** is the species-specific score listed in the `Fish Use Scores - Normalized` tab and the `species_aggregate_score` field. Its maximum possible value is 6. It is not a direct multiplier in the risk equations.\n  - **Overall Fish Use Score** is the BSR-level 0-to-1 score in the `fish_use_score_decimal` field. It is not a direct multiplier in the risk equations.\n\n### Vulnerability inputs\n\n- **Vulnerability Rank** is specific to the species, life stage, and limiting factor. Rank 1 represents the limiting factor with the most likely impact on a given species and life stage, and rank 15 represents the lowest impact.\n- The vulnerability score used in the calculations maps these 1-to-15 rankings on a scale from 0.01 (lowest species/life stage impact) to 1.0 (highest species/life stage imapct)\n- For the combined Migration life stage, the calculation uses the higher of the adult and juvenile vulnerability scores for each species and limiting factor.\n\n### Limiting-factor inputs\n\n- **Limiting-Factor Condition** is specific to the BSR and limiting factor. The current input table does not contain a life-stage-specific condition field.\n- The limiting-factor condition score used in the calculations maps the source 1-to-5 rating to 0.01 to 1.0\n- The provided `lf_condition_score` field uses a previous 0.10-to-1.0 transformation. It is retained for reference, but the calculations use the score recalculated from `lf_condition_score_raw_1_5` using the equation above.\n- Within a BSR, the same condition score for a limiting factor is applied to every species and life-stage pathway.\n\n### Population-priority inputs\n\n- **Population Priority** is specific to the basin, species, and life stage.\n- The population priorities for each species sum to 100%.\n\n## Level 1 risk calculations\n\n### Risk score components\n\nRisk score components vary as follows:\n\n- Life-stage fish use is specific to BSR, species, and life stage. \n- Population priority is specific to basin, species, and life stage.\n- Vulnerability is specific to the species, life stage, and limiting factor. \n- Limiting-factor condition is specific to BSR.\n\nRisk can be aggregated in two equivalent ways:\n\n1. Across the 15 limiting factors for each species and life stage.\n2. Across all species and life stages for each limiting factor.\n\nBoth approaches produce the same Overall Risk Score within a BSR, although the intermediate scores describe different components of the overall score.\n\n### Species and life-stage risk scores\n\nFor a given BSR, species, and life stage, the Life-Stage Fish Use Score and Population Priority are constant across the 15 limiting factors and can be placed in front of the sum:\n\n$$\n\\begin{aligned}\n\\text{life-stage risk score}\n={}&\n\\text{life-stage fish use}\n\\times\n\\text{population priority} \\\\\n&\\times\n\\sum_{\\substack{\\text{15 limiting}\\\\\\text{factors}}}\n\\Bigl(\n\\text{limiting-factor condition}\n\\times\n\\text{vulnerability}\n\\Bigr)\n\\end{aligned}\n$$\n\nThe risk score for the highest-priority life stage is:\n\n$$\n\\begin{aligned}\n\\text{risk score for highest priority life stage}\n={}&\n\\max_{\\substack{\\text{all species and}\\\\\\text{life stages}}}\n\\left(\n\\text{life-stage risk score}\n\\right)\n\\end{aligned}\n$$\n\nThe **Highest Priority Life Stage** is the species and life-stage combination with the largest Life-Stage Risk Score within the BSR.\n\n### Limiting-factor risk scores\n\nLimiting-factor risk scores describe how much risk a given limiting factor is creating for the collection of fish use (all species and life stages) within a BSR. For a given BSR and limiting factor, the Limiting-Factor Condition is constant across species and life stages and can be placed in front of the sum:\n\n$$\n\\begin{aligned}\n\\text{limiting-factor risk score}\n={}&\n\\text{limiting-factor condition} \\\\\n&\\times\n\\sum_{\\substack{\\text{all species and}\\\\\\text{life stages}}}\n\\Bigl(\n\\text{life-stage fish use}\n\\times\n\\text{vulnerability}\n\\times\n\\text{population priority}\n\\Bigr)\n\\end{aligned}\n$$\n\nWithin this sum, life-stage fish use and population priority are specific to the applicable BSR, species, and life stage. Vulnerability is specific to the species, life stage, and limiting factor. Limiting-factor condition is the single condition score for the BSR and limiting factor.\n\nThe risk score for the highest-priority limiting factor is:\n\n$$\n\\begin{aligned}\n\\text{risk score for highest priority limiting factor}\n={}&\n\\max_{\\text{15 limiting factors}}\n\\left(\n\\text{limiting-factor risk score}\n\\right)\n\\end{aligned}\n$$\n\nThe **Highest Priority Limiting Factor** is the limiting factor with the largest Limiting-Factor Risk Score within the BSR.\n\n### Overall BSR risk score\n\nThe Overall Risk Score can be calculated equivalently by summing either all Life-Stage Risk Scores or all 15 Limiting-Factor Risk Scores:\n\n$$\n\\begin{aligned}\n\\text{Overall Risk Score}\n&=\n\\sum_{\\substack{\\text{all species and}\\\\\\text{life stages}}}\n\\left(\n\\text{life-stage risk score}\n\\right) \\\\\n&=\n\\sum_{\\text{15 limiting factors}}\n\\left(\n\\text{limiting-factor risk score}\n\\right)\n\\end{aligned}\n$$\n\n## Level 2 action calculations\n\nThe action weight is calculated from relationship directness and frequency:\n\n$$\n\\text{action weight}\n=\n\\text{relationship directness}\n\\times\n\\text{frequency}\n$$\n\nFor each action, the action-specific benefit components are summed across the 15 limiting factors:\n\n$$\n\\begin{aligned}\n\\text{action-specific benefit score}\n={}&\n\\sum_{\\substack{\\text{15 limiting}\\\\\\text{factors}}}\n\\left(\n\\text{limiting-factor risk score}\n\\times\n\\text{action weight}\n\\right)\n\\end{aligned}\n$$\n\nThe theoretical maximum Action-Specific Benefit Score is **15** for every BSR and action. Set both Limiting-Factor Risk Score and Action Weight to 1 for each of the 15 limiting factors, then sum the resulting 15 benefit components.\n\nThe **Highest Risk-Aligned Action Type** is the action type with the largest Action-Specific Benefit Score within the BSR.'


def render_scoring_methodology() -> None:
    """Render the project-approved scoring explanation from Markdown."""
    with st.expander("How scores are calculated"):
        st.markdown(METHODOLOGY_MARKDOWN)
        st.markdown("### Total possible scores")
        st.markdown(
            "- **Overall Risk Score:** The upper bound is the highest observed "
            "Overall Risk Score among BSRs included in the analysis.\n"
            "- **Species Risk Score:** The upper bound for each species is "
            "its highest observed risk score among included BSRs.\n"
            "- **Life-stage risk scores:** Theoretical totals hold life-stage fish "
            "use and population priority fixed, with vulnerability and "
            "limiting-factor condition set to 1.\n"
            "- **Action-Specific Benefit Score:** The theoretical maximum "
            "is 15 for each action and BSR. The map color scale reaches "
            "the highest observed score for the selected action by default. "
            "With 'Normalize scores to highest ranked action' on, the "
            "color scale uses the highest observed score across all actions "
            "and included BSRs.\n"
            "- **Species Fish Use Score:** The maximum is 6. \n"
            "- **Overall and Life-Stage Fish Use Scores:** The maximum is 1.\n"
            "- **Overall Limiting-Factor Condition Score:** The unweighted "
            "mean of limiting-factor condition scores is out of 1. This is "
            "a display summary; risk calculations use the individual "
            "limiting-factor conditions."
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

    if "risk_balance_difference" in tables["bsr"].columns:
        balance = tables["bsr"]["risk_balance_difference"].abs().max()
        if balance > 1e-9:
            raise ValueError(
                "The BSR summary does not reconcile between life-stage and "
                "limiting-factor aggregation paths."
            )

    return add_score_maxima(tables)


def maximum_column(field: str) -> str:
    """Name a display-only ceiling field without changing any source score."""
    return field + MAXIMUM_SUFFIX


def observed_maximum_column(field: str) -> str:
    """Name a display-only maximum from BSRs included in the analysis."""
    return field + OBSERVED_MAXIMUM_SUFFIX


def observed_included_maximum(table: pd.DataFrame, field: str) -> float:
    """Find the highest observed score among BSRs included in analysis."""
    included = pd.to_numeric(
        table.loc[
            ~table["bsr"].map(normalized_bsr_id).isin(EXCLUDED_TIER_BSRS),
            field,
        ],
        errors="coerce",
    )
    if included.empty or included.isna().any() or not np.isfinite(included).all():
        label = DISPLAY_LABELS.get(field, field)
        raise ValueError(f"Included BSRs must have finite {label} values.")
    return float(included.max())


def add_observed_risk_maximum(bsr: pd.DataFrame) -> pd.DataFrame:
    """Attach the largest observed risk among BSRs included in analysis."""
    result = bsr.copy()
    result[observed_maximum_column("overall_risk_score")] = (
        observed_included_maximum(bsr, "overall_risk_score")
    )
    return result


def score_columns(table: pd.DataFrame, fields: list[str]) -> list[str]:
    """Keep display denominators when selecting map and drill-down fields."""
    return list(dict.fromkeys([
        *fields,
        *[maximum_column(field) for field in fields
          if maximum_column(field) in table.columns],
        *[observed_maximum_column(field) for field in fields
          if observed_maximum_column(field) in table.columns],
    ]))


def add_score_maxima(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Calculate risk ceilings and theoretical action-benefit ceilings.

    These are display metadata only. Source scores and priorities (including
    rounded priorities that sum to 0.99) are never rescaled. A grid component's
    ceiling is LS_corrected_score * population_priority: condition and
    vulnerability are both 1. Each action benefit component is out of 1,
    and an action's 15 components have a theoretical total of 15.
    """
    result = {name: frame.copy() for name, frame in tables.items()}
    grid = result["grid"]
    grid_keys = ["bsr", "species", "life_stage", "limiting_factor"]
    if grid.duplicated(grid_keys).any():
        raise ValueError("The calculation grid contains duplicate scoring pathways.")
    for field in ("population_priority", "vulnerability_score",
                  "LS_corrected_score", "condition_score"):
        grid[field] = pd.to_numeric(grid[field], errors="coerce")
        if (grid[field].isna().any() or not np.isfinite(grid[field]).all()
                or grid[field].lt(0).any() or grid[field].gt(1 + 1e-9).any()):
            raise ValueError(f"{field} must contain finite 0-to-1 values.")
    grid[maximum_column("risk_component")] = (
        grid["LS_corrected_score"] * grid["population_priority"]
    )

    def attach(target: str, source: pd.DataFrame, keys: list[str],
               source_field: str, target_field: str) -> None:
        ceilings = source.groupby(keys, as_index=False)[
            maximum_column(source_field)
        ].sum().rename(columns={
            maximum_column(source_field): maximum_column(target_field)
        })
        result[target] = result[target].merge(
            ceilings, on=keys, how="left", validate="one_to_one"
        )
        if result[target][maximum_column(target_field)].isna().any():
            raise ValueError(f"Missing score ceilings for {target}.")

    attach("life_stage", grid, ["bsr", "species", "life_stage"],
           "risk_component", "risk_score")
    attach("limiting_factor", grid, ["bsr", "limiting_factor"],
           "risk_component", "risk_score")
    attach("bsr", grid, ["bsr"], "risk_component", "overall_risk_score")

    components = result["action_components"].copy()
    weights = pd.to_numeric(components["lfat_score"], errors="coerce")
    if (weights.isna().any() or not np.isfinite(weights).all() or weights.lt(0).any()):
        raise ValueError("Action weights must contain finite, nonnegative values.")
    if components.duplicated(["bsr", "action_id", "limiting_factor"]).any():
        raise ValueError("The action components contain duplicate scoring pathways.")
    components[maximum_column("benefit_component")] = 1.0
    result["action_components"] = components
    result["action"][maximum_column("action_benefit_score")] = (
        ACTION_BENEFIT_MAXIMUM
    )
    attach("bsr", result["action"], ["bsr"],
           "action_benefit_score", "overall_benefit_score")
    result["bsr"][maximum_column("highest_action_benefit_score")] = (
        ACTION_BENEFIT_MAXIMUM
    )

    result["bsr"] = add_observed_risk_maximum(result["bsr"])

    for table in result.values():
        for field in UNIT_SCORE_FIELDS.intersection(table.columns):
            table[maximum_column(field)] = 1.0
        if "species_aggregate_score" in table:
            table[maximum_column("species_aggregate_score")] = SPECIES_FISH_USE_MAXIMUM
        for field in list(table.columns):
            ceiling = maximum_column(field)
            if ceiling not in table:
                continue
            observed = pd.to_numeric(table[field], errors="coerce")
            if (observed > table[ceiling] + 1e-8).any():
                raise ValueError(f"{field} exceeds its calculated total possible score.")
    return result


def summarize_species_risk(life_stage: pd.DataFrame) -> pd.DataFrame:
    """Sum life-stage risk and attach each species' observed map upper bound."""
    summary = life_stage.groupby(["bsr", "basin", "species"], as_index=False).agg(
        species_risk_score=("risk_score", "sum"),
        **{maximum_column("species_risk_score"):
           (maximum_column("risk_score"), "sum")},
    )
    included = summary.loc[
        ~summary["bsr"].map(normalized_bsr_id).isin(EXCLUDED_TIER_BSRS)
    ].copy()
    included["species_risk_score"] = pd.to_numeric(
        included["species_risk_score"], errors="coerce"
    )
    if (included.empty or included["species_risk_score"].isna().any()
            or not np.isfinite(included["species_risk_score"]).all()):
        raise ValueError("Included BSRs must have finite Species Risk Scores.")
    observed = included.groupby("species")["species_risk_score"].max()
    summary[observed_maximum_column("species_risk_score")] = (
        summary["species"].map(observed)
    )
    if summary[observed_maximum_column("species_risk_score")].isna().any():
        raise ValueError("Every species needs an included BSR to set its map upper bound.")
    return summary


def summarize_overall_limiting_factors(limiting: pd.DataFrame) -> pd.DataFrame:
    """Summarize all factor conditions equally; do not recompute risk scores."""
    if limiting.duplicated(["bsr", "limiting_factor"]).any():
        raise ValueError("Duplicate limiting factors prevent an overall condition mean.")
    if limiting["condition_score"].isna().any():
        raise ValueError("Missing limiting-factor conditions prevent an overall mean.")
    summary = limiting.groupby(["bsr", "basin"], as_index=False).agg(
        overall_limiting_factor_score=("condition_score", "mean"),
        limiting_factor_count=("limiting_factor", "nunique"),
        overall_risk_score=("risk_score", "sum"),
        **{maximum_column("overall_risk_score"):
           (maximum_column("risk_score"), "sum")},
    )
    if not summary["limiting_factor_count"].eq(limiting["limiting_factor"].nunique()).all():
        raise ValueError("BSRs do not contain the same complete set of limiting factors.")
    summary[maximum_column("overall_limiting_factor_score")] = 1.0
    return add_observed_risk_maximum(summary)


def is_score_field(field: str) -> bool:
    """Identify scores, excluding ranks, counts, and hidden ceiling metadata."""
    if field == "lfat_score":
        # This is a fixed source coefficient, displayed as Action Weight.
        # Its scale is not defined in the summary exports.
        return False
    return (
        field in UNIT_SCORE_FIELDS
        or field.endswith(("_score", "_risk", "_component", " Fish Use Score"))
    ) and not field.endswith(MAXIMUM_SUFFIX)


def score_maximum(row: pd.Series, field: str) -> Any:
    """Read a score's displayed denominator, with known scale fallbacks."""
    if field == "species_aggregate_score" or field.endswith(" Species Fish Use Score"):
        return SPECIES_FISH_USE_MAXIMUM
    if field in {"action_benefit_score", "highest_action_benefit_score"} or re.fullmatch(
        r"action_rank_\d+_score", field
    ):
        return ACTION_BENEFIT_MAXIMUM
    if field in {"benefit_component", "highest_benefit_component_score"}:
        return 1.0
    if field in {"overall_risk_score", "species_risk_score"} and observed_maximum_column(field) in row:
        return row[observed_maximum_column(field)]
    if maximum_column(field) in row:
        return row[maximum_column(field)]
    if field in UNIT_SCORE_FIELDS or field.endswith(" Life-Stage Fish Use Score"):
        return 1.0
    return None


def format_score_value(row: pd.Series, field: str) -> str:
    """Format a score and its observed or calculated display denominator."""
    value = row[field]
    if pd.isna(value):
        return "Not available"
    if not pd.api.types.is_number(value):
        return str(value)
    if is_rank_field(field) or field.endswith("_count"):
        return f"{float(value):,.0f}"
    if not is_score_field(field):
        return format_score(value)
    ceiling = score_maximum(row, field)
    if ceiling is None or pd.isna(ceiling):
        return f"{format_score(value)} (maximum not supplied)"
    if isinstance(ceiling, str):
        return f"{format_score(value)} ({ceiling})"
    return f"{format_score(value)} / {format_score(ceiling)}"


def score_axis_label(table: pd.DataFrame, field: str, label: str) -> str:
    """State a common denominator or direct readers to row-specific totals."""
    if not is_score_field(field):
        return label
    maxima = table.apply(lambda row: score_maximum(row, field), axis=1).dropna()
    if maxima.empty:
        return f"{label}<br>(maximum not supplied)"
    numeric = pd.to_numeric(maxima, errors="coerce")
    if numeric.notna().all() and np.allclose(numeric, numeric.iloc[0]):
        if field in {"overall_risk_score", "species_risk_score"} and observed_maximum_column(field) in table:
            return f"{label}<br>(maximum observed: {format_score(numeric.iloc[0])})"
        return f"{label}<br>(out of {format_score(numeric.iloc[0])})"
    return f"{label}<br>(total possible varies; see hover)"


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
    if 0 < abs(float(value)) < 0.005:
        return f"{float(value):.3g}"
    return f"{float(value):,.{digits}f}"


def round_float_columns(table: pd.DataFrame) -> pd.DataFrame:
    """Round floating-point values for display without changing source tables."""
    rounded = table.copy()
    float_columns = rounded.select_dtypes(include=["floating"]).columns
    rounded[float_columns] = rounded[float_columns].round(2)
    return rounded


def is_rank_field(column: str) -> bool:
    """Return whether a field name represents an ordinal rank."""
    return (not column.endswith("_score")
            and re.search(r"(^|_)rank($|_)", column.casefold()) is not None)


def normalized_bsr_id(value: Any) -> str:
    """Normalize a BSR identifier for tier exclusions and tie-breaking."""
    return re.sub(r"[^A-Z0-9]+", "", str(value).upper())


def short_bsr_label(value: Any) -> str:
    """Use U# and C# on maps while preserving canonical IDs for joins/clicks."""
    identifier = normalized_bsr_id(value)
    match = re.fullmatch(r"(UGR|CC|U|C)0*(\d+)", identifier)
    if match:
        prefix = "U" if match.group(1) in {"UGR", "U"} else "C"
        return f"{prefix}{int(match.group(2))}"
    return str(value)


def tier_group_sizes(record_count: int) -> tuple[int, int, int]:
    """Split records into three near-equal groups, with remainders first."""
    return tuple(
        len(group) for group in np.array_split(np.arange(record_count), 3)
    )


def add_ranked_thirds(
    table: pd.DataFrame,
    score_column: str,
    tier_column: str,
    *,
    group_column: str | None = None,
) -> pd.DataFrame:
    """Assign Tier 1 to the highest third and Tier 3 to the lowest."""
    tiered = table.copy()
    tiered[score_column] = pd.to_numeric(
        tiered[score_column], errors="coerce"
    )
    tiered[tier_column] = pd.Series(pd.NA, index=tiered.index, dtype="object")
    normalized_ids = tiered["bsr"].map(normalized_bsr_id)
    excluded = normalized_ids.isin(EXCLUDED_TIER_BSRS)
    tiered.loc[excluded, tier_column] = "Not included"

    groups = (
        [tiered.index]
        if group_column is None
        else list(tiered.groupby(group_column, dropna=False).groups.values())
    )
    for indices in groups:
        eligible_indices = [
            index
            for index in indices
            if not excluded.loc[index]
            and pd.notna(tiered.at[index, score_column])
        ]
        ordered_indices = sorted(
            eligible_indices,
            key=lambda index: (
                -float(tiered.at[index, score_column]),
                natural_sort_key(tiered.at[index, "bsr"]),
            ),
        )
        start = 0
        for tier_number, group_size in enumerate(
            tier_group_sizes(len(ordered_indices)),
            start=1,
        ):
            stop = start + group_size
            tiered.loc[
                ordered_indices[start:stop], tier_column
            ] = f"Tier {tier_number}"
            start = stop
    return tiered


def add_preliminary_tiers(bsr: pd.DataFrame) -> pd.DataFrame:
    """Assign preliminary tiers from consecutive overall-risk thirds."""
    tiered = add_ranked_thirds(
        bsr,
        "overall_risk_score",
        "preliminary_tier",
    )
    tiered["preliminary_tier_basis"] = np.where(
        tiered["preliminary_tier"].eq("Not included"),
        "Excluded from analysis",
        "Overall risk score",
    )
    return tiered


def complete_category_color_map(
    categories: pd.Series | list[str],
    preferred: dict[str, str] | None = None,
) -> dict[str, str]:
    """Return stable colors for all categorical values in a map."""
    values = sorted(
        {str(value) for value in categories if not pd.isna(value)},
        key=natural_sort_key,
    )
    colors = dict(preferred or {})
    available_colors = [
        color
        for color in REFERENCE_CATEGORY_COLORS
        if color not in set(colors.values())
    ] or REFERENCE_CATEGORY_COLORS
    color_index = 0
    for value in values:
        if value in colors:
            continue
        if "; " in value:
            colors[value] = "#94a3b8"
        else:
            colors[value] = available_colors[
                color_index % len(available_colors)
            ]
            color_index += 1
    return colors


def summarize_ranked_categories(
    table: pd.DataFrame,
    category_column: str,
    score_column: str,
    prefix: str,
    ranks: int = 3,
) -> pd.DataFrame:
    """Summarize dense-ranked categories and tied labels for each BSR."""
    ranked = table[score_columns(table, ["bsr", category_column, score_column])].copy()
    ranked[score_column] = pd.to_numeric(
        ranked[score_column], errors="coerce"
    )
    if ranked[score_column].isna().any():
        raise ValueError(
            f"Cannot rank {category_column}: {score_column} contains missing "
            "or nonnumeric values."
        )
    ranked["_dense_rank"] = (
        ranked.groupby("bsr")[score_column]
        .rank(method="dense", ascending=False)
        .astype(int)
    )
    summary = ranked[["bsr"]].drop_duplicates().copy()
    for rank in range(1, ranks + 1):
        label_column = f"{prefix}_{rank}_label"
        value_column = f"{prefix}_{rank}_score"
        detail = (
            ranked.loc[ranked["_dense_rank"].eq(rank)]
            .groupby("bsr", as_index=False)
            .agg(
                **{
                    label_column: (
                        category_column,
                        lambda values: "; ".join(
                            sorted(set(values.astype(str)))
                        ),
                    ),
                    value_column: (score_column, "first"),
                }
            )
        )
        summary = summary.merge(
            detail,
            on="bsr",
            how="left",
            validate="one_to_one",
        )
        ceiling_field = maximum_column(score_column)
        if ceiling_field in ranked:
            ceiling_rows = []
            for bsr, group in ranked.loc[ranked["_dense_rank"].eq(rank)].groupby("bsr"):
                if np.allclose(group[ceiling_field], group[ceiling_field].iloc[0]):
                    ceiling = group[ceiling_field].iloc[0]
                else:
                    ceiling = "totals by tied category: " + "; ".join(
                        f"{row[category_column]}: {format_score(row[ceiling_field])}"
                        for _, row in group.sort_values(category_column).iterrows()
                    )
                ceiling_rows.append({"bsr": bsr, maximum_column(value_column): ceiling})
            detail_maxima = pd.DataFrame(
                ceiling_rows, columns=["bsr", maximum_column(value_column)]
            )
            summary = summary.merge(detail_maxima, on="bsr", how="left", validate="one_to_one")
    return summary


def render_shared_legend(
    title: str,
    categories: pd.Series | list[str],
    color_map: dict[str, str],
) -> None:
    """Render one compact color legend for a group of categorical maps."""
    values = sorted(
        {str(value) for value in categories if not pd.isna(value)},
        key=natural_sort_key,
    )
    items = "".join(
        (
            '<span style="display:inline-flex;align-items:center;gap:0.35rem;'
            'margin:0 0.9rem 0.35rem 0;">'
            f'<span style="width:0.85rem;height:0.85rem;border:1px solid '
            f'#64748b;background:{color_map[value]};display:inline-block;">'
            '</span>'
            f'<span>{escape(value)}</span></span>'
        )
        for value in values
    )
    st.markdown(
        f'<div style="font-size:0.82rem;line-height:1.4;margin-bottom:0.4rem;">'
        f'<strong>{escape(title)}</strong><br>{items}</div>',
        unsafe_allow_html=True,
    )


def summarize_priority_life_stages(
    bsr: pd.DataFrame,
    life_stage: pd.DataFrame,
) -> pd.DataFrame:
    """Add the three highest-risk species/life-stage records for each BSR."""
    ranked = life_stage.copy()
    ranked["risk_score"] = pd.to_numeric(
        ranked["risk_score"], errors="coerce"
    )
    if ranked["risk_score"].isna().any():
        raise ValueError(
            "life_stage_scores.csv contains missing or nonnumeric Life-Stage "
            "Risk Scores."
        )
    ranked = ranked.sort_values(
        ["bsr", "risk_score", "species", "life_stage"],
        ascending=[True, False, True, True],
        kind="mergesort",
    )
    ranked["_priority_position"] = ranked.groupby("bsr").cumcount() + 1

    summary = bsr.copy()
    for position in range(1, 4):
        score_column = f"priority_life_stage_{position}_risk_score"
        label_column = f"priority_life_stage_{position}_hover_label"
        detail = ranked.loc[
            ranked["_priority_position"].eq(position),
            score_columns(ranked, ["bsr", "species", "life_stage", "risk_score"]),
        ].copy()
        detail[label_column] = (
            detail["species"].astype(str)
            + " | "
            + detail["life_stage"].astype(str)
            + " Life-Stage Risk Score"
        )
        detail = detail.rename(columns={
            "risk_score": score_column,
            maximum_column("risk_score"): maximum_column(score_column),
        })
        detail = detail[score_columns(detail, ["bsr", score_column, label_column])]
        summary = summary.merge(
            detail,
            on="bsr",
            how="left",
            validate="one_to_one",
        )
    return summary


def summarize_priority_limiting_factors(
    bsr: pd.DataFrame,
    limiting_factor: pd.DataFrame,
) -> pd.DataFrame:
    """Add dense-ranked second and third limiting-factor labels by BSR."""
    ranked = summarize_ranked_categories(
        limiting_factor,
        "limiting_factor",
        "risk_score",
        "_priority_limiting_factor",
    )
    summary = bsr.copy()
    for position, ordinal in ((2, "second"), (3, "third")):
        source = f"_priority_limiting_factor_{position}_label"
        target = f"{ordinal}_highest_risk_limiting_factor"
        summary = summary.merge(
            ranked[["bsr", source]].rename(columns={source: target}),
            on="bsr",
            how="left",
            validate="one_to_one",
        )
    return summary

def show_score_table(
    table: pd.DataFrame,
    column_labels: dict[str, str] | None = None,
) -> None:
    """Display score tables with calculation-consistent, readable labels."""
    display_fields = [field for field in table if not field.endswith(MAXIMUM_SUFFIX)]
    labels = {
        column: DISPLAY_LABELS.get(
            column,
            column.replace("_", " ").title(),
        )
        for column in display_fields
    }
    labels.update(column_labels or {})
    rounded = round_float_columns(table)[display_fields]
    for field in display_fields:
        if is_score_field(field):
            rounded[field] = table.apply(lambda row: format_score_value(row, field), axis=1)
    rounded = rounded.rename(columns=labels)
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


def map_center_zoom(
    geometry: gpd.GeoDataFrame,
    height: int,
    *,
    bottom_margin: int,
    right_margin: int,
    padding_fraction: float = 0.06,
) -> tuple[dict[str, float], float]:
    """Frame the complete BSR extent in Web Mercator with a small margin.

    Plotly's fitbounds does not fit choroplethmap polygons, so calculate a zoom
    for the usable map height and a conservative 700 px chart width. Streamlit
    charts can expand beyond that width on a desktop without clipping.
    """
    west, south, east, north = map(float, geometry.total_bounds)
    south, north = np.clip([south, north], -85.0, 85.0)
    y_south, y_north = (1 - np.arcsinh(np.tan(np.radians([south, north]))) / np.pi) / 2
    y_mid = (y_south + y_north) / 2
    center = {
        "lon": (west + east) / 2,
        "lat": float(np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * y_mid))))),
    }
    x_span = max((east - west) / 360, 1e-9)
    y_span = max(y_south - y_north, 1e-9)
    map_width = 700 - 15 - right_margin
    map_height = height - 55 - bottom_margin
    padded = 1 + 2 * padding_fraction
    zoom = min(
        np.log2(map_width / (512 * x_span * padded)),
        np.log2(map_height / (512 * y_span * padded)),
        12.0,
    )
    return center, float(zoom)


def selected_point_bsr(point: dict[str, Any]) -> str | None:
    """Extract a BSR identifier from a Plotly selection point."""
    custom = point.get("customdata")
    if isinstance(custom, (list, tuple, np.ndarray)) and len(custom):
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
    available = set(st.session_state.get("available_bsrs", []))
    candidates = [selected_point_bsr(point) for point in points]
    candidates = [value for value in candidates if value in available]
    current = st.session_state.get("selected_bsr")
    changed = [value for value in candidates if value != current]
    if candidates:
        st.session_state["selected_bsr"] = (changed or candidates)[-1]


def apply_bsr_selection_style(
    figure: Any,
    selected_bsr: str | None,
    chart_key: str,
) -> None:
    """Sync clicks with the sidebar while leaving every polygon fill intact."""
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
            selected={"marker": {"opacity": MAP_FILL_OPACITY}},
            unselected={"marker": {"opacity": MAP_FILL_OPACITY}},
        )

    # Changing this value tells Plotly to accept the dropdown-controlled
    # selection instead of retaining a previous map click.
    figure.update_layout(
        selectionrevision=f"{chart_key}:{selected_value or 'none'}",
        clickmode="event+select",
        dragmode="pan",
        uirevision=chart_key,
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


def add_excluded_bsr_hatching(
    figure: Any,
    mapped: gpd.GeoDataFrame,
    plot_data: pd.DataFrame,
) -> None:
    """Cover excluded BSRs with a neutral fill and diagonal hatching."""
    if mapped.empty or not figure.data:
        return
    excluded_mask = mapped["bsr"].map(normalized_bsr_id).isin(
        EXCLUDED_TIER_BSRS
    )
    excluded = mapped.loc[excluded_mask, ["bsr", "geometry"]].copy()
    if excluded.empty:
        return

    hover_lookup = plot_data.set_index("bsr")["_hover_text"].to_dict()
    customdata = [
        [
            bsr,
            hover_lookup.get(
                bsr,
                f"<b>BSR:</b> {escape(str(bsr))}<br>"
                "<b>Analysis status:</b> Not included",
            ),
        ]
        for bsr in excluded["bsr"]
    ]
    overlay_arguments = {
        "geojson": json.loads(excluded.to_json()),
        "locations": excluded["bsr"],
        "featureidkey": "properties.bsr",
        "z": [0] * len(excluded),
        "zmin": 0,
        "zmax": 1,
        "colorscale": [
            [0, EXCLUDED_BSR_FILL_COLOR],
            [1, EXCLUDED_BSR_FILL_COLOR],
        ],
        "showscale": False,
        "showlegend": False,
        "customdata": customdata,
        "hovertemplate": "%{customdata[1]}<extra></extra>",
        "marker": {
            "line": {
                "color": MAP_BOUNDARY_COLOR,
                "width": MAP_BOUNDARY_WIDTH,
            },
            "opacity": 1,
        },
        "name": "Not included in analysis: UGR8 (U8)",
    }
    trace_type = getattr(figure.data[0], "type", "")
    if trace_type == "choroplethmap":
        figure.add_trace(go.Choroplethmap(**overlay_arguments))
        scatter_class = getattr(go, "Scattermap", None)
    else:
        figure.add_trace(go.Choroplethmapbox(**overlay_arguments))
        scatter_class = go.Scattermapbox
    if scatter_class is None:
        scatter_class = go.Scattermapbox

    projected = excluded.to_crs(epsg=3857)
    hatch_segments = []
    for polygon in projected.geometry:
        min_x, min_y, max_x, max_y = polygon.bounds
        width = max_x - min_x
        height = max_y - min_y
        spacing = max(max(width, height) / 14.0, 1.0)
        for x_start in np.arange(
            min_x - height,
            max_x + spacing,
            spacing,
        ):
            diagonal = LineString(
                [(x_start, min_y), (x_start + height, max_y)]
            )
            clipped = polygon.intersection(diagonal)
            if clipped.is_empty:
                continue
            if clipped.geom_type == "LineString":
                hatch_segments.append(clipped)
            elif clipped.geom_type in {"MultiLineString", "GeometryCollection"}:
                hatch_segments.extend(
                    part
                    for part in clipped.geoms
                    if part.geom_type == "LineString" and not part.is_empty
                )
    if not hatch_segments:
        return

    hatch_lines = gpd.GeoSeries(
        hatch_segments,
        crs=projected.crs,
    ).to_crs(epsg=4326)
    longitudes: list[float | None] = []
    latitudes: list[float | None] = []
    for line in hatch_lines:
        coordinates = list(line.coords)
        longitudes.extend([coordinate[0] for coordinate in coordinates])
        latitudes.extend([coordinate[1] for coordinate in coordinates])
        longitudes.append(None)
        latitudes.append(None)

    figure.add_trace(
        scatter_class(
            lon=longitudes,
            lat=latitudes,
            mode="lines",
            line={
                "color": EXCLUDED_BSR_HATCH_COLOR,
                "width": EXCLUDED_BSR_HATCH_WIDTH,
            },
            hoverinfo="skip",
            showlegend=False,
            name="Not included in analysis: UGR8 (U8)",
        )
    )


def add_excluded_bsr_key(figure: Any) -> None:
    """Place a hatched swatch in the right map margin, clear of color keys."""
    # Half the previous width and height gives the swatch one-quarter its area.
    x0, x1 = 1.04, 1.06
    y0, y1 = 0.885, 0.915
    figure.add_shape(
        type="rect", xref="paper", yref="paper",
        x0=x0, x1=x1, y0=y0, y1=y1,
        fillcolor=EXCLUDED_BSR_FILL_COLOR,
        line={"color": MAP_BOUNDARY_COLOR, "width": 0.7},
    )
    # Clip each diagonal to the swatch, using the same color as the map hatch.
    for start in np.arange(x0 - (y1 - y0), x1, 0.0085):
        left = max(x0, start)
        right = min(x1, start + (y1 - y0))
        if right <= left:
            continue
        figure.add_shape(
            type="line", xref="paper", yref="paper",
            x0=left, y0=y0 + left - start,
            x1=right, y1=y0 + right - start,
            line={"color": EXCLUDED_BSR_HATCH_COLOR, "width": 1},
        )
    figure.add_annotation(
        xref="paper", yref="paper", x=1.11, y=(y0 + y1) / 2,
        text="Not included in analysis:<br>UGR8 (U8)",
        xanchor="left", yanchor="middle", align="left",
        showarrow=False, font={"size": 11, "color": "#334155"},
    )


def add_bsr_labels(
    figure: Any,
    mapped: gpd.GeoDataFrame,
) -> None:
    """Draw off-white BSR IDs over a thin charcoal text halo."""
    if mapped.empty or not figure.data:
        return
    label_points = mapped[["bsr", "geometry"]].to_crs(epsg=3857)
    label_points["geometry"] = label_points.geometry.representative_point()
    label_points = label_points.to_crs(epsg=4326)
    trace_type = getattr(figure.data[0], "type", "")
    scatter_class = (
        getattr(go, "Scattermap", None)
        if trace_type == "choroplethmap"
        else go.Scattermapbox
    )
    if scatter_class is None:
        scatter_class = go.Scattermapbox

    common = {
        "lon": label_points.geometry.x,
        "lat": label_points.geometry.y,
        "text": label_points["bsr"].map(short_bsr_label),
        "customdata": [[str(bsr)] for bsr in label_points["bsr"]],
        "ids": label_points["bsr"].astype(str),
        "textposition": "middle center",
        "hoverinfo": "none",
        "showlegend": False,
    }
    # A one-pixel charcoal edge separates the off-white glyphs from light fills.
    figure.add_trace(
        scatter_class(
            **common,
            mode="text",
            textfont={"family": "Open Sans Bold", "size": 14, "color": "#414141"},
            name="BSR label halo",
        )
    )
    figure.add_trace(
        scatter_class(
            **common,
            mode="markers+text",
            marker={"size": 26, "color": "#ffffff", "opacity": 0.001},
            textfont={"family": "Open Sans Bold", "size": 12, "color": "#414141"},
            name="BSR labels",
        )
    )


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
    hover_label_overrides: dict[str, str] | None = None,
    hover_label_columns: dict[str, str] | None = None,
    color_scale: str | list[list[Any]] = DEFAULT_COLOR_SCALE,
    color_column: str | None = None,
    color_label: str | None = None,
    color_discrete_map: dict[str, str] | None = None,
    category_order: list[str] | None = None,
    range_color: tuple[float, float] | None = None,
    height: int = 600,
    show_legend: bool = True,
) -> None:
    """Render an interactive BSR choropleth and register click selection."""
    color_column = color_column or metric
    color_label = color_label or metric_label
    hover_columns = [
        column
        for column in (hover_columns or [])
        if column not in {"bsr", "basin"}
    ]
    hover_label_overrides = hover_label_overrides or {}
    hover_label_columns = {
        field: label_column
        for field, label_column in (hover_label_columns or {}).items()
        if field in values.columns and label_column in values.columns
    }
    missing_map_fields = [
        column
        for column in (metric, color_column)
        if column not in values.columns
    ]
    if missing_map_fields:
        st.error(
            f"Cannot render {metric_label}: required fields "
            f"{missing_map_fields!r} are "
            "missing from the current score table. Re-run the revised "
            "integrated-scoring notebook and deploy the updated outputs."
        )
        return
    requested = [
        "bsr",
        metric,
        color_column,
        *hover_columns,
        *hover_label_columns.values(),
    ]
    requested = list(dict.fromkeys(column for column in requested if column in values.columns))
    requested = score_columns(values, requested)
    value_table = values[requested].drop_duplicates("bsr")
    map_candidates = geometry.merge(
        value_table,
        on="bsr",
        how="inner",
        validate="one_to_one",
    )
    excluded_candidate_mask = map_candidates["bsr"].map(
        normalized_bsr_id
    ).isin(EXCLUDED_TIER_BSRS)
    mapped = map_candidates.loc[
        map_candidates[metric].notna()
        & map_candidates[color_column].notna()
        & ~excluded_candidate_mask
    ].copy()

    if mapped.empty:
        st.warning(f"No mapped BSRs contain values for {metric_label}.")
        return

    excluded_candidates = map_candidates.loc[excluded_candidate_mask]
    displayed_geometry = gpd.GeoDataFrame(
        pd.concat([mapped, excluded_candidates], ignore_index=True)
        .drop_duplicates("bsr"),
        geometry="geometry",
        crs=geometry.crs,
    )
    has_excluded = displayed_geometry["bsr"].map(
        normalized_bsr_id
    ).isin(EXCLUDED_TIER_BSRS).any()
    # Reserve the right margin for the hatch key and the bottom for color scales.
    right_margin = 240 if has_excluded and show_legend else 15
    bottom_margin = 15 if categorical else 95
    center, zoom = map_center_zoom(
        displayed_geometry, height,
        bottom_margin=bottom_margin, right_margin=right_margin,
    )

    geojson = json.loads(mapped[["bsr", "geometry"]].to_json())
    plot_data = pd.DataFrame(mapped.drop(columns="geometry"))

    hover_fields = list(dict.fromkeys(["bsr", *hover_columns]))
    if metric not in hover_fields:
        hover_fields.insert(1, metric)
    hover_fields = [
        column for column in hover_fields if column in plot_data.columns
    ]
    hover_labels = {
        column: DISPLAY_LABELS.get(
            column,
            column.replace("_", " ").title(),
        )
        for column in hover_fields
    }
    hover_labels[metric] = metric_label
    hover_labels.update(hover_label_overrides)

    def hover_text(row: pd.Series) -> str:
        lines = []
        for column in hover_fields:
            value = row[column]
            displayed = format_score_value(row, column)
            label_column = hover_label_columns.get(column)
            dynamic_label = (
                row[label_column]
                if label_column is not None
                and label_column in row.index
                and not pd.isna(row[label_column])
                else hover_labels[column]
            )
            label = escape(str(dynamic_label))
            lines.append(f"<b>{label}:</b> {escape(displayed)}")
        return "<br>".join(lines)

    plot_data["_hover_text"] = plot_data.apply(hover_text, axis=1)

    common = {
        "data_frame": plot_data,
        "geojson": geojson,
        "locations": "bsr",
        "featureidkey": "properties.bsr",
        "color": color_column,
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
        | {metric: metric_label, color_column: color_label},
    }

    if categorical:
        # Plotly Express otherwise repeats the entire watershed GeoJSON in
        # every category trace. Assign only each trace's polygons below.
        common["geojson"] = {"type": "FeatureCollection", "features": []}
        color_discrete_map = complete_category_color_map(
            plot_data[color_column], color_discrete_map
        )
        common["color_discrete_map"] = color_discrete_map
        if category_order:
            common["category_orders"] = {color_column: category_order}
    else:
        common["color_continuous_scale"] = color_scale
        numeric = pd.to_numeric(plot_data[color_column], errors="coerce")
        if range_color is not None:
            common["range_color"] = range_color
        else:
            minimum = float(numeric.min())
            maximum = float(numeric.max())
            if minimum == maximum:
                pad = max(abs(minimum) * 0.05, 0.01)
                common["range_color"] = (max(0.0, minimum - pad), maximum + pad)
            else:
                common["range_color"] = (minimum, maximum)

    if hasattr(px, "choropleth_map"):
        figure = px.choropleth_map(map_style=map_style, **common)
        figure.update_layout(
            map={"bearing": 0, "pitch": 0},
            margin={"r": 0, "t": 55, "l": 0, "b": 0},
            height=height,
            legend_title_text=color_label,
            showlegend=False,
            coloraxis_showscale=show_legend,
        )
    else:
        figure = px.choropleth_mapbox(mapbox_style=map_style, **common)
        figure.update_layout(
            mapbox={"bearing": 0, "pitch": 0},
            margin={"r": 0, "t": 55, "l": 0, "b": 0},
            height=height,
            legend_title_text=color_label,
            showlegend=False,
            coloraxis_showscale=show_legend,
        )

    if categorical:
        features_by_bsr = {
            str(feature["properties"]["bsr"]): feature
            for feature in geojson["features"]
        }
        for trace in figure.data:
            trace.geojson = {
                "type": "FeatureCollection",
                "features": [features_by_bsr[str(bsr)] for bsr in trace.locations],
            }

    figure.update_traces(
        marker_line_width=MAP_BOUNDARY_WIDTH,
        marker_line_color=MAP_BOUNDARY_COLOR,
        hovertemplate="%{customdata[1]}<extra></extra>",
        showlegend=False,
    )
    apply_bsr_selection_style(
        figure,
        st.session_state.get("selected_bsr"),
        chart_key,
    )
    add_excluded_bsr_hatching(
        figure,
        displayed_geometry,
        plot_data,
    )
    add_selected_bsr_outline(
        figure,
        displayed_geometry,
        plot_data,
        st.session_state.get("selected_bsr"),
    )
    add_bsr_labels(figure, displayed_geometry)
    if has_excluded and show_legend:
        add_excluded_bsr_key(figure)
    figure.update_layout(
        showlegend=False,
        margin={"r": right_margin, "t": 55, "l": 15, "b": bottom_margin},
    )
    if not categorical:
        figure.update_layout(
            coloraxis_colorbar={
                "orientation": "h", "x": 0.5, "xanchor": "center",
                "y": -0.04,
                "yanchor": "top", "len": 0.8,
                "thickness": 12,
                "title": {"text": score_axis_label(plot_data, color_column, color_label),
                          "side": "bottom"},
            },
        )
    st.plotly_chart(
        figure,
        width="stretch",
        key=chart_key,
        on_select=partial(capture_map_selection, chart_key),
        selection_mode="points",
        config={"displaylogo": False, "scrollZoom": True},
    )
    if categorical and show_legend:
        render_shared_legend(color_label, plot_data[color_column], color_discrete_map)


def horizontal_bar(
    table: pd.DataFrame,
    value: str,
    category: str,
    title: str,
    *,
    color: str | None = None,
    value_label: str | None = None,
    hover_columns: list[str] | None = None,
    hover_label_overrides: dict[str, str] | None = None,
) -> None:
    """Render a consistently formatted horizontal comparison chart."""
    ordered = table.copy().sort_values(value, ascending=True)
    hover_columns = [
        column
        for column in (hover_columns or [])
        if column in ordered.columns and column not in {category, value}
    ]
    hover_label_overrides = hover_label_overrides or {}

    def bar_hover_text(row: pd.Series) -> str:
        fields = [category, value, *hover_columns]
        lines = []
        for column in fields:
            displayed_value = row[column]
            if pd.isna(displayed_value):
                if column in hover_columns:
                    continue
                displayed = "Not available"
            else:
                displayed = format_score_value(row, column)
            if column == category:
                label = DISPLAY_LABELS.get(category, category.replace("_", " ").title())
            elif column == value:
                label = value_label or DISPLAY_LABELS.get(
                    value, value.replace("_", " ").title()
                )
            else:
                label = hover_label_overrides.get(
                    column,
                    DISPLAY_LABELS.get(
                        column, column.replace("_", " ").title()
                    ),
                )
            lines.append(f"<b>{escape(str(label))}:</b> {escape(displayed)}")
        return "<br>".join(lines)

    ordered["_hover_text"] = ordered.apply(bar_hover_text, axis=1)
    ordered["_score_text"] = ordered.apply(lambda row: format_score_value(row, value), axis=1)
    figure = px.bar(
        ordered,
        x=value,
        y=category,
        color=color,
        orientation="h",
        title=title,
        labels={value: score_axis_label(ordered, value, value_label or value), category: ""},
        text="_score_text",
        custom_data=["_hover_text"],
    )
    figure.update_layout(
        height=max(360, 30 * len(ordered) + 110),
        margin={"r": 80, "t": 55, "l": 10, "b": 65},
        legend_title_text="Species" if color == "species" else color,
    )
    figure.update_traces(
        textposition="outside",
        cliponaxis=False,
        hovertemplate=(
            "%{customdata[0]}<extra></extra>"
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
    # Reattach display metadata in case an older cached table lacks it.
    all_bsr = add_preliminary_tiers(add_observed_risk_maximum(tables["bsr"]))
    bsr = filter_table(all_bsr, basin)
    life = filter_table(tables["life_stage"], basin)
    limiting = filter_table(tables["limiting_factor"], basin)

    st.header("Level 1: Overall Risk")
    st.caption(
        "Click a polygon or use the sidebar BSR selector. The charts below "
        "partition the selected BSR score by species and life stage and by "
        "limiting factor."
    )
    map_selection = st.selectbox(
        "Risk score map",
        options=[
            "Overall Risk Score",
            "Overall Preliminary Tiers",
            "Chinook Risk Score",
            "Steelhead Risk Score",
            "Bull Trout Risk Score",
        ],
        key="risk_map_selection",
    )
    common_hover = [
        "overall_risk_score",
        "fish_use_score",
        "highest_risk_species_life_stage",
        "highest_risk_limiting_factor",
    ]
    species_map = None
    if map_selection == "Overall Risk Score":
        render_choropleth(
            geometry,
            bsr,
            "overall_risk_score",
            "Overall Risk Score",
            "Overall Risk Score",
            "map_overall_risk",
            map_style,
            hover_columns=common_hover,
            color_scale=RISK_COLOR_SCALE,
            range_color=(
                0.0,
                float(bsr[observed_maximum_column("overall_risk_score")].iloc[0]),
            ),
        )
        st.caption(
            "Blue represents lower risk and dark red represents higher risk."
        )
    elif map_selection == "Overall Preliminary Tiers":
        render_choropleth(
            geometry,
            bsr,
            "preliminary_tier",
            "Overall Preliminary Tiers",
            "Overall Preliminary Tiers",
            "map_overall_risk",
            map_style,
            categorical=True,
            hover_columns=[
                "preliminary_tier_basis",
                *common_hover,
            ],
            color_discrete_map=TIER_COLORS,
            category_order=TIER_ORDER,
        )
        st.caption(
            "Preliminary tiers assign consecutive thirds by Overall Risk "
            "Score. Tier 1 is the highest third and Tier 3 is the lowest."
        )
    else:
        species = map_selection.removesuffix(" Risk Score")
        species_scores = summarize_species_risk(life)
        species_map = species_scores.loc[species_scores["species"].eq(species)].copy()
        species_observed_maximum = (
            float(species_map[
                observed_maximum_column("species_risk_score")
            ].iloc[0])
            if not species_map.empty else None
        )
        render_choropleth(
            geometry, species_map, "species_risk_score", map_selection,
            map_selection, "map_overall_risk", map_style,
            hover_columns=["species"], color_scale=RISK_COLOR_SCALE,
            range_color=(0.0, species_observed_maximum)
            if species_observed_maximum is not None else None,
        )
        st.caption(f"{species} risk is the sum of its species/life-stage risk scores within each BSR.")

    row = bsr.loc[bsr["bsr"].eq(selected_bsr)].iloc[0]
    metric_columns = st.columns(4)
    metric_columns[0].metric("Selected BSR", selected_bsr)
    metric_columns[1].metric(
        "Overall Risk Score", format_score_value(row, "overall_risk_score"),
        help=OVERALL_RISK_MAXIMUM_HELP,
    )
    metric_columns[2].metric(
        "Preliminary Tier", row["preliminary_tier"]
    )
    if species_map is None:
        secondary_label = "Overall Fish Use Score"
        secondary_value = format_score_value(row, "fish_use_score")
        secondary_help = OVERALL_FISH_USE_HELP
    else:
        secondary_label = f"{species} Risk Score"
        selected_species = species_map.loc[species_map["bsr"].eq(selected_bsr)]
        secondary_value = (
            format_score_value(selected_species.iloc[0], "species_risk_score")
            if not selected_species.empty else "Not available"
        )
        secondary_help = SPECIES_RISK_MAXIMUM_HELP
    metric_columns[3].metric(
        secondary_label, secondary_value, help=secondary_help,
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
                score_columns(life_selected, [
                    "species",
                    "species_aggregate_score",
                    "life_stage",
                    "LS_corrected_score",
                    "population_priority",
                    "risk_score",
                ])
            ].sort_values("risk_score", ascending=False),
            column_labels={
                "risk_score": "Life-Stage Risk Score",
            },
        )
        st.subheader("Limiting factors")
        show_score_table(
            factor_selected[
                score_columns(factor_selected, [
                    "limiting_factor",
                    "condition_score",
                    "risk_score",
                ])
            ].sort_values("risk_score", ascending=False),
            column_labels={
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
        life[score_columns(life, ["bsr", "species", "species_aggregate_score"])]
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
    priority_life_stage_map = summarize_priority_life_stages(bsr, life)
    priority_life_stage_scores = [
        f"priority_life_stage_{position}_risk_score"
        for position in range(1, 4)
    ]
    priority_life_stage_labels = {
        score_column: score_column.replace(
            "_risk_score", "_hover_label"
        )
        for score_column in priority_life_stage_scores
    }
    priority_life_stage_colors = complete_category_color_map(
        priority_life_stage_map["highest_risk_species_life_stage"],
        LIFE_STAGE_COLOR_MAP,
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
        st.subheader("Overall Fish Use Score", help=OVERALL_FISH_USE_HELP)
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
                score_columns(species_rows, [
                    "bsr",
                    "species",
                    "species_aggregate_score",
                ])
            ].drop_duplicates("bsr").merge(
                fish_hover_values[
                    score_columns(fish_hover_values, ["bsr", "fish_use_score", *species_hover_columns])
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
                score_columns(species_rows, [
                    "bsr",
                    "species",
                    "life_stage",
                    "LS_corrected_score",
                    "species_aggregate_score",
                ]),
            ].merge(
                fish_hover_values[
                    score_columns(fish_hover_values, ["bsr", "fish_use_score", *species_hover_columns])
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
        color_scale=FISH_USE_COLOR_SCALE,
    )
    st.caption(
        "Colors span the lowest to highest BSR score on this map. White is "
        "the lowest displayed score, which may be greater than zero."
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
        format_score_value(selected_bsr_row, "fish_use_score"),
        help=OVERALL_FISH_USE_HELP,
    )
    if map_level == "Overall":
        summary_columns[2].metric(
            "Highest Risk-Aligned Life Stage",
            selected_bsr_row["highest_risk_species_life_stage"],
        )
    else:
        summary_columns[2].metric(
            map_metric_label,
            (format_score_value(selected_map_row.iloc[0], map_metric)
             if not selected_map_row.empty else "Not available"),
        )

    species_summary = (
        selected[score_columns(selected, ["species", "species_aggregate_score"])]
        .drop_duplicates()
        .sort_values("species")
    )
    life_stage_hover = selected.pivot(
        index="species",
        columns="life_stage",
        values="LS_corrected_score",
    ).sort_index(axis=1)
    life_stage_hover.columns = [
        f"{life_stage} Life-Stage Fish Use Score"
        for life_stage in life_stage_hover.columns
    ]
    life_stage_hover.columns.name = None
    life_stage_hover_columns = life_stage_hover.columns.tolist()
    species_summary = species_summary.merge(
        life_stage_hover.reset_index(),
        on="species",
        how="left",
        validate="one_to_one",
    )
    left, right = st.columns(2)
    with left:
        horizontal_bar(
            species_summary,
            "species_aggregate_score",
            "species",
            f"{selected_bsr}: fish use by species",
            value_label="Species Fish Use Score",
            hover_columns=life_stage_hover_columns,
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
            score_columns(selected, ["life_stage", "LS_corrected_score"]),
        ]
        horizontal_bar(
            life_stage_summary,
            "LS_corrected_score",
            "life_stage",
            f"{selected_bsr}: {chart_species} fish use by life stage",
            value_label="Life-Stage Fish Use Score",
        )

    with st.expander("Highest Priority Life Stage", expanded=True):
        st.caption(
            "This placeholder indicator is based on the population-weighted "
            "Level 1 risk score, not fish use alone, and is retained for "
            "future consideration."
        )
        render_choropleth(
            geometry,
            priority_life_stage_map,
            "highest_risk_species_life_stage",
            "Species | Life Stage",
            "Highest Risk-Aligned Life Stage",
            "map_top_life_stage",
            map_style,
            categorical=True,
            hover_columns=[
                "fish_use_score",
                "highest_risk_species_life_stage",
                *priority_life_stage_scores,
            ],
            hover_label_columns=priority_life_stage_labels,
            color_discrete_map=priority_life_stage_colors,
            category_order=sorted(
                priority_life_stage_map[
                    "highest_risk_species_life_stage"
                ].dropna().astype(str).unique()
            ),
        )

    with st.expander(f"Show species and life-stage data for BSR: {selected_bsr}"):
        show_score_table(
            selected[
                score_columns(selected, [
                    "species",
                    "species_aggregate_score",
                    "life_stage",
                    "LS_corrected_score",
                ])
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
    """Render factor-condition maps and biological risk drill-downs."""
    limiting = filter_table(tables["limiting_factor"], basin)
    grid = filter_table(tables["grid"], basin)
    ranked_factor_maps = summarize_ranked_categories(
        limiting,
        "limiting_factor",
        "risk_score",
        "limiting_factor_rank",
    )
    factor_category_values = pd.concat(
        [
            ranked_factor_maps[f"limiting_factor_rank_{rank}_label"]
            for rank in range(1, 4)
        ],
        ignore_index=True,
    )
    factor_colors = complete_category_color_map(
        factor_category_values,
        LIMITING_FACTOR_COLOR_MAP,
    )

    st.header("Level 1: Limiting Factors")
    overall_conditions = summarize_overall_limiting_factors(limiting)
    st.subheader("Overall Limiting-Factor Condition Score")
    st.caption(
        "Unweighted mean of all limiting-factor condition scores within each "
        "BSR, out of 1. Higher values indicate more limiting conditions."
    )
    render_choropleth(
        geometry, overall_conditions, "overall_limiting_factor_score",
        "Overall Limiting-Factor Condition Score",
        "Overall Limiting-Factor Condition Score", "map_overall_limiting_factor",
        map_style, hover_columns=["limiting_factor_count", "overall_risk_score"],
        color_scale=LIMITING_FACTOR_COLOR_SCALE,
    )
    st.caption(
        "Map colors span the observed BSR scores. White is the lowest "
        "displayed score, which may be greater than zero."
    )
    st.subheader("Specific Limiting-Factor Map")
    factor_options = sorted(limiting["limiting_factor"].dropna().unique())
    selected_factor = st.selectbox(
        "Limiting factor",
        factor_options,
        help=LIMITING_FACTOR_SELECTION_HELP,
    )
    factor_map = limiting.loc[
        limiting["limiting_factor"].eq(selected_factor)
    ].copy()
    factor_map_selection = st.radio(
        "Limiting-factor map value",
        options=[
            "Limiting-Factor Condition Score",
            "Limiting-Factor Risk Score",
        ],
        horizontal=True,
    )
    factor_map_metric = (
        "condition_score"
        if factor_map_selection == "Limiting-Factor Condition Score"
        else "risk_score"
    )
    render_choropleth(
        geometry,
        factor_map,
        factor_map_metric,
        factor_map_selection,
        f"{selected_factor}: {factor_map_selection}",
        "map_specific_limiting_factor",
        map_style,
        hover_columns=[
            "condition_score",
            "risk_score",
        ],
        color_scale=LIMITING_FACTOR_COLOR_SCALE,
    )
    st.caption(
        "Colors span the observed BSR scores for the selected limiting factor."
    )

    st.subheader("Specific Limiting-Factor Drill-Down")

    left, right = st.columns(2)
    bsr_factors = limiting.loc[limiting["bsr"].eq(selected_bsr)].copy()
    with left:
        horizontal_bar(
            bsr_factors,
            "condition_score",
            "limiting_factor",
            f"{selected_bsr}: limiting-factor condition comparison",
            value_label="Limiting-Factor Condition Score",
        )

    biological = grid.loc[
        grid["bsr"].eq(selected_bsr)
        & grid["limiting_factor"].eq(selected_factor)
    ].copy()
    biological["species_life_stage_label"] = (
        biological["species"] + " | " + biological["life_stage"]
    )
    with right:
        horizontal_bar(
            biological,
            "risk_component",
            "species_life_stage_label",
            f"{selected_bsr}: {selected_factor} risk by species and life stage",
            color="species",
            value_label="Risk Component",
            hover_columns=[
                "population_priority",
                "vulnerability_score",
            ],
        )

    with st.expander(
        f"Show limiting-factor data for: {selected_factor} in {selected_bsr}"
    ):
        show_score_table(
            biological[
                score_columns(biological, [
                    "species",
                    "species_aggregate_score",
                    "life_stage",
                    "LS_corrected_score",
                    "fish_use_score",
                    "population_priority",
                    "condition_score",
                    "vulnerability_score",
                    "risk_component",
                ])
            ].sort_values("risk_component", ascending=False),
            column_labels={
                "condition_score": "Limiting-Factor Condition Score",
            },
        )

    st.subheader("Top Three Limiting Factors")
    st.caption(
        "Ranks are based on Limiting-Factor Risk within each BSR. Tied "
        "factors are retained together."
    )
    for rank in range(1, 4):
        label_column = f"limiting_factor_rank_{rank}_label"
        score_column = f"limiting_factor_rank_{rank}_score"
        render_choropleth(
            geometry,
            ranked_factor_maps,
            label_column,
            "Limiting Factor",
            f"Rank {rank} Limiting Factor",
            f"map_limiting_factor_rank_{rank}",
            map_style,
            categorical=True,
            hover_columns=[score_column],
            hover_label_overrides={
                score_column: "Limiting-Factor Risk",
            },
            color_discrete_map=factor_colors,
            category_order=sorted(
                ranked_factor_maps[label_column]
                .dropna()
                .astype(str)
                .unique()
            ),
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
            "Limiting-Factor Risk Ã— Action Weight."
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


def selected_action_map(actions: pd.DataFrame, action_type: str) -> pd.DataFrame:
    """Rebuild map values and tiers from the currently selected action."""
    selected = actions.loc[actions["action_type"].eq(action_type)].copy()
    if selected.empty or selected["bsr"].duplicated().any():
        raise ValueError("An action map requires exactly one score per BSR.")
    return add_ranked_thirds(
        selected, "action_benefit_score", "action_preliminary_tier"
    )


def render_actions(
    tables: dict[str, pd.DataFrame],
    geometry: gpd.GeoDataFrame,
    basin: str,
    selected_bsr: str,
    map_style: str,
) -> None:
    """Render Level 2 action-benefit maps, rankings, and components."""
    bsr = filter_table(add_preliminary_tiers(tables["bsr"]), basin)
    actions = filter_table(tables["action"], basin)
    limiting = filter_table(tables["limiting_factor"], basin)
    action_components = filter_table(tables["action_components"], basin)
    summarize_action_benefits(
        bsr,
        actions,
        limiting,
        action_components,
    )
    ranked_action_maps = summarize_ranked_categories(
        actions,
        "action_type",
        "action_benefit_score",
        "action_rank",
    )
    action_rank_labels = [
        f"action_rank_{rank}_label" for rank in range(1, 4)
    ]
    action_colors = complete_category_color_map(
        pd.concat(
            [
                actions["action_type"],
                *[
                    ranked_action_maps[column]
                    for column in action_rank_labels
                ],
            ],
            ignore_index=True,
        )
    )

    st.header("Level 2: Action Benefits")

    action_options = (
        actions[["action_id", "action_type"]]
        .drop_duplicates()
        .sort_values("action_id")
    )
    selected_action = st.selectbox(
        "Action type",
        action_options["action_type"].tolist(),
        help=ACTION_MAP_HELP,
        key="selected_action_type",
    )
    action_map = selected_action_map(actions, selected_action)
    st.subheader("Action-Specific Benefit Map")
    action_map_selection = st.radio(
        "Action-benefit map value",
        options=[
            "Action-Specific Benefit Score",
            "Action-Specific Preliminary Tiers",
        ],
        horizontal=True,
    )
    normalize_action_colors = st.toggle(
        "Normalize scores to highest ranked action",
        value=False,
        key="normalize_action_benefit_colors",
        disabled=action_map_selection != "Action-Specific Benefit Score",
        help=(
            "Use the highest observed Action-Specific Benefit Score across all "
            "action types and BSRs included in the analysis as the common "
            "color-scale upper bound, with zero at the low end. When off, "
            "colors span the displayed BSRs for the selected action. "
            "Scores do not change."
        ),
    )
    # Reset the Plotly selection widget when the action, map mode, or color
    # scale changes so the previous figure cannot persist on the new map.
    action_chart_key = (
        f"map_action_specific:{selected_action}:{action_map_selection}:"
        f"{'shared' if normalize_action_colors else 'selected'}"
    )
    if action_map_selection == "Action-Specific Benefit Score":
        map_scale_maximum = (
            observed_included_maximum(actions, "action_benefit_score")
            if normalize_action_colors else None
        )
        render_choropleth(
            geometry,
            action_map,
            "action_benefit_score",
            "Action-Specific Benefit Score",
            f"{selected_action}: Action-Specific Benefit Score",
            action_chart_key,
            map_style,
            hover_columns=[
                "action_type",
                "benefit_rank_within_bsr",
            ],
            color_scale=ACTION_BENEFIT_COLOR_SCALE,
            range_color=(0.0, map_scale_maximum)
            if map_scale_maximum is not None else None,
        )
        if normalize_action_colors:
            st.caption(
                f"Map colors span 0 to {format_score(map_scale_maximum)} "
                "for all action types, based on the highest observed score "
                "among BSRs included in the analysis. Scores are reported out of 15."
            )
        else:
            st.caption(
                f"Map colors span the lowest to highest displayed BSR for "
                f"{selected_action}. White marks the lowest displayed score, "
                "which may be greater than zero; scores are reported out of 15."
            )
    else:
        render_choropleth(
            geometry,
            action_map,
            "action_preliminary_tier",
            "Action-Specific Preliminary Tiers",
            f"{selected_action}: Preliminary Tiers",
            action_chart_key,
            map_style,
            categorical=True,
            hover_columns=[
                "action_type",
                "action_benefit_score",
            ],
            color_discrete_map=TIER_COLORS,
            category_order=TIER_ORDER,
        )
        st.caption(
            "Tiers rank BSRs in consecutive thirds using the selected action's "
            "benefit score. Tier 1 is the highest third. These tiers update "
            "when Action type changes."
        )

    selected = actions.loc[actions["bsr"].eq(selected_bsr)].copy()
    st.caption(
        "Action-Specific Benefit Score (out of 15 for every BSR and action)",
        help=ACTION_BENEFIT_MAXIMUM_HELP,
    )
    horizontal_bar(
        selected,
        "action_benefit_score",
        "action_type",
        f"{selected_bsr}: Action-Specific Benefit Score by action",
        value_label="Action-Specific Benefit Score",
        hover_columns=["benefit_rank_within_bsr"],
    )

    component_rows = action_components.loc[
        action_components["bsr"].eq(selected_bsr)
        & action_components["action_type"].eq(selected_action),
        score_columns(action_components, [
            "bsr",
            "limiting_factor",
            "lfat_score",
            "benefit_component",
        ]),
    ].merge(
        limiting[
            score_columns(limiting, [
                "bsr",
                "limiting_factor",
                "condition_score",
                "risk_score",
            ])
        ],
        on=["bsr", "limiting_factor"],
        how="left",
        validate="many_to_one",
    )
    with st.expander(
        "Show limiting-factor contributions to the selected action: "
        f"{selected_action}"
    ):
        horizontal_bar(
            component_rows,
            "benefit_component",
            "limiting_factor",
            f"{selected_bsr}: benefit components for {selected_action}",
            value_label="Action-Specific Benefit Component",
            hover_columns=[
                "condition_score",
                "risk_score",
                "lfat_score",
            ],
            hover_label_overrides={
                "risk_score": "Limiting-Factor Risk",
            },
        )
        show_score_table(
            component_rows[
                score_columns(component_rows, [
                    "limiting_factor",
                    "condition_score",
                    "risk_score",
                    "lfat_score",
                    "benefit_component",
                ])
            ].sort_values("benefit_component", ascending=False),
            column_labels={
                "risk_score": "Limiting-Factor Risk",
            },
        )

    with st.expander(f"Show action table for BSR: {selected_bsr}"):
        show_score_table(
            selected[
                score_columns(selected, [
                    "action_id",
                    "action_type",
                    "action_benefit_score",
                    "benefit_rank_within_bsr",
                ])
            ]
            .sort_values("benefit_rank_within_bsr")
            .assign(
                benefit_rank_within_bsr=lambda frame: frame[
                    "benefit_rank_within_bsr"
                ].astype(int)
            )
        )

    st.subheader("Top Three Action-Specific Benefits")
    st.caption(
        "Ranks are based on Action-Specific Benefit Score within each BSR. "
        "Tied action types are retained together."
    )
    for rank in range(1, 4):
        label_column = f"action_rank_{rank}_label"
        score_column = f"action_rank_{rank}_score"
        render_choropleth(
            geometry,
            ranked_action_maps,
            label_column,
            "Action Type",
            f"Rank {rank} Action-Specific Benefit",
            f"map_action_rank_{rank}",
            map_style,
            categorical=True,
            hover_columns=[score_column],
            hover_label_overrides={
                score_column: "Action-Specific Benefit Score",
            },
            color_discrete_map=action_colors,
            category_order=sorted(
                ranked_action_maps[label_column]
                .dropna()
                .astype(str)
                .unique()
            ),
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


def select_sidebar_view(page: str) -> None:
    """Set the active page from a grouped sidebar navigation button."""
    st.session_state["atlas_view"] = page


def render_sidebar_view_selector() -> str:
    """Render Level 1 and Level 2 navigation as grouped controls."""
    pages = [
        "Overall Risk",
        "Fish Use",
        "Limiting Factor",
        "Action benefits",
    ]
    if st.session_state.get("atlas_view") not in pages:
        st.session_state["atlas_view"] = pages[0]

    st.sidebar.markdown("#### Level 1")
    for page in pages[:3]:
        st.sidebar.button(
            page,
            key=f"sidebar_view_{page}",
            on_click=select_sidebar_view,
            args=(page,),
            type=(
                "primary"
                if st.session_state["atlas_view"] == page
                else "secondary"
            ),
            use_container_width=True,
        )
    st.sidebar.markdown("#### Level 2")
    st.sidebar.button(
        pages[3],
        key=f"sidebar_view_{pages[3]}",
        on_click=select_sidebar_view,
        args=(pages[3],),
        type=(
            "primary"
            if st.session_state["atlas_view"] == pages[3]
            else "secondary"
        ),
        use_container_width=True,
    )
    return st.session_state["atlas_view"]


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
    basin = "All basins"
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
    page = render_sidebar_view_selector()
    st.sidebar.caption("Map labels: U = Upper Grande Ronde; C = Catherine Creek.")
    st.caption(
        "Numerical scores show score / displayed upper bound. See 'How scores "
        "are calculated' for each score's definition."
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

    if page == "Overall Risk":
        render_overall_risk(tables, geometry, basin, selected_bsr, MAP_STYLE)
    elif page == "Fish Use":
        render_fish_use(tables, geometry, basin, selected_bsr, MAP_STYLE)
    elif page == "Limiting Factor":
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
