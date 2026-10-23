"""Interactive Atlas scoring viewer for Streamlit.

The app reads the Level 1 and Level 2 outputs created by
Atlas_Integrated_Scoring.ipynb, including the scored BSR GeoPackage. The BSR
feature layer is detected from the score_bsr field written by the notebook.
Fish-use views distinguish the BSR-level fish_use_score, species-level
species_aggregate_score, and life-stage LS_corrected_score fields. Level 2
views use action_benefit_score for individual actions. Displayed risk ceilings
hold life-stage fish use and population priority fixed and set vulnerability
and condition to 1. The action-specific benefit ceiling is 45.
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
SCORE_SCHEMA_VERSION = "2026-10-14-observed-percentages-v13"

# Framework dimensions used to derive absolute score ceilings.
MODELED_SPECIES_COUNT = 3
LIMITING_FACTOR_COUNT = 15
MAXIMUM_ACTION_WEIGHT = 1.0

SPECIES_FISH_USE_THEORETICAL_MAXIMUM_BY_BASIN = {
    "Upper Grande Ronde": 13.3,
    "Catherine Creek": 6.0,
}
LIMITING_FACTOR_RISK_ABSOLUTE_MAXIMUM = float(MODELED_SPECIES_COUNT)
BENEFIT_COMPONENT_MAXIMUM = (
    LIMITING_FACTOR_RISK_ABSOLUTE_MAXIMUM * MAXIMUM_ACTION_WEIGHT
)
ACTION_BENEFIT_MAXIMUM = (
    LIMITING_FACTOR_COUNT * BENEFIT_COMPONENT_MAXIMUM
)
MAXIMUM_SUFFIX = "__maximum"
OBSERVED_MAXIMUM_SUFFIX = "__observed_maximum"
DISPLAY_MAXIMUM_SUFFIX = "__display_maximum"
UNIT_SCORE_FIELDS = {
    "fish_use_score", "LS_corrected_score", "condition_score",
    "overall_limiting_factor_score", "population_priority",
    "vulnerability_score", "lfat_score",
}
OVERALL_FISH_USE_HELP = (
    "Overall Fish Use Score summarizes fish use of the BSR across all species "
    "and life stages. The supplied Overall Fish Use Score is on a 0-to-1 "
    "scale and is provided for context. Percentages compare it with the "
    "highest Overall Fish Use Score among included BSRs. It is not a "
    "separate multiplier in the risk equations."
)
SPECIES_FISH_USE_MAXIMUM_HELP = (
    "Species Fish Use Score summarizes a species' life-stage fish use in a BSR, "
    "weighted by each life stage's population priority and summed. The app "
    "displays the supplied Species Fish Use Score for context. Percentages compare "
    "each species with its own highest species fish-use score among included "
    "BSRs across both basins. A value of 100% is that species' observed "
    "maximum, not the percentage of all fish using the BSR. This score is "
    "not a separate multiplier in the risk equations."
)
LIFE_STAGE_FISH_USE_MAXIMUM_HELP = (
    "Life-Stage Fish Use Score rates how heavily a species uses a BSR for one life "
    "stage, based on Negligible, Low, Medium, or High use ratings; High "
    "indicates a core use area. Percentages compare the Life-Stage Fish Use Score with the "
    "highest life-stage fish-use score for the same species across included "
    "BSRs and all its life stages. A value of 100% matches that observed "
    "maximum; it does not mean all fish use this BSR. The raw 0-to-1 Life-Stage Fish Use Score "
    "is the fish-use multiplier used in the risk equations."
)
CONDITION_SCORE_MAXIMUM_HELP = (
    "A specific Limiting-Factor Condition Score describes the degree of "
    "degradation for one factor in a BSR. Limiting-Factor Condition Scores range from 0.01 "
    "(least degraded) to 1 (most degraded); these correspond to source "
    "ratings of 1 and 5. Shown as a percentage of the highest Limiting-Factor "
    "Condition Score among included BSRs, across all limiting factors. The observed "
    "maximum is the most degraded BSR and limiting-factor combination "
    "in the included data, and can be less than 1. A percentage of 100% "
    "therefore does not necessarily mean the factor is maximally degraded."
)
VULNERABILITY_SCORE_MAXIMUM_HELP = (
    "Vulnerability Score describes how strongly a limiting factor can affect a "
    "species and life stage. Vulnerability Scores range from 0.01 (least vulnerable, "
    "rank 15) to 1 (most vulnerable, rank 1). Percentages express this "
    "0-to-1 input scale, with 100% equal to 1."
)
POPULATION_PRIORITY_MAXIMUM_HELP = (
    "Population priority is a life stage's share of the priority weight "
    "within its basin and species. A population-priority weight of 0 assigns no weight; 1 "
    "assigns all weight to that life stage. Percentages are shares of 1, "
    "and priorities sum to approximately 100%, allowing source rounding."
)
OVERALL_LIMITING_FACTOR_MAXIMUM_HELP = (
    "Overall Limiting-Factor Condition Score is the unweighted mean of the "
    "15 specific Limiting-Factor Condition Scores in a BSR. A mean of 0.01 "
    "is least degraded and 1 is most degraded. Percentages compare it with "
    "the highest Overall Limiting-Factor Condition Score among included "
    "BSRs. Overall Limiting-Factor Condition Score is provided for "
    "demonstration and is not incorporated into Overall Risk Score."
)
RISK_COMPONENT_MAXIMUM_HELP = (
    "A Risk Component Score combines Life-Stage Fish Use Score, population "
    "priority, Limiting-Factor Condition Score, and Vulnerability Score for "
    "one species, life stage, and limiting factor in a BSR. Percentages "
    "compare it with the highest Risk Component Score "
    "across all included BSRs, species, life stages, and factors. They are "
    "relative risk indices, not the probability of harm or shares of total risk."
)
LIFE_STAGE_RISK_MAXIMUM_HELP = (
    "Life-Stage Risk Score combines Life-Stage Fish Use Score and population "
    "priority with Limiting-Factor Condition Score and Vulnerability Score "
    "across all 15 limiting factors. Percentages "
    "compare it with the highest life-stage risk score across all included "
    "BSRs, species, and life stages. A value of 100% matches the highest "
    "Life-Stage Risk Score; it is not a probability of harm or a share of "
    "the BSR's Overall Risk Score."
)
LIMITING_FACTOR_RISK_MAXIMUM_HELP = (
    "Limiting-Factor Risk Score combines the Limiting-Factor Condition Score "
    "with Life-Stage Fish Use Score, population priority, and Vulnerability "
    "Score across all species and life "
    "stages. Percentages compare it with the highest limiting-factor risk "
    "score across all included BSRs and factors. A value of 100% matches "
    "the highest Limiting-Factor Risk Score, rather than the highest "
    "Limiting-Factor Condition Score. It is not a probability of harm "
    "or a share of the BSR's Overall Risk Score."
)
OVERALL_RISK_MAXIMUM_HELP = (
    "Overall Risk Score sums Life-Stage Risk Scores across all species and "
    "life stages in a BSR. Percentages compare it with the highest "
    "Overall Risk Score among included BSRs across both basins. A value of "
    "100% identifies the highest observed Overall Risk Score; 50% is half "
    "that Overall Risk Score. These percentages are not probabilities of harm."
)
SPECIES_RISK_MAXIMUM_HELP = (
    "Species Risk Score sums a species' Life-Stage Risk Scores in a BSR. "
    "Percentages compare it with the highest Species Risk Score for the same "
    "species among included BSRs. A value of 100% matches that species' "
    "observed maximum, not a probability of harm."
)
BENEFIT_COMPONENT_MAXIMUM_HELP = (
    "An Action-Specific Benefit Component Score is a Limiting-Factor Risk "
    "Score multiplied by its relationship weight for an action. Percentages "
    "compare it with the highest Action-Specific Benefit Component Score "
    "across all included BSRs, actions, and limiting factors. "
    "They express relative action alignment, not predicted habitat improvement."
)
ACTION_BENEFIT_MAXIMUM_HELP = (
    "Action-Specific Benefit Score sums Action-Specific Benefit Component "
    "Scores across all limiting factors for an action. Percentages compare "
    "it with the highest Action-Specific Benefit Score across all actions "
    "and included BSRs. A value of 100% matches the highest Action-Specific "
    "Benefit Score, not an expected "
    "percentage reduction in risk or improvement in fish abundance."
)

OVERALL_RISK_DESCRIPTION = (
    "Indicates whether the critical limiting factors within a BSR affect "
    "the fish present, by combining fish use and life-stage population "
    "priority with limiting-factor condition and biological vulnerability."
)
FISH_USE_DESCRIPTION = (
    "Rates how heavily fish use each BSR from the Negligible, Low, Medium, "
    "and High use ratings assigned to each species and life stage."
)
LIFE_STAGE_RISK_DESCRIPTION = (
    "Shows how much modeled risk a life stage faces in the selected BSR, based "
    "on fish use, the life stage's population priority, degradation across "
    "all limiting factors, and vulnerability to those factors."
)
LIMITING_FACTOR_RISK_DESCRIPTION = (
    "Shows the modeled risk posed by a limiting factor in a BSR, based on its "
    "condition, fish vulnerability across all species and life stages, and "
    "the importance of the BSR to those fish."
)
ACTION_BENEFIT_DESCRIPTION = (
    "Shows how strongly an action aligns with the existing modeled risk in a "
    "BSR. It does not estimate feasibility, cost, or expected effectiveness."
)

OVERALL_RISK_INTRODUCTION = """
**Overall Risk:** The overall risk score is intended to quantify whether the
critical limiting factors within a BSR affect the fish present in that BSR.
It integrates (a) the presence and relative importance of fish species and
life stages, (b) the condition of limiting factors, and (c) the vulnerability
of the species and life stages present to those limiting factors.

The score is intended to address three questions:

1. How heavily do fish use the BSR?
2. Which limiting factors are most critical within the BSR?
3. Do those critical limiting factors affect the fish present in the BSR?

The overall risk score is provided for demonstration purposes and is not a
direct input to Level 2 scores. Level 2 uses the individual limiting-factor
risk scores that also contribute to Overall Risk.
""".strip()

FISH_USE_INTRODUCTION = """
Rates how heavily fish use each BSR, based on the Negligible / Low / Medium /
High use ratings assigned to each species and life stage. There are three
levels: overall, species, and life stage. Life-stage fish-use scores enter
Overall Risk; overall and species scores are provided for context.

Reported percentages compare a BSR with the highest observed score for the
corresponding level among included BSRs across both basins:

1. **Life-Stage Fish Use:** How heavily a species uses this BSR for one life
   stage, from Negligible to High (a core use area). Percentages use the
   highest life-stage fish-use score for that species across included BSRs
   and all its life stages.
2. **Species Fish Use:** Life-stage fish use for a species in a BSR, weighted
   by each life stage's population priority and summed. The app displays the
   supplied species score. Percentages use the highest species fish-use
   score for that same species among included BSRs.
3. **Overall Fish Use:** Fish use across all modeled species and life stages.
   The raw input is on a 0-to-1 scale; displayed percentages compare it with
   the highest overall fish-use score among included BSRs, which is 100%.

A percentage of 50% means half the corresponding maximum score. It does
not mean half the fish use the BSR, and percentages need not sum to 100%.
""".strip()

LIMITING_FACTOR_INTRODUCTION = """
**Overall Limiting-Factor Condition Score:** Mean of the 15 specific
limiting-factor condition scores within each BSR. This overall limiting-factor condition
score is provided for demonstration and is not incorporated into Overall Risk.
Percentages compare each BSR's mean with the highest mean among included BSRs.

**Specific Limiting-Factor Condition Score:** Degree of degradation for one
limiting factor in a BSR. Raw scores range from **0.01 (least degraded)** to
**1 (most degraded)**. Percentages compare the score with the highest
limiting-factor condition score across all included BSRs and limiting factors, representing
the most degraded BSR and factor combination in the included data. That
observed maximum can be less than 1, so 100% need not mean maximum degradation.

**Limiting-Factor Risk Score:** Modeled risk posed by a limiting factor in a
BSR, based on its condition, the vulnerability of each species and life
stage, and the importance of the BSR to those fish through fish use and
population priority. Percentages compare the score with the highest
limiting-factor risk across all included BSRs and factors.
""".strip()

ACTION_BENEFIT_INTRODUCTION = """
**Action-Specific Benefits:** The action-specific benefit score is intended
to quantify how strongly an action aligns with the modeled risk within a
BSR. It combines:

1. The individual limiting-factor risk scores, which account for fish use,
   population priority, condition, and vulnerability
2. Action weights describing the strength and frequency of the documented
   relationship between each action and limiting factor.

For each action, each limiting-factor risk score is multiplied by its
corresponding action weight. These weighted contributions are then summed
across all 15 limiting factors to calculate the Action-Specific Benefit
Score. Reported percentages compare each score with the highest observed
Action-Specific Benefit Score across all actions and included BSRs in both
basins.
""".strip()

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
    "Select an action to update its score map, preliminary tiers, and "
    "selected BSR comparisons."
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
        [data-testid="stSidebar"] .st-key-atlas_limiting_factor_nav {
            margin-left: 1.15rem;
            padding-left: 0.75rem;
            border-left: 2px solid #cbd5e1;
        }
        [data-testid="stSidebar"] .st-key-atlas_fish_use_nav {
            margin-left: 1.15rem;
            padding-left: 0.75rem;
            border-left: 2px solid #cbd5e1;
        }
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
        .atlas-maxima {
            width: 100%; table-layout: fixed; border-collapse: collapse;
            margin: 0.3rem 0 1.4rem; font-size: 0.92rem;
        }
        .atlas-maxima th, .atlas-maxima td {
            border-bottom: 1px solid #dce3ea; padding: 0.55rem 0.7rem;
            text-align: left; vertical-align: top; overflow-wrap: anywhere;
        }
        .atlas-maxima th {background: #f3f6f8; font-weight: 650;}
        .atlas-maxima tbody tr:nth-child(even) {background: #fafbfc;}
        .atlas-maxima .atlas-max-value {font-variant-numeric: tabular-nums; font-weight: 650;}
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


METHODOLOGY_MARKDOWN = r"""
# How scores are calculated

Each BSR is evaluated for every species, life stage, and limiting factor combination. Higher scores indicate greater modeled importance, impairment, vulnerability, risk, or risk alignment, depending on the score.

## Inputs

### Fish-use scores

**Plain language:** Fish-use scores rate how heavily fish use a BSR, based on Negligible, Low, Medium, and High ratings for each species and life stage. High indicates a core use area. Percentages compare scores with the observed maximum at the corresponding level; they are not percentages of fish abundance.

- **Life-Stage Fish Use Score** is specific to the BSR, species, and life stage. It is the fish-use multiplier used in the risk equations. Source values are divided by the global maximum in the complete fish-use input table, producing raw scores from 0 to 1. Displayed percentages then compare these raw scores with the highest life-stage fish-use score for the same species across included BSRs and all its life stages. This display comparison does not change the risk multiplier.

> ##### Scores reported for context
>
> - **Species Fish Use Score** summarizes a species' life-stage fish use, weighted by life-stage population priority and summed. The app retains the supplied `species_aggregate_score` on its source scale. Percentages compare each species with its own highest species score among included BSRs. It is not a separate multiplier in the risk equations.
> - **Overall Fish Use Score** is the BSR-level `fish_use_score_decimal`, retained on its raw 0-to-1 scale for context. Percentages compare it with the highest overall fish-use score among included BSRs. It is not a separate multiplier in the risk equations.

### Vulnerability scores

**Plain language:** Vulnerability describes how strongly each limiting factor can affect a given species and life stage.

- Vulnerability Rank is specific to the species, life stage, and limiting factor.
- Rank 1 is mapped to a Vulnerability Score of 1, and rank 15 is mapped to 0.01. Intermediate ranks are mapped linearly.
- For the combined Migration life stage, the calculation uses the higher of the adult and juvenile vulnerability scores for each species and limiting factor.

### Limiting-factor condition scores

**Plain language:** Specific limiting-factor condition scores describe the degree of degradation for each limiting factor in each BSR. A raw score of 0.01 is least degraded; 1 is most degraded.

- A raw rating of 1 is mapped to 0.01 and a raw rating of 5 is mapped to 1. Intermediate ratings are mapped linearly.
- The same BSR and limiting-factor condition score is applied to every species and life-stage pathway.
- The legacy `lf_condition_score` field is retained for reference. Calculations use the score derived from `lf_condition_score_raw_1_5`.
- Percentages compare each specific limiting-factor condition score with the highest limiting-factor condition score across included BSRs and all limiting factors. This observed maximum is the most degraded BSR and factor combination in the included data; it can be less than 1.
- **Overall Limiting-Factor Condition** is the unweighted mean of the 15 specific limiting-factor condition scores within a BSR. Its percentages use the highest BSR mean among included BSRs. This demonstration score does not enter Overall Risk.

### Population priority

**Plain language:** Population priority represents the relative importance of each life stage within its basin and species.

- Population Priority is specific to basin, species, and life stage.
- Priorities sum to 1 across life stages within each basin and species.

## Level 1 risk calculations

### Risk component

**Plain language:** A risk component is the contribution of one species, life stage, and limiting factor pathway to risk in a BSR.

$$
\text{risk component}
=
\text{life-stage fish use}
\times
\text{population priority}
\times
\text{limiting-factor condition}
\times
\text{vulnerability}
$$

### Life-stage risk score

**Plain language:** Life-Stage Risk Score indicates how much risk a life stage faces in a BSR, based on fish use, population priority, degradation across all limiting factors, and vulnerability to those factors. Percentages compare it with the highest life-stage risk across all included BSRs, species, and life stages; they are not probabilities of harm or shares of a BSR's total risk.

$$
\begin{aligned}
\text{life-stage risk score}
={}&
\text{life-stage fish use}
\times
\text{population priority} \\
&\times
\sum_{\text{15 limiting factors}}
\left(
\text{limiting-factor condition}
\times
\text{vulnerability}
\right)
\end{aligned}
$$

The Highest Priority Life Stage is the species and life-stage combination with the largest Life-Stage Risk Score within the BSR.

### Limiting-factor risk score

**Plain language:** Limiting-Factor Risk Score indicates the risk posed by a limiting factor in a BSR, based on its condition, the vulnerability of each species and life stage, and the importance of the BSR to those fish through fish use and population priority. Percentages use the highest limiting-factor risk across all included BSRs and factors, rather than a condition-score maximum.

$$
\begin{aligned}
\text{limiting-factor risk score}
={}&
\text{limiting-factor condition} \\
&\times
\sum_{\substack{\text{all species and}\\\text{life stages}}}
\left(
\text{life-stage fish use}
\times
\text{vulnerability}
\times
\text{population priority}
\right)
\end{aligned}
$$

The Highest Priority Limiting Factor is the limiting factor with the largest Limiting-Factor Risk Score within the BSR.

### Overall risk score

**Plain language:** Overall Risk indicates whether the critical limiting factors within a BSR affect the fish present, by combining fish use and population priority with limiting-factor condition and biological vulnerability. It addresses fish use, the most critical limiting factors, and whether those factors affect the fish using the BSR. Percentages compare each BSR with the highest observed Overall Risk Score among included BSRs.

Overall Risk is provided for demonstration purposes. Its aggregate score is not a direct input to Level 2; Level 2 uses the individual limiting-factor risk scores that also contribute to Overall Risk.

The Overall Risk Score can be calculated equivalently by summing all Life-Stage Risk Scores or all 15 Limiting-Factor Risk Scores:

$$
\begin{aligned}
\text{Overall Risk Score}
&=
\sum_{\substack{\text{all species and}\\\text{life stages}}}
\left(
\text{life-stage risk score}
\right) \\
&=
\sum_{\text{15 limiting factors}}
\left(
\text{limiting-factor risk score}
\right)
\end{aligned}
$$

## Level 2 action calculations

### Action weight

**Plain language:** Action Weight describes the strength and frequency of the documented relationship between an action and a limiting factor.

$$
\text{action weight}
=
\text{relationship directness}
\times
\text{frequency}
$$

Action Weight ranges from 0 to 1.

### Action-Specific Benefit Score

**Plain language:** Action-Specific Benefit Score indicates how strongly an action aligns with the existing modeled risk in a BSR. It does not estimate feasibility, cost, or expected effectiveness.

For each action, the weighted limiting-factor contributions are summed:

$$
\begin{aligned}
\text{action-specific benefit score}
={}&
\sum_{\text{15 limiting factors}}
\left(
\text{limiting-factor risk score}
\times
\text{action weight}
\right)
\end{aligned}
$$

The Highest Risk-Aligned Action Type is the action type with the largest Action-Specific Benefit Score within the BSR.
""".strip()


def scoring_maxima_sections(
    tables: dict[str, pd.DataFrame] | None = None,
) -> tuple[tuple[str, tuple[tuple[str, str, str, str], ...]], ...]:
    """Pair framework ceilings with raw maxima from the current included BSRs."""
    sources = dict(tables or {})
    if tables:
        sources["species_risk"] = summarize_species_risk(tables["life_stage"])

    def observed(table_name: str, field: str, basin: str | None = None) -> str:
        source = sources.get(table_name)
        if source is None or source.empty:
            return "Not available"
        if basin is not None:
            source = source.loc[source["basin"].eq(basin)]
        source = source.loc[
            ~source["bsr"].map(normalized_bsr_id).isin(EXCLUDED_TIER_BSRS)
        ]
        if source.empty:
            return "Not available"
        return format_score(observed_included_maximum(source, field))

    return (
        ("Scores reported for context", (
            ("Overall Fish Use Score", "1", "BSR-level source scale (normalized)",
             observed("bsr", "fish_use_score")),
            ("Species Fish Use Score (Upper Grande Ronde)", "13.3", "Species fish-use source scale",
             observed("life_stage", "species_aggregate_score", "Upper Grande Ronde")),
            ("Species Fish Use Score (Catherine Creek)", "6", "Species fish-use source scale",
             observed("life_stage", "species_aggregate_score", "Catherine Creek")),
        )),
        ("Framework inputs", (
            ("Life-Stage Fish Use Score", "1", "Source fish-use maximum (normalized)",
             observed("life_stage", "LS_corrected_score")),
            ("Limiting-Factor Condition Score", "1", "Limiting-factor condition rating scale (normalized)",
             observed("limiting_factor", "condition_score")),
            ("Vulnerability Score", "1", "Vulnerability rank scale (normalized)",
             observed("grid", "vulnerability_score")),
            ("Population Priority", "1", "Life-stage proportion",
             observed("grid", "population_priority")),
        )),
        ("Framework outputs", (
            ("Life-Stage Risk Score", "15", "Lower with fish use and priority held fixed",
             observed("life_stage", "risk_score")),
            ("Species Risk Score", "15", "Lower with fish use and priority held fixed",
             observed("species_risk", "species_risk_score")),
            ("Limiting-Factor Risk Score", "3", "Lower with fish use and priority held fixed",
             observed("limiting_factor", "risk_score")),
            ("Overall Risk Score", "45", "Lower with fish use and priority held fixed",
             observed("bsr", "overall_risk_score")),
            ("Action-Specific Benefit Score", "45", "15 factors x 3 factor risk x 1 action weight",
             observed("action", "action_benefit_score")),
        )),
    )


def render_scoring_methodology(tables: dict[str, pd.DataFrame] | None = None) -> None:
    """Render the project-approved scoring explanation from Markdown."""
    with st.expander("How scores are calculated"):
        st.markdown(METHODOLOGY_MARKDOWN)
        st.markdown("### Reading the displayed percentages")
        st.markdown(
            "Score labels read **percentage (raw score)**. The percentage is "
            "100 times the raw score divided by the **Observed Maximum** for "
            "that comparison. For example, 50% means half the highest observed "
            "score, not half the fish or a 50% probability of harm. The chart "
            "caption identifies the comparison. All included BSRs across both "
            "basins set the maxima; UGR8 is excluded. Species Fish Use, "
            "Life-Stage Fish Use, and Species Risk use separate maxima per "
            "species. Specific limiting-factor condition scores share a maximum "
            "across BSRs and factors; overall limiting-factor condition scores "
            "use the highest BSR mean. "
            "Vulnerability, Population Priority, and Action Weight remain "
            "percentages of their fixed input-scale maximum of 1. Raw "
            "scores and scoring equations are unchanged by display normalization."
        )
        st.markdown("### Theoretical maxima")
        st.caption(
            "(normalized) identifies score inputs expressed on a common 0-to-1 "
            "index; population-priority proportions are used as supplied. "
            "Observed Maximum reports the highest raw score among included "
            "BSRs for each row. Species fish-use rows are basin-specific. "
            "Chart percentages use the comparison specified in each title's help."
        )
        for heading, rows in scoring_maxima_sections(tables):
            st.markdown(f"#### {heading}")
            body = "".join(
                "<tr>"
                f"<td>{escape(component)}</td>"
                f"<td class='atlas-max-value'>{escape(maximum)}</td>"
                f"<td>{escape(basis)}</td>"
                f"<td class='atlas-max-value'>{escape(observed)}</td>"
                "</tr>"
                for component, maximum, basis, observed in rows
            )
            st.markdown(
                "<table class='atlas-maxima'>"
                "<colgroup><col style='width:30%'><col style='width:17%'>"
                "<col style='width:36%'><col style='width:17%'></colgroup>"
                "<thead><tr><th>Score component</th><th>Theoretical maximum</th>"
                "<th>Basis</th><th>Observed Maximum</th></tr></thead>"
                f"<tbody>{body}</tbody></table>",
                unsafe_allow_html=True,
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


def display_maximum_column(field: str) -> str:
    """Name the comparison denominator for a displayed percentage."""
    return field + DISPLAY_MAXIMUM_SUFFIX


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


def observed_maxima_by_species(table: pd.DataFrame, field: str) -> pd.Series:
    """Use the same species' included BSRs across both basins as the reference."""
    included = table.loc[
        ~table["bsr"].map(normalized_bsr_id).isin(EXCLUDED_TIER_BSRS)
    ].copy()
    values = pd.to_numeric(included[field], errors="coerce")
    if (included.empty or values.isna().any() or not np.isfinite(values).all()
            or values.lt(0).any()):
        raise ValueError(f"Included BSRs must have finite nonnegative {field} values.")
    included[field] = values
    maxima = included.groupby("species")[field].max()
    if not set(table["species"]).issubset(maxima.index):
        raise ValueError(f"Every species needs an included BSR to compare {field}.")
    return maxima


def positive_axis_maximum(
    table: pd.DataFrame,
    field: str,
    label: str,
) -> float:
    """Return a stable positive axis bound from all displayed comparison rows."""
    values = pd.to_numeric(table[field], errors="coerce")
    if values.isna().any() or not np.isfinite(values).all():
        raise ValueError(f"{label} values must be finite.")
    maximum = float(values.max())
    return maximum if maximum > 0 else 1.0


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
        *[display_maximum_column(field) for field in fields
          if display_maximum_column(field) in table.columns],
    ]))


def add_score_maxima(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Calculate risk ceilings and theoretical action-benefit ceilings.

    These are display metadata only. Source scores and priorities (including
    rounded priorities that sum to 0.99) are never rescaled. A grid component's
    ceiling is LS_corrected_score * population_priority: condition and
    vulnerability are both 1. Each action benefit component has an absolute
    ceiling of 3, and an action's 15 components have an absolute theoretical
    total of 45.
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
    if (
        weights.isna().any()
        or not np.isfinite(weights).all()
        or weights.lt(0).any()
        or weights.gt(MAXIMUM_ACTION_WEIGHT + 1e-9).any()
    ):
        raise ValueError("Action weights must contain finite 0-to-1 values.")
    if components.duplicated(["bsr", "action_id", "limiting_factor"]).any():
        raise ValueError("The action components contain duplicate scoring pathways.")
    components[maximum_column("benefit_component")] = (
        BENEFIT_COMPONENT_MAXIMUM
    )
    result["action_components"] = components
    result["action"][maximum_column("action_benefit_score")] = (
        ACTION_BENEFIT_MAXIMUM
    )
    result["bsr"][maximum_column("highest_action_benefit_score")] = (
        ACTION_BENEFIT_MAXIMUM
    )

    result["bsr"] = add_observed_risk_maximum(result["bsr"])

    species_source = result["life_stage"]
    species_observed_maximum = observed_maxima_by_species(
        species_source, "species_aggregate_score"
    )
    life_stage_observed_maximum = observed_maxima_by_species(
        species_source, "LS_corrected_score"
    )
    overall_fish_use_maximum = observed_included_maximum(
        result["bsr"], "fish_use_score"
    )
    condition_observed_maximum = observed_included_maximum(
        result["limiting_factor"], "condition_score"
    )
    unknown_basins = (
        set(species_source["basin"])
        - set(SPECIES_FISH_USE_THEORETICAL_MAXIMUM_BY_BASIN)
    )
    if unknown_basins:
        raise ValueError(f"Unknown basin for Species Fish Use maximum: {sorted(unknown_basins)}")

    observed_score_fields = {
        "overall_risk_score", "risk_score", "risk_component",
        "action_benefit_score", "highest_action_benefit_score",
        "benefit_component",
    }
    for table_name, table in result.items():
        for field in UNIT_SCORE_FIELDS.intersection(table.columns):
            table[maximum_column(field)] = 1.0
            table[display_maximum_column(field)] = 1.0
        # Input ceilings stay fixed at 1. Display comparisons use included data.
        for field, maximum in (
            ("fish_use_score", overall_fish_use_maximum),
            ("condition_score", condition_observed_maximum),
        ):
            if field in table:
                table[display_maximum_column(field)] = maximum
        if "LS_corrected_score" in table:
            table[display_maximum_column("LS_corrected_score")] = table[
                "species"
            ].map(life_stage_observed_maximum)
        if "species_aggregate_score" in table:
            table[maximum_column("species_aggregate_score")] = table[
                "basin"
            ].map(SPECIES_FISH_USE_THEORETICAL_MAXIMUM_BY_BASIN)
            table[display_maximum_column("species_aggregate_score")] = (
                table["species"].map(species_observed_maximum)
            )
        for field in observed_score_fields.intersection(table.columns):
            # Each table represents a distinct score component: life-stage risk,
            # limiting-factor risk, action benefit, or an individual component.
            # Excluded BSRs do not set the denominator.
            table[display_maximum_column(field)] = observed_included_maximum(
                table, field
            )
        for field in list(table.columns):
            ceiling = maximum_column(field)
            if ceiling not in table:
                continue
            observed = pd.to_numeric(table[field], errors="coerce")
            if (observed > table[ceiling] + 1e-8).any():
                raise ValueError(f"{field} exceeds its calculated total possible score.")
        for field in list(table.columns):
            denominator = display_maximum_column(field)
            if denominator in table and (table[denominator].isna().any()
                    or table[denominator].lt(0).any()):
                raise ValueError(f"{table_name}: {field} needs a nonnegative observed maximum.")
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
    summary[display_maximum_column("species_risk_score")] = (
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
    summary[display_maximum_column("overall_limiting_factor_score")] = (
        observed_included_maximum(summary, "overall_limiting_factor_score")
    )
    summary[display_maximum_column("overall_risk_score")] = (
        observed_included_maximum(summary, "overall_risk_score")
    )
    return add_observed_risk_maximum(summary)


def is_score_field(field: str) -> bool:
    """Identify scores, excluding ranks, counts, and hidden ceiling metadata."""
    return (
        field in UNIT_SCORE_FIELDS
        or field.endswith(("_score", "_risk", "_component", " Fish Use Score"))
    ) and not field.endswith((MAXIMUM_SUFFIX, DISPLAY_MAXIMUM_SUFFIX))


def score_maximum(row: pd.Series, field: str) -> Any:
    """Read the display comparison denominator for a score."""
    if display_maximum_column(field) in row:
        return row[display_maximum_column(field)]
    if field == "species_aggregate_score" or field.endswith(" Species Fish Use Score"):
        return row.get(display_maximum_column("species_aggregate_score"))
    if field in {"action_benefit_score", "highest_action_benefit_score"} or re.fullmatch(
        r"action_rank_\d+_score", field
    ):
        return row.get(display_maximum_column("action_benefit_score"))
    if field in {"benefit_component", "highest_benefit_component_score"}:
        return row.get(display_maximum_column("benefit_component"))
    if field in {"overall_risk_score", "species_risk_score"} and observed_maximum_column(field) in row:
        return row[observed_maximum_column(field)]
    if field in UNIT_SCORE_FIELDS or field.endswith(" Life-Stage Fish Use Score"):
        return 1.0
    return None


def score_title(field: str, label: str = "") -> str:
    """Use the same explicit score name in labels, scales, and maximum details."""
    if field == "risk_score":
        return ("Limiting-Factor Risk Score" if "limiting" in label.casefold()
                else "Life-Stage Risk Score")
    if re.fullmatch(r"priority_life_stage_\d+_risk_score", field):
        return "Life-Stage Risk Score"
    if re.fullmatch(r"limiting_factor_rank_\d+_score", field):
        return "Limiting-Factor Risk Score"
    if re.fullmatch(r"action_rank_\d+_score", field):
        return "Action-Specific Benefit Score"
    return {
        "overall_risk_score": "Overall Risk Score",
        "species_risk_score": "Species Risk Score",
        "fish_use_score": "Overall Fish Use Score",
        "species_aggregate_score": "Species Fish Use Score",
        "LS_corrected_score": "Life-Stage Fish Use Score",
        "condition_score": "Limiting-Factor Condition Score",
        "overall_limiting_factor_score": "Overall Limiting-Factor Condition Score",
        "risk_component": "Risk Component Score",
        "benefit_component": "Action-Specific Benefit Component Score",
        "action_benefit_score": "Action-Specific Benefit Score",
        "highest_action_benefit_score": "Action-Specific Benefit Score",
    }.get(field, label or DISPLAY_LABELS.get(field, field.replace("_", " ").title()))


def score_maximum_help(field: str, label: str | None = None) -> str | None:
    """Explain how the displayed denominator for a score is determined."""
    normalized_label = (label or DISPLAY_LABELS.get(field, field)).casefold()

    if field == "overall_risk_score" or "overall risk score" in normalized_label:
        return OVERALL_RISK_MAXIMUM_HELP
    if field == "species_risk_score" or "species risk score" in normalized_label:
        return SPECIES_RISK_MAXIMUM_HELP
    if field == "fish_use_score" or "overall fish use" in normalized_label:
        return OVERALL_FISH_USE_HELP
    if (
        field == "species_aggregate_score"
        or field.endswith(" Species Fish Use Score")
        or "species fish use" in normalized_label
    ):
        return SPECIES_FISH_USE_MAXIMUM_HELP
    if (
        field == "LS_corrected_score"
        or field.endswith(" Life-Stage Fish Use Score")
        or "life-stage fish use" in normalized_label
    ):
        return LIFE_STAGE_FISH_USE_MAXIMUM_HELP
    if field == "overall_limiting_factor_score":
        return OVERALL_LIMITING_FACTOR_MAXIMUM_HELP
    if field == "condition_score" or "condition score" in normalized_label:
        return CONDITION_SCORE_MAXIMUM_HELP
    if field == "vulnerability_score" or "vulnerability score" in normalized_label:
        return VULNERABILITY_SCORE_MAXIMUM_HELP
    if field == "population_priority" or "population priority" in normalized_label:
        return POPULATION_PRIORITY_MAXIMUM_HELP
    if field == "lfat_score" or "action weight" in normalized_label:
        return (
            "Action Weight is relationship directness times frequency: 0 "
            "indicates no scored relationship and 1 the strongest relationship. "
            "Percentages express the fixed input scale, with 100% equal to 1."
        )
    if field == "risk_component" or "risk component" in normalized_label:
        return RISK_COMPONENT_MAXIMUM_HELP
    if field == "risk_score":
        return (LIMITING_FACTOR_RISK_MAXIMUM_HELP
                if "limiting" in normalized_label else LIFE_STAGE_RISK_MAXIMUM_HELP)
    if (
        field in {"action_benefit_score", "highest_action_benefit_score"}
        or re.fullmatch(r"action_rank_\d+_score", field)
        or "action-specific benefit score" in normalized_label
    ):
        return ACTION_BENEFIT_MAXIMUM_HELP
    if (
        field in {"benefit_component", "highest_benefit_component_score"}
        or "benefit component" in normalized_label
    ):
        return BENEFIT_COMPONENT_MAXIMUM_HELP
    if (
        re.fullmatch(r"priority_life_stage_\d+_risk_score", field)
        or "life-stage risk" in normalized_label
    ):
        return LIFE_STAGE_RISK_MAXIMUM_HELP
    if (
        re.fullmatch(r"limiting_factor_rank_\d+_score", field)
        or "limiting-factor risk" in normalized_label
        or "limiting factor risk" in normalized_label
    ):
        return LIMITING_FACTOR_RISK_MAXIMUM_HELP
    return None


def score_meaning(field: str, label: str = "") -> str | None:
    """Use one current definition per score in all title and metric tooltips."""
    definition = score_maximum_help(field, label or None)
    if definition:
        return definition
    if field == "preliminary_tier":
        return "BSRs grouped into top, middle, and bottom thirds based on Overall Risk score. Tier I has the highest scores."
    if field == "action_preliminary_tier":
        return "BSRs grouped into consecutive thirds by the selected action's benefit score. Tier 1 has the highest scores."
    if field == "highest_risk_species_life_stage":
        return "The species and life stage with the highest Life-Stage Risk Score in each BSR."
    return None


def format_score_value(row: pd.Series, field: str) -> str:
    """Lead with percentage of observed maximum, then give the raw score."""
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
    if ceiling is not None and not pd.isna(ceiling) and float(ceiling) == 0 and float(value) == 0:
        return f"0% ({format_score(value)})"
    if ceiling is None or pd.isna(ceiling) or float(ceiling) <= 0:
        return f"Not available ({format_score(value)})"
    return f"{format_score(100 * float(value) / float(ceiling), 1)}% ({format_score(value)})"


def score_axis_label(table: pd.DataFrame, field: str, label: str) -> str:
    """Show the raw denominator alongside the percentage on score axes."""
    if not is_score_field(field) or table.empty:
        return label
    maxima = pd.to_numeric(
        table.apply(lambda row: score_maximum(row, field), axis=1),
        errors="coerce",
    )
    basis = "input-scale maximum" if field in {
        "population_priority", "vulnerability_score", "lfat_score"
    } else "Observed Maximum"
    if maxima.isna().any() or maxima.lt(0).any():
        return f"{label} (% of {basis}; maximum unavailable)"
    unique = sorted(set(float(value) for value in maxima))
    if len(unique) == 1:
        maximum_text = format_score(unique[0])
    elif "species" in table:
        by_species = table.assign(_maximum=maxima).groupby("species")["_maximum"].first()
        maximum_text = "; ".join(
            f"{species} {format_score(maximum)}"
            for species, maximum in by_species.items()
        )
    else:
        maximum_text = ", ".join(format_score(value) for value in unique)
    return f"{label} (% of {basis}, {maximum_text})"


def maximum_source(table: pd.DataFrame, field: str, label: str = "") -> tuple[pd.DataFrame, str]:
    """Find complete scored records behind a displayed denominator."""
    sources = st.session_state.get("atlas_score_source_tables", {})
    if not sources:
        return pd.DataFrame(), field
    if field == "species_risk_score":
        return summarize_species_risk(sources["life_stage"]), field
    if field == "overall_limiting_factor_score":
        return summarize_overall_limiting_factors(sources["limiting_factor"]), field
    if field == "overall_risk_score":
        return sources["bsr"], field
    if field in {"fish_use_score", "highest_action_benefit_score"}:
        return (sources["bsr"], field) if field == "fish_use_score" else (
            sources["action"], "action_benefit_score"
        )
    if field in {"species_aggregate_score", "LS_corrected_score"}:
        return sources["life_stage"], field
    if field.endswith(" Species Fish Use Score"):
        species = field.removesuffix(" Species Fish Use Score")
        return sources["life_stage"].loc[
            sources["life_stage"]["species"].eq(species)
        ], "species_aggregate_score"
    if field.endswith(" Life-Stage Fish Use Score"):
        stage = field.removesuffix(" Life-Stage Fish Use Score")
        return sources["life_stage"].loc[
            sources["life_stage"]["life_stage"].eq(stage)
        ], "LS_corrected_score"
    if field == "risk_score":
        return (
            (sources["limiting_factor"], field)
            if "limiting_factor" in table.columns or "limiting" in label.casefold() else
            (sources["life_stage"], field)
        )
    if re.fullmatch(r"priority_life_stage_\d+_risk_score", field):
        return sources["life_stage"], "risk_score"
    if re.fullmatch(r"limiting_factor_rank_\d+_score", field):
        return sources["limiting_factor"], "risk_score"
    if re.fullmatch(r"action_rank_\d+_score", field):
        return sources["action"], "action_benefit_score"
    for name, fields in (
        ("action", {"action_benefit_score"}),
        ("action_components", {"benefit_component", "lfat_score"}),
        ("limiting_factor", {"condition_score"}),
        ("grid", {"risk_component", "vulnerability_score", "population_priority"}),
    ):
        if field in fields:
            return sources[name], field
    return pd.DataFrame(), field


def score_maximum_note(table: pd.DataFrame, field: str, label: str) -> str:
    """Describe each denominator and the included records that attain it."""
    if table.empty or not is_score_field(field):
        return ""
    label = score_title(field, label)
    table = table.loc[table[field].notna()]
    if table.empty:
        return ""
    denominators = table.apply(lambda row: score_maximum(row, field), axis=1)
    numeric = pd.to_numeric(denominators, errors="coerce")
    if numeric.isna().any() or numeric.lt(0).any():
        return f"{label}: observed maximum unavailable."
    if field in {"population_priority", "vulnerability_score", "lfat_score"}:
        return f"{label}: input-scale maximum 1."
    source, source_field = maximum_source(table, field, label)
    if not source.empty and source_field in source:
        source = source.loc[
            ~source["bsr"].map(normalized_bsr_id).isin(EXCLUDED_TIER_BSRS)
        ].copy()
        source["_source_value"] = pd.to_numeric(source[source_field], errors="coerce")
    notes = []
    for maximum in sorted(numeric.unique()):
        heading = label
        if source_field in {"species_aggregate_score", "LS_corrected_score", "species_risk_score"} and "species" in table:
            species = table.loc[np.isclose(numeric, maximum), "species"].dropna().unique()
            if len(species):
                heading += " (" + ", ".join(sorted(map(str, species))) + ")"
        if source.empty:
            notes.append(f"{heading}: observed maximum {format_score(maximum)}.")
            continue
        candidates = source.loc[
            np.isclose(source["_source_value"], maximum, rtol=1e-9, atol=1e-8)
        ]
        if source_field in {"species_aggregate_score", "LS_corrected_score", "species_risk_score"}:
            species_maximum_field = display_maximum_column(source_field)
            if species_maximum_field in candidates:
                candidates = candidates.loc[np.isclose(
                    candidates[species_maximum_field], maximum, rtol=1e-9, atol=1e-8
                )]
            if "species" in table:
                candidates = candidates.loc[
                    candidates["species"].isin(
                        table.loc[np.isclose(numeric, maximum), "species"].dropna().unique()
                    )
                ]
        if field == "species_risk_score" and "species" in table:
            candidates = candidates.loc[
                candidates["species"].isin(table["species"].dropna().unique())
            ]
        if field == "species_aggregate_score" or field.endswith(" Species Fish Use Score"):
            identity = ("bsr", "basin", "species")
        else:
            identity = ("bsr", "species", "life_stage", "limiting_factor", "action_type")
        columns = [name for name in identity if name in candidates]
        records = candidates[columns].drop_duplicates() if columns else candidates
        names = [" | ".join(str(value) for value in row)
                 for row in records.itertuples(index=False, name=None)]
        if names:
            examples = "; ".join(names[:10])
            if len(names) > 10:
                examples += f"; plus {len(names) - 10} other tied cases"
            notes.append(
                f"{heading}: Observed Maximum = {format_score(maximum)}. "
                f"Set by: {examples}."
            )
        else:
            notes.append(
                f"{heading}: observed maximum {format_score(maximum)}. "
                "No included BSR in this comparison attains that value."
            )
    return "\n\n".join(notes)


def score_percentage_caption(field: str, label: str) -> str:
    """Separate the label format, comparison, and example into readable lines."""
    normalized = label.casefold()
    name = score_title(field, label).casefold()
    if field == "species_aggregate_score":
        reference = (
            "For each species, 100% is its highest species fish-use score "
            "among included BSRs."
        )
    elif field == "LS_corrected_score":
        reference = (
            "For this species, 100% is its highest life-stage fish-use score "
            "across included BSRs and all its life stages."
        )
    elif field == "fish_use_score":
        reference = "100% is the highest overall fish-use score among included BSRs."
    elif field == "condition_score":
        reference = (
            "100% is the highest limiting-factor condition score across included BSRs and "
            "all limiting factors, the most degraded BSR and factor combination observed."
        )
    elif field == "overall_limiting_factor_score":
        reference = "100% is the highest overall limiting-factor condition score among included BSRs."
    elif field == "risk_component":
        reference = (
            "100% is the highest risk component score across all included BSRs, "
            "species, life stages, and limiting factors."
        )
    elif field == "risk_score" and "limiting" in normalized:
        reference = (
            "100% is the highest limiting-factor risk across included BSRs "
            "and all limiting factors."
        )
    elif field == "risk_score":
        reference = (
            "100% is the highest life-stage risk across all included BSRs, "
            "species, and life stages."
        )
    elif field == "benefit_component":
        reference = (
            "100% is the highest action-specific benefit component score across all included "
            "BSRs, actions, and limiting factors."
        )
    elif field in {"action_benefit_score", "highest_action_benefit_score"}:
        reference = "100% is the highest action-specific benefit score across all actions and included BSRs."
    elif field == "species_risk_score":
        reference = "100% is the highest species risk score for the same species among included BSRs."
    else:
        reference = "100% is the highest overall risk score among included BSRs."
    return (
        f"**Labels:** percentage of observed maximum score **({name})**  \n"
        f"**Reference:** {reference}  \n"
        "**Example:** If the observed maximum is 0.80, **50% (0.40)** "
        f"means a {name} of **0.40**, half that maximum."
    )


def display_percentages(table: pd.DataFrame, field: str) -> pd.Series:
    """Convert scores for display, including an all-zero comparison."""
    values = pd.to_numeric(table[field], errors="coerce")
    maxima = pd.to_numeric(
        table.apply(lambda row: score_maximum(row, field), axis=1),
        errors="coerce",
    )
    if (maxima.isna().any() or not np.isfinite(maxima).all()
            or maxima.lt(0).any() or ((maxima == 0) & (values != 0)).any()):
        raise ValueError(f"{field} needs a valid observed maximum for display.")
    return values.div(maxima.where(maxima > 0)).fillna(0) * 100


def score_scale_description(field: str, label: str, *, raw_condition: bool = False,
                            unit_color_scale: bool = False) -> str:
    """Put the scale endpoints beside the colorbar or score axis."""
    name = score_title(field, label)
    if raw_condition:
        return f"{name} scale: 0.01 = least degraded; 1 = most degraded."
    if field in {"condition_score", "overall_limiting_factor_score"}:
        return (
            f"Raw {name.casefold()} scale: 0.01 = least degraded; 1 = most degraded. "
            f"100% identifies the highest observed {name.casefold()} across all BSRs."
        )
    upper = "1" if unit_color_scale else "100%"
    lower = "0" if unit_color_scale else "0%"
    return (
        f"{lower} = {name.casefold()} of 0; "
        f"{upper} = highest observed {name.casefold()} across all BSRs"
    )


def fish_use_hover_table(table: pd.DataFrame, index: str, category: str,
                         field: str, suffix: str) -> tuple[pd.DataFrame, list[str]]:
    """Pivot fish-use scores and carry each score's own display denominator."""
    maximum_field = display_maximum_column(field)
    rows = table[[index, category, field, maximum_field]].drop_duplicates()
    scores = rows.pivot(index=index, columns=category, values=field).sort_index(axis=1)
    maxima = rows.pivot(index=index, columns=category, values=maximum_field).reindex(
        columns=scores.columns
    )
    scores.columns = [f"{name} {suffix}" for name in scores.columns]
    maxima.columns = [display_maximum_column(name) for name in scores.columns]
    labels = scores.columns.tolist()
    result = pd.concat([scores, maxima], axis=1)
    result.columns.name = None
    return result.reset_index(), labels


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
        display_field = display_maximum_column(score_column)
        if display_field in ranked:
            display_by_bsr = ranked.groupby("bsr", as_index=False)[
                display_field
            ].first().rename(columns={
                display_field: display_maximum_column(value_column)
            })
            summary = summary.merge(
                display_by_bsr, on="bsr", how="left", validate="one_to_one"
            )
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
            display_maximum_column("risk_score"): display_maximum_column(score_column),
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
    display_fields = [field for field in table if not field.endswith(
        (MAXIMUM_SUFFIX, OBSERVED_MAXIMUM_SUFFIX, DISPLAY_MAXIMUM_SUFFIX)
    )]
    labels = {
        column: DISPLAY_LABELS.get(
            column,
            column.replace("_", " ").title(),
        )
        for column in display_fields
    }
    labels.update(column_labels or {})
    notes = [score_maximum_note(table, field, labels[field])
             for field in display_fields if is_score_field(field)]
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
    for field in display_fields:
        if not is_score_field(field):
            continue
        help_text = score_meaning(field, labels[field])
        if help_text:
            column_config[labels[field]] = st.column_config.TextColumn(
                help=help_text
            )
    st.dataframe(
        rounded,
        width="stretch",
        hide_index=True,
        column_config=column_config,
    )
    if notes:
        st.caption("Score column maximum details", help="\n\n".join(filter(None, notes)))


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
        "name": "Not incl. in analysis: UGR8 (U8)",
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
            name="Not incl. analysis: UGR8 (U8)",
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
        xref="paper", yref="paper", x=1.07, y=(y0 + y1) / 2,
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
    unit_color_scale: bool = False,
    height: int = 600,
    show_legend: bool = True,
    title_help: str | None = None,
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

    title_details = title_help or score_meaning(metric, metric_label)
    st.subheader(title, help=title_details or None)

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
    score_color = not categorical and is_score_field(color_column)
    raw_condition_colors = score_color and unit_color_scale and color_column in {
        "condition_score", "overall_limiting_factor_score"
    }
    colorbar_title = color_label

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
        "title": None,
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
        if score_color:
            denominators = pd.to_numeric(
                plot_data.apply(lambda row: score_maximum(row, color_column), axis=1),
                errors="coerce",
            )
            percentages = display_percentages(plot_data, color_column)
            plot_data["_display_color"] = (
                numeric if raw_condition_colors else
                percentages / 100 if unit_color_scale else percentages
            )
            common["color"] = "_display_color"
            colorbar_title = (
                score_title(color_column, color_label) if raw_condition_colors else
                f"{score_title(color_column, color_label)} (1 = Observed Maximum)" if unit_color_scale else
                "% of Observed Maximum"
            )
            common["labels"]["_display_color"] = colorbar_title
            if raw_condition_colors:
                common["range_color"] = (0.01, 1)
            elif unit_color_scale:
                common["range_color"] = (0, 1)
            elif range_color is not None:
                if not np.allclose(denominators, denominators.iloc[0]):
                    raise ValueError("A shared raw color range needs a shared observed maximum.")
                common["range_color"] = (
                    tuple(endpoint / float(denominators.iloc[0]) * 100
                          for endpoint in range_color)
                    if denominators.iloc[0] > 0 else (0, 100)
                )
            else:
                common["range_color"] = (0, 100)
        elif range_color is not None:
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
            margin={"r": 0, "t": 12, "l": 0, "b": 0},
            height=height,
            legend_title_text=color_label,
            showlegend=False,
            coloraxis_showscale=show_legend,
        )
    else:
        figure = px.choropleth_mapbox(mapbox_style=map_style, **common)
        figure.update_layout(
            mapbox={"bearing": 0, "pitch": 0},
            margin={"r": 0, "t": 12, "l": 0, "b": 0},
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
        margin={"r": right_margin, "t": 12, "l": 15, "b": bottom_margin},
    )
    if not categorical:
        scale_ticks = (
            {"tickvals": [0.01, 0.25, 0.5, 0.75, 1],
             "ticktext": ["0.01", "0.25", "0.5", "0.75", "1"]}
            if raw_condition_colors else
            {"tickvals": [0, 0.25, 0.5, 0.75, 1],
             "ticktext": ["0", "0.25", "0.5", "0.75", "1"]}
            if unit_color_scale else
            {"tickvals": list(np.linspace(*common["range_color"], 5)),
             "ticktext": [f"{round(float(tick), 1):g}%"
                          for tick in np.linspace(*common["range_color"], 5)]}
            if score_color else {}
        )
        figure.update_layout(
            coloraxis_colorbar={
                "orientation": "h", "x": 0.5, "xanchor": "center",
                "y": -0.04,
                "yanchor": "top", "len": 0.8,
                "thickness": 12,
                "title": {"text": colorbar_title, "side": "bottom"},
                **scale_ticks,
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
    if score_color:
        scale_note = score_scale_description(
            color_column, color_label, raw_condition=raw_condition_colors,
            unit_color_scale=unit_color_scale,
        )
        if not unit_color_scale and common["range_color"][1] < 100 - 1e-8:
            scale_note += (
                f" The color scale ends at {format_score(common['range_color'][1], 1)}%, "
                "the highest score in the displayed comparison."
            )
        st.caption(
            scale_note,
            help=score_maximum_note(plot_data, color_column, color_label) or None,
        )
    elif categorical:
        notes = [
            score_maximum_note(plot_data, field, hover_labels[field])
            for field in hover_fields if is_score_field(field)
        ]
        notes = list(dict.fromkeys(note for note in notes if note))
        if notes:
            st.caption("Observed Maximum details", help="\n\n".join(notes))
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
    x_axis_maximum: float | None = None,
    description: str | None = None,
) -> None:
    """Render a consistently formatted horizontal comparison chart."""
    ordered = table.copy().sort_values(value, ascending=True)
    hover_columns = [
        column
        for column in (hover_columns or [])
        if column in ordered.columns and column not in {category, value}
    ]
    hover_label_overrides = hover_label_overrides or {}
    displayed_value_label = value_label or DISPLAY_LABELS.get(
        value, value.replace("_", " ").title()
    )
    if is_score_field(value):
        displayed_value_label = score_title(value, displayed_value_label)

    def field_label(column: str) -> str:
        if column == category:
            return DISPLAY_LABELS.get(
                category, category.replace("_", " ").title()
            )
        if column == value:
            return displayed_value_label
        return hover_label_overrides.get(
            column,
            DISPLAY_LABELS.get(
                column, column.replace("_", " ").title()
            ),
        )

    bar_fields = [category, value, *hover_columns]

    def bar_hover_text(row: pd.Series) -> str:
        lines = []
        for column in bar_fields:
            displayed_value = row[column]
            if pd.isna(displayed_value):
                if column in hover_columns:
                    continue
                displayed = "Not available"
            else:
                displayed = format_score_value(row, column)
            label = field_label(column)
            lines.append(f"<b>{escape(str(label))}:</b> {escape(displayed)}")
        return "<br>".join(lines)

    ordered["_hover_text"] = ordered.apply(bar_hover_text, axis=1)
    ordered["_score_text"] = ordered.apply(lambda row: format_score_value(row, value), axis=1)
    st.subheader(title, help=score_meaning(value, displayed_value_label) or description)
    score_value = is_score_field(value)
    if score_value:
        st.caption(score_percentage_caption(value, displayed_value_label))
        ordered["_display_percent"] = display_percentages(ordered, value)
    figure = px.bar(
        ordered,
        x="_display_percent" if score_value else value,
        y=category,
        color=color,
        orientation="h",
        title=None,
        labels={
            ("_display_percent" if score_value else value): "",
            category: "",
        },
        text="_score_text",
        custom_data=["_hover_text"],
    )
    figure.update_layout(
        height=max(360, 30 * len(ordered) + 110),
        margin={"r": 110, "t": 12, "l": 10, "b": 72},
        legend_title_text="Species" if color == "species" else color,
    )
    if score_value:
        # The observed maximum defines the common 0-to-100% chart scale.
        figure.update_xaxes(
            range=[0.0, 100.0],
            tickvals=[0, 25, 50, 75, 100],
            ticktext=["0%", "25%", "50%", "75%", "100%"],
            title={"text": "% of Observed Maximum",
                   "standoff": 18},
            automargin=True,
        )
    elif x_axis_maximum is not None:
        if not np.isfinite(x_axis_maximum) or x_axis_maximum <= 0:
            raise ValueError("The horizontal-bar x-axis maximum must be positive and finite.")
        figure.update_xaxes(range=[0.0, float(x_axis_maximum)])
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
    if score_value:
        st.caption(
            score_scale_description(value, displayed_value_label),
            help=score_maximum_note(ordered, value, displayed_value_label) or None,
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
    grid = filter_table(tables["grid"], basin)

    st.header("Level 1: Overall Risk", help=OVERALL_RISK_DESCRIPTION)
    st.markdown(OVERALL_RISK_INTRODUCTION)
    st.divider()
    st.caption(
        "Click a polygon or select a BSR in the sidebar to see its risk by "
        "species and life stage and by limiting factor."
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

    row = bsr.loc[bsr["bsr"].eq(selected_bsr)].iloc[0]
    metric_columns = st.columns(4)
    metric_columns[0].metric("Selected BSR", selected_bsr)
    metric_columns[1].metric(
        "Overall Risk Score", format_score_value(row, "overall_risk_score"),
        help=score_meaning("overall_risk_score"),
    )
    metric_columns[2].metric(
        "Preliminary Tier", row["preliminary_tier"],
        help=score_meaning("preliminary_tier"),
    )
    if species_map is None:
        secondary_label = "Overall Fish Use Score"
        secondary_value = format_score_value(row, "fish_use_score")
        secondary_help = score_meaning("fish_use_score")
    else:
        secondary_label = f"{species} Risk Score"
        selected_species = species_map.loc[species_map["bsr"].eq(selected_bsr)]
        secondary_value = (
            format_score_value(selected_species.iloc[0], "species_risk_score")
            if not selected_species.empty else "Not available"
        )
        secondary_help = score_meaning("species_risk_score")
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
    condition_vulnerability = (
        grid.loc[grid["bsr"].eq(selected_bsr)]
        .assign(_product=lambda frame: frame["condition_score"] * frame["vulnerability_score"])
        .groupby(["bsr", "species", "life_stage"], as_index=False)["_product"]
        .sum()
        .rename(columns={"_product": "condition_vulnerability_sum"})
    )
    life_selected = life_selected.merge(
        condition_vulnerability,
        on=["bsr", "species", "life_stage"],
        how="left",
        validate="one_to_one",
    )

    with left:
        horizontal_bar(
            life_selected,
            "risk_score",
            "species_life_stage_label",
            f"{selected_bsr}: risk by species and life stage",
            color="species",
            value_label="Life-Stage Risk Score",
            description=LIFE_STAGE_RISK_DESCRIPTION,
        )
    with right:
        horizontal_bar(
            factor_selected,
            "risk_score",
            "limiting_factor",
            f"{selected_bsr}: risk by limiting factor",
            value_label="Limiting-Factor Risk",
            description=LIMITING_FACTOR_RISK_DESCRIPTION,
        )

    with st.expander(f"Show score tables for BSR: {selected_bsr}"):
        st.subheader("Species and life stages")
        show_score_table(
            life_selected[
                score_columns(life_selected, [
                    "species",
                    "life_stage",
                    "LS_corrected_score",
                    "population_priority",
                    "condition_vulnerability_sum",
                    "risk_score",
                ])
            ].sort_values("risk_score", ascending=False),
            column_labels={
                "condition_vulnerability_sum": "Sum of Condition × Vulnerability",
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

    species_hover_scores, species_hover_columns = fish_use_hover_table(
        life, "bsr", "species", "species_aggregate_score", "Species Fish Use Score"
    )
    fish_hover_values = bsr.merge(
        species_hover_scores,
        on="bsr",
        how="left",
        validate="one_to_one",
    )
    st.header("Level 1: Fish Use", help=FISH_USE_DESCRIPTION)
    st.markdown(FISH_USE_INTRODUCTION)
    st.divider()
    st.caption("Choose an Overall, Species, or Life stage map below.")

    map_level = st.radio(
        "Fish-use map level",
        options=["Overall", "Species", "Life stage"],
        index=2,
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
    st.caption("Hover over a BSR for its raw score and other fish-use scores.")

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
        help=score_meaning("fish_use_score"),
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
            help=score_meaning(map_metric, map_metric_label),
        )

    species_summary = (
        selected[score_columns(selected, ["species", "species_aggregate_score"])]
        .drop_duplicates()
        .sort_values("species")
    )
    life_stage_hover, life_stage_hover_columns = fish_use_hover_table(
        selected, "species", "life_stage", "LS_corrected_score", "Life-Stage Fish Use Score"
    )
    species_summary = species_summary.merge(
        life_stage_hover,
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
            description=(
                "Summarizes each species' life-stage fish use, weighted by "
                "population priority and summed; the supplied species scores "
                "are provided for context."
            ),
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
            score_columns(selected, ["species", "life_stage", "LS_corrected_score"]),
        ]
        horizontal_bar(
            life_stage_summary,
            "LS_corrected_score",
            "life_stage",
            f"{selected_bsr}: {chart_species} fish use by life stage",
            value_label="Life-Stage Fish Use Score",
            description=(
                f"Rates how heavily {chart_species} uses the selected BSR "
                "for each life stage, from Negligible to High (a core use area)."
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


def render_highest_risk_aligned_life_stage(
    tables: dict[str, pd.DataFrame],
    geometry: gpd.GeoDataFrame,
    basin: str,
    map_style: str,
) -> None:
    """Show the highest risk life stage in each BSR on its own page."""
    bsr = filter_table(tables["bsr"], basin)
    life = filter_table(tables["life_stage"], basin)
    priority_map = summarize_priority_life_stages(bsr, life)
    score_fields = [
        f"priority_life_stage_{position}_risk_score"
        for position in range(1, 4)
    ]
    hover_labels = {
        field: field.replace("_risk_score", "_hover_label")
        for field in score_fields
    }
    colors = complete_category_color_map(
        priority_map["highest_risk_species_life_stage"],
        LIFE_STAGE_COLOR_MAP,
    )

    st.header("Level 1: Fish Use", help=FISH_USE_DESCRIPTION)
    st.markdown(FISH_USE_INTRODUCTION)
    st.divider()
    st.caption(
        "This placeholder indicator is based on the population-weighted "
        "Level 1 risk score, not fish use alone, and is retained for "
        "future consideration."
    )
    render_choropleth(
        geometry,
        priority_map,
        "highest_risk_species_life_stage",
        "Species | Life Stage",
        "Highest Risk-Aligned Life Stage",
        "map_top_life_stage",
        map_style,
        categorical=True,
        hover_columns=[
            "fish_use_score",
            "highest_risk_species_life_stage",
            *score_fields,
        ],
        hover_label_columns=hover_labels,
        color_discrete_map=colors,
        category_order=sorted(
            priority_map["highest_risk_species_life_stage"]
            .dropna().astype(str).unique()
        ),
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

    st.header("Level 1: Limiting Factors", help=LIMITING_FACTOR_RISK_DESCRIPTION)
    st.markdown(LIMITING_FACTOR_INTRODUCTION)
    st.divider()
    overall_conditions = filter_table(
        summarize_overall_limiting_factors(tables["limiting_factor"]), basin
    )
    render_choropleth(
        geometry, overall_conditions, "overall_limiting_factor_score",
        "Overall Limiting-Factor Condition Score",
        "Overall Limiting-Factor Condition Score", "map_overall_limiting_factor",
        map_style, hover_columns=["limiting_factor_count", "overall_risk_score"],
        color_scale=LIMITING_FACTOR_COLOR_SCALE,
        unit_color_scale=True,
    )
    st.subheader("Specific Limiting-Factor Map")
    st.caption("Choose a factor and compare its condition or risk across BSRs.")
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
        hover_label_overrides={
            "risk_score": "Limiting-Factor Risk",
        },
        color_scale=LIMITING_FACTOR_COLOR_SCALE,
        unit_color_scale=True,
    )

    st.subheader("Specific Limiting-Factor Drill-Down")
    st.caption(
        "Compare conditions in the selected BSR and the species and life-stage "
        "contributions to the selected factor's risk."
    )

    left, right = st.columns(2)
    bsr_factors = limiting.loc[limiting["bsr"].eq(selected_bsr)].copy()
    with left:
        horizontal_bar(
            bsr_factors,
            "condition_score",
            "limiting_factor",
            f"{selected_bsr}: limiting-factor condition comparison",
            value_label="Limiting-Factor Condition Score",
            description=(
                "Shows the degree of degradation for each limiting factor "
                "in the selected BSR: raw scores range from 0.01 "
                "(least degraded) to 1 (most degraded)."
            ),
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
            description=(
                "Shows how each species and life stage contributes to the "
                "selected limiting factor's risk in the selected BSR."
            ),
        )

    with st.expander(
        f"Show limiting-factor data for: {selected_factor} in {selected_bsr}"
    ):
        show_score_table(
            biological[
                score_columns(biological, [
                    "species",
                    "life_stage",
                    "LS_corrected_score",
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


def render_top_five_limiting_factors(
    tables: dict[str, pd.DataFrame],
    geometry: gpd.GeoDataFrame,
    basin: str,
    map_style: str,
) -> None:
    """Render the top-five limiting-factor maps in a dedicated view."""
    limiting = filter_table(tables["limiting_factor"], basin)
    ranked_factor_maps = summarize_ranked_categories(
        limiting,
        "limiting_factor",
        "risk_score",
        "limiting_factor_rank",
        ranks=5,
    )
    factor_category_values = pd.concat(
        [
            ranked_factor_maps[f"limiting_factor_rank_{rank}_label"]
            for rank in range(1, 6)
        ],
        ignore_index=True,
    )
    factor_colors = complete_category_color_map(
        factor_category_values,
        LIMITING_FACTOR_COLOR_MAP,
    )

    st.header("Level 1: Limiting Factors", help=LIMITING_FACTOR_RISK_DESCRIPTION)
    st.markdown(LIMITING_FACTOR_INTRODUCTION)
    st.divider()
    st.subheader(
        "Top Five Limiting Factors",
        help="Shows the top five Limiting-Factor Risk ranks within each BSR; ties remain together.",
    )
    rank_names = ("highest", "second-highest", "third-highest", "fourth-highest", "fifth-highest")
    for rank in range(1, 6):
        rank_name = rank_names[rank - 1]
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
            title_help=(
                f"This map shows the limiting factor with the {rank_name} "
                "risk score in each BSR. Tied factors share a rank.\n\n"
                + LIMITING_FACTOR_RISK_MAXIMUM_HELP
            ),
        )


def validate_action_benefits(
    bsr: pd.DataFrame,
    actions: pd.DataFrame,
    limiting: pd.DataFrame,
    action_components: pd.DataFrame,
) -> None:
    """Check action scores and their limiting-factor components."""
    bsr_values = bsr.copy()
    bsr_values["highest_action_benefit_score"] = pd.to_numeric(
        bsr_values["highest_action_benefit_score"], errors="coerce"
    )
    if bsr_values["highest_action_benefit_score"].isna().any():
        raise ValueError(
            "bsr_scores.csv contains missing or nonnumeric highest "
            "Action-Specific Benefit Scores."
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

    missing_component_bsrs = sorted(
        set(bsr_values["bsr"]) - set(highest_components["bsr"])
    )
    if missing_component_bsrs:
        raise ValueError(
            "No highest-benefit component could be identified for BSRs: "
            + ", ".join(missing_component_bsrs)
        )


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
    validate_action_benefits(
        bsr,
        actions,
        limiting,
        action_components,
    )

    st.header("Level 2: Action Benefits", help=ACTION_BENEFIT_DESCRIPTION)
    st.markdown(ACTION_BENEFIT_INTRODUCTION)
    st.divider()

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
            "colors span zero to the highest BSR score for the selected action. "
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
            if normalize_action_colors else observed_included_maximum(
                action_map, "action_benefit_score"
            )
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
            range_color=(0.0, map_scale_maximum),
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

    selected = actions.loc[actions["bsr"].eq(selected_bsr)].copy()
    horizontal_bar(
        selected,
        "action_benefit_score",
        "action_type",
        f"{selected_bsr}: Action-Specific Benefit Score by action",
        value_label="Action-Specific Benefit Score",
        hover_columns=["benefit_rank_within_bsr"],
        description=None,
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
            description=(
                "Shows which limiting factors contribute most to the selected "
                "action's score in the selected BSR."
            ),
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


def render_top_five_action_benefits(
    tables: dict[str, pd.DataFrame],
    geometry: gpd.GeoDataFrame,
    basin: str,
    map_style: str,
) -> None:
    """Render the top-five action-benefit maps in a dedicated view."""
    actions = filter_table(tables["action"], basin)
    ranked_action_maps = summarize_ranked_categories(
        actions,
        "action_type",
        "action_benefit_score",
        "action_rank",
        ranks=5,
    )
    action_rank_labels = [
        f"action_rank_{rank}_label" for rank in range(1, 6)
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

    st.header("Level 2: Action Benefits", help=ACTION_BENEFIT_DESCRIPTION)
    st.subheader(
        "Top Five Action-Specific Benefits",
        help="Shows the top five Action-Specific Benefit Score ranks within each BSR; ties remain together.",
    )
    rank_names = ("highest", "second-highest", "third-highest", "fourth-highest", "fifth-highest")
    for rank in range(1, 6):
        rank_name = rank_names[rank - 1]
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
            title_help=(
                f"This map shows the action type with the {rank_name} "
                "benefit score in each BSR. Tied actions share a rank.\n\n"
                + ACTION_BENEFIT_MAXIMUM_HELP
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
        "Highest Risk-Aligned Life Stage",
        "Limiting Factor Risk",
        "Top Five Limiting Factors",
        "Action-Specific Benefits",
        "Top Five Action-Specific Benefits",
    ]
    if st.session_state.get("atlas_view") not in pages:
        st.session_state["atlas_view"] = pages[0]

    st.sidebar.markdown("#### Level 1")
    page = pages[0]
    st.sidebar.button(
        page,
        key=f"sidebar_view_{page}",
        on_click=select_sidebar_view,
        args=(page,),
        type=("primary" if st.session_state["atlas_view"] == page else "secondary"),
        use_container_width=True,
    )
    with st.sidebar.container(key="atlas_fish_use_nav"):
        st.markdown("##### Fish Use")
        for page in pages[1:3]:
            st.button(
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
    with st.sidebar.container(key="atlas_limiting_factor_nav"):
        st.markdown("##### Limiting Factors")
        for page in pages[3:5]:
            st.button(
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
    for page in pages[5:]:
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
    return st.session_state["atlas_view"]


def main() -> None:
    """Run the Streamlit application."""
    configure_page()
    st.title("Atlas Integrated Scoring")
    st.caption("Interactive Level 1 risk and Level 2 action-benefit maps")

    try:
        score_signature = score_file_signature(str(SCORE_DIR))
        tables = load_score_tables(
            str(SCORE_DIR), SCORE_SCHEMA_VERSION, score_signature
        )
    except Exception as error:
        render_scoring_methodology()
        st.error(str(error))
        st.stop()

    # Keep complete included score tables available for attribution in chart help.
    st.session_state["atlas_score_source_tables"] = tables
    render_scoring_methodology(tables)
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
        "Score labels read % of Observed Maximum (raw score). A value of "
        "100% matches the highest observed score for the comparison "
        "described beside each chart."
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
    elif page == "Highest Risk-Aligned Life Stage":
        render_highest_risk_aligned_life_stage(tables, geometry, basin, MAP_STYLE)
    elif page == "Limiting Factor Risk":
        render_limiting_factors(tables, geometry, basin, selected_bsr, MAP_STYLE)
    elif page == "Top Five Limiting Factors":
        render_top_five_limiting_factors(tables, geometry, basin, MAP_STYLE)
    elif page == "Action-Specific Benefits":
        render_actions(tables, geometry, basin, selected_bsr, MAP_STYLE)
    else:
        render_top_five_action_benefits(tables, geometry, basin, MAP_STYLE)

    st.divider()
    st.caption(
        "Scores are relative prioritization indicators derived from the provisional "
        "Atlas framework."
    )


if __name__ == "__main__":
    main()
