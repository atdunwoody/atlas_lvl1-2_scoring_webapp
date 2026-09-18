# Atlas integrated scoring framework

This document describes the calculations implemented in the current Atlas integrated scoring notebook. The notebook combines fish use, limiting-factor condition, biological vulnerability, and population priority into relative risk scores for each BSR. It then weights limiting-factor risk by action relationships to score individual action types. The scores do not evaluate individual projects or predict fish abundance, habitat gain, or restoration effectiveness.

## Source inputs

The notebook reads five processed CSVs and one BSR polygon GeoPackage from `data/inputs`, or from the folder set by `INPUT_DIR_OVERRIDE`. The original workbooks and supporting source table in `Original Excel/` are reference material; the notebook does not read them directly.

| Processed input | Original source file |
|---|---|
| `Fish Use Scores.csv`; `Population scores.csv` | `Original Excel/Fish Use Score calculator - Normalized.xlsx` |
| `Vulnerability table.csv` | `Original Excel/Combined - Lifestage to Limiting Factor Crosswalk Table.xlsx` |
| `LFAT.csv` | `Original Excel/LFAT Atlas Action-LimFact Crosswalk scoring working.xlsx` |
| `Limiting factor scores.csv` | `Original Excel/BSR_LF_cell_stats.csv` |
| `bsr.gpkg` | Input BSR polygons |

The fish-use import selects only the specified fields. Overall BSR fish use comes from `fish_use_score_decimal`.

## Scoring scales

| Input | Calculation used by the notebook | Interpretation of a larger value |
|---|---|---|
| Life-stage fish use | $\text{normalized life-stage fish use} = \dfrac{\text{source life-stage fish use}}{\max_{\text{all input rows}}(\text{source life-stage fish use})}$ | More fish use on the source index |
| Limiting-factor condition | $\text{condition score} = 0.01 + (\text{raw rating} - 1)\dfrac{0.99}{4}$ | Greater impairment |
| Biological vulnerability | $\text{vulnerability score} = 1 - (\text{rank} - 1)\dfrac{0.99}{14}$ | Greater vulnerability to that limiting factor |
| Population priority | $\text{population priority} = \text{source priority}$ | More weight for a life stage within its basin and species |
| Action relationship weight | $\text{action relationship weight} = \text{directness} \times \text{frequency}$ | Stronger relationship between an action and a limiting factor |

The notebook retains the original life-stage value as `LS_corrected_score_source`, stores the normalized result as `LS_corrected_score`, and records the denominator in `life_stage_fish_use_normalization_max`. The action relationship weight is `lfat_score`, the product of `directness_value` and `frequency_value`. A zero source fish-use value remains zero. The condition and vulnerability transformations map their least contributing endpoints to 0.01, not zero. The assumed direction of the condition rating and the linear transformations are modeling choices; passing numerical checks does not validate the source rubric.

Population priorities are used as supplied. They are checked to sum to 1 within each basin and species. There is no additional between-species priority multiplier.

`species_aggregate_score` is retained unchanged for species fish-use context and may exceed 1. The BSR-level `fish_use_score` is the source `fish_use_score_decimal` and ranges from 0 to 1. Neither of these scores are an additional risk multiplier. Because the life-stage denominator is set by the complete input table, changes to that maximum rescale risk and risk-weighted action scores. Compare the recorded denominator when comparing runs.

## Input coverage and migration vulnerability

Each BSR must have the same complete set of species and life stages in fish use, with matching population-priority and vulnerability records. Condition, vulnerability, and action-crosswalk tables must cover the same 15 named limiting factors. Every action has a row for each factor, including zero-weight relationships. The notebook checks unique keys and complete pathway coverage before export.

Chinook and Steelhead fish use and population priority each have one `Migration` life stage, while the vulnerability input has separate adult migration and juvenile emigration records. For each species and limiting factor, the notebook uses the larger vulnerability score:

$$\text{migration vulnerability} = \max(\text{adult migration vulnerability},\ \text{juvenile migration vulnerability})$$

Other life stages pass through unchanged. The source stages, score ranges, and review flags remain available in the vulnerability review output.

## Level 1: integrated risk

The calculation unit is one BSR, species, life stage, and limiting factor pathway. The notebook calculates one four-factor contribution for each pathway:

$$\text{risk component} = \text{life-stage fish use} \times \text{population priority} \times \text{limiting-factor condition} \times \text{vulnerability}$$

It then sums those contributions in several ways:

$$\text{life-stage risk} = \sum_{\text{15 limiting factors}} \text{risk component}$$

$$\text{species risk} = \sum_{\text{life stages of the species}} \text{life-stage risk}$$

$$\text{limiting-factor risk} = \sum_{\text{all species and life stages}} \text{risk component}$$

$$\text{overall BSR risk} = \sum_{\text{all species and life stages}} \text{life-stage risk} = \sum_{\text{15 limiting factors}} \text{limiting-factor risk}$$

The notebook stores pathway values as `risk_component`, the grouped life-stage, species, and limiting-factor values as `risk_score`, and the BSR total as `overall_risk_score`. All grouping routes must give the same BSR total. A pathway with zero life-stage fish use has zero risk. Aggregate scores can exceed 1 and are relative indices, not probabilities.

Within each BSR, life-stage, species, and limiting-factor scores receive dense ranks. The BSR summary retains the leading life-stage and limiting-factor labels and their tie counts. If all scores are zero, tied leading labels indicate a zero-score tie.

## Level 2: action alignment

For each BSR, limiting factor, and action, the notebook multiplies Level 1 limiting-factor risk by that action's relationship weight:

$$\text{action benefit score} = \sum_{\text{15 limiting factors}} (\text{limiting-factor risk} \times \text{action relationship weight})$$

The `action_benefit_score` is reported separately for each BSR and action and ranked within the BSR, with equal scores sharing a rank. The BSR summary records `highest_action_benefit_score` and `highest_risk_aligned_action_type` labels. One limiting factor can contribute to several action scores.

`action_benefit_score` is an action-alignment index based on existing risk. It does not estimate condition improvement, habitat gain, fish response, cost, feasibility, or site-specific effectiveness.

## Outputs and review limits

The notebook writes eight core CSVs to `data/outputs`:

| Output | Contents |
|---|---|
| `bsr_scores.csv` | One row per BSR with overall risk, highest action score, leading labels, and source-review status |
| `fish_use_scores.csv`; `population_scores.csv` | Scoring inputs and retained fish-use context |
| `life_stage_scores.csv`; `species_scores.csv` | Level 1 risk grouped by life stage and species |
| `limiting_factor_scores_integrated.csv` | Level 1 risk grouped by limiting factor |
| `action_scores.csv` | Separate Level 2 alignment score for each BSR and action |
| `calculation_grid.csv` | Individual risk pathways and their inputs |

`bsr_scores.gpkg` copies the input BSR polygons, adds score and review fields, and registers nonspatial fish-use, population, life-stage, species, limiting-factor, and action tables. The `QC/` folder includes action components, source-review tables, normalization and population-priority summaries, a field dictionary, input hashes, run metadata, and the check record.