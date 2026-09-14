# Atlas integrated scoring framework

This framework combines fish use, limiting-factor condition, biological vulnerability, and population priority into relative BSR-level prioritization indices. It then weights limiting-factor risk by action relationships to describe Level 2 action alignment. The results do not evaluate individual projects or predict fish abundance, habitat gain, or restoration effectiveness.

## Source inputs

The scoring code reads five processed CSVs and one BSR polygon GeoPackage. The original workbooks and supporting source table are retained under `inputs/Original Excel/` for review of formulas, ratings, and notes.

| Processed scoring input | Original source file |
|---|---|
| `Fish Use Scores.csv`; `Population scores.csv` | `Original Excel/Fish Use Score calculator - Normalized.xlsx` |
| `Vulnerability table.csv` | `Original Excel/Combined - Lifestage to Limiting Factor Crosswalk Table.xlsx` |
| `LFAT.csv` | `Original Excel/LFAT Atlas Action-LimFact Crosswalk scoring working.xlsx` |
| `Limiting factor scores.csv` | `Original Excel/BSR_LF_cell_stats.csv` |

The code reads only `fish_use_score_decimal` for overall fish use. It does not import alternate overall fish-use fields.

## Scoring scales

| Input | Calculation | Interpretation of a larger value |
|---|---|---|
| Life-stage fish use | $\text{life-stage fish use} = \dfrac{\text{source life-stage fish use}}{\text{maximum source life-stage fish use}}$ | More fish use on the source index |
| Limiting-factor condition | $\text{condition score} = 0.01 + (\text{raw rating} - 1) \times \dfrac{0.99}{4}$ | Greater impairment, assuming the source rubric runs from least impaired at 1 to most impaired at 5 |
| Biological vulnerability | $\text{vulnerability score} = 1 - (\text{rank} - 1) \times \dfrac{0.99}{14}$ | Greater vulnerability to the limiting factor |
| Population priority | $\text{population priority} = \text{source priority}$ | More weight for the life stage within its basin and species |
| Action relationship weight | $\text{action relationship weight} = \text{directness} \times \text{frequency}$ | A stronger action and limiting-factor relationship |

The condition and vulnerability transformations retain a 0.01 floor. A source life-stage fish-use score of zero remains zero and therefore produces zero impact and risk.

Normalization uses one maximum across the complete life-stage fish-use input table. If that maximum changes between runs, absolute impact, risk, and risk-based action scores also rescale. Compare the recorded `life_stage_fish_use_normalization_max` before interpreting score changes between runs.

## Migration vulnerability

Chinook and Steelhead vulnerability inputs contain separate adult and juvenile migration records. For each species and limiting factor, the framework uses the larger score:

$$\text{migration vulnerability} = \max(\text{adult migration vulnerability},\ \text{juvenile migration vulnerability})$$

This represents the more vulnerable migration pathway without adding migration twice.

## Level 1 integrated risk

The basic calculation unit is one BSR, species, life stage, and limiting factor pathway:

$$\text{impact component} = \text{life-stage fish use} \times \text{limiting-factor condition} \times \text{vulnerability}$$

$$\text{risk component} = \text{impact component} \times \text{population priority}$$

Impact describes the unweighted overlap of fish use, condition, and vulnerability. Risk adds the life-stage population-priority weight. Risk is a relative index, not a probability.

The pathway components are summed to produce species/life-stage, species, limiting-factor, and overall BSR scores:

$$\text{species/life-stage risk} = \sum_{\text{15 limiting factors}} \text{risk component}$$

$$\text{limiting-factor risk} = \sum_{\text{species and life stages}} \text{risk component}$$

$$\text{overall BSR risk} = \sum_{\text{all pathways in the BSR}} \text{risk component}$$

These independent grouping routes must produce the same overall BSR risk. Aggregate scores are sums and may exceed 1.

## Level 2 action alignment

The framework applies each action relationship weight to condition, limiting-factor impact, and limiting-factor risk. The established Streamlit app fields are retained.

| Output field | Calculation for one BSR and action |
|---|---|
| `condition_improvement_score` | $\sum_{\text{limiting factors}} (\text{condition score} \times \text{action relationship weight})$ |
| `limiting_factor_amelioration_score` | $\sum_{\text{limiting factors}} (\text{limiting-factor impact} \times \text{action relationship weight})$ |
| `action_benefit_score` | $\sum_{\text{limiting factors}} (\text{limiting-factor risk} \times \text{action relationship weight})$ |

The BSR-level totals use the established fields `overall_condition_improvement_score`, `overall_limiting_factor_amelioration_score`, and `overall_benefit_score`. The final total is:

$$\text{overall benefit score} = \sum_{\text{actions}} \text{action benefit score}$$

Because one limiting factor can be related to multiple actions, the same limiting-factor contribution can enter more than one action score. These values are alignment indices. They are not additive estimates of realized benefit and do not account for feasibility, cost, implementation constraints, landowner willingness, or site-specific effectiveness.

## Interpretation and review limits

- Larger scores indicate greater overlap among the scored inputs or stronger correspondence with action relationships.
- `species_aggregate_score` and `fish_use_score` are contextual fields. They are not additional multipliers in the Level 1 equations.
- Population priorities distribute weight among life stages within basin and species. The framework does not add a separate between-species multiplier.
- `highest_risk_*` fields identify the largest calculated contributions. They do not establish a complete restoration priority.
- Equal scores share a dense rank and all tied top labels are retained.
- Source BSR crosswalk statuses and source condition and vulnerability review flags remain visible. Passing numerical QC does not independently validate those source judgments.
