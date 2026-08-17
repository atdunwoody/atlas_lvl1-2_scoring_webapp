# How risk scores are calculated

Each BSR is evaluated for every species, life stage, and limiting factor combination. Higher input values increase the calculated risk score.

## Inputs

### Fish-use inputs

- **Life-Stage Fish Use Score** is specific to the BSR, species, and life stage. It is the fish-use multiplier used in the risk calculations.
- The Life-Stage Fish Use Score is the **Corrected\_{life stage}** score in the `Normalized Score Calculator` tab and the `LS_corrected_score` field in the fish-use input table.
- Life-Stage Fish Use Scores are normalized from 0 to 1.
- **Species Fish Use Score** is the species-specific score listed in the `Fish Use Scores - Normalized` tab and the `species_aggregate_score` field. It is reported for context but is not a direct multiplier in the risk equations.
- **Overall Fish Use Score** is the BSR-level 0-to-1 score in the `fish_use_score_decimal` field. It is reported for context but is not a direct multiplier in the risk equations.

### Vulnerability inputs

- **Vulnerability Rank** is specific to the species, life stage, and limiting factor. Rank 1 represents the highest vulnerability, and rank 15 represents the lowest vulnerability.
- The vulnerability score used in the calculations maps rank 1 to 1.0 and rank 15 to 0.01:

$$
\text{vulnerability}
=
1
-
\left(
\text{vulnerability rank} - 1
\right)
\times
\frac{0.99}{14}
$$

- For the combined Migration life stage, the calculation uses the higher of the adult and juvenile vulnerability scores for each species and limiting factor.

### Limiting-factor inputs

- **Limiting-Factor Condition** is specific to the BSR and limiting factor. The current input table does not contain a life-stage-specific condition field.
- The limiting-factor condition score used in the calculations maps the source 1-to-5 rating to 0.01 to 1.0:

$$
\text{limiting-factor condition}
=
0.01
+
\left(
\text{raw condition rating} - 1
\right)
\times
\frac{0.99}{4}
$$

- The provided `lf_condition_score` field uses a previous 0.10-to-1.0 transformation. It is retained for reference, but the calculations use the score recalculated from `lf_condition_score_raw_1_5` using the equation above.
- Within a BSR, the same condition score for a limiting factor is applied to every species and life-stage pathway.

### Population-priority inputs

- **Population Priority** is specific to the basin, species, and life stage.
- The population priorities for each species sum to 100%.

## Level 1 risk calculations

### Pathway risk components

For each BSR, species, life stage, and limiting factor combination:

$$
\begin{aligned}
\text{risk component}
={}&
\text{life-stage fish use}
\times
\text{limiting-factor condition} \\
&\times
\text{vulnerability}
\times
\text{population priority}
\end{aligned}
$$

Life-stage fish use and population priority are specific to the applicable BSR, species, and life stage. Vulnerability is specific to the species, life stage, and limiting factor. Limiting-factor condition is specific to the BSR and limiting factor and is applied to all species and life-stage pathways for that limiting factor within the BSR.

Risk can be aggregated in two equivalent ways:

1. Across the 15 limiting factors for each species and life stage.
2. Across all species and life stages for each limiting factor.

Both approaches produce the same Overall Risk Score within a BSR, although the intermediate scores describe different components of the overall score.

### Species and life-stage risk scores

For a given BSR, species, and life stage, the Life-Stage Fish Use Score and Population Priority are constant across the 15 limiting factors and can be placed in front of the sum:

$$
\begin{aligned}
\text{life-stage risk score}
={}&
\text{life-stage fish use}
\times
\text{population priority} \\
&\times
\sum_{\substack{\text{15 limiting}\\\text{factors}}}
\Bigl(
\text{limiting-factor condition}
\times
\text{vulnerability}
\Bigr)
\end{aligned}
$$

The risk score for the highest-priority life stage is:

$$
\begin{aligned}
\text{risk score for highest priority life stage}
={}&
\max_{\substack{\text{all species and}\\\text{life stages}}}
\left(
\text{life-stage risk score}
\right)
\end{aligned}
$$

The **Highest Priority Life Stage** is the species and life-stage combination with the largest Life-Stage Risk Score within the BSR.

### Limiting-factor risk scores

For a given BSR and limiting factor, the Limiting-Factor Condition is constant across species and life stages and can be placed in front of the sum:

$$
\begin{aligned}
\text{limiting-factor risk score}
={}&
\text{limiting-factor condition} \\
&\times
\sum_{\substack{\text{all species and}\\\text{life stages}}}
\Bigl(
\text{life-stage fish use}
\times
\text{vulnerability}
\times
\text{population priority}
\Bigr)
\end{aligned}
$$

Within this sum, life-stage fish use and population priority are specific to the applicable BSR, species, and life stage. Vulnerability is specific to the species, life stage, and limiting factor. Limiting-factor condition is the single condition score for the BSR and limiting factor.

The risk score for the highest-priority limiting factor is:

$$
\begin{aligned}
\text{risk score for highest priority limiting factor}
={}&
\max_{\text{15 limiting factors}}
\left(
\text{limiting-factor risk score}
\right)
\end{aligned}
$$

The **Highest Priority Limiting Factor** is the limiting factor with the largest Limiting-Factor Risk Score within the BSR.

### Overall BSR risk score

The Overall Risk Score can be calculated equivalently by summing either all Life-Stage Risk Scores or all 15 Limiting-Factor Risk Scores:

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

The action weight is calculated from relationship directness and frequency:

$$
\text{action weight}
=
\text{relationship directness}
\times
\text{frequency}
$$

For each limiting factor and action:

$$
\text{action-specific benefit component}
=
\text{limiting-factor risk score}
\times
\text{action weight}
$$

For each action, the action-specific benefit components are summed across the 15 limiting factors:

$$
\begin{aligned}
\text{action-specific benefit score}
={}&
\sum_{\substack{\text{15 limiting}\\\text{factors}}}
\left(
\text{limiting-factor risk score}
\times
\text{action weight}
\right)
\end{aligned}
$$

The Overall Benefit Score is the sum of the Action-Specific Benefit Scores across all actions:

$$
\begin{aligned}
\text{overall benefit score}
={}&
\sum_{\text{all actions}}
\left(
\text{action-specific benefit score}
\right)
\end{aligned}
$$

The **Highest Risk-Aligned Action Type** is the action type with the largest Action-Specific Benefit Score within the BSR.