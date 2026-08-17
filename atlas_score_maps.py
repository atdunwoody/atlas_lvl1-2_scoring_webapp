"""Create publication-ready maps from the Atlas scoring GeoPackage.

The scoring notebook writes a scored BSR feature layer plus the
``species_scores`` and ``action_type_scores`` attribute tables to
``data/outputs/bsr_scores.gpkg``. This script reads those outputs and creates:

* overall risk;
* one risk map per species;
* highest risk-aligned species/life stage;
* highest risk-aligned limiting factor;
* highest risk-aligned benefit action type; and
* one benefit-score map per action type;
* preliminary BSR tiers based on overall-risk thirds; and
* preliminary BSR tiers seeded by each species' two highest-risk BSRs.

All maps include a CartoDB Positron basemap, BSR outlines and labels, a north
arrow, a ground-distance scale bar, and a legend. UGR 8 is excluded from score
colors and shared scale calculations and is shown with a neutral hatch. Species
maps share one color scale; action-type benefit maps share a second color scale
so maps within each group can be compared directly. Overall and species risk
maps also label each analyzed BSR's rank beneath its BSR identifier.

Required packages: geopandas, pandas, numpy, matplotlib, contextily, certifi.
"""

from __future__ import annotations

import argparse
import colorsys
import math
import re
import sqlite3
import ssl
import textwrap
import unicodedata
from pathlib import Path

import certifi
import geopandas as gpd
import geopy.geocoders
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# geopy 2.4 and later uses the system certificate store by default. A malformed
# certificate in the Windows store can therefore prevent contextily from being
# imported. Use Conda's certifi CA bundle explicitly while retaining full TLS
# certificate verification.
geopy.geocoders.options.default_ssl_context = ssl.create_default_context(
    cafile=certifi.where()
)

import contextily as cx


DEFAULT_GPKG = Path("data/outputs/bsr_scores.gpkg")
DEFAULT_OUTPUT_DIR = Path("data/outputs/maps")
WEB_MERCATOR_CRS = "EPSG:3857"

BOUNDARY_COLOR = "#54616b"
NO_DATA_COLOR = "#dedede"
MAP_FACE_COLOR = "#f7f7f4"
EXCLUDED_BSR_IDS = {"UGR8", "UGR08"}
EXCLUDED_BSR_COLOR = "#f1efeb"
EXCLUDED_BSR_HATCH = "////"
TIER_COLORS = {
    "Tier 1": "#c94f62",
    "Tier 2": "#e5b567",
    "Tier 3": "#A7AAA5",
}

RISK_CMAP = LinearSegmentedColormap.from_list(
    "two_hue_risk",
    ["#4f9ca8", "#b9d9d8", "#f7f1e8", "#e8a29d", "#b7435a"],
)
BENEFIT_CMAP = LinearSegmentedColormap.from_list(
    "pastel_benefit",
    ["#fff5e8", "#f7dec1", "#efc3b3", "#dfa6b5", "#bd8fb8"],
)

# The first colors are hand-selected for clear categorical separation. Extra
# colors are generated in pastel HLS space when a map has more categories.
PASTEL_CATEGORIES = [
    "#a8d5ba",
    "#f4c6a6",
    "#b8c7e8",
    "#e6b3c8",
    "#c8d6a5",
    "#c5b5df",
    "#f1d49b",
    "#9fcbd4",
    "#d7b7a3",
    "#b6d8d0",
    "#efb6a8",
    "#c9c4e8",
    "#d8e0ad",
    "#b4c7d9",
    "#e2c4a8",
    "#b9d2a8",
]


def quote_identifier(value: str) -> str:
    """Safely quote a SQLite identifier."""
    return '"' + str(value).replace('"', '""') + '"'


def slugify(value: object) -> str:
    """Return a stable, filesystem-safe label."""
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "unnamed"


def pastel_colors(count: int) -> list[str]:
    """Return ``count`` distinct pastel colors."""
    if count <= len(PASTEL_CATEGORIES):
        return PASTEL_CATEGORIES[:count]

    colors = PASTEL_CATEGORIES.copy()
    remaining = count - len(colors)
    for index in range(remaining):
        hue = (index / max(remaining, 1) + 0.06) % 1.0
        red, green, blue = colorsys.hls_to_rgb(hue, 0.78, 0.42)
        colors.append(mpl.colors.to_hex((red, green, blue)))
    return colors


def feature_layer_with_bsr(connection: sqlite3.Connection) -> str:
    """Find the one feature layer containing a BSR identifier."""
    layers = pd.read_sql_query(
        "SELECT table_name FROM gpkg_contents WHERE data_type = 'features'",
        connection,
    )["table_name"].tolist()

    candidates: list[str] = []
    for layer in layers:
        columns = pd.read_sql_query(
            f"PRAGMA table_info({quote_identifier(layer)})", connection
        )["name"].str.lower()
        if columns.isin(["bsr", "score_bsr"]).any():
            candidates.append(layer)

    if len(candidates) != 1:
        raise ValueError(
            "Expected exactly one feature layer with a 'bsr' or 'score_bsr' "
            f"field; found {len(candidates)}: {candidates}"
        )
    return candidates[0]


def read_attribute_table(
    connection: sqlite3.Connection, table_name: str
) -> pd.DataFrame:
    """Read a required nonspatial GeoPackage attribute table."""
    available = set(
        pd.read_sql_query(
            "SELECT table_name FROM gpkg_contents", connection
        )["table_name"]
    )
    if table_name not in available:
        raise ValueError(
            f"Required GeoPackage table '{table_name}' was not found. "
            f"Available tables: {sorted(available)}"
        )
    return pd.read_sql_query(
        f"SELECT * FROM {quote_identifier(table_name)}", connection
    )


def require_columns(
    table: pd.DataFrame, required: list[str], table_name: str
) -> None:
    """Raise a readable error when an expected scoring field is absent."""
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(
            f"{table_name} is missing required fields: {', '.join(missing)}"
        )


def read_scoring_outputs(
    gpkg_path: str | Path,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame]:
    """Read BSR geometry, species scores, and action scores."""
    gpkg_path = Path(gpkg_path).expanduser().resolve()
    if not gpkg_path.is_file():
        raise FileNotFoundError(f"Scoring GeoPackage not found: {gpkg_path}")

    with sqlite3.connect(gpkg_path) as connection:
        layer_name = feature_layer_with_bsr(connection)
        species_scores = read_attribute_table(connection, "species_scores")
        action_scores = read_attribute_table(connection, "action_type_scores")

    bsr = gpd.read_file(gpkg_path, layer=layer_name)
    if bsr.crs is None:
        raise ValueError("The BSR feature layer has no coordinate reference system.")

    require_columns(
        bsr,
        [
            "overall_risk_score",
            "highest_risk_species_life_stage",
            "highest_risk_limiting_factor",
            "highest_risk_aligned_action_type",
        ],
        layer_name,
    )
    require_columns(
        species_scores, ["bsr", "species", "risk_score"], "species_scores"
    )
    require_columns(
        action_scores,
        ["bsr", "action_type", "action_benefit_score"],
        "action_type_scores",
    )

    normalized_columns = {column.lower(): column for column in bsr.columns}
    bsr_key = normalized_columns.get(
        "bsr", normalized_columns.get("score_bsr")
    )
    if bsr_key is None:
        raise ValueError("The BSR feature layer has no BSR identifier field.")
    bsr = bsr.loc[bsr.geometry.notna() & ~bsr.geometry.is_empty].copy()
    bsr["_bsr_key"] = bsr[bsr_key].astype(str).str.strip()
    bsr["_bsr_label"] = bsr[bsr_key].astype(str).str.strip()
    normalized_bsr_labels = (
        bsr["_bsr_label"]
        .str.upper()
        .str.replace(r"[^A-Z0-9]+", "", regex=True)
    )
    bsr["_excluded_from_analysis"] = normalized_bsr_labels.isin(
        EXCLUDED_BSR_IDS
    )
    if bsr["_bsr_key"].duplicated().any():
        duplicates = sorted(bsr.loc[bsr["_bsr_key"].duplicated(), "_bsr_key"])
        raise ValueError(f"Duplicate BSR geometries found: {duplicates}")

    for table in (species_scores, action_scores):
        table["_bsr_key"] = table["bsr"].astype(str).str.strip()

    # CartoDB tiles use Web Mercator. Keeping all map layers in this CRS also
    # makes the basemap request and the scale-bar drawing deterministic.
    return bsr.to_crs(WEB_MERCATOR_CRS), species_scores, action_scores


def join_one_score(
    bsr: gpd.GeoDataFrame,
    scores: pd.DataFrame,
    score_field: str,
) -> gpd.GeoDataFrame:
    """Join one unique score per BSR to the geometry."""
    selected = scores[["_bsr_key", score_field]].copy()
    if selected["_bsr_key"].duplicated().any():
        raise ValueError(
            f"The score table has more than one '{score_field}' row per BSR."
        )
    return bsr.merge(selected, on="_bsr_key", how="left", validate="one_to_one")


def padded_extent(gdf: gpd.GeoDataFrame, padding_fraction: float = 0.045) -> tuple:
    """Return a slightly padded plotting extent."""
    xmin, ymin, xmax, ymax = gdf.total_bounds
    width = max(xmax - xmin, 1.0)
    height = max(ymax - ymin, 1.0)
    return (
        xmin - width * padding_fraction,
        xmax + width * padding_fraction,
        ymin - height * padding_fraction,
        ymax + height * padding_fraction,
    )


def add_positron_basemap(ax: plt.Axes, zoom: int | str = "auto") -> None:
    """Add the light CartoDB Positron basemap."""
    try:
        cx.add_basemap(
            ax,
            source=cx.providers.CartoDB.Positron,
            crs=WEB_MERCATOR_CRS,
            zoom=zoom,
            attribution_size=6,
        )
    except Exception as error:
        raise RuntimeError(
            "Could not retrieve the CartoDB Positron basemap. Confirm that "
            "the computer has internet access, or call create_all_maps(..., "
            "use_basemap=False) for an offline draft."
        ) from error


def make_axes(
    gdf: gpd.GeoDataFrame,
    title: str,
    use_basemap: bool,
    basemap_zoom: int | str,
) -> tuple[plt.Figure, plt.Axes]:
    """Create a common map frame and set the spatial extent."""
    fig, ax = plt.subplots(figsize=(11.0, 8.5), constrained_layout=True)
    left, right, bottom, top = padded_extent(gdf)
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.set_aspect("equal")
    ax.set_facecolor(MAP_FACE_COLOR)
    ax.set_axis_off()
    ax.set_title(
        textwrap.fill(title, width=56),
        fontsize=17,
        fontweight="semibold",
        color="#27343b",
        pad=14,
    )
    if use_basemap:
        add_positron_basemap(ax, zoom=basemap_zoom)
    return fig, ax


def nice_scale_length(target_ground_metres: float) -> float:
    """Round a target scale-bar distance down to a 1, 2, or 5 interval."""
    if not np.isfinite(target_ground_metres) or target_ground_metres <= 0:
        return 1.0
    exponent = 10 ** math.floor(math.log10(target_ground_metres))
    candidates = np.array([1.0, 2.0, 5.0, 10.0]) * exponent
    valid = candidates[candidates <= target_ground_metres]
    return float(valid[-1] if len(valid) else candidates[0])


def add_scale_bar(ax: plt.Axes, gdf: gpd.GeoDataFrame) -> None:
    """Add an approximate ground-distance scale bar to a Web Mercator map."""
    xmin, xmax = ax.get_xlim()
    lonlat_bounds = gdf.to_crs("EPSG:4326").total_bounds
    centre_latitude = (lonlat_bounds[1] + lonlat_bounds[3]) / 2.0
    cosine_latitude = max(math.cos(math.radians(centre_latitude)), 0.05)

    ground_width = (xmax - xmin) * cosine_latitude
    ground_length = nice_scale_length(ground_width * 0.20)
    web_mercator_length = ground_length / cosine_latitude
    axis_fraction = web_mercator_length / (xmax - xmin)

    start_x = 0.055
    end_x = start_x + axis_fraction
    y = 0.055
    line_style = {
        "transform": ax.transAxes,
        "color": "#27343b",
        "linewidth": 2.2,
        "solid_capstyle": "butt",
        "zorder": 10,
    }
    ax.add_line(Line2D([start_x, end_x], [y, y], **line_style))
    ax.add_line(Line2D([start_x, start_x], [y - 0.008, y + 0.008], **line_style))
    ax.add_line(Line2D([end_x, end_x], [y - 0.008, y + 0.008], **line_style))

    if ground_length >= 1_000:
        label = f"{ground_length / 1_000:g} km"
    else:
        label = f"{ground_length:g} m"
    ax.text(
        (start_x + end_x) / 2.0,
        y + 0.012,
        label,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        color="#27343b",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.82,
            "pad": 1.8,
        },
        zorder=11,
    )


def add_bsr_labels(
    ax: plt.Axes,
    gdf: gpd.GeoDataFrame,
    rank_field: str | None = None,
) -> None:
    """Label each BSR and optionally show a compact score rank."""
    if "_bsr_label" not in gdf.columns:
        raise ValueError("Map data are missing the internal BSR label field.")

    label_points = gdf.geometry.representative_point()
    polygon_count = len(gdf)
    font_size = max(5.5, min(8.0, 8.25 - polygon_count * 0.025))

    show_ranks = rank_field is not None and rank_field in gdf.columns
    for row_index, (label, point) in enumerate(
        zip(gdf["_bsr_label"], label_points)
    ):
        if pd.isna(label) or point.is_empty:
            continue
        source_index = gdf.index[row_index]
        rank = gdf.at[source_index, rank_field] if show_ranks else pd.NA
        has_rank = pd.notna(rank)
        ax.annotate(
            str(label),
            xy=(point.x, point.y),
            xytext=(0, 1.0 if has_rank else 0),
            textcoords="offset points",
            ha="center",
            va="bottom" if has_rank else "center",
            fontsize=font_size,
            fontweight="semibold",
            color="#25343b",
            clip_on=True,
            zorder=8,
            path_effects=[
                path_effects.Stroke(linewidth=3.0, foreground="white"),
                path_effects.Normal(),
            ],
        )
        if has_rank:
            ax.annotate(
                f"Rank {int(rank)}",
                xy=(point.x, point.y),
                xytext=(0, -1.0),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=max(font_size - 1.3, 4.6),
                fontweight="normal",
                color="#35464e",
                clip_on=True,
                zorder=8,
                path_effects=[
                    path_effects.Stroke(linewidth=2.5, foreground="white"),
                    path_effects.Normal(),
                ],
            )


def add_north_arrow(ax: plt.Axes) -> None:
    """Add a clean filled pointer arrow in axes coordinates."""
    ax.annotate(
        "",
        xy=(0.94, 0.925),
        xytext=(0.94, 0.825),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops={
            "arrowstyle": "simple",
            "facecolor": "#27343b",
            "edgecolor": "white",
            "linewidth": 1.1,
            "mutation_scale": 24,
        },
        zorder=12,
    )
    ax.text(
        0.94,
        0.955,
        "N",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#27343b",
        zorder=13,
        path_effects=[
            path_effects.Stroke(linewidth=3.0, foreground="white"),
            path_effects.Normal(),
        ],
    )


def score_formatter(value: float, _position: int | None = None) -> str:
    """Format a numeric legend without unnecessary trailing zeros."""
    absolute = abs(value)
    if absolute >= 100:
        return f"{value:,.0f}"
    if absolute >= 10:
        return f"{value:,.1f}"
    if absolute >= 1:
        return f"{value:,.2f}"
    return f"{value:,.3f}"


def exclusion_mask(gdf: gpd.GeoDataFrame) -> pd.Series:
    """Return the analysis-exclusion flag aligned to a map table."""
    if "_excluded_from_analysis" not in gdf.columns:
        return pd.Series(False, index=gdf.index, dtype=bool)
    return gdf["_excluded_from_analysis"].fillna(False).astype(bool)


def plot_excluded_bsrs(ax: plt.Axes, gdf: gpd.GeoDataFrame) -> None:
    """Overlay excluded BSRs with a neutral hatch."""
    excluded = exclusion_mask(gdf)
    if not excluded.any():
        return
    gdf.loc[excluded].plot(
        ax=ax,
        facecolor=EXCLUDED_BSR_COLOR,
        edgecolor=BOUNDARY_COLOR,
        linewidth=0.9,
        hatch=EXCLUDED_BSR_HATCH,
        zorder=6,
    )


def excluded_bsr_legend_patch() -> Patch:
    """Return the standard legend symbol for excluded BSRs."""
    return Patch(
        facecolor=EXCLUDED_BSR_COLOR,
        edgecolor=BOUNDARY_COLOR,
        hatch=EXCLUDED_BSR_HATCH,
        label="Not included in analysis: UGR 8",
    )


def finalize_map(
    fig: plt.Figure,
    ax: plt.Axes,
    gdf: gpd.GeoDataFrame,
    output_path: Path,
    dpi: int,
    rank_field: str | None = None,
) -> None:
    """Add common map furniture and write a PNG."""
    add_bsr_labels(ax, gdf, rank_field=rank_field)
    add_scale_bar(ax, gdf)
    add_north_arrow(ax)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "Atlas score mapping script"},
    )
    plt.close(fig)


def plot_numeric_map(
    gdf: gpd.GeoDataFrame,
    value_field: str,
    title: str,
    legend_title: str,
    output_path: Path,
    cmap: mpl.colors.Colormap,
    scale_min: float,
    scale_max: float,
    dpi: int,
    use_basemap: bool,
    basemap_zoom: int | str,
    rank_field: str | None = None,
) -> None:
    """Plot a continuous score map with a color-bar legend."""
    mapped = gdf.copy()
    mapped[value_field] = pd.to_numeric(mapped[value_field], errors="coerce")
    excluded = exclusion_mask(mapped)
    valid = mapped[value_field].notna() & ~excluded
    missing = mapped[value_field].isna() & ~excluded

    fig, ax = make_axes(mapped, title, use_basemap, basemap_zoom)
    if missing.any():
        mapped.loc[missing].plot(
            ax=ax,
            color=NO_DATA_COLOR,
            edgecolor=BOUNDARY_COLOR,
            linewidth=0.45,
            zorder=2,
        )
    if valid.any():
        mapped.loc[valid].plot(
            ax=ax,
            column=value_field,
            cmap=cmap,
            vmin=scale_min,
            vmax=scale_max,
            alpha=0.84,
            edgecolor=BOUNDARY_COLOR,
            linewidth=0.55,
            zorder=3,
        )
    plot_excluded_bsrs(ax, mapped)

    normalization = mpl.colors.Normalize(vmin=scale_min, vmax=scale_max)
    scalar_mappable = mpl.cm.ScalarMappable(norm=normalization, cmap=cmap)
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(
        scalar_mappable,
        ax=ax,
        fraction=0.036,
        pad=0.018,
        shrink=0.76,
    )
    colorbar.set_label(legend_title, fontsize=10, labelpad=9)
    colorbar.ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(score_formatter))
    colorbar.outline.set_edgecolor("#9aa4a9")

    special_handles = []
    if missing.any():
        special_handles.append(
            Patch(
                facecolor=NO_DATA_COLOR,
                edgecolor=BOUNDARY_COLOR,
                label="No data",
            )
        )
    if excluded.any():
        special_handles.append(excluded_bsr_legend_patch())
    if special_handles:
        ax.legend(
            handles=special_handles,
            loc="lower right",
            frameon=True,
            framealpha=0.9,
            fontsize=8,
        )
    finalize_map(
        fig,
        ax,
        mapped,
        output_path,
        dpi,
        rank_field=rank_field,
    )


def plot_categorical_map(
    gdf: gpd.GeoDataFrame,
    value_field: str,
    title: str,
    legend_title: str,
    output_path: Path,
    dpi: int,
    use_basemap: bool,
    basemap_zoom: int | str,
    category_color_overrides: dict[str, str] | None = None,
) -> None:
    """Plot a categorical map with a discrete pastel legend."""
    mapped = gdf.copy()
    excluded = exclusion_mask(mapped)
    has_value = (
        mapped[value_field].notna()
        & mapped[value_field].astype(str).str.strip().ne("")
    )
    valid = has_value & ~excluded
    missing = ~has_value & ~excluded
    categories = sorted(
        mapped.loc[valid, value_field].astype(str).unique(), key=str.casefold
    )
    if category_color_overrides is None:
        category_colors = dict(
            zip(categories, pastel_colors(len(categories)))
        )
    else:
        missing_colors = [
            category
            for category in categories
            if category not in category_color_overrides
        ]
        if missing_colors:
            raise ValueError(
                "No map color was provided for categories: "
                + ", ".join(missing_colors)
            )
        category_colors = {
            category: category_color_overrides[category]
            for category in categories
        }

    fig, ax = make_axes(mapped, title, use_basemap, basemap_zoom)
    for category in categories:
        selected = mapped[value_field].astype(str).eq(category) & ~excluded
        mapped.loc[selected].plot(
            ax=ax,
            color=category_colors[category],
            alpha=0.86,
            edgecolor=BOUNDARY_COLOR,
            linewidth=0.55,
            zorder=3,
        )
    if missing.any():
        mapped.loc[missing].plot(
            ax=ax,
            color=NO_DATA_COLOR,
            edgecolor=BOUNDARY_COLOR,
            linewidth=0.55,
            zorder=2,
        )
    plot_excluded_bsrs(ax, mapped)

    handles = [
        Patch(
            facecolor=category_colors[category],
            edgecolor=BOUNDARY_COLOR,
            label=textwrap.fill(category, width=38),
        )
        for category in categories
    ]
    if missing.any():
        handles.append(
            Patch(
                facecolor=NO_DATA_COLOR,
                edgecolor=BOUNDARY_COLOR,
                label="No data",
            )
        )
    if excluded.any():
        handles.append(excluded_bsr_legend_patch())
    ax.legend(
        handles=handles,
        title=legend_title,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
        frameon=True,
        framealpha=0.94,
        edgecolor="#b8c0c4",
        fontsize=8.5,
        title_fontsize=9.5,
        labelspacing=0.65,
        ncol=2 if len(handles) > 12 else 1,
        columnspacing=1.0,
    )
    finalize_map(fig, ax, mapped, output_path, dpi)


def action_type_only(value: object) -> object:
    """Remove action IDs from the notebook's ``ID | type`` display field."""
    if pd.isna(value):
        return value
    actions = []
    for item in str(value).split(";"):
        actions.append(item.split("|", maxsplit=1)[-1].strip())
    return "; ".join(sorted(set(actions), key=str.casefold))


def nonzero_max(values: pd.Series) -> float:
    """Return a usable upper bound for a nonnegative continuous legend."""
    numeric = pd.to_numeric(values, errors="coerce")
    maximum = numeric.max(skipna=True)
    if pd.isna(maximum) or maximum <= 0:
        return 1.0
    return float(maximum)


def rank_included_scores(
    gdf: gpd.GeoDataFrame, value_field: str
) -> pd.Series:
    """Rank analyzed BSRs from highest to lowest, retaining tied ranks."""
    numeric = pd.to_numeric(gdf[value_field], errors="coerce")
    eligible = numeric.notna() & ~exclusion_mask(gdf)
    ranks = pd.Series(pd.NA, index=gdf.index, dtype="Int64")
    ranks.loc[eligible] = (
        numeric.loc[eligible]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )
    return ranks


def ordered_overall_risk(bsr: gpd.GeoDataFrame) -> pd.DataFrame:
    """Return analyzed BSRs in deterministic descending risk order."""
    ordered = bsr.loc[
        ~exclusion_mask(bsr),
        ["_bsr_key", "_bsr_label", "overall_risk_score"],
    ].copy()
    ordered["overall_risk_score"] = pd.to_numeric(
        ordered["overall_risk_score"], errors="coerce"
    )
    ordered = ordered.dropna(subset=["overall_risk_score"])
    ordered["overall_risk_rank"] = (
        ordered["overall_risk_score"]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )
    ordered = ordered.sort_values(
        ["overall_risk_score", "_bsr_label"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)
    ordered["overall_sort_position"] = np.arange(1, len(ordered) + 1)
    return ordered


def tier_sizes(record_count: int) -> tuple[int, int, int]:
    """Split a record count into three near-equal groups, largest first."""
    groups = np.array_split(np.arange(record_count), 3)
    return tuple(len(group) for group in groups)


def scenario_i_tiers(bsr: gpd.GeoDataFrame) -> pd.DataFrame:
    """Assign tiers from consecutive thirds of overall risk scores."""
    ordered = ordered_overall_risk(bsr)
    sizes = tier_sizes(len(ordered))
    tiers = np.empty(len(ordered), dtype=object)
    start = 0
    for tier_number, group_size in enumerate(sizes, start=1):
        stop = start + group_size
        tiers[start:stop] = f"Tier {tier_number}"
        start = stop
    ordered["tier"] = tiers
    ordered["tier_assignment_basis"] = "Overall risk score"

    assignment = bsr[
        ["_bsr_key", "_bsr_label", "overall_risk_score"]
    ].merge(
        ordered[
            [
                "_bsr_key",
                "overall_risk_rank",
                "overall_sort_position",
                "tier",
                "tier_assignment_basis",
            ]
        ],
        on="_bsr_key",
        how="left",
        validate="one_to_one",
    )
    excluded = assignment["_bsr_key"].isin(
        set(bsr.loc[exclusion_mask(bsr), "_bsr_key"])
    )
    assignment.loc[excluded, "tier"] = "Not included"
    assignment.loc[excluded, "tier_assignment_basis"] = "Excluded from analysis"
    return assignment


def scenario_ii_tiers(
    bsr: gpd.GeoDataFrame,
    species_scores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign thirds while requiring each species' top two BSRs in Tier 1."""
    ordered = ordered_overall_risk(bsr)
    sizes = tier_sizes(len(ordered))
    included_keys = set(ordered["_bsr_key"])

    species_context = species_scores.loc[
        species_scores["_bsr_key"].isin(included_keys),
        ["_bsr_key", "species", "risk_score"],
    ].copy()
    species_context["risk_score"] = pd.to_numeric(
        species_context["risk_score"], errors="coerce"
    )
    species_context = species_context.dropna(
        subset=["species", "risk_score"]
    ).merge(
        ordered[
            ["_bsr_key", "_bsr_label", "overall_risk_score"]
        ],
        on="_bsr_key",
        how="left",
        validate="many_to_one",
    )
    species_context["species_risk_rank"] = (
        species_context.groupby("species")["risk_score"]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )

    # Exact top-two selection is required to preserve the target Tier 1 size.
    # Ties are resolved by overall risk and then BSR identifier.
    species_context = species_context.sort_values(
        ["species", "risk_score", "overall_risk_score", "_bsr_label"],
        ascending=[True, False, False, True],
        kind="stable",
    )
    species_top_two = (
        species_context.groupby("species", sort=False, group_keys=False)
        .head(2)
        .copy()
    )
    species_top_two["species_selection_order"] = (
        species_top_two.groupby("species", sort=False).cumcount() + 1
    )
    species_top_two = species_top_two.sort_values(
        ["species", "species_selection_order"], kind="stable"
    ).reset_index(drop=True)
    species_top_two["scenario_ii_tier"] = "Tier 1"
    species_top_two["selection_basis"] = "Species top-two requirement"

    required_tier_one_keys = set(species_top_two["_bsr_key"])
    tier_one_target = sizes[0]
    if len(required_tier_one_keys) > tier_one_target:
        raise ValueError(
            "Scenario II cannot satisfy both requirements: the union of the "
            "top two BSRs for each species contains "
            f"{len(required_tier_one_keys)} BSRs, but the top third contains "
            f"only {tier_one_target}."
        )

    overall_order = ordered["_bsr_key"].tolist()
    tier_one_keys = set(required_tier_one_keys)
    for bsr_key in overall_order:
        if len(tier_one_keys) >= tier_one_target:
            break
        tier_one_keys.add(bsr_key)

    remaining = [
        bsr_key for bsr_key in overall_order if bsr_key not in tier_one_keys
    ]
    tier_two_keys = set(remaining[: sizes[1]])
    tier_three_keys = set(remaining[sizes[1] :])

    tier_lookup = {
        **{key: "Tier 1" for key in tier_one_keys},
        **{key: "Tier 2" for key in tier_two_keys},
        **{key: "Tier 3" for key in tier_three_keys},
    }
    required_species = (
        species_top_two.groupby("_bsr_key")["species"]
        .agg(lambda values: "; ".join(sorted(set(values), key=str.casefold)))
        .to_dict()
    )

    assignment = bsr[
        ["_bsr_key", "_bsr_label", "overall_risk_score"]
    ].copy()
    assignment["overall_risk_rank"] = assignment["_bsr_key"].map(
        ordered.set_index("_bsr_key")["overall_risk_rank"]
    ).astype("Int64")
    assignment["overall_sort_position"] = assignment["_bsr_key"].map(
        ordered.set_index("_bsr_key")["overall_sort_position"]
    ).astype("Int64")
    assignment["tier"] = assignment["_bsr_key"].map(tier_lookup)
    assignment["tier1_species_top_two_for"] = assignment["_bsr_key"].map(
        required_species
    )
    assignment["tier_assignment_basis"] = "Overall risk score"
    required = assignment["_bsr_key"].isin(required_tier_one_keys)
    assignment.loc[
        required, "tier_assignment_basis"
    ] = "Top-two species risk requirement"
    excluded = assignment["_bsr_key"].isin(
        set(bsr.loc[exclusion_mask(bsr), "_bsr_key"])
    )
    assignment.loc[excluded, "tier"] = "Not included"
    assignment.loc[excluded, "tier_assignment_basis"] = "Excluded from analysis"
    return assignment, species_top_two


def export_assignment_table(table: pd.DataFrame, output_path: Path) -> None:
    """Write a tier audit table with user-facing field names."""
    exported = table.rename(
        columns={"_bsr_label": "BSR", "_bsr_key": "bsr_join_key"}
    )
    exported.to_csv(output_path, index=False)


def create_all_maps(
    gpkg_path: str | Path = DEFAULT_GPKG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    dpi: int = 300,
    use_basemap: bool = True,
    basemap_zoom: int | str = "auto",
) -> pd.DataFrame:
    """Create all requested maps and return a file manifest.

    Parameters
    ----------
    gpkg_path:
        Scored GeoPackage written by ``atlas_integrated_scoring``.
    output_dir:
        Parent directory for map PNGs and ``map_manifest.csv``.
    dpi:
        Output resolution. The default is suitable for reports and posters.
    use_basemap:
        Use CartoDB Positron tiles. Set to ``False`` only for offline drafts.
    basemap_zoom:
        Contextily zoom level or ``"auto"``.
    """
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    bsr, species_scores, action_scores = read_scoring_outputs(gpkg_path)
    manifest: list[dict[str, object]] = []
    included_bsr_keys = set(
        bsr.loc[~exclusion_mask(bsr), "_bsr_key"].astype(str)
    )
    bsr["_overall_risk_rank"] = rank_included_scores(
        bsr, "overall_risk_score"
    )

    overall_max = nonzero_max(
        bsr.loc[~exclusion_mask(bsr), "overall_risk_score"]
    )
    overall_path = output_dir / "overall_risk.png"
    plot_numeric_map(
        bsr,
        value_field="overall_risk_score",
        title="Overall Risk",
        legend_title="Overall risk score (higher = greater risk)",
        output_path=overall_path,
        cmap=RISK_CMAP,
        scale_min=0.0,
        scale_max=overall_max,
        dpi=dpi,
        use_basemap=use_basemap,
        basemap_zoom=basemap_zoom,
        rank_field="_overall_risk_rank",
    )
    manifest.append(
        {
            "map_group": "Overall risk",
            "category": pd.NA,
            "score_field": "overall_risk_score",
            "scale_min": 0.0,
            "scale_max": overall_max,
            "output_file": str(overall_path),
        }
    )

    species_max = nonzero_max(
        species_scores.loc[
            species_scores["_bsr_key"].isin(included_bsr_keys),
            "risk_score",
        ]
    )
    species_dir = output_dir / "risk_by_species"
    for species in sorted(
        species_scores["species"].dropna().astype(str).unique(), key=str.casefold
    ):
        selected = species_scores.loc[
            species_scores["species"].astype(str).eq(species),
            ["_bsr_key", "risk_score"],
        ]
        mapped = join_one_score(bsr, selected, "risk_score")
        mapped["_species_risk_rank"] = rank_included_scores(
            mapped, "risk_score"
        )
        output_path = species_dir / f"risk_{slugify(species)}.png"
        plot_numeric_map(
            mapped,
            value_field="risk_score",
            title=f"Risk by Species: {species}",
            legend_title="Species risk score (higher = greater risk)",
            output_path=output_path,
            cmap=RISK_CMAP,
            scale_min=0.0,
            scale_max=species_max,
            dpi=dpi,
            use_basemap=use_basemap,
            basemap_zoom=basemap_zoom,
            rank_field="_species_risk_rank",
        )
        manifest.append(
            {
                "map_group": "Risk by species",
                "category": species,
                "score_field": "risk_score",
                "scale_min": 0.0,
                "scale_max": species_max,
                "output_file": str(output_path),
            }
        )

    categorical_maps = [
        (
            "highest_risk_species_life_stage",
            "Highest Risk-Aligned Life Stage",
            "Species | life stage",
            output_dir / "highest_risk_aligned_life_stage.png",
        ),
        (
            "highest_risk_limiting_factor",
            "Highest Risk-Aligned Limiting Factor",
            "Limiting factor",
            output_dir / "highest_risk_aligned_limiting_factor.png",
        ),
        (
            "_highest_risk_aligned_action_type_display",
            "Highest Risk-Aligned Benefit Action Type",
            "Action type",
            output_dir / "highest_risk_aligned_benefit_action_type.png",
        ),
    ]
    bsr["_highest_risk_aligned_action_type_display"] = bsr[
        "highest_risk_aligned_action_type"
    ].map(action_type_only)

    for field, title, legend_title, output_path in categorical_maps:
        plot_categorical_map(
            bsr,
            value_field=field,
            title=title,
            legend_title=legend_title,
            output_path=output_path,
            dpi=dpi,
            use_basemap=use_basemap,
            basemap_zoom=basemap_zoom,
        )
        manifest.append(
            {
                "map_group": title,
                "category": pd.NA,
                "score_field": field,
                "scale_min": pd.NA,
                "scale_max": pd.NA,
                "output_file": str(output_path),
            }
        )

    scenario_i = scenario_i_tiers(bsr)
    scenario_ii, species_top_two = scenario_ii_tiers(bsr, species_scores)
    bsr["_scenario_i_tier"] = bsr["_bsr_key"].map(
        scenario_i.set_index("_bsr_key")["tier"]
    )
    bsr["_scenario_ii_tier"] = bsr["_bsr_key"].map(
        scenario_ii.set_index("_bsr_key")["tier"]
    )

    scenario_i_table_path = output_dir / "bsr_tiers_scenario_i.csv"
    scenario_ii_table_path = output_dir / "bsr_tiers_scenario_ii.csv"
    species_top_two_path = output_dir / "scenario_ii_species_top_two.csv"
    export_assignment_table(scenario_i, scenario_i_table_path)
    export_assignment_table(scenario_ii, scenario_ii_table_path)
    species_top_two.rename(
        columns={
            "_bsr_label": "BSR",
            "_bsr_key": "bsr_join_key",
            "risk_score": "species_risk_score",
        }
    ).to_csv(species_top_two_path, index=False)

    tier_maps = [
        (
            "_scenario_i_tier",
            "Preliminary BSR Tiers, Scenario I: Overall Risk Thirds",
            output_dir / "preliminary_bsr_tiers_scenario_i.png",
        ),
        (
            "_scenario_ii_tier",
            "Preliminary BSR Tiers, Scenario II: Species Top-Two Requirement",
            output_dir / "preliminary_bsr_tiers_scenario_ii.png",
        ),
    ]
    for field, title, output_path in tier_maps:
        plot_categorical_map(
            bsr,
            value_field=field,
            title=title,
            legend_title="Preliminary BSR tier",
            output_path=output_path,
            dpi=dpi,
            use_basemap=use_basemap,
            basemap_zoom=basemap_zoom,
            category_color_overrides=TIER_COLORS,
        )
        manifest.append(
            {
                "map_group": title,
                "category": pd.NA,
                "score_field": field,
                "scale_min": pd.NA,
                "scale_max": pd.NA,
                "output_file": str(output_path),
            }
        )

    # Multiple action IDs can share one action type. Sum them here because the
    # requested maps are by action type, not action ID.
    action_by_type = (
        action_scores.groupby(["_bsr_key", "action_type"], as_index=False)
        .agg(action_benefit_score=("action_benefit_score", "sum"))
    )
    action_max = nonzero_max(
        action_by_type.loc[
            action_by_type["_bsr_key"].isin(included_bsr_keys),
            "action_benefit_score",
        ]
    )
    action_dir = output_dir / "benefit_by_action_type"
    for action_type in sorted(
        action_by_type["action_type"].dropna().astype(str).unique(),
        key=str.casefold,
    ):
        selected = action_by_type.loc[
            action_by_type["action_type"].astype(str).eq(action_type),
            ["_bsr_key", "action_benefit_score"],
        ]
        mapped = join_one_score(bsr, selected, "action_benefit_score")
        output_path = action_dir / f"benefit_{slugify(action_type)}.png"
        plot_numeric_map(
            mapped,
            value_field="action_benefit_score",
            title=f"Benefit Score by Action Type: {action_type}",
            legend_title="Action benefit score",
            output_path=output_path,
            cmap=BENEFIT_CMAP,
            scale_min=0.0,
            scale_max=action_max,
            dpi=dpi,
            use_basemap=use_basemap,
            basemap_zoom=basemap_zoom,
        )
        manifest.append(
            {
                "map_group": "Benefit score by action type",
                "category": action_type,
                "score_field": "action_benefit_score",
                "scale_min": 0.0,
                "scale_max": action_max,
                "output_file": str(output_path),
            }
        )

    manifest_table = pd.DataFrame(manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_table.to_csv(output_dir / "map_manifest.csv", index=False)
    return manifest_table


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpkg",
        type=Path,
        default=DEFAULT_GPKG,
        help="Scored GeoPackage from the integrated scoring notebook.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for PNG maps and map_manifest.csv.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--no-basemap",
        action="store_true",
        help="Create an offline draft without basemap tiles.",
    )
    parser.add_argument(
        "--basemap-zoom",
        default="auto",
        help="Contextily zoom level, or 'auto' (default).",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    zoom: int | str = arguments.basemap_zoom
    if zoom != "auto":
        zoom = int(zoom)
    manifest = create_all_maps(
        gpkg_path=arguments.gpkg,
        output_dir=arguments.output_dir,
        dpi=arguments.dpi,
        use_basemap=not arguments.no_basemap,
        basemap_zoom=zoom,
    )
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
