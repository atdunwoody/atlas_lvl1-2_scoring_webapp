"""Create publication-ready maps from the Atlas scoring GeoPackage.

The scoring notebook writes a scored BSR feature layer plus the
``species_scores`` and ``action_type_scores`` attribute tables to
``data/outputs/bsr_scores.gpkg``. This script reads those outputs and creates:

* overall risk;
* one risk map per species;
* highest risk-aligned species/life stage;
* highest risk-aligned limiting factor;
* highest risk-aligned benefit action type; and
* one benefit-score map per action type.

All maps include a CartoDB Positron basemap, BSR outlines and labels, a north
arrow, a ground-distance scale bar, and a legend. Species maps share one color
scale; action-type benefit maps share a second color scale so maps within each
group can be compared directly.

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

RISK_CMAP = LinearSegmentedColormap.from_list(
    "pastel_risk",
    ["#f6f1e8", "#d8ebe4", "#b5d9d1", "#8fc2bd", "#69a5aa"],
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


def add_bsr_labels(ax: plt.Axes, gdf: gpd.GeoDataFrame) -> None:
    """Label each BSR at an interior point using haloed map text."""
    if "_bsr_label" not in gdf.columns:
        raise ValueError("Map data are missing the internal BSR label field.")

    label_points = gdf.geometry.representative_point()
    polygon_count = len(gdf)
    font_size = max(6.5, min(9.5, 10.0 - polygon_count * 0.035))

    for label, point in zip(gdf["_bsr_label"], label_points):
        if pd.isna(label) or point.is_empty:
            continue
        ax.text(
            point.x,
            point.y,
            str(label),
            ha="center",
            va="center",
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


def finalize_map(
    fig: plt.Figure,
    ax: plt.Axes,
    gdf: gpd.GeoDataFrame,
    output_path: Path,
    dpi: int,
) -> None:
    """Add common map furniture and write a PNG."""
    add_bsr_labels(ax, gdf)
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
) -> None:
    """Plot a continuous score map with a color-bar legend."""
    mapped = gdf.copy()
    mapped[value_field] = pd.to_numeric(mapped[value_field], errors="coerce")
    valid = mapped[value_field].notna()

    fig, ax = make_axes(mapped, title, use_basemap, basemap_zoom)
    if (~valid).any():
        mapped.loc[~valid].plot(
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

    if (~valid).any():
        ax.legend(
            handles=[Patch(facecolor=NO_DATA_COLOR, edgecolor=BOUNDARY_COLOR)],
            labels=["No data"],
            loc="lower right",
            frameon=True,
            framealpha=0.9,
            fontsize=8,
        )
    finalize_map(fig, ax, mapped, output_path, dpi)


def plot_categorical_map(
    gdf: gpd.GeoDataFrame,
    value_field: str,
    title: str,
    legend_title: str,
    output_path: Path,
    dpi: int,
    use_basemap: bool,
    basemap_zoom: int | str,
) -> None:
    """Plot a categorical map with a discrete pastel legend."""
    mapped = gdf.copy()
    valid = mapped[value_field].notna() & mapped[value_field].astype(str).str.strip().ne("")
    categories = sorted(
        mapped.loc[valid, value_field].astype(str).unique(), key=str.casefold
    )
    category_colors = dict(zip(categories, pastel_colors(len(categories))))

    fig, ax = make_axes(mapped, title, use_basemap, basemap_zoom)
    for category in categories:
        selected = mapped[value_field].astype(str).eq(category)
        mapped.loc[selected].plot(
            ax=ax,
            color=category_colors[category],
            alpha=0.86,
            edgecolor=BOUNDARY_COLOR,
            linewidth=0.55,
            zorder=3,
        )
    if (~valid).any():
        mapped.loc[~valid].plot(
            ax=ax,
            color=NO_DATA_COLOR,
            edgecolor=BOUNDARY_COLOR,
            linewidth=0.55,
            zorder=2,
        )

    handles = [
        Patch(
            facecolor=category_colors[category],
            edgecolor=BOUNDARY_COLOR,
            label=textwrap.fill(category, width=38),
        )
        for category in categories
    ]
    if (~valid).any():
        handles.append(
            Patch(
                facecolor=NO_DATA_COLOR,
                edgecolor=BOUNDARY_COLOR,
                label="No data",
            )
        )
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
    bsr, species_scores, action_scores = read_scoring_outputs(gpkg_path)
    manifest: list[dict[str, object]] = []

    overall_max = nonzero_max(bsr["overall_risk_score"])
    overall_path = output_dir / "overall_risk.png"
    plot_numeric_map(
        bsr,
        value_field="overall_risk_score",
        title="Overall Risk",
        legend_title="Overall risk score",
        output_path=overall_path,
        cmap=RISK_CMAP,
        scale_min=0.0,
        scale_max=overall_max,
        dpi=dpi,
        use_basemap=use_basemap,
        basemap_zoom=basemap_zoom,
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

    species_max = nonzero_max(species_scores["risk_score"])
    species_dir = output_dir / "risk_by_species"
    for species in sorted(
        species_scores["species"].dropna().astype(str).unique(), key=str.casefold
    ):
        selected = species_scores.loc[
            species_scores["species"].astype(str).eq(species),
            ["_bsr_key", "risk_score"],
        ]
        mapped = join_one_score(bsr, selected, "risk_score")
        output_path = species_dir / f"risk_{slugify(species)}.png"
        plot_numeric_map(
            mapped,
            value_field="risk_score",
            title=f"Risk by Species: {species}",
            legend_title="Species risk score",
            output_path=output_path,
            cmap=RISK_CMAP,
            scale_min=0.0,
            scale_max=species_max,
            dpi=dpi,
            use_basemap=use_basemap,
            basemap_zoom=basemap_zoom,
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

    # Multiple action IDs can share one action type. Sum them here because the
    # requested maps are by action type, not action ID.
    action_by_type = (
        action_scores.groupby(["_bsr_key", "action_type"], as_index=False)
        .agg(action_benefit_score=("action_benefit_score", "sum"))
    )
    action_max = nonzero_max(action_by_type["action_benefit_score"])
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
