import matplotlib.patheffects as pe
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

# Global look
LABEL_FONTSIZE = 10           # <- same for both figures
EDGE_LW_COUNTRY = 0.6
EDGE_LW_HILITE  = 0.8
GREY = "#d0d0d0"
RED  = "#d9534f"

def _mainland_bounds(gdf):
    """Return tight bounds around the mainland by dropping tiny offshore polygons."""
    # Work in projected CRS so areas are in m^2
    geom_list = []
    for geom in gdf.geometry:
        if geom is None:
            continue
        if isinstance(geom, MultiPolygon):
            geom_list.extend([p for p in geom.geoms if not p.is_empty])
        elif isinstance(geom, Polygon):
            geom_list.append(geom)

    if not geom_list:
        return gdf.total_bounds

    areas = [p.area for p in geom_list]
    amax  = max(areas)
    # keep polygons larger than 0.5% of the largest (tunes out far islands)
    keep  = [p for p in geom_list if p.area >= 0.005 * amax]
    mainland = unary_union(keep) if keep else unary_union(geom_list)
    return mainland.bounds  # (minx, miny, maxx, maxy)

def plot_country_highlight(world, country_col, region_col,
                           country_name, highlight_name, out_path,
                           exclude_region_substrings=None,
                           epsg=None, figsize=(6, 10),
                           label_text=None, label_fontsize=LABEL_FONTSIZE,
                           dpi=600, also_pdf=True):
    import numpy as np
    import matplotlib as mpl

    # Embed fonts in vector outputs
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["svg.fonttype"] = "none"

    g = world.copy()
    g = g[g[country_col].str.lower() == country_name.lower()].copy()

    # Optional coarse name-based exclusions (Ross, Chatham, Antarctica, etc.)
    if exclude_region_substrings:
        mask = np.zeros(len(g), dtype=bool)
        for s in exclude_region_substrings:
            mask |= g[region_col].str.contains(s, case=False, na=False)
        g = g[~mask].copy()

    # Highlight region (exact → contains)
    hi = g[g[region_col].str.lower() == highlight_name.lower()]
    if hi.empty:
        hi = g[g[region_col].str.contains(highlight_name, case=False, na=False)]
    if hi.empty:
        raise SystemExit(f"Could not find '{highlight_name}' in {country_name}.")

    # Local CRS for nice shapes
    if epsg:
        g  = g.to_crs(epsg=epsg)
        hi = hi.to_crs(epsg=epsg)

    # Center: bounds around the mainland only
    minx, miny, maxx, maxy = _mainland_bounds(g)
    pad_x = (maxx - minx) * 0.025
    pad_y = (maxy - miny) * 0.025

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    g.plot(ax=ax, color=GREY, edgecolor="white", linewidth=EDGE_LW_COUNTRY)
    hi.plot(ax=ax, color=RED,  edgecolor="white", linewidth=EDGE_LW_HILITE)

    # Single label placed inside the highlighted region
    label = label_text if label_text else str(hi.iloc[0][region_col])
    pt = hi.geometry.unary_union.representative_point()
    ax.text(
        pt.x, pt.y, label,
        fontsize=label_fontsize, ha="center", va="center", color="#222222",
        path_effects=[pe.withStroke(linewidth=2, foreground="white")]
    )

    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)
    ax.set_aspect("equal")
    ax.set_axis_off()

    plt.tight_layout()
    # 600-dpi PNG
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    if also_pdf:
        base, _ = os.path.splitext(out_path)
        plt.savefig(base + ".pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved: {out_path} {'and ' + base + '.pdf' if also_pdf else ''}")


# Chile — VII Del Maule Region
plot_country_highlight(
    world, country_col, region_col,
    country_name="Chile",
    highlight_name="Maule",
    out_path=os.path.join(output_dir, "Maule_map.png"),
    exclude_region_substrings=["antarc", "antártica", "easter", "pascua", "rapa", "juan fernández", "san félix", "san ambrosio"],
    epsg=32719,                   # UTM 19S
    figsize=(6, 10),
    label_text="VII Del Maule Region",
    label_fontsize=LABEL_FONTSIZE,
    dpi=600,
)

# New Zealand — Canterbury
plot_country_highlight(
    world, country_col, region_col,
    country_name="New Zealand",
    highlight_name="Canterbury",
    out_path=os.path.join(output_dir, "Canterbury_map.png"),
    exclude_region_substrings=["ross", "chatham"],
    epsg=2193,                    # NZTM2000
    figsize=(6, 10),
    label_text="Canterbury",
    label_fontsize=LABEL_FONTSIZE,
    dpi=600,
)
