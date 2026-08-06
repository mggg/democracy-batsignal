import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
from gerrychain import Graph
from gerrytools.plotting import GeoPlot, LabelOptions

ROOT_DIR = Path(__file__).resolve().parents[2]
FIGURES_DIR = ROOT_DIR / "figures" / "plan_maps"

PLAN_COLUMN = "district"
ALTERNATE_PLAN_COLUMN = "alternate_district"
PARTISAN_COLUMN = "pres_20_dem_share"
PHILADELPHIA_VTD_PREFIX = "vtd:42101"


def load_pa_geodata() -> gpd.GeoDataFrame:
    """Loads the PA geometry and adds enacted-plan and partisan columns.

    Returns:
        gpd.GeoDataFrame: PA voting districts in a projected CRS.

    Raises:
        ValueError: If the graph and geometry rows do not describe the same voting districts.
    """
    graph = Graph.from_json(str(ROOT_DIR / "JSON_dualgraphs" / "pa_dualgraph.json"))
    gdf = gpd.read_parquet(ROOT_DIR / "data" / "pa_gdf.parquet").to_crs("EPSG:5070")

    graph_paths = pd.Series({node: graph.node_data(node)["path"] for node in graph.nodes})
    if not graph_paths.reindex(gdf.index).equals(gdf["path"]):
        raise ValueError("PA graph nodes and geometry rows are not aligned")

    seed_plan = pd.Series({node: graph.node_data(node)["seed_plan"] for node in graph.nodes})
    gdf[PLAN_COLUMN] = seed_plan.reindex(gdf.index).add(1).astype(int)

    graph_nodes = list(graph.nodes)

    # A rustrecom plan is an assignment vector in graph-node order rather than a keyed mapping.
    alternate_plan = json.loads((ROOT_DIR / "data" / "alt_plan_pa.json").read_text())
    if len(alternate_plan) != len(graph_nodes):
        raise ValueError("Alternate plan length does not match the PA graph")
    gdf[ALTERNATE_PLAN_COLUMN] = (
        pd.Series(alternate_plan, index=graph_nodes).reindex(gdf.index).add(1).astype(int)
    )

    two_party_votes = gdf["pres_20_dem"] + gdf["pres_20_rep"]
    gdf[PARTISAN_COLUMN] = gdf["pres_20_dem"].div(two_party_votes.where(two_party_votes > 0))
    return gdf


def save_philadelphia_plan_map(gdf: gpd.GeoDataFrame, output_path: Path) -> None:
    """Saves a labeled Philadelphia County enacted-plan map.

    Args:
        gdf: Philadelphia geometry and enacted district assignments to plot.
        output_path: Destination PNG path.
    """
    plot = GeoPlot(gdf)
    plot.add_districting_plan_layer(
        PLAN_COLUMN,
        dissolve=True,
        show_labels=True,
        label_style="halo",
        edgecolor="black",
        edgewidth=0.6,
    )
    plot.save(str(output_path))


def save_state_plan_map(gdf: gpd.GeoDataFrame, output_path: Path) -> None:
    """Saves a labeled statewide enacted-plan map.

    Args:
        gdf: Geometry and enacted district assignments to plot.
        output_path: Destination PNG path.
    """
    # Set a statewide default, then override labels that crowd around Philadelphia.
    label_sizes = {district: 6 for district in gdf[PLAN_COLUMN].unique()} | {
        2: 2.5,
        3: 3,
        5: 3,
        7: 3,
        11: 2.5,
        14: 3,
        26: 3,
        29: 2.5,
        36: 3,
        37: 3,
        38: 3,
        41: 3,
    }

    plot = GeoPlot(gdf)
    plot.add_districting_plan_layer(
        PLAN_COLUMN,
        dissolve=True,
        show_labels=True,
        label_options=LabelOptions(label_style="halo", fontsize=label_sizes),
        edgecolor="black",
        edgewidth=0.6,
    )
    plot.save(str(output_path))


def save_state_partisan_choropleth(gdf: gpd.GeoDataFrame, output_path: Path) -> None:
    """Saves a statewide presidential two-party vote-share choropleth.

    Args:
        gdf: Geometry and election results to plot.
        output_path: Destination PNG path.
    """

    label_sizes = {district: 6 for district in gdf[PLAN_COLUMN].unique()} | {
        2: 2.5,
        3: 3,
        5: 3,
        7: 3,
        11: 2.5,
        14: 3,
        26: 3,
        29: 2.5,
        36: 3,
        37: 3,
        38: 3,
        41: 3,
    }

    plot = GeoPlot(gdf)
    plot.add_choropleth_layer(
        PARTISAN_COLUMN,
        colormap="RdBu",
        vmin=0,
        vmax=1,
        show_colorbar=True,
        colorbar_label="Democratic two-party vote share",
    )
    plot.add_outline_layer(
        geo_source=gdf,
        dissolve_column=PLAN_COLUMN,
        edgecolor="black",
        edgewidth=0.4,
        label_options=LabelOptions(label_style="badge", fontsize=label_sizes),
        show_labels=True,
    )
    plot.save(str(output_path))


def save_philadelphia_partisan_choropleth(
    gdf: gpd.GeoDataFrame,
    output_path: Path,
) -> None:
    """Saves a Philadelphia County presidential vote-share choropleth.

    Args:
        gdf: Philadelphia geometry and election results to plot.
        output_path: Destination PNG path.
    """
    import matplotlib.pyplot as plt

    # Use an external Matplotlib axes because the district 37 leader lines are ordinary
    # Matplotlib artists added after GerryTools renders its geographic layers.
    figure, ax = plt.subplots(dpi=300)

    plot = GeoPlot(gdf)
    plot.add_choropleth_layer(
        PARTISAN_COLUMN,
        colormap="RdBu",
        vmin=0,
        vmax=1,
        show_colorbar=True,
        colorbar_label="Democratic two-party vote share",
    )
    plot.add_outline_layer(
        geo_source=gdf,
        dissolve_column=PLAN_COLUMN,
        edgecolor="grey",
        edgewidth=1,
        show_labels=True,
        label_options=LabelOptions(
            label_style="badge",
            fontsize={36: 5, 37: 5, 38: 6, 41: 5},
            # Position offsets use the projected geometry's coordinate units (meters here).
            adjustments={
                11: (-1_200, 400),
                26: (2000, 0),
                29: (200, -500),
                36: (1500, 1500),
                37: (-700, 600),
                38: (-300, -200),
                41: (300, -200),
            },
        ),
    )
    # Binding renders the queued layers onto ax and establishes the projected map limits before
    # custom Matplotlib artists are added.
    plot.bind_to_ax(ax)

    # District 37 is a MultiPolygon in Philadelphia. Its displaced badge needs one leader line to
    # an interior point in each disconnected piece.
    district_37 = gdf[gdf[PLAN_COLUMN].eq(37)]
    geometry = district_37.dissolve().geometry.iloc[0]

    # Reproduce the same offset used by LabelOptions so the lines start beneath the badge center.
    label_base = geometry.representative_point()
    label_position = (
        label_base.x - 700,
        label_base.y + 600,
    )

    for piece in geometry.geoms:
        # representative_point() is guaranteed to fall inside the piece; a centroid may not.
        target = piece.representative_point()
        ax.plot(
            [label_position[0], target.x],
            [label_position[1], target.y],
            color="black",
            linewidth=0.6,
            # District labels render above this layer, hiding the lines underneath the badge.
            zorder=2.5,
        )
    figure.savefig(str(output_path), bbox_inches="tight", dpi=300)
    plt.close(figure)


def save_recom_district_comparison(gdf: gpd.GeoDataFrame, output_path: Path) -> None:
    """Saves a green-grey comparison of one district changed by a ReCom step.

    Args:
        gdf: Geometry with starting and alternate district assignments.
        output_path: Destination PNG path.

    Raises:
        ValueError: If the alternate plan does not change any assignments.
    """

    changed = pd.Series(
        gdf[PLAN_COLUMN].ne(gdf[ALTERNATE_PLAN_COLUMN]),
        index=gdf.index,
        dtype=bool,
    )

    if not bool(changed.any()):
        raise ValueError("Alternate plan is identical to the starting plan")

    # Select one of the two recombined districts automatically instead of hard-coding its label.
    changed_districts = sorted(
        set(gdf.loc[changed, PLAN_COLUMN]) | set(gdf.loc[changed, ALTERNATE_PLAN_COLUMN])
    )
    district = changed_districts[0]
    dist_in_starting_plan = gdf[PLAN_COLUMN].eq(district)
    dist_in_alternate_plan = gdf[ALTERNATE_PLAN_COLUMN].eq(district)

    starting_plan_dist = gdf.loc[dist_in_starting_plan].copy()
    alternate_plan_dist = gdf.loc[dist_in_alternate_plan].copy()

    # PA path identifiers begin with a five-digit county FIPS code after the "vtd:" prefix.
    gdf["county"] = gdf["path"].str.split(":").str[1].str[:5]

    plot = GeoPlot(gdf)
    # Highlight layers are drawn in order. Grey shows the starting footprint; green overlays the
    # alternate footprint.
    plot.add_highlight_layer(
        geo_source=starting_plan_dist,
        facecolor="default_grey",
    )
    plot.add_highlight_layer(
        geo_source=alternate_plan_dist,
        facecolor="green",
    )
    plot.add_outline_layer(dissolve_column="county")

    # geometry_mask controls the view limits only; it does not filter or style plotted rows.
    plot.focus_axes(geometry_mask=dist_in_starting_plan | dist_in_alternate_plan, pad=0.5)

    plot.save(str(output_path))


def main() -> None:
    """Generates basic enacted-plan maps for Pennsylvania."""
    gdf = load_pa_geodata()

    # Rotate every geometry around one shared state center. Rotating after projection treats the
    # state as a rigid object; rotating longitude and latitude coordinates would distort it.
    minx, miny, maxx, maxy = gdf.total_bounds
    center = ((minx + maxx) / 2, (miny + maxy) / 2)

    gdf.geometry = gdf.geometry.rotate(-10, origin=center)

    philadelphia = gpd.GeoDataFrame(gdf[gdf["path"].str.startswith(PHILADELPHIA_VTD_PREFIX)].copy())
    if philadelphia.empty:
        raise ValueError("No Philadelphia County voting districts were found")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / "pa_original_plan.png"
    save_state_plan_map(gdf, output_path)
    print(f"Saved to '{output_path}'")

    output_path = FIGURES_DIR / "pa_original_plan_philadelphia.png"
    save_philadelphia_plan_map(philadelphia, output_path)
    print(f"Saved to '{output_path}'")

    output_path = FIGURES_DIR / "pa_pres_20_partisan_choropleth.png"
    save_state_partisan_choropleth(gdf, output_path)
    print(f"Saved to '{output_path}'")

    output_path = FIGURES_DIR / "pa_pres_20_partisan_choropleth_philadelphia.png"
    save_philadelphia_partisan_choropleth(
        philadelphia,
        output_path,
    )
    print(f"Saved to '{output_path}'")

    output_path = FIGURES_DIR / "pa_recom_district_comparison.png"
    save_recom_district_comparison(gdf, output_path)
    print(f"Saved to '{output_path}'")


if __name__ == "__main__":
    main()
