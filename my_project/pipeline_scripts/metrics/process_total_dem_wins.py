import jsonlines as jl
from gerrychain import Graph
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
import numpy as np
import geopandas as gpd
from pathlib import Path
from pyben import PyBenDecoder
import os

script_dir = Path(__file__).parent
top_dir = script_dir.parents[1]


def compute_score(
    sample_idx,
    assignment_vector,
    dem_count_matrix,
    rep_count_matrix,
):
    assignment = np.asarray(assignment_vector, dtype=np.int32)
    race_totals = {name: 0 for name in race_names}

    for part in np.unique(assignment):
        mask = assignment == part
        dem_totals = dem_count_matrix[mask].sum(axis=0)
        rep_totals = rep_count_matrix[mask].sum(axis=0)
        dem_wins = dem_totals > rep_totals

        for i, race in enumerate(race_names):
            race_totals[race] += 1 if dem_wins[i] else 0

    return ({"sample": sample_idx, "scores": race_totals},)


if __name__ == "__main__":
    batch_size = 10_000

    CHAIN_FILE = f"{top_dir}/chain_outputs/MN_chain_100000_steps_seed42.jsonl.ben"
    GRAPH_PATH = f"{top_dir}/JSON_dualgraphs/MN_precincts.geojson"
    OUTPUT_PATH = f"{top_dir}/stats/MN_dem_win_scores.jsonl"

    decoder = PyBenDecoder(CHAIN_FILE)
    n_samples = len(decoder)
    samples = list(range(1, n_samples + 1))

    df = gpd.read_file(GRAPH_PATH)
    graph = Graph.from_geodataframe(df)
    df.drop(columns=["geometry"], inplace=True)

    race_names = [
        "PRES16",
        "SSEN16",
    ]

    dem_rep_pairs = [(f"{name}D", f"{name}R") for name in race_names]
    dem_count_matrix = df[[pair[0] for pair in dem_rep_pairs]].to_numpy()
    rep_count_matrix = df[[pair[1] for pair in dem_rep_pairs]].to_numpy()

    all_scores = []
    n_batches = (len(samples) + batch_size - 1) // batch_size
    for batch_no in range(n_batches):
        current_batch = samples[batch_no * batch_size : (batch_no + 1) * batch_size]
        vectors = list(decoder.subsample_indices(current_batch))

        pairs = list(zip(current_batch, vectors))

        with joblib_progress(
            description=f"Computing total dem wins in parallel (batch {batch_no + 1}/{n_batches})",
            total=len(pairs),
        ):
            scores = Parallel(
                n_jobs=os.cpu_count() or 1,
            )(
                delayed(compute_score)(idx, vec, dem_count_matrix, rep_count_matrix)
                for idx, vec in pairs
            )

        all_scores.extend(scores)

    with jl.open(OUTPUT_PATH, "w") as writer:
        writer.write_all(all_scores)
