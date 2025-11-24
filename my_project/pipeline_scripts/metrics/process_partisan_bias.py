import jsonlines as jl
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
from pathlib import Path
import geopandas as gpd
import numpy as np
from pyben import PyBenDecoder
import os

script_dir = Path(__file__).parent
top_dir = script_dir.parents[1]


def compute_score(sample_idx, assignment_vector, vote_arrays):
    assign = np.asarray(assignment_vector, dtype=np.int64)

    k = int(assign.max())  # number of districts

    out = {"sample": sample_idx, "pb_scores": {}}

    for dem_votes, rep_votes, name in vote_arrays:
        dem_tot = np.bincount(assign, weights=dem_votes, minlength=k)
        rep_tot = np.bincount(assign, weights=rep_votes, minlength=k)

        total = dem_tot + rep_tot
        dem_share = np.divide(
            dem_tot, total, out=np.zeros_like(dem_tot, dtype="float64"), where=total > 0
        )

        mean_share = dem_share.mean()
        pb = (dem_share > mean_share).sum() / k - 0.5
        out["pb_scores"][name] = float(pb)

    return out


if __name__ == "__main__":
    batch_size = 10_000

    CHAIN_FILE = f"{top_dir}/chain_outputs/MN_chain_100000_steps_seed42.jsonl.ben"
    GRAPH_PATH = f"{top_dir}/JSON_dualgraphs/MN_precincts.geojson"
    OUTPUT_PATH = f"{top_dir}/stats/MN_partisan_bias_scores.jsonl"

    decoder = PyBenDecoder(CHAIN_FILE)
    n_samples = len(decoder)
    samples = list(range(1, n_samples + 1))

    gdf = gpd.read_file(GRAPH_PATH)

    elections = ["PRES16", "SSEN16"]
    election_pairs = [(f"{name}D", f"{name}R") for name in elections]

    # grab vote columns as numpy arrays once
    vote_arrays = []
    for d_col, r_col in election_pairs:
        vote_arrays.append(
            (
                gdf[d_col].to_numpy(dtype="float64", copy=True),
                gdf[r_col].to_numpy(dtype="float64", copy=True),
                d_col[:-1],
            )
        )
    all_scores = []
    n_batches = (len(samples) + batch_size - 1) // batch_size
    for batch_no in range(n_batches):
        current_batch = samples[batch_no * batch_size : (batch_no + 1) * batch_size]
        vectors = list(decoder.subsample_indices(current_batch))

        pairs = list(zip(current_batch, vectors))

        with joblib_progress(
            description=f"Computing partisan bias (batch {batch_no+1}/{n_batches})",
            total=len(pairs),
        ):
            scores = Parallel(
                n_jobs=os.cpu_count() or 1,
            )(delayed(compute_score)(idx, vec, vote_arrays) for idx, vec in pairs)

        all_scores.extend(scores)

    with jl.open(OUTPUT_PATH, "w") as writer:
        writer.write_all(all_scores)
