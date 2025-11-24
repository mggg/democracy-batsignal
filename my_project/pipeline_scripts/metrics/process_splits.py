import jsonlines as jl
from gerrychain import Graph
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
import numpy as np
from pathlib import Path
from pyben import PyBenDecoder
import os

script_dir = Path(__file__).parent
top_dir = script_dir.parents[1]


def compute_score(
    sample_idx,
    assignment_vector,
    u,
    v,
    same_county_mask,
    county_names,
):
    assignment = np.asarray(assignment_vector, dtype=np.int32)
    cut = assignment[u] != assignment[v]
    cut_edges = int(cut.sum())
    county_splits = len(set(county_names[cut & same_county_mask]))
    return {
        "sample": sample_idx,
        "scores": {
            "county_splits": county_splits,
            "cut_edges": cut_edges,
        },
    }


if __name__ == "__main__":
    batch_size = 10_000

    CHAIN_FILE = f"{top_dir}/chain_outputs/MN_chain_100000_steps_seed42.jsonl.ben"
    GRAPH_PATH = f"{top_dir}/JSON_dualgraphs/MN_precincts.geojson"
    OUTPUT_PATH = f"{top_dir}/stats/MN_split_scores.jsonl"

    decoder = PyBenDecoder(CHAIN_FILE)
    n_samples = len(decoder)
    samples = list(range(1, n_samples + 1))

    graph = Graph.from_file(GRAPH_PATH)
    edges = np.asarray(list(graph.edges()), dtype=np.int64)
    u = edges[:, 0]
    v = edges[:, 1]

    # Masks that depend only on the graph (not on assignments)
    # A "split" counts only when it's a cut edge AND both endpoints share the same county/place.
    same_county_mask = np.fromiter(
        (
            graph.nodes[int(a)]["COUNTYNAME"] == graph.nodes[int(b)]["COUNTYNAME"]
            for a, b in edges
        ),
        dtype=bool,
        count=len(edges),
    )

    # Don't need to record b since we are going to filter to when it has the same value as a
    county_names = np.fromiter(
        (graph.nodes[int(a)]["COUNTYNAME"] for a, _ in edges),
        dtype=object,
        count=len(edges),
    )

    all_scores = []
    n_batches = (len(samples) + batch_size - 1) // batch_size
    for batch_no in range(n_batches):
        current_batch = samples[batch_no * batch_size : (batch_no + 1) * batch_size]
        vectors = list(decoder.subsample_indices(current_batch))

        pairs = list(zip(current_batch, vectors))

        with joblib_progress(
            description=f"Computing all split scores in parallel (batch {batch_no + 1}/{n_batches})",
            total=len(pairs),
        ):
            scores = Parallel(
                n_jobs=os.cpu_count() or 1,
            )(
                delayed(compute_score)(idx, vec, u, v, same_county_mask, county_names)
                for idx, vec in pairs
            )

        all_scores.extend(scores)

    with jl.open(OUTPUT_PATH, "w") as writer:
        writer.write_all(all_scores)
