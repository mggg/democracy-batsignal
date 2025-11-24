import jsonlines as jl
from gerrychain import GeographicPartition, Graph
from gerrychain.metrics import polsby_popper
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
import numpy as np
from pathlib import Path
from pyben import PyBenDecoder
import os

script_dir = Path(__file__).parent
top_dir = script_dir.parents[1]


def compute_score(sample_number, assignment_vector, graph):
    part = GeographicPartition(
        graph, assignment={i: val for i, val in enumerate(assignment_vector)}
    )
    return {"sample": sample_number, "scores": polsby_popper(part)}


if __name__ == "__main__":
    batch_size = 1000
    n_samples = 10_000

    CHAIN_FILE = f"{top_dir}/chain_outputs/MN_chain_100000_steps_seed42.jsonl.ben"
    GRAPH_PATH = f"{top_dir}/JSON_dualgraphs/MN_precincts.geojson"
    OUTPUT_PATH = f"{top_dir}/stats/MN_polsby_scores.jsonl"

    decoder = PyBenDecoder(CHAIN_FILE)
    total_chain_length = len(decoder)

    if n_samples > total_chain_length:
        print(
            "Requested more samples than available in chain; using full chain length."
        )
        n_samples = total_chain_length

    subsamples = sorted(
        map(
            int, np.random.choice(total_chain_length, size=n_samples, replace=False) + 1
        )
    )  # +1 for 1-based indexing

    graph = Graph.from_file(GRAPH_PATH)

    all_scores = []
    n_batches = (len(subsamples) + batch_size - 1) // batch_size
    for batch_no in range(n_batches):
        current_batch = subsamples[batch_no * batch_size : (batch_no + 1) * batch_size]
        vectors = list(decoder.subsample_indices(current_batch))

        pairs = list(zip(current_batch, vectors))

        with joblib_progress(
            description=f"Computing all Polsby-Popper scores in parallel (batch {batch_no + 1}/{n_batches})",
            total=len(pairs),
        ):
            scores = Parallel(
                n_jobs=os.cpu_count() or 1,
            )(delayed(compute_score)(idx, vec, graph) for idx, vec in pairs)

        all_scores.extend(scores)

    with jl.open(OUTPUT_PATH, "w") as writer:
        writer.write_all(all_scores)
