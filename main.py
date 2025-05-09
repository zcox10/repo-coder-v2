import os
from typing import List
import logging

# Disable parallel tokenization for HuggingFace
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.build_windows.windows import Windows
from src.build_embeddings.embeddings import Embeddings
from src.build_retrievals.retrievals import Retrievals
from src.build_prompts.prompts import Prompts
from src.build_predictions.predictions import Predictions
from src.build_scores.scores import Scores

from src.models.codegen_model import CodeGenModel
from src.utils.constants import Constants


def main(
    model_name: str,
    benchmarks: List[str],
    repos: List[str],
    window_sizes: List[int],
    slice_sizes: List[int],
    max_top_k: int,
    vector_type: str,
    max_new_tokens: int,
    batch_size: int,
):
    model = CodeGenModel(model_name)

    # Define modes (rg -> vanilla RAG baseline, gt -> Ground Truth)
    baseline_modes = [Constants.rg, Constants.gt]

    # Initialize helper classes
    embeddings = Embeddings(tokenizer=model)
    windows = Windows()
    sim_search = Retrievals()
    prompts = Prompts(tokenizer=model)
    codegen = Predictions(model=model)
    score = Scores()

    # 1. Build windows for all repositories according to the provided window and slice size
    # Ouptut each repo (window) at: FilePathBuilder.create_repo_window_path()
    windows.build_repo_windows(repos=repos, window_sizes=window_sizes, slice_sizes=slice_sizes)

    # 2. Generate embeddings for all repository context windows
    # Ouptut each repo (embeddings) at: FilePathBuilder.create_repo_window_path() -> FilePathBuilder.create_one_gram_vector_path()
    embeddings.build_repo_window_embeddings(
        repos=repos, window_sizes=window_sizes, slice_sizes=slice_sizes
    )

    # 3. Builds task-based windows for both the baseline (RG1) and oracle (GT) methods.
    # Ouptut each repo (window) at: FilePathBuilder.create_search_window_path()
    windows.build_baseline_windows(
        benchmarks=benchmarks,
        modes=baseline_modes,
        repos=repos,
        window_sizes=window_sizes,
        slice_sizes=slice_sizes,
    )

    # 4. Build embeddings from baseline and ground truth windows
    # Ouptut each repo (embeddings) at: FilePathBuilder.create_search_window_path() -> FilePathBuilder.create_one_gram_vector_path()
    embeddings.build_baseline_window_embeddings(
        benchmarks=benchmarks,
        modes=baseline_modes,
        repos=repos,
        window_sizes=window_sizes,
        slice_sizes=slice_sizes,
    )

    # 5. Perform similarity search between a query's context and all embeddings in a repo
    # Output retrieved embeddings at:  FilePathBuilder.create_search_window_path() -> FilePathBuilder.create_retrieval_results_path()
    sim_search.retrieve_similar_embeddings(
        vectorizer=vector_type,
        max_top_k=max_top_k,
        benchmarks=benchmarks,
        modes=baseline_modes,
        repos=repos,
        window_sizes=window_sizes,
        slice_sizes=slice_sizes,
        generation_type="baseline",
    )

    # 6. Build prompts for baseline (RG1 and GT)
    # Output prompts at: FilePathBuilder.create_prompts_path()
    prompts.build_prompts(
        vectorizer=vector_type,
        max_top_k=max_top_k,
        benchmarks=benchmarks,
        modes=baseline_modes,
        repos=repos,
        window_sizes=window_sizes,
        slice_sizes=slice_sizes,
        generation_type="baseline",
        max_new_tokens=max_new_tokens,
    )

    # 7. Generate predictions from prompts
    # Output predictions at: FilePathBuilder.create_predictions_path()
    codegen.create_predictions(
        model_name=model_name,
        vectorizer=vector_type,
        benchmarks=benchmarks,
        modes=baseline_modes,
        window_sizes=window_sizes,
        slice_sizes=slice_sizes,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )

    # 8. Generate score from predictions
    # Output scores at: FilePathBuilder.create_scores_path()
    score.build_scores(
        model_name=model_name,
        vectorizer=vector_type,
        benchmarks=benchmarks,
        modes=baseline_modes,
        window_sizes=window_sizes,
        slice_sizes=slice_sizes,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="\n%(levelname)s - %(message)s")

    # Repositories to evaluate
    repos = [
        "huggingface_diffusers",
        "nerfstudio-project_nerfstudio",
        "awslabs_fortuna",
        "huggingface_evaluate",
        "google_vizier",
        "alibaba_FederatedScope",
        "pytorch_rl",
        "opendilab_ACE",
    ]

    main(
        model_name="Salesforce/codegen-350M-mono",
        benchmarks=[
            Constants.short_api_benchmark,
            Constants.short_line_benchmark,
            Constants.api_benchmark,
            Constants.line_benchmark,
        ],
        repos=repos,
        window_sizes=[20],
        slice_sizes=[2],
        max_top_k=10,
        vector_type="one-gram",
        max_new_tokens=100,
        batch_size=16,
    )
