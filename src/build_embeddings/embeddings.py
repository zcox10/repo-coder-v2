import itertools
from typing import Callable, List

from src.utils.file_path_builder import FilePathBuilder


class Embeddings:
    """
    Coordinates embeddings of context windows across multiple repos, benchmarks,
    and window/slice configurations using a given embedding model (e.g., CodeGenModel).

    Args:
        tokenizer: An model to generate embeddings for a provided context
    """

    def __init__(self, tokenizer: Callable[[str], object]):
        self.tokenizer = tokenizer

    def build_repo_window_embeddings(
        self, repos: List[str], window_sizes: List[int], slice_sizes: List[int]
    ) -> None:
        """
        Vectorizes context windows created directly from repository source code.

        Args:
            repos: List of repository names
            window_sizes: List of context window sizes
            slice_sizes: List of stride sizes
        """
        for repo, window_size, slice_size in itertools.product(repos, window_sizes, slice_sizes):
            path = FilePathBuilder.create_repo_window_path(repo, window_size, slice_size)
            self.tokenizer.build_embeddings(path)

    def build_baseline_window_embeddings(
        self,
        benchmarks: List[str],
        modes: List[str],
        repos: List[str],
        window_sizes: List[int],
        slice_sizes: List[int],
    ) -> None:
        """
        Generates embeddings for context windows for every permutation of:
            - Benchmark (e.g., `short_line` or `short_api`)
            - Mode (e.g., `rg` or `gt`)
            - Repo (e.g., huggingface_diffusers)
            - Window_size (e.g., 20)
            - Slice_size (e.g., 2)
        """
        for benchmark, mode, repo, window_size, slice_size in itertools.product(
            benchmarks, modes, repos, window_sizes, slice_sizes
        ):
            path = FilePathBuilder.create_search_window_path(
                benchmark, mode, repo, window_size, slice_size
            )
            self.tokenizer.build_embeddings(path)

    def build_prediction_window_embeddings(
        self,
        benchmarks: List[str],
        modes: List[str],
        repos: List[str],
        window_sizes: List[int],
        slice_sizes: List[int],
    ) -> None:
        """
        Generated embeddings for context windows derived from model predictions (e.g., RepoCoder).
        """
        for benchmark, mode, repo, window_size, slice_size in itertools.product(
            benchmarks, modes, repos, window_sizes, slice_sizes
        ):
            # Find the prediction window path
            path = FilePathBuilder.create_prediction_window_path(
                benchmark, mode, repo, window_size, slice_size
            )

            # Store embeddings at output path
            self.tokenizer.build_embeddings(path)
