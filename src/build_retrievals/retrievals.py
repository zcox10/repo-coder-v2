import os
import copy
import itertools
import tqdm
import numpy as np
import logging
from concurrent.futures import as_completed, ProcessPoolExecutor
from typing import List, Dict, Callable, Tuple, Any


from src.utils.file_path_builder import FilePathBuilder
from src.utils.tools import Tools
from src.build_retrievals.similarity import SimilarityScore


class Retrievals:
    """
    Orchestrates the parallelized code search and retrieval process over a set of repositories.
    Depending on the vectorization strategy, it loads precomputed code embeddings and performs
    similarity-based matching between query code and repository code windows.
    """

    def retrieve_similar_embeddings(
        self,
        vectorizer: str,
        max_top_k: int,
        benchmarks: List[str],
        modes: List[str],
        repos: List[str],
        window_sizes: List[int],
        slice_sizes: List[int],
        generation_type: str,
    ):
        """
        Runs code similarity search tasks in parallel across repositories and window/slice sizes.

        Args:
            vectorizer (str): The type of embedding to perform (e.g., one-gram vs. ada002)
            max_top_k (int): The maximum retrieved embeddings to store for a prompt
            benchmarks (List[str]): The list of benchmarks to run code search on (e.g., random_line vs. random_api)
            modes (List[str]): The list of modes to run code search on (e.g., GT vs. RG1)
            repos (List[str]): The list of repos perform retrieval on
            window_sizes (List[int]): Window sizes for chunking code
            slice_sizes (List[int]): Slice sizes for chunking code
            generation_type (str): The type of code search to perform (e.g., prediction vs. baseline)
        """
        # Define similarity metric (sim_scorer), embedding path (embedding_path_builder)
        sim_scorer, embedding_path_builder = self._retrieve_similarity_scorer(vectorizer)

        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            for benchmark, mode, repo, window_size, slice_size in itertools.product(
                benchmarks, modes, repos, window_sizes, slice_sizes
            ):
                # Retrieve embedding lines for repository, embedding lines for query,
                # and output path for storing retrieval results
                repo_embedding_lines, query_embedding_lines, output_path = (
                    self._retrieve_embedding_lines(
                        max_top_k,
                        benchmark,
                        mode,
                        repo,
                        window_size,
                        slice_size,
                        generation_type,
                        embedding_path_builder,
                    )
                )

                futures.append(
                    executor.submit(
                        Retrievals._run_retrieval_job,
                        query_embedding_lines,
                        repo_embedding_lines,
                        max_top_k,
                        sim_scorer,
                        output_path,
                    )
                )

            # Process all retrieval jobs in parallel
            for future in tqdm.tqdm(
                as_completed(futures), total=len(futures), desc="Searching for Retrieval Context"
            ):
                future.result()

    def _retrieve_embedding_lines(
        self,
        max_top_k: int,
        benchmark: str,
        mode: str,
        repo: str,
        window_size: int,
        slice_size: int,
        generation_type: str,
        embedding_path_builder: Callable[[str], str],
    ):
        """
        Returns the repo_embedding_lines, query_embedding_lines, and output path to store retrieval context.
        """

        # Define query's window path
        if generation_type == "prediction":
            query_window_path = FilePathBuilder.create_prediction_window_path(
                benchmark=benchmark,
                mode=mode,
                repo=repo,
                window_size=window_size,
                slice_size=slice_size,
            )
        else:
            query_window_path = FilePathBuilder.create_search_window_path(
                benchmark=benchmark,
                mode=mode,
                repo=repo,
                window_size=window_size,
                slice_size=slice_size,
            )

        # Define query embedding path
        query_embedding_path = embedding_path_builder(query_window_path)

        # Define repo window and repo embedding paths
        repo_window_path = FilePathBuilder.create_repo_window_path(repo, window_size, slice_size)
        repo_embedding_path = embedding_path_builder(repo_window_path)

        # Final output path to store retrieval embeddings
        output_path = FilePathBuilder.create_retrieval_results_path(
            query_embedding_path, repo_embedding_path, max_top_k
        )

        # Load precomputed embeddings
        repo_embedding_lines = Tools.load_pickle(repo_embedding_path)
        query_embedding_lines = Tools.load_pickle(query_embedding_path)

        return repo_embedding_lines, query_embedding_lines, output_path

    def _find_top_k_context(
        self,
        query_embedding_line: Dict[str, Any],
        repo_embedding_lines: List[Dict[str, Any]],
        max_top_k: int,
        sim_scorer: Callable[[np.ndarray, np.ndarray], float],
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Finds the top-K most similar context lines from the repository for a single query line.

        Args:
            query_embedding_line: A single embedding entry from the query.
            repo_embedding_lines: All embedding entries from the repository.
            max_top_k (int): The maximum retrieved embeddings to store for a prompt
            sim_scorer: The type of similarity calculation to use (found in similarity.py)

        Returns:
            List[Tuple[Dict, float]]: A list of tuples containing the top-K context lines and their similarity scores.
        """
        top_k_context = []
        query_embedding = np.array(query_embedding_line["data"][0]["embedding"])

        for repo_embedding_line in repo_embedding_lines:
            # skip repo_embedding_lines that appear after a query's code line
            if self._is_context_after_hole(repo_embedding_line, query_embedding_line):
                continue

            # Compute similarity score and store in top_k_context
            repo_line_embedding = np.array(repo_embedding_line["data"][0]["embedding"])
            similarity_score = sim_scorer(query_embedding, repo_line_embedding)
            top_k_context.append((repo_embedding_line, similarity_score))

        # Sort by similarity (desc), and take the top K highest scores
        top_k_context = sorted(top_k_context, key=lambda x: x[1], reverse=True)[:max_top_k]

        return top_k_context

    def _is_context_after_hole(
        self, repo_embedding_line: Dict[str, Any], query_embedding_line: Dict[str, Any]
    ) -> bool:
        """
        Checks if the context from the repository file appears *after* the hole in the query file.

        Args:
            repo_embedding_line (Dict): A single embedding entry from the repository.
            query_embedding_line (Dict): A single embedding entry from the query.

        Returns:
            bool: True if all repo lines are *not* after the hole; False otherwise.
        """
        hole_fpath_tuple = tuple(query_embedding_line["metadata"]["fpath_tuple"])
        context_is_not_after_hole = []

        for metadata in repo_embedding_line["metadata"]:
            if tuple(metadata["fpath_tuple"]) != hole_fpath_tuple:
                context_is_not_after_hole.append(True)
                continue

            # Same file as the hole, check if context comes before the hole
            if metadata["end_line_no"] <= query_embedding_line["metadata"]["context_start_lineno"]:
                context_is_not_after_hole.append(True)
                continue

            context_is_not_after_hole.append(False)

        # If any context line appears after the hole, return False
        return not any(context_is_not_after_hole)

    def _retrieve_similarity_scorer(self, vectorizer: str):
        """
        Determine similarity calculation and outut file path depending on the defined vectorizer (e.g., one-gram vs. ada002).
        """
        if vectorizer == "one-gram":
            sim_scorer = SimilarityScore.jaccard_similarity
            embedding_path_builder = FilePathBuilder.create_one_gram_vector_path
        elif vectorizer == "ada002":
            sim_scorer = SimilarityScore.cosine_similarity
            embedding_path_builder = FilePathBuilder.create_ada002_vector_path
        else:
            raise ValueError(f"Unsupported vectorizer: {vectorizer}")

        return sim_scorer, embedding_path_builder

    @staticmethod
    def _run_retrieval_job(
        query_embedding_lines,
        repo_embedding_lines,
        max_top_k,
        sim_scorer,
        output_path,
    ):
        query_lines_with_retrieved_results = []

        for query_line in query_embedding_lines:
            new_line = copy.deepcopy(query_line)
            top_k_context = []

            query_embedding = np.array(query_line["data"][0]["embedding"])
            for repo_embedding_line in repo_embedding_lines:
                repo_embedding = np.array(repo_embedding_line["data"][0]["embedding"])
                score = sim_scorer(query_embedding, repo_embedding)
                top_k_context.append((repo_embedding_line, score))

            top_k_context = sorted(top_k_context, key=lambda x: x[1], reverse=True)[:max_top_k]
            new_line["top_k_context"] = top_k_context
            query_lines_with_retrieved_results.append(new_line)

        Tools.dump_pickle(query_lines_with_retrieved_results, output_path)
        logging.info(f"Saved vectors to: {output_path}")
