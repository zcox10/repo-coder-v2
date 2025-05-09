import os
import itertools
import logging
from typing import List, Tuple, Dict, Any

from src.utils.constants import Constants
from src.utils.file_path_builder import FilePathBuilder
from src.utils.tools import Tools


class Prompts:
    def __init__(self, tokenizer):
        """
        Constructs prompts for code generation tasks using retrieved context from repositories.
        This class handles loading retrieval results, formatting context into prompts, and saving
        the output prompts for downstream generation tasks.

        Args:
            tokenizer: The tokenizer used to measure prompt lengths and tokenize final strings.
        """

        # Formatting components
        self.separator = "# " + "-" * 50

        # Define task paths
        self.task_path = {
            Constants.line_benchmark: Constants.random_line_completion_benchmark,
            Constants.api_benchmark: Constants.api_completion_benchmark,
            Constants.short_api_benchmark: Constants.short_api_completion_benchmark,
            Constants.short_line_benchmark: Constants.short_random_line_completion_benchmark,
        }

        # Define embedding model to tokenize inputs
        self.tokenizer = tokenizer

    def build_prompts(
        self,
        vectorizer: str,
        max_top_k: int,
        benchmarks: List[str],
        modes: List[str],
        repos: List[str],
        window_sizes: List[int],
        slice_sizes: List[int],
        generation_type: str,
        max_new_tokens: int,
    ):
        """
        Builds prompts by retrieving top-k similar context windows and formatting them
        into usable prompts for generation tasks. Prompts are constructed for every combination
        of benchmark, mode, repo, window size, and slice size, and saved to disk.

        Args:
            vectorizer: Name of the vector model used for retrieval (e.g., "one-gram", "ada002").
            max_top_k: Number of top similar context windows to retrieve.
            benchmarks: List of benchmarks to generate prompts for.
            modes: List of retrieval modes (e.g., "rg", "gt").
            repos: List of repository names to use as source data.
            window_sizes: Context window sizes for retrieval.
            slice_sizes: Stride sizes for sliding window retrieval.
            generation_type: Type of generation task (e.g., "prediction", "search").
            max_new_tokens: Number of new tokens the model is allowed to generate.
        """

        # max_prompt_length: the total number of tokens to include in a prompt
        # max_context_length defines total number of tokens allowed
        # max_new_tokens defines number of tokens generated as a response
        max_prompt_length = self.tokenizer.max_context_length - max_new_tokens

        # Iterate through all benchmarks, modes, window sizes, and slice sizes
        for benchmark, mode, window_size, slice_size in itertools.product(
            benchmarks, modes, window_sizes, slice_sizes
        ):
            # Define output path to store prompts
            output_file_path = FilePathBuilder.create_prompts_path(
                vectorizer, benchmark, mode, window_size, slice_size
            )

            # Define task path
            task_path = self.task_path[benchmark]

            lines = []
            for repo in repos:
                # Obtain retrieval code lines with top K context per repo
                retrieval_code_lines = self._get_retrieval_code_lines(
                    vectorizer,
                    max_top_k,
                    benchmark,
                    mode,
                    repo,
                    window_size,
                    slice_size,
                    generation_type,
                )

                # Construct prompts per repo and append to list
                lines.extend(
                    self._build_prompts_by_task_id(
                        retrieval_code_lines, task_path, mode, max_prompt_length
                    )
                )

            Tools.dump_jsonl(lines, output_file_path)

    def _get_retrieval_code_lines(
        self,
        vectorizer: str,
        max_top_k: int,
        benchmark: str,
        mode: str,
        repo: str,
        window_size: int,
        slice_size: int,
        generation_type: str,
    ):
        """
        Loads retrieved context lines results for a specific configuration.

        Returns:
            List of retrieval results, each with query metadata and top-k context.
        """

        # Obtain embedding_path_builder
        embedding_path_builder = {
            "one-gram": FilePathBuilder.create_one_gram_vector_path,
            "ada002": FilePathBuilder.create_ada002_vector_path,
        }[vectorizer]

        # Define query path where query embeddings are stored
        if generation_type == "prediction":
            query_window_path = FilePathBuilder.create_prediction_window_path(
                benchmark, mode, repo, window_size, slice_size
            )
        else:
            query_window_path = FilePathBuilder.create_search_window_path(
                benchmark, mode, repo, window_size, slice_size
            )

        # Define query_embedding_path and repo_embedding_path to extract a retrieval embedding path
        query_embedding_path = embedding_path_builder(query_window_path)
        repo_embedding_path = embedding_path_builder(
            FilePathBuilder.create_repo_window_path(repo, window_size, slice_size)
        )

        # Define the retrieval embedding path to retrieve relevant code lines
        retrieval_path = FilePathBuilder.create_retrieval_results_path(
            query_embedding_path, repo_embedding_path, max_top_k
        )
        return Tools.load_pickle(retrieval_path)

    def _build_prompts_by_task_id(
        self,
        retrieval_code_lines: List[Dict[str, Any]],
        task_path: str,
        mode: str,
        max_retrieval_length: int,
    ) -> List[Dict[str, Any]]:
        """
        Builds prompts by pairing retrieved context with task-specific generation prompts.
        Each prompt is formed by prepending relevant code blocks to a task-specific hole.

        Returns:
            A list of fully-formed prompts, each with prompt text and full metadata.
        """

        # Load tasks and map by task_id for lookup
        tasks_by_task_id = {
            task["metadata"]["task_id"]: task for task in Tools.load_jsonl(task_path)
        }

        # Iterate through retrieved code lines to construct prompts per query
        prompts = []
        for query in retrieval_code_lines:
            task_id = query["metadata"]["task_id"]
            task = tasks_by_task_id[task_id]

            # Construct prompt with additional context and metadata
            prompt, context = self._build_prompt(
                mode, task["prompt"], task["metadata"], query["top_k_context"], max_retrieval_length
            )
            full_prompt_context = {
                "prompt": prompt,
                "metadata": {
                    **task["metadata"],
                    "query_window": {
                        "context": query["context"],
                        "metadata": query["metadata"],
                    },
                    "top_k_context": [
                        {
                            "context": c[0]["context"],
                            "metadata": c[0]["metadata"],
                            "sim_score": c[1],
                        }
                        for c in context
                    ],
                    "window_size": query["metadata"]["window_size"],
                    "slice_size": (context[0][0]["metadata"][0]["slice_size"] if context else None),
                },
            }

            # Structure output prompt with necessary metadata for tracing
            prompts.append(full_prompt_context)

        return prompts

    def _build_prompt(
        self,
        mode: str,
        prompt: str,
        task_metadata: Dict[str, Any],
        top_k_context: List[Tuple[Dict[str, Any], float]],
        max_retrieval_length: int,
    ) -> Tuple[str, List[Tuple[Dict[str, Any], float]]]:
        """
        Constructs a full prompt by prepending retrieved context blocks to the task prompt.
        Ensures the prompt stays within a specified token limit.

        Returns:
            A tuple containing the full prompt string and the list of chosen context blocks.
        """

        # Define prompt header to calculate current token length of header + prompt
        prompt_header = (
            "# Here are some relevant code fragments from other files of the repo:\n"
            + self.separator
            + "\n"
        )
        current_token_length = len(self.tokenizer.tokenize(prompt_header + "\n" + prompt))

        # Select block construction method based on retrieval mode
        make_block = self._make_extended_block if mode == Constants.rg else self._make_block
        blocks = []
        chosen_context = []

        # Prioritizes top-most relevant items last for prepending
        for retrieved_context in top_k_context:
            # Prepare arguments for the block function
            kwargs = {"retrieved_context": retrieved_context}
            if mode == Constants.rg:
                kwargs["task_metadata"] = task_metadata

            block_str, token_len = make_block(**kwargs)

            # Check token budget before appending
            if current_token_length + token_len < max_retrieval_length:
                blocks.insert(0, block_str)
                current_token_length += token_len
                chosen_context.append(retrieved_context)

        final_prompt = prompt_header + "".join(blocks) + "\n" + prompt

        prompt_token_len = len(self.tokenizer.tokenize(final_prompt))
        if prompt_token_len >= max_retrieval_length:
            logging.error(f"Final prompt length: {prompt_token_len} >= {max_retrieval_length}\n")

        return final_prompt, chosen_context

    def _make_block(self, retrieved_context: Tuple[Dict[str, Any], float]) -> Tuple[str, int]:
        """
        Formats a single context block into a comment-based structure suitable for prompt prepending.
        Intended for non-overlapping (GT mode) context that does not intersect the task's location.

        Returns:
            A tuple of (formatted block string, token count).
        """

        content, _ = retrieved_context
        metadata = content["metadata"]

        # Format file paths
        f_paths = ["/".join(x["fpath_tuple"][1:]) for x in metadata]
        f_paths_str = "\n".join([f"# {f_path}" for f_path in f_paths])

        # Convert context code into comments
        comment_lines = [f"# {line}" for line in content["context"].splitlines()]

        block = "\n".join(
            [
                "# the below code fragment can be found in:",
                f_paths_str,
                self.separator,
                *comment_lines,
                self.separator,
                "",
            ]
        )
        token_len = len(self.tokenizer.tokenize(block))
        return block, token_len

    def _make_extended_block(
        self, task_metadata: Dict[str, Any], retrieved_context: Tuple[Dict[str, Any], float]
    ) -> Tuple[str, int]:
        """
        Constructs an extended context block that reads raw source code and dynamically adjusts
        context window size based on retrieved metadata. Skips overlapping code segments.

        Returns:
            A tuple of (formatted block string, token count). Returns empty string if no valid block found.
        """

        content, _ = retrieved_context
        for meta in content["metadata"]:
            # Skip if the metadata overlaps with or is after the hole
            if (
                meta["fpath_tuple"] == tuple(task_metadata["fpath_tuple"])
                and meta["end_line_no"] >= task_metadata["line_no"]
            ):
                continue

            # Read the raw source code from disk
            file_path = os.path.join(Constants.base_repos_dir, *meta["fpath_tuple"])
            code_lines = Tools.read_code(file_path).splitlines()

            # Dynamically adjust context window around retrieved region
            new_end = min(
                meta["end_line_no"] + meta["window_size"] // meta["slice_size"], len(code_lines)
            )
            new_start = max(0, new_end - meta["window_size"])
            snippet = code_lines[new_start:new_end]

            # Format code as comments
            comment_lines = [f"# {line}" for line in snippet]
            f_paths_str = "\n".join(
                [f"# {'/'.join(x['fpath_tuple'][1:])}" for x in content["metadata"]]
            )

            block = "\n".join(
                [
                    "# the below code fragment can be found in:",
                    f_paths_str,
                    self.separator,
                    *comment_lines,
                    self.separator,
                    "",
                ]
            )
            token_len = len(self.tokenizer.tokenize(block))
            return block, token_len

        return "", 0  # Fallback if no valid snippet found
