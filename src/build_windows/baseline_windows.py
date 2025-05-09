import logging
import itertools
from typing import Any, Dict, List

from src.utils.file_path_builder import FilePathBuilder
from src.utils.tools import Tools
from src.build_windows.window_utils import WindowUtils


class BaselineWindows:
    """
    Constructs context windows for the retrieve-and-generate baseline approach and ground-truth approach.
    For each task, extracts lines leading up to the target line (exclusive), forming a context window.
    These are later used for retrieval-augmented generation evaluation and ground-truth evaluation.
    """

    def __init__(self):
        self.utils = WindowUtils()

    def build_baseline_windows(
        self,
        benchmarks: List[str],
        modes: List[str],
        repos: List[str],
        window_sizes: List[int],
        slice_sizes: List[int],
    ):
        for benchmark, mode, repo, window_size, slice_size in itertools.product(
            benchmarks, modes, repos, window_sizes, slice_sizes
        ):
            self._build_baseline_window(benchmark, mode, repo, window_size, slice_size)

    def _build_baseline_window(
        self,
        benchmark: str,
        mode: str,
        repo: str,
        window_size: int,
        slice_size: int,
    ) -> None:
        """
        Builds and saves context windows for each matching task.
        There are variations depending on the `mode` when defining the `start_line` and `end_line`.
        Details for these differences are in `self.utils._retrieve_code_context()`
        """
        code_windows: List[Dict[str, Any]] = []

        # Load all tasks for a given benchmark
        task_path = Tools.retrieve_task_path(benchmark)
        tasks = Tools.load_jsonl(task_path)

        for task in tasks:
            # Skip tasks not from this repo
            if task["metadata"]["task_id"].split("/")[0] != repo:
                continue

            fpath_tuple, line_no, context_start_lineno, code_lines, start_line, end_line = (
                self.utils._retrieve_code_context(task, window_size, mode)
            )

            context = self.utils._get_context_window(code_lines, start_line, end_line)
            metadata = self.utils._make_metadata(
                fpath_tuple=fpath_tuple,
                repo=repo,
                window_size=window_size,
                line_no=line_no,
                start_line=start_line,
                end_line=end_line,
                context_start_lineno=context_start_lineno,
                task_id=task["metadata"]["task_id"],
            )

            code_windows.append(
                {
                    "context": context,
                    "metadata": metadata,
                }
            )

        # Save code windows to output path
        output_path = FilePathBuilder.create_search_window_path(
            benchmark, mode, repo, window_size, slice_size
        )
        Tools.dump_pickle(code_windows, output_path)

        logging.info(
            f"Built {len(code_windows)} windows for {repo} (mode: {mode}) with window size {window_size}"
        )
