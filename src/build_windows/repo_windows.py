from typing import Any, Dict, List, Tuple
from collections import defaultdict
import logging
import itertools

from src.utils.file_path_builder import FilePathBuilder
from src.utils.tools import Tools
from src.utils.constants import Constants


class RepoWindows:
    """
    Generates context windows across all Python files in a repository.

    Each window is a symmetric slice of lines around a target line, sampled
    at a regular interval (slice step). These windows are useful for building
    search indexes or training context models.
    """

    def build_repo_windows(self, repos, window_sizes, slice_sizes):
        """
        Build windows for all permutations of repositories, window sizes, and slice sizes.
        """
        for repo, window_size, slice_size in itertools.product(repos, window_sizes, slice_sizes):
            self._build_repo_window(repo, window_size, slice_size)

    def _build_repo_window(self, repo, window_size, slice_size) -> None:
        """
        Builds windows for entire repository and writes to a pickle file.

        Args:
            repo (str): Repository name or relative path (within `data/repositories/`).
            window_size (int): Total number of lines in each context window.
            slice_size (int): Controls how densely windows are sampled across a file.
        """

        all_code_windows: List[Dict[str, Any]] = []
        source_code_files = Tools.iterate_repository(repo)

        for fpath_tuple, code in source_code_files.items():
            all_code_windows.extend(
                self._build_windows_for_file(fpath_tuple, code, repo, window_size, slice_size)
            )

        merged_windows = self._merge_windows_with_same_context(all_code_windows)
        output_path = FilePathBuilder.create_repo_window_path(repo, window_size, slice_size)
        Tools.dump_pickle(merged_windows, output_path)

        logging.info(
            f"Built {len(merged_windows)} windows for repo '{repo}' (window size: {window_size}, slice size: {slice_size}); path: {output_path}"
        )

    def _build_windows_for_file(
        self, fpath_tuple: Tuple[str, ...], code: str, repo: str, window_size: int, slice_size: int
    ) -> List[Dict[str, Any]]:
        """
        Creates a list of context windows from a single source file.

        Args:
            fpath_tuple: Normalized path to the source file as tuple of path parts.
            code: Raw source code string.
            repo (str): Repository name or relative path (within `data/repositories/`).
            window_size (int): Total number of lines in each context window.
            slice_size (int): Controls how densely windows are sampled across a file.

        Returns:
            List of dicts, each representing a code window with metadata.
        """
        code_windows: List[Dict[str, Any]] = []
        code_lines = code.splitlines()
        slice_step = max(1, window_size // slice_size)

        for line_no in range(0, len(code_lines), slice_step):
            start_line = max(0, line_no - slice_step)
            end_line = min(len(code_lines), line_no + window_size - slice_step)

            window_lines = [line for line in code_lines[start_line:end_line]]
            if not window_lines:  # skip empty windows
                continue

            context = "\n".join(window_lines)
            metadata = {
                "fpath_tuple": fpath_tuple,
                "line_no": line_no,
                "start_line_no": start_line,
                "end_line_no": end_line,
                "window_size": window_size,
                "repo": repo,
                "slice_size": slice_size,
            }

            code_windows.append(
                {
                    "context": context,
                    "metadata": metadata,
                }
            )

        return code_windows

    def _merge_windows_with_same_context(
        self, code_windows: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Deduplicates windows by merging metadata of windows with identical context.

        Args:
            code_windows: List of windows (each with 'context' and 'metadata').

        Returns:
            Merged list with unique 'context' entries and grouped metadata.
        """

        merged_code_windows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for window in code_windows:
            context = window["context"]
            merged_code_windows[context].append(window["metadata"])

        return [
            {"context": context, "metadata": metadata_list}
            for context, metadata_list in merged_code_windows.items()
        ]
