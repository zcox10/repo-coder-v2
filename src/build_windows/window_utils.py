from typing import Any, Dict, List, Tuple

from src.utils.tools import Tools
from src.utils.constants import Constants


class WindowUtils:
    """
    Utils class for generating context windows over source code files.
    Handles common functionality such as file loading, context slicing, and dumping output.
    """

    def _get_code_lines(self, fpath_tuple: Tuple[str, ...]) -> List[str]:
        """
        Returns the lines of code for a given file path tuple.
        """
        return Tools.iterate_repository(fpath_tuple[0])[fpath_tuple].splitlines()

    def _get_context_window(self, code_lines: List[str], start_line: int, end_line: int) -> str:
        """
        Returns a joined string of code lines in the [start_line, end_line) range.
        Strips trailing/leading whitespace lines.
        """
        window_lines = [line for line in code_lines[start_line:end_line] if line.strip()]
        return "\n".join(window_lines)

    def _make_metadata(
        self,
        fpath_tuple: Tuple[str, ...],
        repo: str,
        window_size: int,
        line_no: int,
        start_line: int,
        end_line: int,
        context_start_lineno: int = 0,
        task_id: str = "",
        extra_metadata: Dict[str, Any] = {},
    ) -> Dict[str, Any]:
        """
        Standard metadata structure attached to each code window.
        """
        base_metadata = {
            "fpath_tuple": fpath_tuple,
            "line_no": line_no,
            "start_line_no": start_line,
            "end_line_no": end_line,
            "window_size": window_size,
            "context_start_lineno": context_start_lineno,
            "repo": repo,
        }
        if task_id:
            base_metadata["task_id"] = task_id
        base_metadata.update(extra_metadata)
        return base_metadata

    def _retrieve_line_bounds_by_mode(
        self,
        mode: str,
        line_no: int,
        context_start_lineno: int,
        window_size: int,
        total_code_lines: int,
    ):
        """
        Define start and end line depending on mode (e.g., GT vs. RG1)
        """

        # Baseline RAG window method
        if mode == Constants.rg:
            start_line = max(context_start_lineno, line_no - window_size)
            end_line = line_no  # Exclude the line being predicted

        # Grouth truth window method
        elif mode == Constants.gt:
            delta_size = window_size // 2
            start_line = max(context_start_lineno, line_no - delta_size)
            end_line = line_no + window_size - delta_size
            end_line = min(total_code_lines, end_line)  # Clamp to file length

        # Iterative RAG (RepoCoder), same as Constants.gt
        elif mode == Constants.rgrg:
            delta_size = window_size // 2
            start_line = max(context_start_lineno, line_no - delta_size)
            end_line = line_no + window_size - delta_size
            end_line = min(total_code_lines, end_line)

        return start_line, end_line

    def _retrieve_code_context(self, task, window_size, mode):
        fpath_tuple = tuple(task["metadata"]["fpath_tuple"])
        line_no = task["metadata"]["line_no"]
        context_start_lineno = task["metadata"]["context_start_lineno"]

        code_lines = self._get_code_lines(fpath_tuple)
        start_line, end_line = self._retrieve_line_bounds_by_mode(
            mode, line_no, context_start_lineno, window_size, len(code_lines)
        )
        return fpath_tuple, line_no, context_start_lineno, code_lines, start_line, end_line
