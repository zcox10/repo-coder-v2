from typing import Any, Dict, List
import logging
import itertools

from src.utils.tools import Tools
from src.utils.constants import Constants
from src.build_windows.window_utils import WindowUtils
from src.utils.file_path_builder import FilePathBuilder


class PredictionWindows:

    def __init__(self):
        self.utils = WindowUtils()

    def build_prediction_windows(
        self,
        model_name: str,
        vectorizer: str,
        benchmarks: List[str],
        modes: List[str],
        repos: List[str],
        window_sizes: List[int],
        slice_sizes: List[int],
    ):
        for benchmark, mode, repo, window_size, slice_size in itertools.product(
            benchmarks, modes, repos, window_sizes, slice_sizes
        ):
            prediction_path = FilePathBuilder.create_predictions_path(
                model_name,
                vectorizer,
                benchmark,
                mode,
                window_size,
                slice_size,
            )
            self._build_prediction_window(
                benchmark, mode, repo, window_size, slice_size, prediction_path
            )

    def _build_prediction_window(
        self,
        benchmark: str,
        mode: str,
        repo: str,
        window_size: int,
        slice_size: int,
        prediction_path: str,
    ) -> None:
        """
        Builds context windows by inserting model predictions into the original source code.

        Each window is centered around a prediction line. The predicted text is inserted,
        and then a window of lines is extracted around it. This is used to evaluate how
        inserted completions change the local context.

        Args:
            repo (str): The repository name.
            window_size (int): Total number of lines in the prediction context window.
            prediction_path (str): Path to the `.jsonl` file containing model predictions.
            benchmark (str): The benchmark for defining an output path to save results to
        """
        code_windows: List[Dict[str, Any]] = []
        mode_rgrg = Constants.rgrg

        predictions = Tools.load_jsonl(prediction_path)
        for prediction in predictions:
            if prediction["metadata"]["task_id"].split("/")[0] != repo:
                continue

            fpath_tuple, line_no, context_start_lineno, code_lines, start_line, end_line = (
                self.utils._retrieve_code_context(prediction, window_size, mode_rgrg)
            )

            # Get all predicted completions (usually just one)
            for sample in [choice["text"] for choice in prediction["choices"]]:
                sample_lines = [line for line in sample.splitlines() if line.strip()]

                # Insert prediction lines into the original source before the predicted line
                new_code_lines = code_lines[:line_no] + sample_lines + code_lines[line_no:]

                # Calculate window bounds *after* inserting prediction
                start_line, end_line = self.utils._retrieve_line_bounds_by_mode(
                    mode_rgrg,
                    line_no,
                    context_start_lineno,
                    window_size=window_size,
                    total_code_lines=len(new_code_lines),
                )

                window_lines = [
                    line for line in new_code_lines[start_line:end_line] if line.strip()
                ]
                if not window_lines:
                    continue

                context = "\n".join(window_lines)

                metadata = self.utils._make_metadata(
                    fpath_tuple=fpath_tuple,
                    repo=repo,
                    window_size=window_size,
                    line_no=line_no,
                    start_line=start_line,
                    end_line=end_line,
                    context_start_lineno=context_start_lineno,
                    task_id=prediction["metadata"]["task_id"],
                    extra_metadata={"prediction": sample},
                )

                code_windows.append(
                    {
                        "context": context,
                        "metadata": metadata,
                    }
                )

        logging.info(
            f"Build {len(code_windows)} prediction windows for {repo}; window size: {window_size}; slice size: {slice_size}; benchmark: {benchmark}; mode: {mode}"
        )

        output_path = FilePathBuilder.create_prediction_window_path(
            benchmark, mode, repo, window_size, slice_size
        )

        Tools.dump_pickle(code_windows, output_path)
