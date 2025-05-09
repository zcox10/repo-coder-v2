import os
import glob
import pickle
import json
from typing import Any, Dict, List, Tuple

from src.utils.constants import Constants


class Tools:
    """
    Collection of utility functions for file I/O, tokenization, and source code parsing.
    """

    @staticmethod
    def read_code(fname: str) -> str:
        """
        Reads the content of a source code file as a UTF-8 string.
        """
        with open(fname, "r", encoding="utf8") as f:
            return f.read()

    @staticmethod
    def load_pickle(fname: str) -> Any:
        """
        Loads a Python object from a pickle file.
        """
        with open(fname, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def dump_pickle(obj: Any, fname: str) -> None:
        """
        Dumps a Python object to a pickle file.
        """
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        with open(fname, "wb") as f:
            pickle.dump(obj, f)

    @staticmethod
    def dump_json(obj: Any, fname: str) -> None:
        """
        Writes a Python object to a JSON file.
        """
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        with open(fname, "w", encoding="utf8") as f:
            json.dump(obj, f)

    @staticmethod
    def dump_jsonl(obj: List[Any], fname: str) -> None:
        """
        Writes a list of Python objects to a JSONL file (one object per line).
        """
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        with open(fname, "w", encoding="utf8") as f:
            for item in obj:
                f.write(json.dumps(item) + "\n")

    @staticmethod
    def retrieve_task_path(benchmark: str):
        return {
            Constants.line_benchmark: Constants.random_line_completion_benchmark,
            Constants.api_benchmark: Constants.api_completion_benchmark,
            Constants.short_line_benchmark: Constants.short_random_line_completion_benchmark,
            Constants.short_api_benchmark: Constants.short_api_completion_benchmark,
        }[benchmark]

    @staticmethod
    def load_jsonl(fname: str) -> List[Any]:
        """
        Loads a JSONL file into a list of Python objects.
        """
        with open(fname, "r", encoding="utf8") as f:
            return [json.loads(line) for line in f]

    @staticmethod
    def iterate_repository(repo: str) -> Dict[Tuple[str, ...], str]:
        """
        Recursively loads all `.py` files in a given repository and returns a dictionary
        mapping tuple-based file paths to their source code content.
        """
        pattern = os.path.join(f"{Constants.base_repos_dir}/{repo}", "**", "*.py")
        files = glob.glob(pattern, recursive=True)

        skipped_files: List[Tuple[str, Exception]] = []
        loaded_code_files: Dict[Tuple[str, ...], str] = {}
        base_dir_list = os.path.normpath(Constants.base_repos_dir).split(os.sep)

        for fname in files:
            try:
                code = Tools.read_code(fname)
                # Normalize path and remove base_dir prefix
                fpath_tuple = tuple(os.path.normpath(fname).split(os.sep)[len(base_dir_list) :])
                loaded_code_files[fpath_tuple] = code
            except Exception as e:
                skipped_files.append((fname, e))
                continue

        if skipped_files:
            print(f"Skipped {len(skipped_files)} out of {len(files)} files due to I/O errors")
            for fname, e in skipped_files:
                print(f"{fname}: {e}")

        return loaded_code_files
