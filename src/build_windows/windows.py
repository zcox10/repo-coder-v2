from src.build_windows.baseline_windows import BaselineWindows
from src.build_windows.prediction_windows import PredictionWindows
from src.build_windows.repo_windows import RepoWindows


class Windows:
    """
    High-level wrapper for generating context windows for different use cases:
    - Raw repository windows
    - Baseline (RG1) and ground-truth (GT) windows
    - Prediction-derived windows (e.g., RepoCoder)
    """

    def build_baseline_windows(self, **kwargs):
        BaselineWindows().build_baseline_windows(**kwargs)

    def build_prediction_windows(self, **kwargs):
        PredictionWindows().build_prediction_windows(**kwargs)

    def build_repo_windows(self, **kwargs):
        RepoWindows().build_repo_windows(**kwargs)
