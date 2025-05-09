from typing import List
import editdistance
import itertools
import logging

from src.utils.file_path_builder import FilePathBuilder
from src.utils.tools import Tools
from src.utils.constants import Constants


class Scores:

    def build_scores(
        self,
        model_name: str,
        vectorizer: str,
        benchmarks: List[str],
        modes: List[str],
        window_sizes: List[int],
        slice_sizes: List[int],
    ):
        final_scores = []
        for benchmark, mode, window_size, slice_size in itertools.product(
            benchmarks, modes, window_sizes, slice_sizes
        ):
            prediction_path = FilePathBuilder.create_predictions_path(
                model_name,
                vectorizer,
                benchmark,
                mode,
                window_size,
                slice_size,
            )
            prediction_file = Tools.load_jsonl(prediction_path)
            final_scores.append(self._compute_score(benchmark, mode, prediction_file, passk=1))

        scores_path = FilePathBuilder.create_scores_path(model_name, vectorizer)
        Tools.dump_jsonl(obj=final_scores, fname=scores_path)
        logging.info(f"Output scores to: {scores_path}")

    def _compute_EM(self, target, predictions):
        target_lines = [line.strip() for line in target.splitlines() if line.strip()]
        EM_scores = []
        for prediction in predictions:
            prediction_lines = [line.strip() for line in prediction.splitlines() if line.strip()][
                : len(target_lines)
            ]
            if len(target_lines) != len(prediction_lines):
                EM_scores.append(0)
                continue
            if target_lines == prediction_lines:
                EM_scores.append(1)
                continue
            EM_scores.append(0)
        return max(EM_scores)

    def _compute_ES(self, target, predictions):
        target_lines = [line.strip() for line in target.splitlines() if line.strip()]
        target_str = "\n".join(target_lines)
        ES_scores = []
        for prediction in predictions:
            prediction_lines = [line.strip() for line in prediction.splitlines() if line.strip()][
                : len(target_lines)
            ]
            prediction_str = "\n".join(prediction_lines)
            ES_scores.append(
                1
                - (
                    editdistance.eval(target_str, prediction_str)
                    / max(len(target_str), len(prediction_str))
                )
            )
        return max(ES_scores)

    def _compute_score(self, benchmark, mode, prediction_file, passk):

        scores = {
            "benchmark": benchmark,
            "mode": mode,
            "repos": {},
            "scores": {
                "total_samples": None,
                "summed_em_score": None,
                "mean_em_score": None,
                "summed_es_score": None,
                "mean_es_score": None,
            },
        }
        for line in prediction_file:
            repo = line["metadata"]["task_id"].split("/")[0]
            samples = [line["choices"][i]["text"] for i in range(len(line["choices"]))][:passk]
            ground_truth = line["metadata"]["ground_truth"]

            em_score = self._compute_EM(ground_truth, samples)
            es_score = self._compute_ES(ground_truth, samples)

            # Initialize list for repo if it doesn't exist
            if repo not in scores["repos"]:
                scores["repos"][repo] = []

            scores["repos"][repo].append(
                {
                    "ground_truth": ground_truth,
                    "samples": samples,
                    "em_score": em_score,
                    "es_score": es_score,
                }
            )

        return self._update_scores(scores)

    def _update_scores(self, scores):
        total_samples = 0
        summed_em_score = 0
        summed_es_score = 0.0
        for _, samples in scores["repos"].items():
            for sample in samples:
                total_samples += 1
                summed_em_score += sample["em_score"]
                summed_es_score += sample["es_score"]

        scores["scores"]["total_samples"] = total_samples
        scores["scores"]["summed_em_score"] = summed_em_score
        scores["scores"]["summed_es_score"] = summed_es_score
        scores["scores"]["mean_em_score"] = summed_em_score / total_samples
        scores["scores"]["mean_es_score"] = summed_es_score / total_samples

        return scores
