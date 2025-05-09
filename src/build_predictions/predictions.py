import torch
import tqdm
import logging
import itertools
from pprint import pprint
from typing import List, Tuple, Dict, Any

from src.utils.file_path_builder import FilePathBuilder
from src.utils.tools import Tools


class Predictions:
    """
    A utility class to generate code completions using a language model,
    batch the inputs, and save the predictions to disk.
    """

    def __init__(self, model):
        # Input from CodeGenModel()
        self.codegen = model
        self.model = self.codegen.model
        self.tokenizer = self.codegen.tokenizer
        # self.model.cuda()
        # self.tokenizer.cuda()

    def create_predictions(
        self,
        model_name: str,
        vectorizer: str,
        benchmarks: List[str],
        modes: List[str],
        window_sizes: List[int],
        slice_sizes: List[int],
        max_new_tokens: int,
        batch_size: int,
    ):

        # Iterate through all benchmarks, modes, window sizes, and slice sizes
        for benchmark, mode, window_size, slice_size in itertools.product(
            benchmarks, modes, window_sizes, slice_sizes
        ):
            # Generate batch predictions on a set of benchmark/mode prompts
            self._create_batch_predictions(
                model_name,
                vectorizer,
                benchmark,
                mode,
                window_size,
                slice_size,
                max_new_tokens,
                batch_size,
            )

    def _create_batch_predictions(
        self,
        model_name: str,
        vectorizer: str,
        benchmark: str,
        mode: str,
        window_size: int,
        slice_size: int,
        max_new_tokens: int,
        batch_size: int,
    ):
        # Retrieve prompt path
        prompt_path = FilePathBuilder.create_prompts_path(
            vectorizer, benchmark, mode, window_size, slice_size
        )

        # Load JSONL file containing prompts
        lines = Tools.load_jsonl(prompt_path)

        # Append newline to each prompt to maintain separation
        prompts = [f"{line['prompt']}\n" for line in lines]

        # Ground truths to trim output
        ground_truths = [f"{line['metadata']['ground_truth']}\n" for line in lines]

        # Create batches of prompts and ground truths
        batches = self._get_batches(prompts, ground_truths, batch_size)

        output_text = []

        # Generate predictions for each batch
        for batch in tqdm.tqdm(
            batches,
            total=len(batches),
            desc=f"\nGenerating Batch Predictions for {benchmark}/{mode}",
        ):
            gen_text = self._generate_batch(batch, max_new_tokens)
            output_text.extend(gen_text)

            # for prompt, ground_truth, output in zip(
            #     batch["prompt"], batch["ground_truth"], gen_text
            # ):
            #     print("\n========== PROMPT ==========")
            #     print(prompt)

            #     print("\n========== GROUND TRUTH ==========")
            #     print(ground_truth)

            #     print("\n========== OUTPUT ==========")
            #     print(output)

        # Ensure each prompt has a corresponding generation
        assert len(output_text) == len(prompts)

        # Create output lines with predictions formatted in JSONL
        new_lines = []
        for line, gen in zip(lines, output_text):
            new_lines.append(
                {
                    "prompt": line["prompt"],
                    "metadata": line["metadata"],
                    "choices": [{"text": gen}],
                }
            )

        # Save predictions to out_path
        out_path = FilePathBuilder.create_predictions_path(
            model_name, vectorizer, benchmark, mode, window_size, slice_size
        )
        Tools.dump_jsonl(new_lines, out_path)
        logging.info(f"Saved predictions to {out_path}")

    def _get_batches(
        self, prompts: List[Dict[str, Any]], ground_truths: List[Dict[str, Any]], batch_size: int
    ):
        """
        Splits a list of prompts into smaller batches.
        """
        batches = []
        for i in range(0, len(prompts), batch_size):
            batches.append(
                {
                    "prompt": prompts[i : i + batch_size],
                    "ground_truth": ground_truths[i : i + batch_size],
                }
            )
        return batches

    def _generate_batch(self, batch, max_new_tokens):
        """
        Generates completions for a single batch of prompts.

        Returns:
            List[str]: Generated completions corresponding to each prompt.
        """
        prompt_batch = batch["prompt"]
        ground_truth_batch = batch["ground_truth"]

        # Tokenize and pad the prompt batch
        prompts = self.tokenizer(prompt_batch, return_tensors="pt", padding=True, truncation=True)
        input_ids = prompts["input_ids"].to(self.codegen.device)
        attention_mask = prompts["attention_mask"].to(self.codegen.device)

        # Calculate how many new tokens can be generated without exceeding context limit
        max_length = self.codegen.max_context_length
        input_lengths = (attention_mask != 0).sum(dim=1)
        adjusted_max_new_tokens = min(max_new_tokens, int(max_length - input_lengths.max().item()))

        if adjusted_max_new_tokens <= 0:
            raise ValueError(
                f"Prompt too long! Cannot generate any new tokens within the context limit ({max_length})"
            )

        # Generate predicted tokens
        with torch.no_grad():
            gen_tokens = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=adjusted_max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode generated tokens to text
        gen_text = self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

        # Remove original prompt from decoded text to return only the generated completion
        for i in range(len(gen_text)):
            gen_text[i] = gen_text[i][len(prompt_batch[i]) :]

            # trim output to be the same number of lines as the ground truth batch
            num_lines = len(ground_truth_batch[i].splitlines())
            gen_text[i] = "\n".join(gen_text[i].splitlines()[:num_lines])

        return gen_text
