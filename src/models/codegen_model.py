import os
import tqdm
import logging
import torch
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.tools import Tools
from src.utils.file_path_builder import FilePathBuilder


class CodeGenModel:
    """
    Tokenizer/model wrapper for CodeGen using HuggingFace Transformers.
    """

    def __init__(self, model_name) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.tokenizer.add_special_tokens({"pad_token": self.tokenizer.eos_token})

        self.print_model_device()

    @property
    def model_name(self):
        """
        Returns the model name or path
        """
        return self.model.config.name_or_path

    @property
    def max_context_length(self):
        """
        Returns the max context length allowed for a prompt
        """
        return self.model.config.max_position_embeddings

    def print_model_device(self):
        device = next(self.model.parameters()).device
        logging.info(f"Model is on device: {device}")

    def tokenize(self, text: str) -> List[int]:
        """
        Tokenizes the given text using CodeGen tokenizer.
        """
        return self.tokenizer.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        """
        Decodes a sequence of token IDs back to text.
        """
        return self.tokenizer.decode(token_ids)

    def build_embeddings(self, input_file) -> None:
        """
        Builds the 1-gram vectors for the input windows.
        Saves the output as a pickle file where each line includes the context,
        its metadata, and the embedding (token IDs).
        """

        lines = Tools.load_pickle(input_file)
        new_lines: List[Dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(self.tokenize, line["context"]) for line in lines]

            for i in tqdm.tqdm(range(len(futures)), desc=f"Tokenizing windows for {input_file}"):
                tokenized = futures[i].result()
                line = lines[i]
                new_lines.append(
                    {
                        "context": line["context"],
                        "metadata": line["metadata"],
                        "data": [{"embedding": tokenized}],
                    }
                )

        output_file_path = FilePathBuilder.create_one_gram_vector_path(input_file)
        Tools.dump_pickle(new_lines, output_file_path)
        logging.info(f"Saved vectors to: {output_file_path}")
