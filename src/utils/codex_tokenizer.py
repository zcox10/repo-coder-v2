# import tiktoken
# from typing import List

# from src.utils.constants import Constants


# class CodexTokenizer:
#     """
#     Tokenizer wrapper for Codex using TikToken.
#     """

#     def __init__(self) -> None:
#         self.tokenizer = tiktoken.get_encoding(Constants.codex_tokenizer)

#     def tokenize(self, text: str) -> List[int]:
#         """
#         Tokenizes the given text using Codex encoding.
#         """
#         return self.tokenizer.encode_ordinary(text)

#     def decode(self, token_ids: List[int]) -> str:
#         """
#         Decodes a sequence of token IDs back to text.
#         """
#         return self.tokenizer.decode(token_ids)
