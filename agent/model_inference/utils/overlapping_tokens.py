# from typing import Dict, List, Tuple, Union
# from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

# def process_token(
#     token: Tuple[str, int],
#     input_str: str,
#     tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
#     max_tail_len: int,
# ) -> Tuple[int, List[List[int]]]:
#     """
#     Process a single token to find overlapping sequences.

#     Args:
#         token (Tuple[str, int]): A tuple containing the word and its token ID.
#         input_str (str): The input string to find overlaps with.
#         tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast]): Tokenizer instance.
#         max_tail_len (int): Maximum length of the tail sequence.

#     Returns:
#         Tuple[int, List[List[int]]]: A tuple containing:
#             - token_id (int): The ID of the processed token.
#             - sequences (List[List[int]]): List of unique tail token sequences.
#     """
#     word, token_id = token

#     if input_str == word:
#         # Exact match; return the token with an empty tail sequence
#         sequences = [[]]
#     else:
#         sequences_set = set()
#         input_len = len(input_str)
#         word_len = len(word)
#         max_overlap = min(word_len, input_len)

#         # Identify overlapping sequences by matching suffix of the token with the prefix of input_str
#         for overlap_len in range(1, max_overlap + 1):
#             # Skip if the overlap length equals the word length and words are identical
#             if overlap_len == word_len and word == input_str:
#                 continue
#             if word[-overlap_len:] == input_str[:overlap_len]:
#                 tail_str = input_str[overlap_len:]
#                 try:
#                     # Tokenize the tail string
#                     tail_tokenization = tokenizer.encode(tail_str, add_special_tokens=False)
#                     # Limit the tail length if specified
#                     if max_tail_len >= 0:
#                         tail_tokenization = tail_tokenization[:max_tail_len]
#                     sequences_set.add(tuple(tail_tokenization))
#                 except Exception:
#                     # Skip if tokenization fails
#                     continue
#         sequences = [list(seq) for seq in sequences_set]

#     return token_id, sequences


# def get_overlapping_token_sequences(
#     input_str: str,
#     tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
#     vocab: Dict[str, int],
#     max_tail_len: int = -1,
# ) -> Dict[int, List[List[int]]]:
#     """
#     Find overlapping token sequences from a given vocabulary.

#     Args:
#         input_str (str): The input string to find overlapping sequences with.
#         tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast]): Tokenizer instance.
#         vocab (Dict[str, int]): Vocabulary mapping from words to token IDs.
#         max_tail_len (int, optional): Maximum length of the tail sequences. Defaults to -1.

#     Returns:
#         Dict[int, List[List[int]]]: A dictionary mapping token IDs to lists of overlapping sequences.
#     """
#     token_sequences: Dict[int, List[List[int]]] = {}
#     # Process each token in the vocabulary sequentially
#     for word, token_id in vocab.items():
#         token_id, sequences = process_token((word, token_id), input_str, tokenizer, max_tail_len)
#         if sequences:
#             token_sequences[token_id] = sequences

#     return token_sequences
