# sft_data_utils.py
"""
Data utilities for Supervised Fine-Tuning (SFT).
Handles chat-format data with prompt/response masking.
"""
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import torch
from torch.utils.data import Dataset, DataLoader
import json


# Available CoT format options for reference:
# Top-level fields:
#   - "cot_format": Primary CoT (uses first traversal method)
#   - "cot_format_with_context": CoT with PGN context prepended
#   - "cot_format_with_directions": CoT with <up>/<down> or <level> markers
#   - "cot_format_with_directions_and_context": CoT with markers and context
#
# Per-method fields (via cot_by_method.{method}.{field}):
#   Methods: dfs_policy, dfs_verifier, bfs_policy, bfs_verifier,
#            dfs_policy_layers, dfs_verifier_layers, bfs_policy_layers, bfs_verifier_layers
#   Fields: cot_format, cot_format_with_context, cot_format_with_markers, cot_format_with_markers_and_context
#
# Multi-path formats (per-candidate with <sep> separators):
#   - "multi_path_cot_policy.cot_format_with_context"
#   - "multi_path_cot_verifier.cot_format_with_context"


import re as _re

_ENV_TAG = '<call_env>'
_STRIP_ENV_RE = _re.compile(r'\s*<call_env>\s*\S*')


def _strip_env_tokens(text: str) -> str:
    """Remove each '<call_env> MOVE' pair from text, leaving only model outputs.

    NOTE: this DELETES the opponent's reply, so the surviving moves are no longer a
    legal continuation of the prompt position (e.g. "Qf3xc6+ <call_env> Pb7xc6
    Bc4a6# <call_env>" becomes "Qf3xc6+ Bc4a6#", but Bc4a6# is only legal after
    Black plays bxc6).  Prefer env_mode='mask' for continuation-style data; this
    mode is kept for the single-move methods (best_move_only) and for backwards
    compatibility.
    """
    return _STRIP_ENV_RE.sub('', text)


def _split_env_segments(text: str) -> Tuple[str, set]:
    """Split a multi-turn continuation into text-without-tags + opponent word indices.

    Expected pattern (verified over the puzzle_seq shards):

        own <call_env> opp own <call_env> opp own ... <call_env>

    i.e. the first segment is the model's own move, and every segment AFTER a
    <call_env> starts with exactly one opponent reply followed by the model's next
    own move.  The trailing <call_env> yields an empty final segment.

    Returns:
        (text with the <call_env> tags removed, set of word indices that are
         opponent moves in that text)
    """
    words: List[str] = []
    opponent_word_idx = set()
    for seg_i, seg in enumerate(text.split(_ENV_TAG)):
        for word_i, word in enumerate(seg.split()):
            if seg_i >= 1 and word_i == 0:
                # First token after a <call_env> is the environment's reply.
                opponent_word_idx.add(len(words))
            words.append(word)
    return ' '.join(words), opponent_word_idx


def get_nested_field(data: Dict, field_path: str, default: Any = '') -> Any:
    """
    Get a nested field from a dictionary using dot notation.
    
    Examples:
        get_nested_field(data, 'cot_format')
        get_nested_field(data, 'cot_by_method.dfs_policy.cot_format_with_context')
        get_nested_field(data, 'multi_path_cot_policy.cot_format')
    """
    keys = field_path.split('.')
    value = data
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value


class SFTDataset(Dataset):
    """
    Dataset for SFT that loads chat-format data and applies loss masking.
    
    Each sample should be a JSON line with:
    - "prompt": the input/question (will be masked in loss)
    - "response": the target output (will be trained on)
    - Optional "system": system prompt
    
    For BoN/tree-generated data, supports configurable CoT format selection.
    """
    
    def __init__(
        self,
        data_files: List[str],
        tokenizer,
        seq_len: int = 512,
        mask_prompt: bool = True,
        cot_field: str = "cot_format",
        prompt_field: str = "pgn",
        strip_env_tokens: bool = False,
        env_mode: Optional[str] = None,
    ):
        """
        Args:
            data_files: List of JSON/JSONL file paths
            tokenizer: Tokenizer instance
            seq_len: Maximum sequence length
            mask_prompt: Whether to mask prompt tokens in loss calculation
            cot_field: Which CoT field to use for response. Supports dot notation for nested fields.
                Examples:
                - "cot_format" (default, backward compatible)
                - "cot_format_with_context" (includes PGN in response)
                - "cot_by_method.dfs_policy.cot_format_with_context"
                - "cot_by_method.dfs_verifier_layers.cot_format_with_markers_and_context"
                - "multi_path_cot_policy.cot_format_with_context"
            prompt_field: Which field to use for prompt (default: "pgn")
        """
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.mask_prompt = mask_prompt
        self.cot_field = cot_field
        self.prompt_field = prompt_field
        self.strip_env_tokens = strip_env_tokens

        # Resolve env handling.  Defaults reproduce the previous behaviour exactly.
        if env_mode is None:
            env_mode = 'strip' if strip_env_tokens else 'keep'
        if env_mode not in ('strip', 'mask', 'keep'):
            raise ValueError(f"env_mode must be one of 'strip'/'mask'/'keep', got {env_mode!r}")
        self.env_mode = env_mode

        # env_mode='mask' needs (a) force_lan encoding so that token spans are
        # concatenative over whitespace words, and (b) per-move tokenization to
        # compute those spans.  Fail loudly rather than mask the wrong tokens.
        if env_mode == 'mask':
            import inspect
            if 'force_lan' not in inspect.signature(tokenizer.encode).parameters:
                raise ValueError(
                    "env_mode='mask' requires a tokenizer whose encode() accepts "
                    "force_lan=True (e.g. LanTokenizerSFT); got "
                    f"{type(tokenizer).__name__}"
                )
            if not hasattr(tokenizer, '_lan_move_to_tokens'):
                raise ValueError(
                    "env_mode='mask' requires tokenizer._lan_move_to_tokens() to "
                    f"compute per-move token spans; got {type(tokenizer).__name__}"
                )

        # Cache the <T> token ID so __getitem__ can detect thinking responses
        self._t_start_id = tokenizer.get_vocab().get("<T>") if hasattr(tokenizer, "get_vocab") else None
        # Cache EOS ID to strip it from non-thinking responses
        self._eos_id = tokenizer.eos_id() if hasattr(tokenizer, "eos_id") and callable(tokenizer.eos_id) else None
        
        # Load all samples
        self.samples = []
        for file_path in data_files:
            self.samples.extend(self._load_file(file_path))
        
        print(f"[SFT Dataset] Loaded {len(self.samples)} samples from {len(data_files)} files")
        print(f"[SFT Dataset] Using cot_field='{cot_field}', prompt_field='{prompt_field}', env_mode='{self.env_mode}'")
        if self.env_mode == 'mask':
            n_masked = sum(len(s.get('env_spans') or []) for s in self.samples)
            n_with = sum(1 for s in self.samples if s.get('env_spans'))
            print(f"[SFT Dataset] env_mode=mask: {n_masked} opponent moves masked "
                  f"across {n_with}/{len(self.samples)} samples")
            # env_mode='mask' is for non-thinking continuation data (e.g.
            # solution_continuation), which must not carry <T>/</T>/<sep>.  Their
            # presence means the cot_field is a thinking format and the whole
            # own/opponent alternation assumed by _split_env_segments is wrong.
            _thinking = [t for t in ('<T>', '</T>', '<sep>')
                         if any(t in s['response'] for s in self.samples)]
            if _thinking:
                print(f"[SFT Dataset] WARNING: env_mode='mask' but responses contain "
                      f"{_thinking} — this is a thinking/multi-path format, and the "
                      f"opponent-move masking assumes plain move continuations. "
                      f"Check cot_field={self.cot_field!r}.")


    def _encode_response(self, text: str) -> List[int]:
        """Encode a response, forcing the LAN branch when spans must line up."""
        if self.env_mode == 'mask':
            return self.tokenizer.encode(text, force_lan=True)
        return self.tokenizer.encode(text)

    def _env_token_spans(self, text: str, opponent_word_idx: set) -> List[Tuple[int, int]]:
        """Map opponent word indices to [start, end) token spans in response space.

        Response space = self._encode_response(text) with the leading <bos> removed,
        i.e. index 0 is the first LAN token.  Valid because force_lan encoding is
        concatenative over whitespace words.
        """
        spans: List[Tuple[int, int]] = []
        pos = 0
        for i, word in enumerate(text.split()):
            n_tok = len(self.tokenizer._lan_move_to_tokens(word))
            if i in opponent_word_idx:
                spans.append((pos, pos + n_tok))
            pos += n_tok
        return spans

    def _prepare_response(self, raw: str) -> Tuple[str, List[Tuple[int, int]]]:
        """Apply env handling to a raw CoT string. Returns (text, env_spans)."""
        if self.env_mode == 'strip':
            return _strip_env_tokens(raw), []
        if self.env_mode == 'mask':
            text, opponent_word_idx = _split_env_segments(raw)
            return text, self._env_token_spans(text, opponent_word_idx)
        return raw, []


    def _load_file(self, file_path: str) -> List[Dict]:
        """Load samples from a JSON or JSONL file."""
        file_path = Path(file_path)
        
        # Try to detect format by reading the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        return self._load_bon_json(file_path, content)

    
    def _load_bon_json(self, file_path: Path, content: str) -> List[Dict]:
        """Load samples from BoN/tree JSON format (from best_of_n_generator.py or subtree_dfs_generator.py)."""
        samples = []
        try:
            data = json.loads(content)
            results = data.get('results', [])
            
            # Log available fields from first result for debugging
            if results:
                first = results[0]
                available_top_level = [k for k in first.keys() if 'cot' in k.lower()]
                if 'cot_by_method' in first:
                    available_methods = list(first['cot_by_method'].keys())
                    print(f"[SFT Dataset] Available cot_by_method keys: {available_methods}")
                print(f"[SFT Dataset] Available top-level CoT fields: {available_top_level}")
            
            skipped = 0
            filtered_by_length = 0
            for result in results:
                # Get prompt using configured field
                prompt = get_nested_field(result, self.prompt_field, '').replace('\n', ' ').strip()
                
                # Get response using configured CoT field (supports nested access)
                cot_format = get_nested_field(result, self.cot_field, '').strip()
                cot_format, env_spans = self._prepare_response(cot_format)

                if not prompt or not cot_format:
                    skipped += 1
                    continue
                
                # Tokenize the same way as in __getitem__
                prompt_token_ids = self.tokenizer.encode(prompt)[:-1]  # remove the eos token
                response_token_ids = self._encode_response(cot_format)[1:]  # remove the bos token
                # For non-thinking responses, strip trailing EOS so model learns to continue
                is_thinking = (self._t_start_id is not None and len(response_token_ids) > 0
                               and response_token_ids[0] == self._t_start_id)
                if not is_thinking and self._eos_id is not None and len(response_token_ids) > 0 and response_token_ids[-1] == self._eos_id:
                    response_token_ids = response_token_ids[:-1]
                token_ids = prompt_token_ids + response_token_ids
                
                if len(token_ids) > self.seq_len:
                    filtered_by_length += 1
                    continue
                
                samples.append({
                    'prompt': prompt,
                    'response': cot_format,
                    # [start, end) token spans (response space) to mask out of the loss
                    'env_spans': env_spans,
                    # Store metadata for debugging
                    'target_move': result.get('target_move', ''),
                    'target_move_san': result.get('target_move_san', ''),
                })
            
            print(f"[SFT Dataset] Loaded {len(samples)} samples from BoN JSON: {file_path}")
            if skipped > 0:
                print(f"[SFT Dataset] Skipped {skipped} samples (empty prompt or response for field '{self.cot_field}')")
            if filtered_by_length > 0:
                print(f"[SFT Dataset] Filtered out {filtered_by_length} samples (longer than {self.seq_len} tokens)")
        except json.JSONDecodeError as e:
            print(f"[Warning] Failed to parse BoN JSON file {file_path}: {e}")
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            input_ids: Token IDs for the full sequence
            labels: Token IDs for loss calculation (with prompt masked if enabled)
            attention_mask: Mask for valid tokens (1) vs padding (0)
        """
        sample = self.samples[idx]
        prompt = sample.get('prompt', '')
        response = sample.get('response', '')
        
        # Build full text: prompt + space + response
        # For BoN format: "pgn <T> <sep> traj1 <sep> ... <sep> <T> answer"
        # formatted_text = f"{prompt} {response}"
        
        # Encode the full text
        prompt_token_ids = self.tokenizer.encode(prompt)[:-1] # remove the eos token
        response_token_ids = self._encode_response(response)[1:] # remove the bos token

        response_starts_with_T = (
            self._t_start_id is not None
            and len(response_token_ids) > 0
            and response_token_ids[0] == self._t_start_id
        )
        # For non-thinking responses, strip trailing EOS so the model learns to
        # continue predicting rather than terminate after the move.
        if not response_starts_with_T and self._eos_id is not None and len(response_token_ids) > 0 and response_token_ids[-1] == self._eos_id:
            response_token_ids = response_token_ids[:-1]

        token_ids = prompt_token_ids + response_token_ids

        # Calculate prompt_length for masking.
        # If the response starts with <T>, also mask that token (it's part of the prompt frame,
        # not a token the model should learn to predict). For non-thinking data there is no <T>,
        # so we do NOT add the extra +1.
        if self.mask_prompt:
            prompt_length = len(prompt_token_ids) + (1 if response_starts_with_T else 0)
        else:
            prompt_length = 0
        # Truncate if too long
        if len(token_ids) > self.seq_len:
            token_ids = token_ids[:self.seq_len]
            prompt_length = min(prompt_length, self.seq_len - 1)
        
        # Create input_ids and labels
        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)  # All except last token
        labels = torch.tensor(token_ids[1:], dtype=torch.long)      # Shifted by 1 (next token prediction)
        
        # Create attention mask (all 1s for now, will be padded in collate_fn)
        attention_mask = torch.ones_like(input_ids)
        
        # Mask prompt tokens in labels if enabled
        if self.mask_prompt and prompt_length > 1:
            # Adjust prompt_length for the shift (input_ids = tokens[:-1], labels = tokens[1:])
            mask_length = min(prompt_length - 1, len(labels))
            labels[:mask_length] = -100  # -100 is ignored by PyTorch CrossEntropyLoss

        # Mask the environment's (opponent's) replies out of the loss.  They stay in
        # input_ids so the board the model conditions on remains legal; only the
        # gradient is removed.
        #
        # Index algebra: token_ids = prompt + response, so response token r_i lives
        # at token_ids[L + i] with L = len(prompt_token_ids).  Since
        # labels[j] = token_ids[j + 1], r_i is at labels[L + i - 1].
        env_spans = sample.get('env_spans') or []
        if env_spans:
            L = len(prompt_token_ids)
            n_labels = len(labels)
            for start, end in env_spans:
                lo = max(L + start - 1, 0)
                hi = min(L + end - 1, n_labels)
                if hi > lo:
                    labels[lo:hi] = -100

        return input_ids, labels, attention_mask


def collate_fn_sft(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], 
                   pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    """
    Collate function for SFT batching with padding.
    
    Args:
        batch: List of (input_ids, labels, attention_mask) tuples
        pad_token_id: Token ID to use for padding
    
    Returns:
        Dictionary with batched and padded tensors
    """
    input_ids_list, labels_list, attention_mask_list = zip(*batch)
    
    # Find max length in batch
    max_len = max(len(ids) for ids in input_ids_list)
    
    # Pad sequences
    batch_input_ids = []
    batch_labels = []
    batch_attention_mask = []
    
    for input_ids, labels, attention_mask in zip(input_ids_list, labels_list, attention_mask_list):
        pad_len = max_len - len(input_ids)
        
        # Pad input_ids
        padded_input_ids = torch.cat([
            input_ids,
            torch.full((pad_len,), pad_token_id, dtype=torch.long)
        ])
        
        # Pad labels (use -100 for padding so it's ignored in loss)
        padded_labels = torch.cat([
            labels,
            torch.full((pad_len,), -100, dtype=torch.long)
        ])
        
        # Pad attention_mask
        padded_attention_mask = torch.cat([
            attention_mask,
            torch.zeros(pad_len, dtype=torch.long)
        ])
        
        batch_input_ids.append(padded_input_ids)
        batch_labels.append(padded_labels)
        batch_attention_mask.append(padded_attention_mask)
    
    return {
        'input_ids': torch.stack(batch_input_ids),
        'labels': torch.stack(batch_labels),
        'attention_mask': torch.stack(batch_attention_mask),
    }


class MultiTurnSFTDataset(SFTDataset):
    """
    SFT Dataset for multi-turn CoT training with environment interaction masking.

    Extends SFTDataset by additionally masking the environment's response tokens
    that follow each <call_env> token in the CoT.

    Expected sequence pattern:
        [model_move] <call_env> [env_move] [model_move] <call_env> [env_move] ...

    Masking rules applied to `labels` (in-place):
      - <call_env> itself is NOT masked (model learns to generate it).
      - The one move immediately after <call_env> IS masked (opponent's response).
        A "move" consists of: piece, from-sq, [x], to-sq, [=, promo-piece], [+/#].
        Masking stops as soon as the first token of the next model move is seen
        (a piece/castling token that is NOT a promotion target).
    """

    # Chess tokens that START a move (piece initials + castling)
    _MOVE_START_STRS = ("K", "Q", "R", "B", "N", "P", "O-O", "O-O-O")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        vocab = self.tokenizer.get_vocab()

        # <call_env> token ID
        self._call_env_id: Optional[int] = None
        try:
            self._call_env_id = self.tokenizer.call_env_id()
        except (ValueError, AttributeError):
            print(
                "[MultiTurnSFTDataset] Warning: call_env_id() not available on tokenizer. "
                "Env-response masking will be skipped (set include_env_tokens=True in tokenizer config)."
            )

        # IDs of tokens that START a chess move (used to detect move boundaries)
        self._move_start_ids: set = {vocab[t] for t in self._MOVE_START_STRS if t in vocab}

        # IDs of structural tokens that unconditionally end the env-response window
        self._structural_ids: set = set()
        for tok in ("<T>", "</T>", "<sep>", "<bos>", "<eos>"):
            if tok in vocab:
                self._structural_ids.add(vocab[tok])
        # Also treat any other env tokens (verify, reward) as structural terminators
        try:
            for t, tid in self.tokenizer.env_token_ids().items():
                if t != "<call_env>":
                    self._structural_ids.add(tid)
        except (ValueError, AttributeError):
            pass

        # Promotion "=" token ID (needed to distinguish promotion target from move start)
        self._equal_id: Optional[int] = vocab.get("=")

    def _mask_env_responses(self, input_ids: torch.Tensor, labels: torch.Tensor) -> None:
        """
        In-place masking of env response tokens in `labels`.

        input_ids[j] = token_ids[j]  (what the model sees as input at step j)
        labels[j]    = token_ids[j+1] (what the model should predict at step j)

        The env move tokens live in labels, shifted one position earlier than
        they appear in input_ids.  Concretely, after <call_env> at input_ids[j]:
            labels[j]   = first env token  (the opponent's piece)
            labels[j+1] = second env token (from-square)
            ...
        so we must start masking at labels[j], NOT labels[j+1].

        Algorithm (boundary detection uses labels[j], NOT input_ids[j]):
          When input_ids[j] == call_env_id:
            Enter env-response mode; fall through so labels[j] is processed
            in the same iteration.
          While in env-response mode, for each position j:
            • labels[j] is a structural token  → stop (leave unmasked).
            • labels[j] is a move-start token  (K/Q/R/B/N/P/O-O/O-O-O)
              AND the prior label was not '=' (promotion target):
                – first such token  → piece of the env move → mask it.
                – second such token → piece of the model's next move → stop.
            • Otherwise                          → interior of env move → mask.
        """
        if self._call_env_id is None:
            return

        in_env_response = False
        env_move_started = False
        last_was_equal = False
        n = len(input_ids)

        for j in range(n):
            tok_j = input_ids[j].item()

            # ---- Handle <call_env> in input ----
            if tok_j == self._call_env_id:
                in_env_response = True
                env_move_started = False
                last_was_equal = False
                # NOTE: no `continue` — fall through so labels[j] (= the first
                # env-response token) is processed immediately below.

            # ---- Env-response masking: boundary detection via labels[j] ----
            if in_env_response:
                label_j = labels[j].item()

                if label_j in self._structural_ids:
                    # Structural delimiter → stop, leave token unmasked
                    in_env_response = False
                    env_move_started = False
                    last_was_equal = False

                else:
                    is_label_move_start = (
                        label_j in self._move_start_ids and not last_was_equal
                    )
                    if is_label_move_start:
                        if not env_move_started:
                            # Piece that starts the env move → mask it
                            env_move_started = True
                            labels[j] = -100
                            last_was_equal = False  # piece tokens are never '='
                        else:
                            # Piece of the model's next move → stop, leave unmasked
                            in_env_response = False
                            env_move_started = False
                            last_was_equal = False
                    else:
                        # Interior of env move (from-sq, 'x', to-sq, '=', promo, '+', '#')
                        labels[j] = -100
                        last_was_equal = (
                            label_j == self._equal_id
                            if self._equal_id is not None
                            else False
                        )
            else:
                last_was_equal = False

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids, labels, attention_mask = super().__getitem__(idx)
        self._mask_env_responses(input_ids, labels)
        return input_ids, labels, attention_mask


def create_multi_turn_sft_dataloader(
    data_files: List[str],
    tokenizer,
    batch_size: int = 8,
    seq_len: int = 512,
    num_workers: int = 4,
    shuffle: bool = True,
    mask_prompt: bool = True,
    pad_token_id: Optional[int] = None,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    cot_field: str = "cot_format",
    prompt_field: str = "pgn",
    strip_env_tokens: bool = False,
    env_mode: Optional[str] = None,
) -> DataLoader:
    """
    Create a DataLoader for multi-turn SFT training.

    Identical to create_sft_dataloader but uses MultiTurnSFTDataset so that
    environment response tokens (those following <call_env> up to <sep>) are
    masked in the loss, allowing the model to learn only its own outputs.

    Args:
        data_files: List of JSONL file paths
        tokenizer: Tokenizer instance (must have include_env_tokens=True for masking)
        batch_size: Batch size
        seq_len: Maximum sequence length
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        mask_prompt: Whether to mask prompt tokens in loss
        pad_token_id: Token ID for padding (defaults to tokenizer's pad_id or 0)
        prefetch_factor: Number of batches to prefetch per worker
        persistent_workers: Keep workers alive between epochs
        cot_field: Which CoT field to use for response (supports dot notation)
        prompt_field: Which field to use for prompt (default: "pgn")

    Returns:
        DataLoader instance
    """
    dataset = MultiTurnSFTDataset(
        data_files=data_files,
        tokenizer=tokenizer,
        seq_len=seq_len,
        mask_prompt=mask_prompt,
        cot_field=cot_field,
        prompt_field=prompt_field,
        strip_env_tokens=strip_env_tokens,
        env_mode=env_mode,
    )

    if pad_token_id is None:
        if hasattr(tokenizer, 'pad_id') and callable(tokenizer.pad_id):
            pad_token_id = tokenizer.pad_id()
        elif hasattr(tokenizer, 'pad_token_id'):
            pad_token_id = tokenizer.pad_token_id
        else:
            pad_token_id = 0
            print("[Warning] No pad token found in tokenizer, using 0")

    def collate_fn(batch):
        return collate_fn_sft(batch, pad_token_id=pad_token_id)

    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'collate_fn': collate_fn,
        'pin_memory': True,
    }

    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = prefetch_factor
        loader_kwargs['persistent_workers'] = persistent_workers

    return DataLoader(dataset, **loader_kwargs)


def create_sft_dataloader(
    data_files: List[str],
    tokenizer,
    batch_size: int = 8,
    seq_len: int = 512,
    num_workers: int = 4,
    shuffle: bool = True,
    mask_prompt: bool = True,
    pad_token_id: Optional[int] = None,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    cot_field: str = "cot_format",
    prompt_field: str = "pgn",
    strip_env_tokens: bool = False,
    env_mode: Optional[str] = None,
) -> DataLoader:
    """
    Create a DataLoader for SFT training.
    
    Args:
        data_files: List of JSONL file paths
        tokenizer: Tokenizer instance
        batch_size: Batch size
        seq_len: Maximum sequence length
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        mask_prompt: Whether to mask prompt in loss
        pad_token_id: Token ID for padding (defaults to tokenizer's pad_id or 0)
        prefetch_factor: Number of batches to prefetch per worker
        persistent_workers: Keep workers alive between epochs
        cot_field: Which CoT field to use for response. Supports dot notation for nested fields.
            Examples:
            - "cot_format" (default)
            - "cot_format_with_context"
            - "cot_by_method.dfs_policy.cot_format_with_context"
            - "cot_by_method.dfs_verifier_layers.cot_format_with_markers_and_context"
            - "multi_path_cot_policy.cot_format_with_context"
        prompt_field: Which field to use for prompt (default: "pgn")
    
    Returns:
        DataLoader instance
    """
    dataset = SFTDataset(
        data_files=data_files,
        tokenizer=tokenizer,
        seq_len=seq_len,
        mask_prompt=mask_prompt,
        cot_field=cot_field,
        prompt_field=prompt_field,
        strip_env_tokens=strip_env_tokens,
        env_mode=env_mode,
    )

    # Determine pad token ID
    if pad_token_id is None:
        if hasattr(tokenizer, 'pad_id') and callable(tokenizer.pad_id):
            pad_token_id = tokenizer.pad_id()
        elif hasattr(tokenizer, 'pad_token_id'):
            pad_token_id = tokenizer.pad_token_id
        else:
            pad_token_id = 0
            print(f"[Warning] No pad token found in tokenizer, using 0")
    
    # Create collate function with pad token
    def collate_fn(batch):
        return collate_fn_sft(batch, pad_token_id=pad_token_id)
    
    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'collate_fn': collate_fn,
        'pin_memory': True,
    }
    
    # Add prefetch and persistent workers if using multiple workers
    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = prefetch_factor
        loader_kwargs['persistent_workers'] = persistent_workers
    
    return DataLoader(dataset, **loader_kwargs)

