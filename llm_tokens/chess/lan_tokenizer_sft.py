# lan_tokenizer_sft.py
"""
LAN Tokenizer with SFT support (CoT format with <T> and <sep> tokens).

This extends the base LAN tokenizer with SFT-specific functionality:
- <T> token for marking thinking/CoT content
- <sep> token for separating prompt from response
"""
from typing import List, Dict, Optional, Tuple
import io
import chess, chess.pgn
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from .base_tokenizer import BaseTokenizer

_RESULT = {"1-0", "0-1", "1/2-1/2", "*"}
FILES = "abcdefgh"
RANKS = "12345678"
SQUARES = [f+r for f in FILES for r in RANKS]
PROMOS = "QRBN"
DIGITS = set("0123456789")

# SFT special tokens for CoT format
T_TOKEN = "<T>"
T_END_TOKEN = "</T>"
SEP_TOKEN = "<sep>"

# Environment interaction / reward special tokens
CALL_ENV_TOKEN = "<call_env>"
VERIFY_TOKEN = "<verify>"
REWARD_POS_TOKEN = "<+1>"
REWARD_NEG_TOKEN = "<-1>"
REWARD_ZERO_TOKEN = "<0>"
ENV_TOKENS = [CALL_ENV_TOKEN]
REWARD_TOKENS = [VERIFY_TOKEN, REWARD_POS_TOKEN, REWARD_NEG_TOKEN, REWARD_ZERO_TOKEN]

def _vocab_with_sft(
    include_move_numbers: bool,
    keep_result: bool,
    bos: str,
    eos: str,
    unk: str,
    include_env_tokens: bool = False,
    include_reward_tokens: bool = False,
) -> Dict[str, int]:
    """Create vocabulary including SFT special tokens."""
    base = [bos, eos, unk]
    ops = ["x", "=", "+", "#", "O-O", "O-O-O", ".", "..."]
    toks = base + list("KQRBNP") + SQUARES + list(PROMOS) + ops
    if include_move_numbers:
        toks += list("0123456789")
    if keep_result:
        toks += sorted(_RESULT)

    # Add SFT special tokens for CoT format
    sft_tokens = [T_TOKEN, T_END_TOKEN, SEP_TOKEN]
    toks += sft_tokens

    # Add environment / reward tokens when requested
    if include_env_tokens:
        toks += ENV_TOKENS
    if include_reward_tokens:
        toks += REWARD_TOKENS

    return {t: i for i, t in enumerate(dict.fromkeys(toks))}


class LanTokenizerSFT(BaseTokenizer):
    """
    LAN Tokenizer with SFT capabilities.

    This tokenizer extends the base LAN tokenizer with:
    - <T> token for marking thinking/CoT boundaries
    - <sep> token for separating candidate trajectories

    CoT Format: {prompt} <T> <sep> {traj1} <sep> {traj2} <sep> ... <sep> {trajN} <sep> <T> {answer}

    Where:
    - {prompt}: The game history/board state (PGN moves)
    - {trajN}: Candidate reasoning trajectories
    - {answer}: The final best move
    """

    # Special tokens for CoT format
    T = T_TOKEN
    T_END = T_END_TOKEN
    SEP = SEP_TOKEN

    # Environment / reward tokens (class-level constants for easy access)
    CALL_ENV = CALL_ENV_TOKEN   # "<call_env>"
    VERIFY = VERIFY_TOKEN       # "<verify>"
    REWARD_POS = REWARD_POS_TOKEN  # "<+1>"
    REWARD_NEG = REWARD_NEG_TOKEN  # "<-1>"
    REWARD_ZERO = REWARD_ZERO_TOKEN  # "<0>"
    ENV_TOKENS = ENV_TOKENS     # full list

    def __init__(self, config: Optional[dict] = None):
        """
        Args:
            config: Configuration dict with tokenizer settings.
                include_env_tokens (bool): add <call_env>, <verify>, <+1>, <-1>, <0>
                    to the vocabulary.  Default: False.
        """
        config = config or {}

        include_move_numbers = config.get("include_move_numbers", False)
        include_black_tripledots = config.get("include_black_tripledots", False)
        bos = config.get("bos", "<bos>")
        eos = config.get("eos", "<eos>")
        unk = config.get("unk", "<unk>")
        keep_result = config.get("keep_result", False)
        include_env_tokens = config.get("include_env_tokens", False)
        include_reward_tokens = config.get("include_reward_tokens", False)

        self._bos = bos
        self._eos = eos
        self._unk = unk
        self._keep_res = keep_result
        self._include_nums = include_move_numbers
        self._include_black_ellipses = include_black_tripledots
        self._include_env_tokens = include_env_tokens
        self._include_reward_tokens = include_reward_tokens

        # Create vocabulary with SFT tokens
        tok2id = _vocab_with_sft(
            include_move_numbers, keep_result, bos, eos, unk,
            include_env_tokens=include_env_tokens,
            include_reward_tokens=include_reward_tokens,
        )
        self._tok2id = tok2id

        # Initialize tokenizer
        self.tk = Tokenizer(WordLevel(vocab=tok2id, unk_token=self._unk))
        self.tk.pre_tokenizer = WhitespaceSplit()
    
    def _pgn_to_tokens(self, text: str) -> Optional[List[str]]:
        """Convert PGN text to tokens."""
        import os, contextlib
        with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
            g = chess.pgn.read_game(io.StringIO(text))
        if g is None:
            return None
        
        b, out, n = g.board(), [], 1
        for mv in g.mainline_moves():
            if b.turn == chess.WHITE and self._include_nums:
                out += list(str(n)) + (
                    ["..."] if self._include_black_ellipses and b.fullmove_number < n else ["."]
                )
            
            if b.is_castling(mv):
                b.push(mv)
                suf = "#" if b.is_checkmate() else ("+" if b.is_check() else "")
                b.pop()
                out.append("O-O" if chess.square_file(mv.to_square) == 6 else "O-O-O")
                if suf:
                    out.append(suf)
                b.push(mv)
            else:
                piece = b.piece_at(mv.from_square).symbol().upper()
                frm = chess.square_name(mv.from_square)
                to = chess.square_name(mv.to_square)
                is_cap = b.is_capture(mv)
                promo = mv.promotion
                
                b.push(mv)
                suf = "#" if b.is_checkmate() else ("+" if b.is_check() else "")
                
                # Emit LAN tokens
                out.append(piece)
                out.append(frm)
                if is_cap:
                    out.append("x")
                out.append(to)
                if promo:
                    out += ["=", chess.piece_symbol(promo).upper()]
                if suf:
                    out.append(suf)
            
            if b.turn == chess.WHITE:
                n += 1
        
        res = g.headers.get("Result")
        if self._keep_res and res in _RESULT:
            out.append(res)
        
        return out
    
    def _lan_move_to_tokens(self, move: str) -> List[str]:
        """
        Convert a single LAN move to tokens.
        
        LAN format: [Piece][from_square][x]?[to_square][=Promo]?[+#]?
        
        Examples:
            "Ng1f3" -> ["N", "g1", "f3"]
            "Nd4xe6" -> ["N", "d4", "x", "e6"]
            "Pe2e4" -> ["P", "e2", "e4"]
            "Pe4xd5" -> ["P", "e4", "x", "d5"]
            "O-O" -> ["O-O"]
            "O-O-O" -> ["O-O-O"]
            "Pe7e8=Q" -> ["P", "e7", "e8", "=", "Q"]
            "Ng1f3+" -> ["N", "g1", "f3", "+"]
        """
        # Handle castling
        if move in {"O-O", "O-O-O"}:
            return [move]
        if move.rstrip("+#") in {"O-O", "O-O-O"}:
            base = move.rstrip("+#")
            suffix = move[len(base):]
            return [base] + ([suffix] if suffix else [])
        
        out = []
        i = 0
        n = len(move)
        
        # Get piece letter (required in LAN format)
        if i < n and move[i] in "KQRBNP":
            out.append(move[i])
            i += 1
        else:
            # No piece letter - might be malformed, return as-is
            return [move]
        
        # Get from square (required in LAN format)
        if i + 1 < n and move[i] in FILES and move[i + 1] in RANKS:
            out.append(move[i:i+2])
            i += 2
        
        # Handle capture
        if i < n and move[i] == "x":
            out.append("x")
            i += 1
        
        # Get to square (required in LAN format)
        if i + 1 < n and move[i] in FILES and move[i + 1] in RANKS:
            out.append(move[i:i+2])
            i += 2
        
        # Handle promotion
        if i < n and move[i] == "=":
            out.append("=")
            i += 1
            if i < n and move[i] in PROMOS:
                out.append(move[i])
                i += 1
        
        # Handle check/checkmate
        if i < n and move[i] in "+#":
            out.append(move[i])
            i += 1
        
        return out
    
    def _active_env_tokens(self) -> set:
        """Return the set of env tokens that are active for this instance."""
        return set(ENV_TOKENS) if self._include_env_tokens else set()

    def _cot_to_tokens(self, text: str) -> List[str]:
        """
        Convert CoT formatted text to tokens.
        Handles special tokens and LAN moves.
        """
        env_toks = self._active_env_tokens()
        out = []
        for token in text.split():
            if token in {self.T, self.T_END, self.SEP} or token in env_toks:
                # Keep special tokens as-is
                out.append(token)
            elif token in _RESULT:
                # Game result
                out.append(token)
            elif token and token[0].isdigit() and "." in token:
                # Move number like "1." or "15..."
                # Split into digits and dots
                num_part = token.rstrip(".")
                dot_part = token[len(num_part):]
                out.extend(list(num_part))
                if dot_part:
                    out.append("..." if len(dot_part) > 1 else ".")
            elif token and all(c.isdigit() for c in token):
                # Pure number - tokenize each digit
                out.extend(list(token))
            else:
                # LAN move - tokenize it
                out.extend(self._lan_move_to_tokens(token))
        return out
    
    def encode(self, text: str, force_lan: bool = False) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Text to encode (can be PGN or CoT formatted)
            force_lan: Treat `text` as a whitespace-separated list of LAN moves and
                skip both the CoT and the PGN detection branches.

                This matters for SFT *responses*.  `_pgn_to_tokens` is tried first
                below and is accepted whenever it yields ANY moves, but
                python-chess accepts long-algebraic SAN, so a response whose first
                move happens to be legal from the *starting* position is parsed as
                a game and everything after the first illegal move is silently
                dropped:

                    "Pf2f3 Qg4h3 Pd4xc5" -> ['P','f2','f3']       (rest lost)
                    "Pg2g4#"             -> ['P','g2','g4']       ('#' lost)

                force_lan=True also makes encoding *concatenative* over
                whitespace words, which callers rely on to compute per-move token
                spans (see SFTDataset._env_token_spans).

        Returns:
            List of token IDs
        """
        if force_lan:
            lan_tokens = []
            for word in text.split():
                lan_tokens.extend(self._lan_move_to_tokens(word))
            tokens = [self._bos] + lan_tokens + [self._eos]
            return self.tk.encode(" ".join(tokens)).ids

        # Check if this is CoT-formatted text (contains special tokens)
        sft_special = (
            [self.T, self.T_END, self.SEP]
            + (ENV_TOKENS if self._include_env_tokens else [])
        )
        is_cot_format = any(token in text for token in sft_special)
        
        if is_cot_format:
            t_idx = text.index(self.T)
            prompt_part = text[:t_idx].strip()
            rest_part = text[t_idx:]  # starts with <T>

            pgn_tokens = self._pgn_to_tokens(prompt_part) if prompt_part else None
            if pgn_tokens is None:
                pgn_tokens = self._cot_to_tokens(prompt_part) if prompt_part else []
            rest_tokens = self._cot_to_tokens(rest_part)
            tokens = [self._bos] + pgn_tokens + rest_tokens + [self._eos]
        else:
            pgn_tokens = self._pgn_to_tokens(text)
            if pgn_tokens is not None and len(pgn_tokens) > 0:
                tokens = [self._bos] + pgn_tokens + [self._eos]
            else:
                # Not valid PGN — treat each word as a LAN move
                lan_tokens = []
                for word in text.split():
                    lan_tokens.extend(self._lan_move_to_tokens(word))
                tokens = [self._bos] + lan_tokens + [self._eos]
        
        return self.tk.encode(" ".join(tokens)).ids
    
    def decode(self, ids: List[int]) -> str:
        """
        Decode token IDs to text.
        
        Args:
            ids: List of token IDs
        
        Returns:
            Decoded text
        """
        toks = [t for t in self.tk.decode(ids).split() if t not in {self._bos, self._eos}]
        
        # Otherwise, use LAN decoding logic
        out: List[str] = []
        i, n = 0, len(toks)
        
        while i < n:
            t = toks[i]
            
            if t in {self.T, self.T_END, self.SEP} or t in _RESULT or t in self._active_env_tokens():
                out.append(t)
                i += 1
                continue
            
            if t and all(ch in DIGITS for ch in t):
                j = i
                num = []
                while j < n and all(ch in DIGITS for ch in toks[j]):
                    num.append(toks[j])
                    j += 1
                dots = ""
                if j < n and toks[j] in {".", "..."}:
                    dots = toks[j]
                    j += 1
                out.append("".join(num) + dots)
                i = j
                continue
            
            if t in {"O-O", "O-O-O"}:
                j = i + 1
                suf = toks[j] if j < n and toks[j] in {"+", "#"} else ""
                if suf:
                    j += 1
                out.append(t + suf)
                i = j
                continue
            
            if t in set("KQRBNP"):
                piece = t
                j = i + 1
                frm = toks[j] if j < n else ""
                j += 1
                cap = ""
                if j < n and toks[j] == "x":
                    cap = "x"
                    j += 1
                to = toks[j] if j < n else ""
                j += 1
                promo = ""
                if j + 1 <= n - 1 and toks[j] == "=" and toks[j + 1] in set(PROMOS):
                    promo = "=" + toks[j + 1]
                    j += 2
                suf = ""
                if j < n and toks[j] in {"+", "#"}:
                    suf = toks[j]
                    j += 1
                lan = f"{piece}{frm}{cap}{to}{promo}{suf}"
                out.append(lan)
                i = j
                continue
            
            out.append(t)
            i += 1
        
        return " ".join(out)
    
    def get_vocab(self) -> Dict[str, int]:
        """Get token-to-id vocabulary mapping."""
        return self._tok2id
    
    def bos_id(self) -> Optional[int]:
        """Get BOS token ID."""
        return self._tok2id[self._bos]
    
    def eos_id(self) -> Optional[int]:
        """Get EOS token ID."""
        return self._tok2id[self._eos]
    
    def pad_id(self) -> Optional[int]:
        """Get PAD token ID (uses BOS as pad by default)."""
        return self._tok2id.get("<pad>", self.bos_id())
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self._tok2id)
    
    def t_id(self) -> int:
        """Get <T> token ID."""
        return self._tok2id[self.T]
    
    def sep_id(self) -> int:
        """Get <sep> token ID."""
        return self._tok2id[self.SEP]
    
    def t_end_id(self) -> int:
        """Get </T> token ID."""
        return self._tok2id[self.T_END]
    
    # ------------------------------------------------------------------
    # Environment / reward token accessors
    # ------------------------------------------------------------------

    def _require_env_tokens(self) -> None:
        if not self._include_env_tokens:
            raise ValueError(
                "Environment tokens are not enabled. "
                "Pass include_env_tokens=True in the config."
            )

    def call_env_id(self) -> int:
        """Get <call_env> token ID."""
        self._require_env_tokens()
        return self._tok2id[CALL_ENV_TOKEN]

    def verify_id(self) -> int:
        """Get <verify> token ID."""
        self._require_env_tokens()
        return self._tok2id[VERIFY_TOKEN]

    def reward_pos_id(self) -> int:
        """Get <+1> (positive reward) token ID."""
        self._require_env_tokens()
        return self._tok2id[REWARD_POS_TOKEN]

    def reward_neg_id(self) -> int:
        """Get <-1> (negative reward) token ID."""
        self._require_env_tokens()
        return self._tok2id[REWARD_NEG_TOKEN]

    def reward_zero_id(self) -> int:
        """Get <0> (zero reward) token ID."""
        self._require_env_tokens()
        return self._tok2id[REWARD_ZERO_TOKEN]

    def reward_id(self, value) -> int:
        """
        Get reward token ID by numeric value.

        Args:
            value: 1, -1, or 0  (or the strings "+1", "-1", "0")

        Returns:
            Token ID for the corresponding reward token.
        """
        self._require_env_tokens()
        mapping = {1: REWARD_POS_TOKEN, -1: REWARD_NEG_TOKEN, 0: REWARD_ZERO_TOKEN,
                   "+1": REWARD_POS_TOKEN, "-1": REWARD_NEG_TOKEN, "0": REWARD_ZERO_TOKEN}
        if value not in mapping:
            raise ValueError(f"reward value must be one of 1, -1, 0 (or '+1', '-1', '0'), got {value!r}")
        return self._tok2id[mapping[value]]

    def env_token_ids(self) -> Dict[str, int]:
        """Get mapping of all env/reward special tokens to their IDs."""
        self._require_env_tokens()
        return {tok: self._tok2id[tok] for tok in ENV_TOKENS}
    
    def extract_parts(self, text: str) -> Tuple[Optional[str], Optional[List[str]], str]:
        """
        Extract prompt, trajectories and answer from BoN CoT formatted text.
        
        Args:
            text: Text in format: {prompt} <T> <sep> {traj1} <sep> ... <sep> <T> {answer}
        
        Returns:
            prompt: The prompt/context (or None if not present)
            trajectories: List of trajectory strings (or None if not present)
            answer: The final answer
        """
        if self.T not in text:
            return None, None, text
        
        try:
            # Split by <T> to get prompt, thinking section, and answer
            t_parts = text.split(self.T)
            if len(t_parts) < 3:
                return None, None, text
            
            # t_parts[0] is prompt (before first <T>)
            # t_parts[1] is the thinking section with trajectories
            # t_parts[2] is the answer
            prompt = t_parts[0].strip() if t_parts[0].strip() else None
            thinking_section = t_parts[1].strip()
            answer = t_parts[2].strip()
            
            # Split thinking section by <sep> to get trajectories
            trajectories = [t.strip() for t in thinking_section.split(self.SEP) if t.strip()]
            
            return prompt, trajectories, answer
        except (ValueError, IndexError):
            return None, None, text
    
    def extract_thinking_and_answer(self, text: str) -> Tuple[Optional[List[str]], str]:
        """
        Extract trajectories and answer from BoN CoT formatted text (ignores prompt).
        
        Args:
            text: Text in format: {prompt} <T> <sep> {traj1} <sep> ... <sep> <T> {answer}
        
        Returns:
            trajectories: List of trajectory strings (or None if not present)
            answer: The final answer
        """
        _, trajectories, answer = self.extract_parts(text)
        return trajectories, answer
    
    def get_sft_special_tokens(self) -> List[str]:
        """Get list of SFT special tokens (including env/reward tokens if enabled)."""
        toks = [self.T, self.T_END, self.SEP]
        if self._include_env_tokens:
            toks += ENV_TOKENS
        return toks

    def get_sft_token_ids(self) -> Dict[str, int]:
        """Get mapping of SFT special tokens to their IDs."""
        result = {
            self.T: self._tok2id[self.T],
            self.T_END: self._tok2id[self.T_END],
            self.SEP: self._tok2id[self.SEP],
        }
        if self._include_env_tokens:
            for tok in ENV_TOKENS:
                result[tok] = self._tok2id[tok]
        return result
    
    def parse_cot_line(self, line: str) -> Tuple[Optional[List[str]], Optional[str]]:
        """
        Parse a CoT data line in format: <T> <sep> ... <sep> <T> {answer}
        
        Args:
            line: A line from the CoT data file
        
        Returns:
            trajectories: List of trajectory strings
            answer: The final answer/move
        """
        line = line.strip()
        if not line or not line.startswith(self.T):
            return None, None
        
        return self.extract_thinking_and_answer(line)

