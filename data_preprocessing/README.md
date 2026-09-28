# Data preprocessing

End-to-end pipeline that turns raw Lichess data into the tokenized shards consumed
by `pretraining/`. Each stage is a single, self-contained script (one input → one
output) meant to be easy to read and adapt.

## Released dataset (skip the pipeline)

The final tokenized pretraining dataset is **released on Hugging Face**, so you do
not need to rerun any of this to reproduce training:

- **`pavelslab-nyu/pretrain_v1_54B`** — tokenized `.npy` shards (~20B tokens).

```bash
huggingface-cli download chess-pre-to-post/pretrain_v1_20b \
    --repo-type dataset --local-dir /path/to/pretrain_v1_20b
```

Point `pretraining/run_pretrain.sh`'s `DATA_DIR` at that directory. The pipeline
below documents how that dataset was built from scratch.

## Requirements

`python-chess`, `pandas`, `pyarrow`, `duckdb`, `numpy`, `requests`, `tqdm`, plus
the repo-root `llm_tokens/` package (used by the tokenizer step; imported
automatically — no install needed). The `pretraining` conda env covers all of
these.

---

## Pipeline

### 1. Human games — download (website only)

Raw human games come from the public **Lichess game database**:
<https://database.lichess.org/>. Download the monthly standard-game dumps you want
(PGN), and convert them to parquet with columns including `id`, `white_elo`,
`black_elo`, `event` (and the move/text fields). There is no crawler here — just
fetch the dumps from the site.

### 2. Human games — preprocess

`preprocess_human_games.py` cleans a parquet (file or directory): drops bullet
games, computes `avg_elo`, deduplicates by game `id`, and filters to an Elo range.

```bash
python preprocess_human_games.py \
    --input  /path/to/raw_lichess_parquet \
    --output /path/to/clean_human_games \
    --min-elo 800 --max-elo 3000
```

### 3. Puzzle games — crawl

`download_puzzles.py` starts from the **Lichess puzzle database**
(<https://database.lichess.org/#puzzles>, `lichess_db_puzzle.csv`), quality-filters
and rating-balances the puzzles, then fetches each puzzle's *source game* PGN via
the Lichess games-export API.

```bash
python download_puzzles.py \
    --puzzle-csv /path/to/lichess_db_puzzle.csv \
    --output     filtered_puzzles_with_pgn.csv
```

### 4. Puzzle games — preprocess

`preprocess_puzzles.py` deduplicates by the normalized source-game move sequence
and extracts `ctx` (the SAN moves leading up to each puzzle position).

```bash
python preprocess_puzzles.py \
    --input  filtered_puzzles_with_pgn.csv \
    --output puzzles_processed.csv
```

### 5. Decontamination

`decontaminate.py` removes any training game that ever reaches a puzzle **test**
position (FEN match while replaying the game), preventing eval leakage.

```bash
python decontaminate.py \
    --train-parquet /path/to/clean_human_games/part-00000.parquet \
    --puzzle-files  /path/to/puzzles_test.csv \
    --out           /path/to/decontaminated/part-00000.parquet \
    --removed       /path/to/removed/part-00000.parquet
```

### 6. Tokenization

`tokenize_dataset.py` streams the cleaned parquet's text column into size-bounded
`.txt` shards, tokenizes each with the chess tokenizer, and writes `uint32` `.npy`
shards — the exact input format the pretraining data loader expects.

```bash
python tokenize_dataset.py \
    --parquet   /path/to/decontaminated \
    --out-dir   /path/to/pretrain_tokenized \
    --column    ctx \
    --tokenizer LanTokenizer
```

The resulting `--out-dir` is what you pass as `DATA_DIR` to
`pretraining/run_pretrain.sh`.

---

### Pipeline at a glance

| Stage | Script | In → Out |
|-------|--------|----------|
| 1. Human download | *(website)* | Lichess DB → raw parquet |
| 2. Human preprocess | `preprocess_human_games.py` | raw parquet → clean parquet |
| 3. Puzzle crawl | `download_puzzles.py` | puzzle CSV → puzzles+PGN CSV |
| 4. Puzzle preprocess | `preprocess_puzzles.py` | puzzles+PGN CSV → puzzles+`ctx` CSV |
| 5. Decontaminate | `decontaminate.py` | clean parquet + puzzle FENs → decontaminated parquet |
| 6. Tokenize | `tokenize_dataset.py` | parquet → `.npy` token shards |
