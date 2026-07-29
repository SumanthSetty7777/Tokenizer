from pathlib import Path
from time import perf_counter
from src.bpe import train_bpe_slow

input_path = Path("/Users/ssetty/dl_repos/tokenizer/data/TinyStoriesV2-GPT4-valid.txt")
text = input_path.read_text(encoding="utf-8")

start = perf_counter()
spec = train_bpe_slow(
    text=text,
    vocab_size=1000,
    special_tokens=("<|endoftext|>",),
)
elapsed = perf_counter() - start

print("elapsed:", elapsed)
print("vocab size:", len(spec.vocab))
print("merges:", len(spec.merges))

# uv run python -m cProfile -o profile.prof src/profile_bpe.py

"""
uv run python - <<'PY'
import pstats

stats = pstats.Stats("profile.prof")
stats.strip_dirs()
stats.sort_stats("cumulative")
stats.print_stats(30)
PY

_______________


uv run python - <<'PY'
import pstats

stats = pstats.Stats("profile.prof").strip_dirs()

print("\n=== CUMULATIVE TIME ===")
stats.sort_stats("cumulative").print_stats(25)

print("\n=== SELF TIME ===")
stats.sort_stats("tottime").print_stats(25)
PY

__________________________

# to visualize
-> uv run snakeviz profile.prof
"""

"""
ncalls: how many times called
tottime: time spent inside that function only
cumtime: time spent inside that function + functions it called
"""