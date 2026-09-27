import os

# One OpenMP thread avoids FAISS/PyTorch crashes on macOS.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
