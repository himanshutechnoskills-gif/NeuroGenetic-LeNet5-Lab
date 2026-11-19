from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
STATUS_PATH = ARTIFACT_DIR / "progress.json"
MODEL_PATH = ARTIFACT_DIR / "best_model.keras"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"
SUMMARY_PATH = ARTIFACT_DIR / "results_summary.json"
HISTORY_PATH = ARTIFACT_DIR / "experiment_history.json"
EXAMPLES_PATH = ARTIFACT_DIR / "examples.npz"
PAPER_URL = "https://doi.org/10.1109/5.726791"
