import csv
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
from backend.data_loader import load_mnist, save_test_examples
from backend.genetic_algorithm import run_genetic_algorithm, run_random_search
from backend.ml_model import BASELINE, SEARCH_SPACE, build_model, estimate_macs, set_seed, hardware_details
from backend.settings import ARTIFACT_DIR, STATUS_PATH, MODEL_PATH, METADATA_PATH, SUMMARY_PATH, HISTORY_PATH, EXAMPLES_PATH, PAPER_URL


PROFILES = {
    "demo": {
        "population": 5,
        "generations": 3,
        "fit_epochs": 1,
        "ga_train": 5000,
        "ga_val": 1500,
        "final_epochs": 4,
        "final_runs": 1,
        "batch_size": 128,
        "lambda_size": 0.002,
        "crossover_probability": 0.8,
        "mutation_probability": 0.2,
        "tournament_size": 3,
        "elite": 1,
        "seed": 42,
    },
    "full": {
        "population": 8,
        "generations": 5,
        "fit_epochs": 3,
        "ga_train": 12000,
        "ga_val": 5000,
        "final_epochs": 15,
        "final_runs": 3,
        "batch_size": 128,
        "lambda_size": 0.002,
        "crossover_probability": 0.8,
        "mutation_probability": 0.2,
        "tournament_size": 3,
        "elite": 1,
        "seed": 42,
    },
}


def write_json(path, value):
    temp_path = Path(str(path) + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(value, file, indent=2)
    temp_path.replace(path)


def read_status():
    if STATUS_PATH.exists():
        try:
            return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"status": "not_started", "stage": "Not trained yet", "percent": 0, "logs": []}


def update_status(status=None, stage=None, percent=None, message=None, **extra):
    current = read_status()
    if status is not None:
        current["status"] = status
    if stage is not None:
        current["stage"] = stage
    if percent is not None:
        current["percent"] = round(float(percent), 1)
    if message:
        logs = current.get("logs", [])
        logs.append({"time": datetime.now().strftime("%H:%M:%S"), "message": str(message)})
        current["logs"] = logs[-35:]
        print(message, flush=True)
    current.update(extra)
    current["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_json(STATUS_PATH, current)


def create_plots(ga_history, ga_records, random_records, summary, validation_curves, confusion):
    ga_path = ARTIFACT_DIR / "ga_convergence.png"
    comparison_path = ARTIFACT_DIR / "model_comparison.png"
    validation_path = ARTIFACT_DIR / "validation_accuracy.png"
    confusion_path = ARTIFACT_DIR / "confusion_matrix.png"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    generations = [item["generation"] for item in ga_history]
    axes[0].plot(generations, [item["best_fitness"] for item in ga_history], marker="o", label="Best fitness")
    axes[0].plot(generations, [item["mean_fitness"] for item in ga_history], marker="s", label="Mean fitness")
    axes[0].set_title("Genetic Algorithm convergence")
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Fitness")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    for records, label in [(ga_records, "Genetic Algorithm"), (random_records, "Random Search")]:
        ordered = sorted(records, key=lambda item: item["evaluation"])
        best_values = []
        best_value = -1e9
        for item in ordered:
            best_value = max(best_value, item["fitness"])
            best_values.append(best_value)
        axes[1].plot(range(1, len(best_values) + 1), best_values, label=label)
    axes[1].set_title("Best fitness by evaluation")
    axes[1].set_xlabel("Unique evaluations")
    axes[1].set_ylabel("Best fitness so far")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(ga_path, dpi=170)
    plt.close(fig)

    names = [item["name"] for item in summary]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(names, [item["accuracy_mean"] * 100 for item in summary], yerr=[item["accuracy_std"] * 100 for item in summary], capsize=5)
    axes[0].set_title("Test accuracy (mean ± std)")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].tick_params(axis="x", rotation=15)
    axes[1].bar(names, [item["parameters"] for item in summary])
    axes[1].set_title("Trainable parameters")
    axes[1].set_ylabel("Parameter count")
    axes[1].tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(comparison_path, dpi=170)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4.2))
    for name, values in validation_curves.items():
        axis.plot(range(1, len(values) + 1), values, marker="o", label=name)
    axis.set_title("Validation accuracy during final training")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Validation accuracy")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(validation_path, dpi=170)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(confusion, cmap="Blues")
    axis.set_title("GA-LeNet confusion matrix")
    axis.set_xlabel("Predicted digit")
    axis.set_ylabel("Actual digit")
    axis.set_xticks(range(10))
    axis.set_yticks(range(10))
    for row in range(10):
        for column in range(10):
            axis.text(column, row, str(int(confusion[row, column])), ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(confusion_path, dpi=170)
    plt.close(fig)

    return {
        "ga_convergence": ga_path.name,
        "model_comparison": comparison_path.name,
        "validation_accuracy": validation_path.name,
        "confusion_matrix": confusion_path.name,
    }


def save_evaluations(path, records):
    rows = []
    for record in records:
        row = {key: value for key, value in record.items() if key != "genome"}
        row.update(record["genome"])
        rows.append(row)
    if rows:
        with open(path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def run_training(profile="demo"):
    if profile not in PROFILES:
        raise ValueError("Profile must be either 'demo' or 'full'.")
    settings = dict(PROFILES[profile])
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    update_status(status="running", stage="Preparing experiment", percent=1, message=f"Starting {profile.upper()} training profile.", error=None)
    hardware = hardware_details()
    random.seed(settings["seed"])
    np.random.seed(settings["seed"])
    set_seed(settings["seed"])
    try:
        update_status(stage="Loading MNIST", percent=4, message="Loading the real MNIST handwritten-digit dataset.")
        data = load_mnist(settings["seed"], settings["ga_train"], settings["ga_val"])
        update_status(stage="Dataset ready", percent=8, message=f"Dataset split: {len(data['x_train']):,} train, {len(data['x_val']):,} validation, {len(data['x_test']):,} test images.")
        save_test_examples(data, EXAMPLES_PATH)

        evaluation_messages = {"count": 0}
        total_target = settings["population"] * settings["generations"] * 2

        def on_evaluation(record):
            evaluation_messages["count"] += 1
            stage_name = "Genetic search" if record["method"] == "Genetic Algorithm" else "Random-search control"
            percent = 8 + min(48, evaluation_messages["count"] / max(1, total_target) * 48)
            update_status(stage=stage_name, percent=percent, message=f"{record['method']} evaluation {record['evaluation']}: validation accuracy {record['validation_accuracy'] * 100:.2f}% · fitness {record['fitness']:.4f} · {record['parameters']:,} parameters.")

        def on_generation(record):
            update_status(stage=f"Genetic search · generation {record['generation']}", message=f"Generation {record['generation']}: best fitness {record['best_fitness']:.4f}, mean fitness {record['mean_fitness']:.4f}.")

        update_status(stage="Genetic search", percent=10, message="Evolving CNN designs with selection, crossover, mutation and elitism.")
        ga_best, ga_history, ga_cache = run_genetic_algorithm(data, settings, on_generation, on_evaluation)
        update_status(stage="Random-search control", percent=57, message=f"Genetic search finished with {len(ga_cache)} unique candidates. Giving random search the same evaluation budget.")
        random_best, random_cache = run_random_search(data, settings, len(ga_cache), on_evaluation)
        contenders = [
            ("Baseline LeNet-5", dict(BASELINE)),
            ("Random Search", dict(random_best["genome"])),
            ("GA-LeNet (proposed)", dict(ga_best["genome"])),
        ]
        final_runs = []
        validation_curves = {}
        confusion = None
        classification = None
        first_model_saved = False
        total_final = len(contenders) * settings["final_runs"]
        run_index = 0
        for model_name, genome in contenders:
            accuracies = []
            run_parameters = 0
            run_times = []
            for seed_index in range(settings["final_runs"]):
                run_index += 1
                seed = settings["seed"] + seed_index
                keras.backend.clear_session()
                set_seed(seed)
                model = build_model(genome)
                started = time.time()
                history = model.fit(
                    data["x_train"],
                    data["y_train"],
                    validation_data=(data["x_val"], data["y_val"]),
                    epochs=settings["final_epochs"],
                    batch_size=settings["batch_size"],
                    verbose=0,
                )
                train_seconds = time.time() - started
                test_loss, test_accuracy = model.evaluate(data["x_test"], data["y_test"], verbose=0)
                predictions = np.argmax(model.predict(data["x_test"], batch_size=settings["batch_size"], verbose=0), axis=1)
                accuracies.append(float(test_accuracy))
                run_parameters = int(model.count_params())
                run_times.append(float(train_seconds))
                if seed_index == 0:
                    validation_curves[model_name] = [float(value) for value in history.history["val_accuracy"]]
                if model_name == "GA-LeNet (proposed)" and seed_index == 0:
                    confusion = confusion_matrix(data["y_test"], predictions, labels=list(range(10)))
                    classification = classification_report(data["y_test"], predictions, labels=list(range(10)), output_dict=True, zero_division=0)
                    if not first_model_saved:
                        model.save(MODEL_PATH)
                        first_model_saved = True
                final_runs.append({
                    "name": model_name,
                    "seed": seed,
                    "test_accuracy": float(test_accuracy),
                    "test_loss": float(test_loss),
                    "parameters": run_parameters,
                    "training_seconds": float(train_seconds),
                    "validation_curve": validation_curves.get(model_name, []),
                })
                del model
                percent = 62 + run_index / total_final * 31
                update_status(stage=f"Final training · {model_name}", percent=percent, message=f"{model_name} seed {seed_index + 1}/{settings['final_runs']}: test accuracy {test_accuracy * 100:.2f}% after {train_seconds:.1f}s.")
            model_records = [item for item in final_runs if item["name"] == model_name]
            summary_item = {
                "name": model_name,
                "accuracy_mean": float(np.mean(accuracies)),
                "accuracy_std": float(np.std(accuracies)),
                "accuracy_best": float(np.max(accuracies)),
                "parameters": run_parameters,
                "training_seconds_mean": float(np.mean(run_times)),
            }
            if model_name == "Baseline LeNet-5":
                baseline_summary = summary_item
            elif model_name == "Random Search":
                random_summary = summary_item
            else:
                ga_summary = summary_item

        summary = [baseline_summary, random_summary, ga_summary]
        plots = create_plots(ga_history, list(ga_cache.values()), list(random_cache.values()), summary, validation_curves, confusion)
        write_json(SUMMARY_PATH, summary)
        history_payload = {
            "genetic_history": ga_history,
            "genetic_evaluations": list(ga_cache.values()),
            "random_evaluations": list(random_cache.values()),
            "final_runs": final_runs,
            "validation_curves": validation_curves,
            "confusion_matrix": confusion.tolist(),
            "classification_report": classification,
        }
        write_json(HISTORY_PATH, history_payload)
        write_json(ARTIFACT_DIR / "best_genomes.json", {
            "baseline": BASELINE,
            "random_search": random_best["genome"],
            "genetic_algorithm": ga_best["genome"],
        })
        write_json(ARTIFACT_DIR / "classification_report.json", classification)
        save_evaluations(ARTIFACT_DIR / "all_evaluations.csv", list(ga_cache.values()) + list(random_cache.values()))
        plot_names = plots
        metadata = {
            "project_title": "Neuro-Genetic LeNet-5",
            "dataset": "MNIST handwritten digits",
            "profile": profile,
            "dataset_split": {"training": len(data["x_train"]), "validation": len(data["x_val"]), "test": len(data["x_test"]), "ga_training_subset": len(data["x_ga"]), "ga_validation_subset": len(data["x_ga_val"])},
            "search_space_size": int(np.prod([len(values) for values in SEARCH_SPACE.values()])),
            "genes": SEARCH_SPACE,
            "settings": settings,
            "best_genome": ga_best["genome"],
            "best_validation_accuracy": ga_best["validation_accuracy"],
            "best_fitness": ga_best["fitness"],
            "best_parameters": ga_summary["parameters"],
            "estimated_macs_per_image": estimate_macs(ga_best["genome"]),
            "baseline_parameters": 61706,
            "genetic_unique_evaluations": len(ga_cache),
            "random_unique_evaluations": len(random_cache),
            "hardware": hardware,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "paper_url": PAPER_URL,
            "plots": plot_names,
            "model_file": MODEL_PATH.name,
            "model_explanation": "The genetic algorithm chooses the CNN design. Back-propagation with Adam learns the CNN weights. The final softmax layer scores ten digit classes.",
        }
        write_json(METADATA_PATH, metadata)
        write_json(ARTIFACT_DIR / "run_settings.json", settings)
        update_status(status="completed", stage="Experiment complete", percent=100, message="Training and evaluation finished. Open the React dashboard to test handwritten digits and inspect the experiment.", model_ready=True, profile=profile, error=None)
        return metadata
    except Exception as error:
        update_status(status="failed", stage="Training failed", message=f"Training stopped: {type(error).__name__}: {error}", error=f"{type(error).__name__}: {error}")
        raise
