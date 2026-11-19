import argparse
from backend.trainer import run_training


def main():
    parser = argparse.ArgumentParser(description="Train the Neuro-Genetic LeNet-5 project.")
    parser.add_argument("--profile", choices=["demo", "full"], default="demo")
    args = parser.parse_args()
    metadata = run_training(args.profile)
    print("\nTRAINING SUMMARY")
    print(f"Profile: {metadata['profile']}")
    print(f"Dataset: {metadata['dataset']}")
    print(f"Search space: {metadata['search_space_size']:,} possible networks")
    print(f"Best validation accuracy: {metadata['best_validation_accuracy'] * 100:.2f}%")
    print(f"Best genome: {metadata['best_genome']}")
    print(f"Saved model: {metadata['model_file']}")
    print("Open the dashboard using the backend and frontend commands in README.md.")


if __name__ == "__main__":
    main()
