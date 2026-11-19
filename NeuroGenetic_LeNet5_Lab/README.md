# Neuro-Genetic LeNet-5 | Soft Computing Lab

An interactive handwritten-digit recognition laboratory built around the project in the supplied synopsis and report. A LeNet-style Convolutional Neural Network learns to classify digits, while a Genetic Algorithm searches for a compact and accurate CNN configuration.

The React frontend is the user-facing dashboard. Python contains the data loader, neural-network model, Genetic Algorithm and training pipeline as separate modules. FastAPI connects the trained model to the frontend.

## What this project demonstrates

- CNN-based handwritten digit recognition using the MNIST dataset.
- A nine-gene chromosome that controls the CNN architecture and training settings.
- Tournament selection, uniform crossover, mutation and elitism.
- A fitness function that rewards validation accuracy and penalises unnecessary model parameters.
- A fair equal-budget random-search control and a baseline LeNet-5-style model.
- A React UI for drawing, uploading and testing handwritten digits.
- Per-input analysis: class probabilities, top-two margin, confidence description, preprocessing quality, inference time, compute device, parameter count and estimated MACs.
- Experiment charts, final test metrics, candidate-evaluation records, model explanation, testing parameters and the research-paper reference.

## Requirements

Use **64-bit Python 3.11** and Node.js 20 or newer. Keep an internet connection for the first MNIST download and for installing Python and npm packages.

TensorFlow 2.21 is pinned in `requirements.txt`. On native Windows this project uses TensorFlow CPU; the current TensorFlow installation guide explains the native-Windows GPU limitation and the CPU installation path: <https://www.tensorflow.org/install/pip>.

## Run it in VS Code on Windows

1. Extract the ZIP file to a normal folder, for example `Desktop\NeuroGenetic_LeNet5_Lab`.
2. Open that folder in VS Code using **File → Open Folder**.
3. Open **Terminal → New Terminal**. Make sure the terminal path is the project root, the folder containing `train.py`.
4. Run the setup once:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python check_environment.py
cd frontend
npm install
cd ..
```

5. Train the model in the terminal:

```powershell
python train.py
```

The default `demo` profile loads real MNIST, searches CNN designs, compares the GA winner with random search and the baseline, trains the finalists, and writes the model and results. The first run downloads MNIST automatically. Keep the terminal open until it prints `TRAINING SUMMARY` and `Experiment complete`.

6. Open a **second VS Code terminal** at the project root and start the Python API:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000 --reload
```

7. Open a **third VS Code terminal**, start the React development server:

```powershell
cd frontend
npm run dev
```

8. Open this address in Chrome:

<http://127.0.0.1:5173>

Keep the backend and frontend terminals running while you use the dashboard.

### Easier Windows launch files

After the one-time setup, you can double-click `train_demo.bat` to train, `start_backend.bat` to start the API, and `start_frontend.bat` to start React. The first setup can also be started with `install_windows.bat` after Python 3.11 and Node.js have been installed.

## Full experiment for report-quality results

The default demo profile is intended for a live classroom demonstration and uses a smaller search budget, fewer final epochs and one final seed. For the larger experiment specified in the synopsis/report, run:

```powershell
python train.py --profile full
```

The full profile uses population size 8, 5 generations, 3 epochs per candidate, 12,000 candidate-training images, 5,000 candidate-validation images, 15 final epochs and 3 random seeds. It can take substantially longer, especially on a CPU-only laptop. Do not close the training terminal before it finishes.

The code writes all experimental outputs under `backend/artifacts/`. Each training run replaces the previous model and experiment artifacts in that folder.

## What input should you give?

The model classifies **one handwritten digit from 0 to 9**. In Live Studio you have three ways to provide it:

1. Draw a digit directly on the black canvas with your mouse or trackpad.
2. Upload a cropped PNG, JPG or WebP image containing one digit.
3. Click one of the built-in MNIST test examples. Each built-in example has a known digit label, which makes it useful for the professor demonstration.

For a clean prediction, use a single bold digit, centered in the image. Avoid writing a word, multiple digits, a full page, or a photograph with a busy background. The app converts the image to grayscale, normalizes it, crops the foreground and centers it on a 28 × 28 canvas like the MNIST input format.

### Suggested live demo sequence

1. Open **Live Studio** and click an MNIST sample such as `#3` or `#7`.
2. Show the normalized image, predicted digit, top-three class scores, inference time and model compute information.
3. Clear the canvas and handwrite the same digit yourself.
4. Try a visually similar digit and compare the top-two class gap. Discuss uncertainty if the model gives close scores.
5. Open **Experiment Lab** to show the GA convergence plot, comparison with random search and baseline, and confusion matrix.
6. Open **Algorithm Guide** to explain the nine genes, fitness function, selection, crossover, mutation and elitism.
7. Open **Research Notes** to explain the base paper, project upgrade, dataset split and testing parameters.

## How to get more input images

- The built-in samples come from the official MNIST test split loaded during training.
- Keras MNIST documentation: <https://keras.io/api/datasets/mnist/>.
- TensorFlow Datasets MNIST page: <https://www.tensorflow.org/datasets/catalog/mnist>.
- You can also write a digit on paper, photograph it, crop it to one digit and upload the image. A plain, high-contrast image works best.

## What results should appear?

The precise accuracy is **not hard-coded**. It depends on the training run, random seed, profile and hardware. After training, the UI will show the actual values measured by your computer. It does not invent a target score.

A successful training run creates:

| File | What it contains |
|---|---|
| `best_model.keras` | Trained GA-LeNet model used by Live Studio |
| `metadata.json` | Best chromosome, search-space size, hardware and training setup |
| `results_summary.json` | Baseline, random-search and GA-LeNet final comparison |
| `experiment_history.json` | GA generations, candidates, final runs and class metrics |
| `best_genomes.json` | Best baseline, random and GA chromosomes |
| `all_evaluations.csv` | Candidate fitness, validation accuracy, gene values and parameter counts |
| `classification_report.json` | Precision, recall and F1 by digit class |
| `ga_convergence.png` | Fitness trend and equal-evaluation-budget search comparison |
| `model_comparison.png` | Test accuracy and trainable-parameter comparison |
| `validation_accuracy.png` | Validation accuracy over final training epochs |
| `confusion_matrix.png` | GA-LeNet test confusion matrix |
| `examples.npz` | One real test image for each digit class |

## About the algorithm and model

### CNN / LeNet-style model

The network receives a 28 × 28 grayscale image. Two convolution layers learn local stroke features. Pooling reduces the spatial dimensions. Two dense layers combine the learned features. The final ten-unit softmax layer returns a score for each digit class.

The CNN learns its weights through back-propagation and the Adam optimizer. The Genetic Algorithm does not replace back-propagation; it searches for the CNN architecture and hyper-parameters that should be trained.

### Genetic Algorithm

Each candidate is a chromosome with nine genes: `c1`, `c2`, `kernel`, `fc1`, `fc2`, `activation`, `pooling`, `dropout` and `lr`. The search space contains 161,280 combinations. The algorithm uses tournament selection, uniform crossover, per-gene mutation and elitism.

Fitness is:

```text
Fitness = validation accuracy - λ × (parameter count / 100,000)
```

The current size-penalty value is λ = 0.002. Validation data is used during the search; the held-out test set is reserved for final model evaluation.

### How to explain compute or “power” used

Live Studio reports whether TensorFlow used CPU or a supported GPU, the model's trainable parameter count, an estimate of multiply-accumulate operations (MACs) for one input, and the measured inference time. It does **not** claim to measure electrical power consumption in watts. Native Windows TensorFlow 2.11+ uses CPU; supported GPU training can be configured through WSL2 on a suitable NVIDIA setup.

## Base research paper

Y. LeCun, L. Bottou, Y. Bengio and P. Haffner, “Gradient-Based Learning Applied to Document Recognition,” *Proceedings of the IEEE*, vol. 86, no. 11, pp. 2278–2324, November 1998. DOI: 10.1109/5.726791.

Paper link: <https://doi.org/10.1109/5.726791>

The project builds on the LeNet-5/CNN and MNIST foundation, then adds GA-based architecture search, a size-aware fitness function, a baseline chromosome, equal-budget random search and multi-seed final testing. The uploaded report and synopsis templates are retained in `reference/`.

## Project folder structure

```text
NeuroGenetic_LeNet5_Lab/
├── backend/
│   ├── api.py
│   ├── data_loader.py
│   ├── genetic_algorithm.py
│   ├── ml_model.py
│   ├── settings.py
│   └── trainer.py
├── frontend/
│   ├── src/App.jsx
│   ├── src/main.jsx
│   ├── src/styles.css
│   ├── package.json
│   └── vite.config.js
├── reference/
│   ├── Project_Report_template.docx
│   ├── Project_Synopsis_template.docx
│   └── neuro_genetic_lenet_original.py
├── train.py
├── check_environment.py
├── requirements.txt
└── README.md
```

## Troubleshooting

**`py -3.11` is not recognised**: install 64-bit Python 3.11 and select “Add Python to PATH”. Restart VS Code after installing it.

**PowerShell blocks activation**: in the current terminal run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then run `.\.venv\Scripts\Activate.ps1` again.

**MNIST download fails**: check your internet connection and rerun `python train.py`. Keras caches the dataset after a successful download.

**TensorFlow installation fails**: confirm the selected interpreter is Python 3.11, 64-bit, and that the VS Code interpreter is `.venv`. Do not install the project requirements into a different Python environment.

**Frontend says API offline**: keep the backend terminal running at port 8000 and check `http://127.0.0.1:8000/api/health`.

**Dashboard says model not trained**: finish `python train.py` first. Then refresh the React page.

**It predicts a wrong digit**: this is a classifier, not a perfect OCR system. Try a larger, centered, high-contrast digit or compare the top-three class scores.

**Report results**: use the actual CSV/JSON/artifact values produced by the full profile. Do not copy expected outcomes or fabricated numbers into Chapter 5.
