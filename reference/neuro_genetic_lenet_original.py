#!/usr/bin/env python3
"""
Neuro-Genetic LeNet-5
=====================
Genetic-Algorithm optimisation of a LeNet-5 style Convolutional Neural Network
for handwritten digit recognition (MNIST).

Base paper
----------
Y. LeCun, L. Bottou, Y. Bengio and P. Haffner, "Gradient-Based Learning Applied
to Document Recognition," Proceedings of the IEEE, vol. 86, no. 11,
pp. 2278-2324, Nov. 1998.

What is new (compared with the base paper)
------------------------------------------
1. Architecture + hyper-parameters of LeNet-5 are found automatically by a
   Genetic Algorithm (neuro-genetic hybrid) instead of manual trial and error.
2. Compactness-aware fitness:  fitness = val_accuracy - LAMBDA * (params / 1e5)
3. The original LeNet-5 is one point inside the search space, so the baseline
   comparison is perfectly fair.
4. Fair comparison against Random Search with the SAME evaluation budget and
   multi-seed (mean +- std) final evaluation on the untouched MNIST test set.

Usage
-----
    python neuro_genetic_lenet.py                 # full experiment (MNIST)
    python neuro_genetic_lenet.py --quick         # tiny run to check setup
    python neuro_genetic_lenet.py --fake --quick  # offline smoke test (no MNIST)
"""

import argparse
import csv
import json
import os
import random
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# --------------------------------------------------------------------------- #
# 1. Search space (the "chromosome")                                           #
# --------------------------------------------------------------------------- #
SEARCH_SPACE = {
    "c1": [4, 6, 8, 12, 16, 24, 32],          # filters in conv layer 1
    "c2": [8, 16, 24, 32, 48, 64],            # filters in conv layer 2
    "kernel": [3, 5],                         # kernel size (both conv layers)
    "fc1": [64, 120, 168, 256],               # units in first dense layer
    "fc2": [32, 64, 84, 128],                 # units in second dense layer
    "activation": ["tanh", "relu"],
    "pooling": ["avg", "max"],
    "dropout": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    "lr": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],     # Adam learning rate
}
GENES = list(SEARCH_SPACE.keys())

# Original LeNet-5 configuration (base paper) -- lies INSIDE the search space
BASELINE = {
    "c1": 6, "c2": 16, "kernel": 5, "fc1": 120, "fc2": 84,
    "activation": "tanh", "pooling": "avg", "dropout": 0.0, "lr": 1e-3,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# --------------------------------------------------------------------------- #
# 2. Data                                                                      #
# --------------------------------------------------------------------------- #
def make_fake_data(n, rng):
    """Small synthetic 28x28 dataset, only used to test the pipeline offline."""
    yy, xx = np.mgrid[0:28, 0:28]
    templates = []
    for k in range(10):
        cy, cx = 6 + (k // 5) * 14, 5 + (k % 5) * 4
        templates.append(np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 3.0 ** 2)))
    templates = np.stack(templates)
    y = rng.randint(0, 10, n)
    x = templates[y] + rng.rand(n, 28, 28) * 0.3
    return np.clip(x, 0, 1).astype("float32")[..., None], y.astype("int64")


def prepare_data(args):
    if args.fake:
        rng = np.random.RandomState(args.seed)
        x, y = make_fake_data(8000, rng)
        x_te, y_te = make_fake_data(2000, rng)
    else:
        (x, y), (x_te, y_te) = keras.datasets.mnist.load_data()
        x = (x.astype("float32") / 255.0)[..., None]
        x_te = (x_te.astype("float32") / 255.0)[..., None]

    # 90 % train / 10 % validation (validation is used by the GA, NOT the test set)
    idx = np.random.RandomState(args.seed).permutation(len(x))
    n_val = int(0.1 * len(x))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    data = {
        "x_tr": x[tr_idx], "y_tr": y[tr_idx],
        "x_val": x[val_idx], "y_val": y[val_idx],
        "x_te": x_te, "y_te": y_te,
    }
    # smaller subsets so that every GA fitness evaluation is fast
    data["x_ga"], data["y_ga"] = data["x_tr"][:args.ga_train], data["y_tr"][:args.ga_train]
    data["x_gav"], data["y_gav"] = data["x_val"][:args.ga_val], data["y_val"][:args.ga_val]
    print(f"[data] train={len(data['x_tr'])} val={len(data['x_val'])} test={len(data['x_te'])} "
          f"| GA subset train={len(data['x_ga'])} val={len(data['x_gav'])}")
    return data


# --------------------------------------------------------------------------- #
# 3. Model builder: genome -> CNN                                              #
# --------------------------------------------------------------------------- #
def build_model(g):
    Pool = layers.AveragePooling2D if g["pooling"] == "avg" else layers.MaxPooling2D
    k, act = g["kernel"], g["activation"]
    inp = keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(g["c1"], k, padding="same", activation=act)(inp)   # C1
    x = Pool(2)(x)                                                       # S2
    x = layers.Conv2D(g["c2"], k, padding="valid", activation=act)(x)    # C3
    x = Pool(2)(x)                                                       # S4
    x = layers.Flatten()(x)
    x = layers.Dense(g["fc1"], activation=act)(x)                        # C5 / FC
    x = layers.Dropout(g["dropout"])(x)
    x = layers.Dense(g["fc2"], activation=act)(x)                        # F6
    out = layers.Dense(10, activation="softmax")(x)                      # output
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=g["lr"]),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def genome_key(g):
    return tuple(g[k] for k in GENES)


# --------------------------------------------------------------------------- #
# 4. Fitness function                                                          #
# --------------------------------------------------------------------------- #
def evaluate_genome(g, data, args, cache, gen, method):
    """Train briefly on the GA subset and return the record (cached)."""
    key = genome_key(g)
    if key in cache:
        return cache[key]
    keras.backend.clear_session()
    set_seed(args.seed)
    model = build_model(g)
    t0 = time.time()
    model.fit(data["x_ga"], data["y_ga"], epochs=args.fit_epochs,
              batch_size=args.batch_size, verbose=0)
    _, val_acc = model.evaluate(data["x_gav"], data["y_gav"], verbose=0)
    params = model.count_params()
    fitness = val_acc - args.lam * params / 1e5
    rec = {"method": method, "eval_id": len(cache) + 1, "gen": gen, "genome": dict(g),
           "val_acc": float(val_acc), "params": int(params), "fitness": float(fitness),
           "time_s": time.time() - t0}
    cache[key] = rec
    return rec


# --------------------------------------------------------------------------- #
# 5. Genetic Algorithm                                                         #
# --------------------------------------------------------------------------- #
def random_genome(rng):
    return {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}


def tournament(pop, fits, rng, k):
    contenders = rng.sample(range(len(pop)), min(k, len(pop)))
    return pop[max(contenders, key=lambda i: fits[i])]


def crossover(a, b, rng):
    """Uniform crossover: every gene is copied from parent A or B with p = 0.5."""
    return {k: (a[k] if rng.random() < 0.5 else b[k]) for k in GENES}


def mutate(g, rng, pm):
    """Each gene is re-sampled from its allowed values with probability pm."""
    child = dict(g)
    for k in GENES:
        if rng.random() < pm:
            child[k] = rng.choice(SEARCH_SPACE[k])
    return child


def run_ga(data, args):
    rng = random.Random(args.seed)
    cache, history = {}, []
    pop = [random_genome(rng) for _ in range(args.pop)]
    for gen in range(1, args.gens + 1):
        recs = [evaluate_genome(g, data, args, cache, gen, "GA") for g in pop]
        fits = [r["fitness"] for r in recs]
        best_i = int(np.argmax(fits))
        history.append({"generation": gen, "best_fitness": max(fits),
                        "mean_fitness": float(np.mean(fits)), "worst_fitness": min(fits),
                        "best_val_acc": recs[best_i]["val_acc"],
                        "best_params": recs[best_i]["params"], "evals_so_far": len(cache)})
        print(f"[GA] gen {gen}/{args.gens}  best fit={max(fits):.4f}  "
              f"mean={np.mean(fits):.4f}  best val-acc={recs[best_i]['val_acc']:.4f}  "
              f"params={recs[best_i]['params']}  unique evals={len(cache)}")
        if gen == args.gens:
            break
        order = sorted(range(len(pop)), key=lambda i: fits[i], reverse=True)
        new_pop = [dict(pop[i]) for i in order[:args.elite]]          # elitism
        while len(new_pop) < args.pop:
            p1 = tournament(pop, fits, rng, args.tourn)               # selection
            p2 = tournament(pop, fits, rng, args.tourn)
            child = crossover(p1, p2, rng) if rng.random() < args.pc else dict(p1)
            new_pop.append(mutate(child, rng, args.pm))               # mutation
        pop = new_pop
    best = max(cache.values(), key=lambda r: r["fitness"])
    return best, history, cache


def run_random_search(data, args, n_evals):
    rng = random.Random(args.seed + 1)
    cache = {}
    while len(cache) < n_evals:
        evaluate_genome(random_genome(rng), data, args, cache, len(cache) + 1, "Random")
    best = max(cache.values(), key=lambda r: r["fitness"])
    print(f"[RS] {len(cache)} evaluations  best fit={best['fitness']:.4f}  "
          f"val-acc={best['val_acc']:.4f}  params={best['params']}")
    return best, cache


# --------------------------------------------------------------------------- #
# 6. Final training / testing                                                  #
# --------------------------------------------------------------------------- #
def final_train(name, g, data, args, seed):
    keras.backend.clear_session()
    set_seed(seed)
    model = build_model(g)
    t0 = time.time()
    hist = model.fit(data["x_tr"], data["y_tr"],
                     validation_data=(data["x_val"], data["y_val"]),
                     epochs=args.final_epochs, batch_size=args.batch_size, verbose=0)
    train_time = time.time() - t0
    test_loss, test_acc = model.evaluate(data["x_te"], data["y_te"], verbose=0)
    y_pred = np.argmax(model.predict(data["x_te"], verbose=0), axis=1)
    print(f"[final] {name:<16} seed={seed}  test-acc={test_acc:.4f}  "
          f"params={model.count_params()}  time={train_time:.0f}s")
    return {"name": name, "seed": seed, "test_acc": float(test_acc), "test_loss": float(test_loss),
            "params": int(model.count_params()), "train_time_s": train_time,
            "val_curve": [float(v) for v in hist.history["val_accuracy"]],
            "loss_curve": [float(v) for v in hist.history["loss"]], "y_pred": y_pred}


# --------------------------------------------------------------------------- #
# 7. Plots and saving                                                          #
# --------------------------------------------------------------------------- #
def best_so_far(cache):
    recs = sorted(cache.values(), key=lambda r: r["eval_id"])
    out, best = [], -1e9
    for r in recs:
        best = max(best, r["fitness"])
        out.append(best)
    return out


def plot_convergence(history, ga_cache, rs_cache, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    gens = [h["generation"] for h in history]
    ax[0].plot(gens, [h["best_fitness"] for h in history], "o-", label="Best")
    ax[0].plot(gens, [h["mean_fitness"] for h in history], "s--", label="Mean")
    ax[0].set_xticks(gens)
    ax[0].set_xlabel("Generation"); ax[0].set_ylabel("Fitness")
    ax[0].set_title("GA convergence"); ax[0].grid(alpha=.3); ax[0].legend()
    ax[1].plot(range(1, len(ga_cache) + 1), best_so_far(ga_cache), label="Genetic Algorithm")
    ax[1].plot(range(1, len(rs_cache) + 1), best_so_far(rs_cache), label="Random Search")
    ax[1].set_xlabel("Number of fitness evaluations"); ax[1].set_ylabel("Best fitness so far")
    ax[1].set_title("GA vs Random Search (equal budget)"); ax[1].grid(alpha=.3); ax[1].legend()
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()


def plot_comparison(summary, path):
    names = [s["name"] for s in summary]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(names, [s["acc_mean"] * 100 for s in summary],
              yerr=[s["acc_std"] * 100 for s in summary], capsize=5,
              color=["#9aa5b1", "#e9a23b", "#2f80ed"][:len(names)])
    lo = min(s["acc_mean"] for s in summary) * 100
    ax[0].set_ylim(max(0, lo - 2), 100)
    ax[0].set_ylabel("Test accuracy (%)"); ax[0].set_title("Test accuracy (mean +- std)")
    ax[1].bar(names, [s["params"] for s in summary],
              color=["#9aa5b1", "#e9a23b", "#2f80ed"][:len(names)])
    ax[1].set_ylabel("Trainable parameters"); ax[1].set_title("Model size")
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()


def plot_val_curves(first_runs, path):
    plt.figure(figsize=(6, 4))
    for name, r in first_runs.items():
        plt.plot(range(1, len(r["val_curve"]) + 1), r["val_curve"], label=name)
    plt.xlabel("Epoch"); plt.ylabel("Validation accuracy")
    plt.title("Validation accuracy during final training")
    plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
    plt.savefig(path, dpi=200); plt.close()


def plot_confusion(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=range(10))
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title(title); plt.xlabel("Predicted"); plt.ylabel("True")
    plt.xticks(range(10)); plt.yticks(range(10))
    thr = cm.max() / 2
    for i in range(10):
        for j in range(10):
            plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                     color="white" if cm[i, j] > thr else "black")
    plt.colorbar(); plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()


def save_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- #
# 8. Main                                                                      #
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fake", action="store_true", help="use synthetic data (offline test)")
    p.add_argument("--quick", action="store_true", help="tiny settings to check that it runs")
    p.add_argument("--pop", type=int, default=8, help="GA population size")
    p.add_argument("--gens", type=int, default=5, help="GA generations")
    p.add_argument("--pc", type=float, default=0.8, help="crossover probability")
    p.add_argument("--pm", type=float, default=0.2, help="per-gene mutation probability")
    p.add_argument("--tourn", type=int, default=3, help="tournament size")
    p.add_argument("--elite", type=int, default=1, help="number of elites copied unchanged")
    p.add_argument("--lam", type=float, default=0.002, help="size-penalty weight in fitness")
    p.add_argument("--fit_epochs", type=int, default=3, help="epochs per fitness evaluation")
    p.add_argument("--ga_train", type=int, default=12000)
    p.add_argument("--ga_val", type=int, default=5000)
    p.add_argument("--final_epochs", type=int, default=15)
    p.add_argument("--final_runs", type=int, default=3, help="seeds for final evaluation")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results")
    a = p.parse_args()
    if a.quick:
        a.pop, a.gens, a.fit_epochs = 4, 2, 1
        a.ga_train, a.ga_val = 2000, 1000
        a.final_epochs, a.final_runs = 2, 1
    return a


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)
    print("Settings:", vars(args))
    data = prepare_data(args)

    # ---- search ---------------------------------------------------------- #
    print("\n=== Genetic Algorithm search ===")
    ga_best, history, ga_cache = run_ga(data, args)
    print("\n=== Random search (same budget) ===")
    rs_best, rs_cache = run_random_search(data, args, len(ga_cache))

    print("\nBest GA genome   :", ga_best["genome"])
    print("Best random genome:", rs_best["genome"])

    # ---- final comparison ------------------------------------------------ #
    print("\n=== Final training on full training set, testing on MNIST test set ===")
    contenders = {"Baseline LeNet-5": BASELINE,
                  "Random Search": rs_best["genome"],
                  "GA-LeNet (ours)": ga_best["genome"]}
    runs, first_runs = [], {}
    for name, genome in contenders.items():
        for s in range(args.final_runs):
            r = final_train(name, genome, data, args, seed=args.seed + s)
            runs.append(r)
            first_runs.setdefault(name, r)

    summary = []
    for name in contenders:
        rr = [r for r in runs if r["name"] == name]
        accs = [r["test_acc"] for r in rr]
        summary.append({"name": name, "acc_mean": float(np.mean(accs)),
                        "acc_std": float(np.std(accs)), "acc_best": float(np.max(accs)),
                        "params": rr[0]["params"],
                        "train_time_s": float(np.mean([r["train_time_s"] for r in rr]))})

    # ---- outputs --------------------------------------------------------- #
    out = lambda f: os.path.join(args.out, f)
    save_csv(out("ga_history.csv"), history)
    all_evals = []
    for cache in (ga_cache, rs_cache):
        for r in sorted(cache.values(), key=lambda r: r["eval_id"]):
            row = {k: v for k, v in r.items() if k != "genome"}
            row.update(r["genome"])
            all_evals.append(row)
    save_csv(out("all_evaluations.csv"), all_evals)
    skip = ("y_pred", "val_curve", "loss_curve")
    save_csv(out("final_runs.csv"),
             [{k: v for k, v in r.items() if k not in skip} for r in runs])
    save_csv(out("results_summary.csv"), summary)
    with open(out("best_genomes.json"), "w") as f:
        json.dump({"baseline": BASELINE, "random_search": rs_best["genome"],
                   "genetic_algorithm": ga_best["genome"]}, f, indent=2)

    plot_convergence(history, ga_cache, rs_cache, out("fig_ga_convergence.png"))
    plot_comparison(summary, out("fig_comparison.png"))
    plot_val_curves(first_runs, out("fig_validation_curves.png"))
    ga_run = first_runs["GA-LeNet (ours)"]
    plot_confusion(data["y_te"], ga_run["y_pred"], "Confusion matrix - GA-LeNet (test set)",
                   out("fig_confusion_matrix.png"))
    with open(out("classification_report.txt"), "w") as f:
        f.write(classification_report(data["y_te"], ga_run["y_pred"], digits=4))

    print("\n================ SUMMARY (test set) ================")
    print(f"{'Model':<20}{'Acc mean (%)':>14}{'Std (%)':>10}{'Params':>10}{'Time (s)':>10}")
    for s in summary:
        print(f"{s['name']:<20}{s['acc_mean']*100:>14.2f}{s['acc_std']*100:>10.2f}"
              f"{s['params']:>10}{s['train_time_s']:>10.0f}")
    print(f"\nAll files saved in: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
