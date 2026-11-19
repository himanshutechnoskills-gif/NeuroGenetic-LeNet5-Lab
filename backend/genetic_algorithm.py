import random
import numpy as np
from tensorflow import keras
from backend.ml_model import SEARCH_SPACE, build_model, set_seed

GENES = list(SEARCH_SPACE.keys())


def random_genome(rng):
    return {gene: rng.choice(values) for gene, values in SEARCH_SPACE.items()}


def genome_key(genome):
    return tuple(genome[gene] for gene in GENES)


def tournament_selection(population, fitnesses, rng, size):
    candidates = rng.sample(range(len(population)), min(size, len(population)))
    winner = max(candidates, key=lambda index: fitnesses[index])
    return dict(population[winner])


def uniform_crossover(parent_a, parent_b, rng):
    return {gene: parent_a[gene] if rng.random() < 0.5 else parent_b[gene] for gene in GENES}


def mutate(genome, rng, probability):
    child = dict(genome)
    for gene in GENES:
        if rng.random() < probability:
            options = [value for value in SEARCH_SPACE[gene] if value != child[gene]]
            child[gene] = rng.choice(options or SEARCH_SPACE[gene])
    return child


def evaluate_genome(genome, data, settings, cache, generation, method, on_event=None):
    key = genome_key(genome)
    if key in cache:
        return cache[key]
    keras.backend.clear_session()
    set_seed(settings["seed"])
    model = build_model(genome)
    model.fit(
        data["x_ga"],
        data["y_ga"],
        epochs=settings["fit_epochs"],
        batch_size=settings["batch_size"],
        verbose=0,
    )
    _, validation_accuracy = model.evaluate(data["x_ga_val"], data["y_ga_val"], verbose=0)
    parameter_count = int(model.count_params())
    fitness = float(validation_accuracy - settings["lambda_size"] * parameter_count / 100000)
    record = {
        "method": method,
        "evaluation": len(cache) + 1,
        "generation": generation,
        "genome": dict(genome),
        "validation_accuracy": float(validation_accuracy),
        "parameters": parameter_count,
        "fitness": fitness,
    }
    cache[key] = record
    if on_event:
        on_event(record)
    del model
    return record


def run_genetic_algorithm(data, settings, on_generation=None, on_evaluation=None):
    rng = random.Random(settings["seed"])
    cache = {}
    population = [random_genome(rng) for _ in range(settings["population"])]
    history = []
    for generation in range(1, settings["generations"] + 1):
        records = [
            evaluate_genome(genome, data, settings, cache, generation, "Genetic Algorithm", on_evaluation)
            for genome in population
        ]
        fitnesses = [record["fitness"] for record in records]
        best_index = int(np.argmax(fitnesses))
        summary = {
            "generation": generation,
            "best_fitness": float(max(fitnesses)),
            "mean_fitness": float(np.mean(fitnesses)),
            "worst_fitness": float(min(fitnesses)),
            "best_validation_accuracy": records[best_index]["validation_accuracy"],
            "best_parameters": records[best_index]["parameters"],
            "evaluations_so_far": len(cache),
        }
        history.append(summary)
        if on_generation:
            on_generation(summary)
        if generation == settings["generations"]:
            break
        ranked = sorted(range(len(population)), key=lambda index: fitnesses[index], reverse=True)
        next_population = [dict(population[index]) for index in ranked[:settings["elite"]]]
        while len(next_population) < settings["population"]:
            first = tournament_selection(population, fitnesses, rng, settings["tournament_size"])
            second = tournament_selection(population, fitnesses, rng, settings["tournament_size"])
            child = uniform_crossover(first, second, rng) if rng.random() < settings["crossover_probability"] else first
            next_population.append(mutate(child, rng, settings["mutation_probability"]))
        population = next_population
    best = max(cache.values(), key=lambda record: record["fitness"])
    return best, history, cache


def run_random_search(data, settings, evaluation_budget, on_evaluation=None):
    rng = random.Random(settings["seed"] + 1)
    cache = {}
    attempts = 0
    while len(cache) < evaluation_budget:
        attempts += 1
        if attempts > evaluation_budget * 1000:
            raise RuntimeError("Random search could not find enough unique candidates.")
        genome = random_genome(rng)
        evaluate_genome(genome, data, settings, cache, len(cache) + 1, "Random Search", on_evaluation)
    best = max(cache.values(), key=lambda record: record["fitness"])
    return best, cache
