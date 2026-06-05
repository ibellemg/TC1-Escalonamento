import numpy as np
from config import (
    N_RUNS,
    MAX_ITER,
    SEED_BASE,
    MAX_NO_IMPROVE_GLOBAL,
    MAX_NO_IMPROVE_LOCAL,
    MAX_PARETO_POINTS,
)
from optimization import run_multiple_times
from visualization import print_summary_table, plot_convergence, plot_best_schedule


# =====================================================================
# Mono-objetivo
# =====================================================================

def run_single_objective(evaluator_configs, n_machine, pt, we, due_date):
    summaries = []

    for evaluator_config in evaluator_configs:
        summary = run_multiple_times(
            evaluator_config["evaluator"],
            evaluator_config["tasks"],
            n_machine,
            n_runs=N_RUNS,
            max_iter=MAX_ITER,
            seed_base=SEED_BASE,
            max_no_improve_global=MAX_NO_IMPROVE_GLOBAL,
            max_no_improve_local=MAX_NO_IMPROVE_LOCAL,
        )

        print_summary_table(summary, evaluator_config["name"])
        plot_convergence(
            summary,
            "Curvas de convergência - " + evaluator_config["name"],
            filename="convergencia_" + evaluator_config["name"].split()[0] + ".png",
        )
        plot_best_schedule(
            summary["best_solution"],
            "Melhor solução para " + evaluator_config["name"],
            pt,
            we,
            due_date,
            filename="schedule_" + evaluator_config["name"].split()[0] + ".png",
        )
        summaries.append(summary)

    return summaries


# =====================================================================
# Funções auxiliares multiobjetivo
# =====================================================================

def weighted_sum(weight, f1_norm, f2_norm):
    return weight * f1_norm + (1 - weight) * f2_norm


def safe_normalize(value, minimum, maximum):
    denominator = maximum - minimum
    if abs(denominator) < 1e-12:
        return 0.0
    return (value - minimum) / denominator


def build_normalized_values(solution, evaluators, ideal, nadir):
    f1 = evaluators[0]["evaluator"](solution)
    f2 = evaluators[1]["evaluator"](solution)

    f1_norm = safe_normalize(f1, ideal[0], nadir[0])
    f2_norm = safe_normalize(f2, ideal[1], nadir[1])

    return f1, f2, f1_norm, f2_norm


def estimate_nadir(evaluators, n_machines, n_runs=5, max_iter=100, seed_base=42):
    """
    Estima ponto nadir a partir das soluções obtidas ao otimizar f1 e f2 separadamente.
    Evita maximizar -f, pois isso pode gerar soluções artificiais e pouco úteis.
    """
    candidate_values = []

    for config in evaluators:
        summary = run_multiple_times(
            config["evaluator"],
            config["tasks"],
            n_machines,
            n_runs=n_runs,
            max_iter=max_iter,
            seed_base=seed_base,
            max_no_improve_global=MAX_NO_IMPROVE_GLOBAL,
            max_no_improve_local=MAX_NO_IMPROVE_LOCAL,
        )

        for item in summary["all_results"]:
            sol = item["solution"]
            f1 = evaluators[0]["evaluator"](sol)
            f2 = evaluators[1]["evaluator"](sol)
            candidate_values.append((f1, f2))

    f1_max = max(value[0] for value in candidate_values)
    f2_max = max(value[1] for value in candidate_values)

    return f1_max, f2_max


def get_all_objective_points(summary, evaluators, method, parameter):
    points = []

    for item in summary["all_results"]:
        sol = item["solution"]
        f1 = evaluators[0]["evaluator"](sol)
        f2 = evaluators[1]["evaluator"](sol)
        points.append({
            "f1": f1,
            "f2": f2,
            "solution": sol,
            "run": item["run"],
            "seed": item["seed"],
            "method": method,
            "parameter": parameter,
        })

    return points


def filter_non_dominated(points):
    non_dominated = []

    for i, point in enumerate(points):
        dominated = False

        for j, other in enumerate(points):
            if i == j:
                continue

            dominates = (
                other["f1"] <= point["f1"]
                and other["f2"] <= point["f2"]
                and (other["f1"] < point["f1"] or other["f2"] < point["f2"])
            )

            if dominates:
                dominated = True
                break

        if not dominated:
            non_dominated.append(point)

    # Remove duplicados por par (f1, f2)
    unique = {}
    for point in non_dominated:
        key = (round(point["f1"], 8), round(point["f2"], 8))
        if key not in unique:
            unique[key] = point

    return sorted(unique.values(), key=lambda p: (p["f1"], p["f2"]))


def limit_pareto_points(points, max_points=MAX_PARETO_POINTS):
    """Mantém no máximo max_points bem distribuídos ao longo de f1."""
    if len(points) <= max_points:
        return points

    sorted_points = sorted(points, key=lambda p: (p["f1"], p["f2"]))
    indices = np.linspace(0, len(sorted_points) - 1, max_points)
    indices = sorted(set(int(round(i)) for i in indices))

    limited = [sorted_points[i] for i in indices]

    # Garante exatamente max_points se arredondamentos removerem índices repetidos.
    i = 0
    while len(limited) < max_points and i < len(sorted_points):
        if sorted_points[i] not in limited:
            limited.append(sorted_points[i])
        i += 1

    return sorted(limited, key=lambda p: (p["f1"], p["f2"]))


def build_frontiers_by_run(points, max_points=MAX_PARETO_POINTS):
    """
    Constrói uma fronteira não-dominada separada para cada execução.

    Isso atende ao item (e) da orientação: apresentar, para cada abordagem
    escalar, as 05 fronteiras obtidas a partir das diferentes execuções
    do método.
    """
    frontiers = {}

    runs = sorted(set(point["run"] for point in points))
    for run in runs:
        run_points = [point for point in points if point["run"] == run]
        pareto_run = filter_non_dominated(run_points)
        pareto_run = limit_pareto_points(pareto_run, max_points)
        frontiers[run] = pareto_run

    return frontiers


# =====================================================================
# Soma Ponderada
# =====================================================================

def run_gvns_weighted_sum(ideal, nadir, evaluators, weight, n_machines,
                          n_runs=5, max_iter=100, seed_base=42):
    if weight >= 0.5:
        tasks = evaluators[0]["tasks"][:]
    else:
        tasks = evaluators[1]["tasks"][:]

    def obj_func(solution):
        _, _, f1_norm, f2_norm = build_normalized_values(solution, evaluators, ideal, nadir)
        return weighted_sum(weight, f1_norm, f2_norm)

    return run_multiple_times(
        obj_func,
        tasks,
        n_machines,
        n_runs=n_runs,
        max_iter=max_iter,
        seed_base=seed_base,
        max_no_improve_global=MAX_NO_IMPROVE_GLOBAL,
        max_no_improve_local=MAX_NO_IMPROVE_LOCAL,
    )


# Compatibilidade com nome antigo.
def run_vns_soma_ponderada(maximos, evaluators, summaries, peso, n_machines,
                           n_runs=5, max_iter=100, seed_base=42):
    ideal = (summaries[0]["best_value"], summaries[1]["best_value"])
    return run_gvns_weighted_sum(
        ideal,
        maximos,
        evaluators,
        peso,
        n_machines,
        n_runs=n_runs,
        max_iter=max_iter,
        seed_base=seed_base,
    )


def run_weighted_sum_frontier(evaluators, summaries, n_machines, weights):
    ideal = (summaries[0]["best_value"], summaries[1]["best_value"])
    nadir = estimate_nadir(evaluators, n_machines, n_runs=N_RUNS, max_iter=100, seed_base=SEED_BASE)

    print(f"\nPonto ideal estimado: f1={ideal[0]:.2f}, f2={ideal[1]:.2f}")
    print(f"Ponto nadir estimado: f1={nadir[0]:.2f}, f2={nadir[1]:.2f}")

    all_points = []

    for weight in weights:
        summary = run_gvns_weighted_sum(
            ideal,
            nadir,
            evaluators,
            weight,
            n_machines,
            n_runs=N_RUNS,
            max_iter=MAX_ITER,
            seed_base=SEED_BASE,
        )

        points = get_all_objective_points(summary, evaluators, "Soma Ponderada", weight)
        all_points.extend(points)
        print(f"Peso {weight:.2f}: {len(points)} soluções geradas")

    pareto = limit_pareto_points(filter_non_dominated(all_points), MAX_PARETO_POINTS)
    frontiers_by_run = build_frontiers_by_run(all_points, MAX_PARETO_POINTS)
    return all_points, pareto, frontiers_by_run


# =====================================================================
# Epsilon-restrito
# =====================================================================

def run_epsilon_restricted(evaluators, epsilon, n_machines, primary="f1",
                           n_runs=5, max_iter=100, seed_base=42):
    """
    Método epsilon-restrito com penalização.

    primary='f1': minimiza f1 sujeito a f2 <= epsilon.
    primary='f2': minimiza f2 sujeito a f1 <= epsilon.
    """
    penalty = 10**6

    if primary == "f1":
        primary_eval = evaluators[0]["evaluator"]
        secondary_eval = evaluators[1]["evaluator"]
        tasks = evaluators[0]["tasks"][:]
    elif primary == "f2":
        primary_eval = evaluators[1]["evaluator"]
        secondary_eval = evaluators[0]["evaluator"]
        tasks = evaluators[1]["tasks"][:]
    else:
        raise ValueError("primary deve ser 'f1' ou 'f2'.")

    def obj_func(solution):
        primary_value = primary_eval(solution)
        secondary_value = secondary_eval(solution)
        violation = max(0.0, secondary_value - epsilon)
        return primary_value + penalty * violation

    return run_multiple_times(
        obj_func,
        tasks,
        n_machines,
        n_runs=n_runs,
        max_iter=max_iter,
        seed_base=seed_base,
        max_no_improve_global=MAX_NO_IMPROVE_GLOBAL,
        max_no_improve_local=MAX_NO_IMPROVE_LOCAL,
    )


def run_epsilon_frontier(evaluators, summaries, n_machines, epsilons, primary="f1"):
    all_points = []

    for epsilon in epsilons:
        summary = run_epsilon_restricted(
            evaluators,
            epsilon,
            n_machines,
            primary=primary,
            n_runs=N_RUNS,
            max_iter=MAX_ITER,
            seed_base=SEED_BASE,
        )

        points = get_all_objective_points(summary, evaluators, "Epsilon-restrito", epsilon)
        all_points.extend(points)
        print(f"Epsilon {epsilon:.2f}: {len(points)} soluções geradas")

    pareto = limit_pareto_points(filter_non_dominated(all_points), MAX_PARETO_POINTS)
    frontiers_by_run = build_frontiers_by_run(all_points, MAX_PARETO_POINTS)
    return all_points, pareto, frontiers_by_run
