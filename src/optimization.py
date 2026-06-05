import numpy as np
import random
from heuristic import constructive_initial_solution as initial_solution


# =====================================================================
# Estruturas de vizinhança
# =====================================================================

def clone_solution(solution):
    return [machine[:] for machine in solution]


def intra_machine_swap(solution, rng):
    sol = clone_solution(solution)
    candidate_machines = [k for k in range(len(sol)) if len(sol[k]) >= 2]

    if not candidate_machines:
        return sol

    k = rng.choice(candidate_machines)
    i, j = rng.sample(range(len(sol[k])), 2)
    sol[k][i], sol[k][j] = sol[k][j], sol[k][i]
    return sol


def inter_machine_insert(solution, rng):
    sol = clone_solution(solution)
    source_candidates = [k for k in range(len(sol)) if len(sol[k]) >= 1]

    if not source_candidates:
        return sol

    k_from = rng.choice(source_candidates)
    idx_from = rng.randrange(len(sol[k_from]))
    task = sol[k_from].pop(idx_from)

    k_to = rng.randrange(len(sol))
    idx_to = rng.randrange(len(sol[k_to]) + 1)
    sol[k_to].insert(idx_to, task)

    return sol


def inter_machine_swap(solution, rng):
    sol = clone_solution(solution)
    candidate_machines = [k for k in range(len(sol)) if len(sol[k]) >= 1]

    if len(candidate_machines) < 2:
        return sol

    k1, k2 = rng.sample(candidate_machines, 2)
    i = rng.randrange(len(sol[k1]))
    j = rng.randrange(len(sol[k2]))
    sol[k1][i], sol[k2][j] = sol[k2][j], sol[k1][i]
    return sol


def intra_machine_2opt(solution, rng):
    sol = clone_solution(solution)
    candidate_machines = [k for k in range(len(sol)) if len(sol[k]) >= 3]

    if not candidate_machines:
        return sol

    k = rng.choice(candidate_machines)
    i, j = sorted(rng.sample(range(len(sol[k])), 2))
    sol[k][i:j + 1] = sol[k][i:j + 1][::-1]
    return sol


def or_opt(solution, rng):
    """Move um bloco de duas tarefas para outra máquina."""
    sol = clone_solution(solution)
    candidate_machines = [k for k in range(len(sol)) if len(sol[k]) >= 2]

    if not candidate_machines:
        return sol

    k_from = rng.choice(candidate_machines)
    idx = rng.randrange(len(sol[k_from]) - 1)
    block = sol[k_from][idx:idx + 2]
    sol[k_from] = sol[k_from][:idx] + sol[k_from][idx + 2:]

    possible_targets = [k for k in range(len(sol)) if k != k_from]
    if not possible_targets:
        return sol

    k_to = rng.choice(possible_targets)
    pos = rng.randrange(len(sol[k_to]) + 1)
    sol[k_to] = sol[k_to][:pos] + block + sol[k_to][pos:]

    return sol


def block_exchange(solution, rng):
    """Troca blocos de uma ou duas tarefas entre duas máquinas."""
    sol = clone_solution(solution)
    candidate_machines = [k for k in range(len(sol)) if len(sol[k]) >= 1]

    if len(candidate_machines) < 2:
        return sol

    k1, k2 = rng.sample(candidate_machines, 2)
    size1 = rng.randint(1, min(2, len(sol[k1])))
    size2 = rng.randint(1, min(2, len(sol[k2])))

    i = rng.randrange(len(sol[k1]) - size1 + 1)
    j = rng.randrange(len(sol[k2]) - size2 + 1)

    block1 = sol[k1][i:i + size1]
    block2 = sol[k2][j:j + size2]

    sol[k1] = sol[k1][:i] + block2 + sol[k1][i + size1:]
    sol[k2] = sol[k2][:j] + block1 + sol[k2][j + size2:]

    return sol


NEIGHBORHOODS = [
    ("swap_intra", intra_machine_swap),
    ("insert_inter", inter_machine_insert),
    ("swap_inter", inter_machine_swap),
    ("2opt_intra", intra_machine_2opt),
    ("or_opt", or_opt),
    ("block_exchange", block_exchange),
]


# =====================================================================
# RVND - Random Variable Neighborhood Descent
# =====================================================================

def rvnd(solution, evaluator, rng, max_no_improve=60, samples_per_neighborhood=10):
    """
    Busca local RVND.

    A ordem das vizinhanças é embaralhada. Se uma vizinhança melhora a solução,
    a lista é reiniciada. Se não melhora, a vizinhança é removida da rodada.
    """
    current = clone_solution(solution)
    current_value = evaluator(current)
    no_improve = 0

    while no_improve < max_no_improve:
        neighborhoods = NEIGHBORHOODS[:]
        rng.shuffle(neighborhoods)
        improved_in_round = False

        while neighborhoods and no_improve < max_no_improve:
            _, neighborhood = neighborhoods.pop(0)

            best_candidate = None
            best_candidate_value = current_value

            for _ in range(samples_per_neighborhood):
                candidate = neighborhood(current, rng)
                candidate_value = evaluator(candidate)

                if candidate_value < best_candidate_value:
                    best_candidate = candidate
                    best_candidate_value = candidate_value

            if best_candidate is not None:
                current = best_candidate
                current_value = best_candidate_value
                no_improve = 0
                improved_in_round = True

                neighborhoods = NEIGHBORHOODS[:]
                rng.shuffle(neighborhoods)
            else:
                no_improve += 1

        if not improved_in_round:
            no_improve += 1

    return current, current_value


# =====================================================================
# GVNS - General Variable Neighborhood Search
# =====================================================================

def shake(solution, k, rng):
    """Perturba a solução. A intensidade cresce com k."""
    shaken = clone_solution(solution)
    n_moves = rng.randint(k + 2, k + 5)

    for _ in range(n_moves):
        _, neighborhood = rng.choice(NEIGHBORHOODS)
        shaken = neighborhood(shaken, rng)

    return shaken


def run_gvns(
    evaluator,
    tasks,
    n_machine,
    rng,
    max_iter=400,
    max_no_improve_global=80,
    max_no_improve_local=60,
):
    current = initial_solution(evaluator, tasks, n_machine, rng)
    current, current_value = rvnd(
        current,
        evaluator,
        rng,
        max_no_improve=max_no_improve_local,
    )

    best = clone_solution(current)
    best_value = current_value

    history = [best_value]
    iter_count = 0
    no_improve_global = 0

    while iter_count < max_iter and no_improve_global < max_no_improve_global:
        k = 0

        while (
            k < len(NEIGHBORHOODS)
            and iter_count < max_iter
            and no_improve_global < max_no_improve_global
        ):
            shaken = shake(best, k, rng)
            candidate, candidate_value = rvnd(
                shaken,
                evaluator,
                rng,
                max_no_improve=max_no_improve_local,
            )

            if candidate_value < best_value:
                best = candidate
                best_value = candidate_value
                k = 0
                no_improve_global = 0
            else:
                k += 1
                no_improve_global += 1

            history.append(best_value)
            iter_count += 1

    return best, best_value, history


# Mantém compatibilidade com chamadas antigas.
run_vns = run_gvns


# =====================================================================
# Execuções múltiplas
# =====================================================================

def run_multiple_times(
    evaluator,
    tasks,
    n_machine,
    n_runs=5,
    max_iter=400,
    seed_base=42,
    max_no_improve_global=80,
    max_no_improve_local=60,
):
    results = []
    best_global_sol = None
    best_global_val = float("inf")
    best_global_history = None

    for r in range(n_runs):
        seed = seed_base + r
        rng = random.Random(seed)

        best_sol, best_val, history = run_gvns(
            evaluator,
            tasks,
            n_machine,
            rng,
            max_iter=max_iter,
            max_no_improve_global=max_no_improve_global,
            max_no_improve_local=max_no_improve_local,
        )

        results.append({
            "run": r + 1,
            "seed": seed,
            "value": best_val,
            "history": history,
            "solution": best_sol,
        })

        if best_val < best_global_val:
            best_global_val = best_val
            best_global_sol = best_sol
            best_global_history = history

    values = np.array([r["value"] for r in results], dtype=float)

    return {
        "min": float(np.min(values)),
        "std": float(np.std(values, ddof=0)),
        "max": float(np.max(values)),
        "best_solution": best_global_sol,
        "best_value": best_global_val,
        "best_history": best_global_history,
        "all_results": results,
    }
