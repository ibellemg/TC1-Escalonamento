import numpy as np


def evaluate_weighted_tardiness(solution, we, pt, due_date):
    """Avalia soma ponderada dos atrasos: sum_j w_j * T_j."""
    n_tasks = len(we)
    completion_times = np.zeros(n_tasks)

    for k, machine_tasks in enumerate(solution):
        current_time = 0.0
        for task in machine_tasks:
            current_time += pt[task, k]
            completion_times[task] = current_time

    tardiness = np.maximum(completion_times - due_date, 0.0)
    return float(np.sum(we * tardiness))


def evaluate_makespan(solution, we, pt):
    """Avalia o makespan: Cmax."""
    machine_completion = np.zeros(len(solution))

    for k, machine_tasks in enumerate(solution):
        current_time = 0.0
        for task in machine_tasks:
            current_time += pt[task, k]
        machine_completion[k] = current_time

    return float(np.max(machine_completion))


def evaluate_tc1(solution, we, pt, due_date):
    """
    Função objetivo do TC1:
    FO = Cmax + soma_j w_j * T_j
    """
    cmax = evaluate_makespan(solution, we, pt)
    weighted_tardiness = evaluate_weighted_tardiness(solution, we, pt, due_date)
    return cmax + weighted_tardiness


def build_tc1_evaluator(we, pt, due_date):
    """Constrói o avaliador da função objetivo do TC1."""
    return lambda solution: evaluate_tc1(solution, we, pt, due_date)


def evaluate_objectives(solution, we, pt, due_date):
    """Retorna Cmax, atraso ponderado e FO combinada."""
    cmax = evaluate_makespan(solution, we, pt)
    weighted_tardiness = evaluate_weighted_tardiness(solution, we, pt, due_date)
    total = cmax + weighted_tardiness
    return cmax, weighted_tardiness, total