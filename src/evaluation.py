import numpy as np


def build_weighted_tardiness_evaluator(we, pt, due_date):
    """Constrói a função objetivo f2: soma ponderada dos atrasos."""
    return lambda solution: evaluate_weighted_tardiness(solution, we, pt, due_date)


def evaluate_weighted_tardiness(solution, we, pt, due_date):
    """Avalia f2 = soma_j w_j * T_j."""
    n_tasks = len(we)
    completion_times = np.zeros(n_tasks)

    for k, machine_tasks in enumerate(solution):
        current_time = 0.0
        for task in machine_tasks:
            current_time += pt[task, k]
            completion_times[task] = current_time

    tardiness = np.maximum(completion_times - due_date, 0.0)
    return float(np.sum(we * tardiness))


def build_makespan_evaluator(we, pt):
    """Constrói a função objetivo f1: makespan."""
    return lambda solution: evaluate_makespan(solution, we, pt)


def evaluate_makespan(solution, we, pt):
    """Avalia f1 = Cmax."""
    machine_completion = np.zeros(len(solution))

    for k, machine_tasks in enumerate(solution):
        current_time = 0.0
        for task in machine_tasks:
            current_time += pt[task, k]
        machine_completion[k] = current_time

    return float(np.max(machine_completion))


def evaluate_objectives(solution, evaluator_f1, evaluator_f2):
    """Retorna o par objetivo (f1, f2) de uma solução."""
    return evaluator_f1(solution), evaluator_f2(solution)
