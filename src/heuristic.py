def create_empty_solution(n_machines):
    return [[] for _ in range(n_machines)]


def clone_solution(solution):
    return [machine[:] for machine in solution]


def constructive_initial_solution(evaluator, tasks, n_machines, rng, randomization_rate=0.15):
    """
    Constrói uma solução inicial viável por inserção gulosa.

    A lista de tarefas já chega ordenada por uma regra de prioridade definida no main.
    A pequena aleatoriedade evita que todas as execuções partam exatamente do mesmo ponto.
    """
    ordered_tasks = tasks[:]

    for i in range(len(ordered_tasks) - 1):
        if rng.random() < randomization_rate:
            j = rng.randint(i, len(ordered_tasks) - 1)
            ordered_tasks[i], ordered_tasks[j] = ordered_tasks[j], ordered_tasks[i]

    solution = create_empty_solution(n_machines)

    for task in ordered_tasks:
        best_solution = None
        best_value = float("inf")

        for k in range(n_machines):
            candidate = clone_solution(solution)
            candidate[k].append(task)
            value = evaluator(candidate)

            if value < best_value:
                best_value = value
                best_solution = candidate

        solution = best_solution

    return solution


# Mantém compatibilidade com versões anteriores do código.
greedy_initial_solution = constructive_initial_solution
