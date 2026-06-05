import pandas as pd
import numpy as np
from config import FILE_PATH, N_RUNS, MAX_ITER, SEED_BASE
from run_functions import (
    run_single_objective,
    run_weighted_sum_frontier,
    run_epsilon_frontier,
)
from visualization import (
    plot_pareto_frontier,
    plot_two_frontiers,
    print_pareto_points,
    plot_frontiers_by_run,
    print_frontiers_by_run,
)
from evaluation import build_makespan_evaluator, build_weighted_tardiness_evaluator


# =====================================================================
# Leitura da instância
# =====================================================================

def load_instance_from_excel(file_path):
    """
    Lê o arquivo Excel:
    - Coluna 0: tarefa;
    - Colunas M1...M5: tempos de processamento;
    - Coluna Peso: penalidade por atraso;
    - Linha DueDate: data comum de entrega.
    """
    df = pd.read_excel(file_path)
    df.columns = ["Tarefa", "M1", "M2", "M3", "M4", "M5", "Peso"]
    df = df.dropna(how="all").reset_index(drop=True)

    due_row = df[df["Tarefa"].astype(str).str.strip().str.lower() == "duedate"]
    if due_row.empty:
        raise ValueError("Não foi encontrada a linha do DueDate.")

    due_date = int(due_row.iloc[0]["M1"])

    df = df[df["Tarefa"].astype(str).str.strip().str.lower() != "duedate"].copy()
    df = df[df["Tarefa"].notna()].copy()
    df["Tarefa"] = df["Tarefa"].astype(int)
    df = df.sort_values("Tarefa").reset_index(drop=True)

    pt = df[["M1", "M2", "M3", "M4", "M5"]].astype(float).to_numpy()
    we = df["Peso"].astype(float).to_numpy()

    n_tasks = pt.shape[0]
    n_machines = pt.shape[1]

    return pt, we, due_date, n_tasks, n_machines


def print_instance_info(n_tasks, n_machines, due_date, pt, we):
    print("Instância carregada com sucesso.")
    print(f"Número de tarefas: {n_tasks}")
    print(f"Número de máquinas: {n_machines}")
    print(f"Due date: {due_date}")
    print(f"Formato PT: {pt.shape}")
    print(f"Formato WE: {we.shape}")


# =====================================================================
# Configuração dos objetivos
# =====================================================================

def build_evaluator_configs(pt, we, due_date, n_tasks):
    evaluator_configs = []

    # Para makespan, prioriza tarefas mais longas primeiro.
    tasks_f1 = list(range(n_tasks))
    tasks_f1.sort(key=lambda j: -np.min(pt[j, :]))

    evaluator_configs.append({
        "evaluator": build_makespan_evaluator(we, pt),
        "tasks": tasks_f1,
        "name": "f1 (makespan)",
    })

    # Para atraso ponderado, prioriza maior peso e menor tempo mínimo.
    # Como a due date é comum, EDD puro não diferencia as tarefas.
    tasks_f2 = list(range(n_tasks))
    tasks_f2.sort(key=lambda j: (-we[j], np.min(pt[j, :])))

    evaluator_configs.append({
        "evaluator": build_weighted_tardiness_evaluator(we, pt, due_date),
        "tasks": tasks_f2,
        "name": "f2 (soma ponderada dos atrasos)",
    })

    return evaluator_configs


# =====================================================================
# Main
# =====================================================================

def main():
    pt, we, due_date, n_tasks, n_machines = load_instance_from_excel(FILE_PATH)
    print_instance_info(n_tasks, n_machines, due_date, pt, we)

    evaluator_configs = build_evaluator_configs(pt, we, due_date, n_tasks)

    print("\nExecutando otimizações mono-objetivo com GVNS + RVND...")
    summaries = run_single_objective(evaluator_configs, n_machines, pt, we, due_date)

    # -----------------------------------------------------------------
    # Soma Ponderada
    # -----------------------------------------------------------------
    print("\nExecutando abordagem multiobjetivo por Soma Ponderada...")
    weights = np.linspace(0, 1, 11)
    weighted_all_points, weighted_pareto, weighted_frontiers_by_run = run_weighted_sum_frontier(
        evaluator_configs,
        summaries,
        n_machines,
        weights,
    )

    print_pareto_points(weighted_pareto, "Fronteira não-dominada - Soma Ponderada")
    plot_pareto_frontier(
        weighted_pareto,
        title="Fronteira de Pareto estimada - Soma Ponderada",
        filename="pareto_soma_ponderada.png",
    )
    print_frontiers_by_run(
        weighted_frontiers_by_run,
        "05 fronteiras por execução - Soma Ponderada",
    )
    plot_frontiers_by_run(
        weighted_frontiers_by_run,
        title="05 fronteiras estimadas - Soma Ponderada",
        filename="pareto_soma_ponderada_5_execucoes.png",
    )

    # -----------------------------------------------------------------
    # Epsilon-restrito
    # -----------------------------------------------------------------
    print("\nExecutando abordagem multiobjetivo por Epsilon-restrito...")

    # Minimiza f1 sujeito a f2 <= epsilon.
    # Os epsilons são gerados entre o melhor f2 encontrado e o pior f2 observado
    # nas soluções da Soma Ponderada, evitando valores totalmente fora da escala.
    f2_min = min(point["f2"] for point in weighted_all_points)
    f2_max = max(point["f2"] for point in weighted_all_points)
    epsilons = np.linspace(f2_min, f2_max, 11)

    epsilon_all_points, epsilon_pareto, epsilon_frontiers_by_run = run_epsilon_frontier(
        evaluator_configs,
        summaries,
        n_machines,
        epsilons,
        primary="f1",
    )

    print_pareto_points(epsilon_pareto, "Fronteira não-dominada - Epsilon-restrito")
    plot_pareto_frontier(
        epsilon_pareto,
        title="Fronteira de Pareto estimada - Epsilon-restrito",
        filename="pareto_epsilon_restrito.png",
    )
    print_frontiers_by_run(
        epsilon_frontiers_by_run,
        "05 fronteiras por execução - Epsilon-restrito",
    )
    plot_frontiers_by_run(
        epsilon_frontiers_by_run,
        title="05 fronteiras estimadas - Epsilon-restrito",
        filename="pareto_epsilon_restrito_5_execucoes.png",
    )

    # -----------------------------------------------------------------
    # Comparação final
    # -----------------------------------------------------------------
    plot_two_frontiers(weighted_pareto, epsilon_pareto, filename="pareto_comparativo.png")

    print("\nResumo final:")
    print(f"Soma Ponderada: {len(weighted_all_points)} soluções geradas; {len(weighted_pareto)} não-dominadas selecionadas.")
    print(f"Epsilon-restrito: {len(epsilon_all_points)} soluções geradas; {len(epsilon_pareto)} não-dominadas selecionadas.")
    print("Gráficos salvos na pasta img/.")
    print("Figuras do item (e): pareto_soma_ponderada_5_execucoes.png e pareto_epsilon_restrito_5_execucoes.png.")


if __name__ == "__main__":
    main()
