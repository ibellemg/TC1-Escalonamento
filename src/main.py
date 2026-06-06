import pandas as pd
import numpy as np

from config import (
    FILE_PATH,
    N_RUNS,
    MAX_ITER,
    SEED_BASE,
    MAX_NO_IMPROVE_GLOBAL,
    MAX_NO_IMPROVE_LOCAL,
)

from optimization import run_multiple_times
from visualization import print_summary_table, plot_convergence, plot_best_schedule
from evaluation import build_tc1_evaluator, evaluate_objectives


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

    due_date = float(due_row.iloc[0]["M1"])

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


def build_tasks_order(pt, we):
    """
    Define a ordem inicial das tarefas para a heurística construtiva.

    Como a due date é comum, EDD não diferencia as tarefas.
    Por isso, usamos uma regra baseada em:
    - maior peso primeiro;
    - menor tempo mínimo como desempate.
    """
    n_tasks = len(we)
    tasks = list(range(n_tasks))
    tasks.sort(key=lambda j: (-we[j], np.min(pt[j, :])))
    return tasks


def main():
    pt, we, due_date, n_tasks, n_machines = load_instance_from_excel(FILE_PATH)
    print_instance_info(n_tasks, n_machines, due_date, pt, we)

    evaluator = build_tc1_evaluator(we, pt, due_date)
    tasks = build_tasks_order(pt, we)

    print("\nExecutando GVNS + RVND para a função objetivo do TC1...")
    print("FO = Cmax + soma ponderada dos atrasos")

    summary = run_multiple_times(
        evaluator,
        tasks,
        n_machines,
        n_runs=N_RUNS,
        max_iter=MAX_ITER,
        seed_base=SEED_BASE,
        max_no_improve_global=MAX_NO_IMPROVE_GLOBAL,
        max_no_improve_local=MAX_NO_IMPROVE_LOCAL,
    )

    print_summary_table(summary, "GVNS + RVND - FO TC1")

    best_solution = summary["best_solution"]
    cmax, atraso_ponderado, fo_total = evaluate_objectives(best_solution, we, pt, due_date)

    print("\n===== Melhor solução global - GVNS + RVND =====")
    print(f"FO total = {fo_total:.4f}")
    print(f"Cmax = {cmax:.4f}")
    print(f"Atraso ponderado = {atraso_ponderado:.4f}")

    plot_convergence(
        summary,
        "Curvas de convergência - GVNS + RVND",
        filename="convergencia_gvns_rvnd.png",
    )

    plot_best_schedule(
        best_solution,
        "Melhor solução encontrada pelo GVNS + RVND",
        pt,
        we,
        due_date,
        filename="gantt_gvns_rvnd.png",
    )

    print("\nGráficos gerados:")
    print("img/convergencia_gvns_rvnd.png")
    print("img/gantt_gvns_rvnd.png")


if __name__ == "__main__":
    main()