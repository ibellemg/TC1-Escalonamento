import os
import time
import numpy as np
import pandas as pd
from contextlib import redirect_stdout

from mip_puro import carregar_instancia as carregar_mip
from mip_puro import resolver_mip_puro, plotar_gantt

from main import load_instance_from_excel, build_tasks_order
from evaluation import build_tc1_evaluator, evaluate_objectives
from optimization import run_multiple_times
from visualization import plot_convergence, plot_best_schedule


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "resultados")
IMG_DIR = os.path.join(BASE_DIR, "img")


INSTANCIAS = [
    "i5x25.xlsx",
    "i5x50.xlsx",
    "i5x100.xlsx",
]


LIMITES_MIP = {
    "i5x25.xlsx": 900,
    "i5x50.xlsx": 900,
    "i5x100.xlsx": 1800,
}


CONFIG_GVNS = {
    "n_runs": 10,
    "max_iter": 1000,
    "seed_base": 42,
    "max_no_improve_global": 200,
    "max_no_improve_local": 100,
}


def nome_base(nome_arquivo):
    return nome_arquivo.replace(".xlsx", "")


def rodar_mip(nome_arquivo):
    base = nome_base(nome_arquivo)
    caminho = os.path.join(DATA_DIR, nome_arquivo)
    time_limit = LIMITES_MIP[nome_arquivo]

    saida_txt = os.path.join(
        RESULTS_DIR,
        f"MIP_{base}_{time_limit}s.txt"
    )

    with open(saida_txt, "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            print(f"===== MIP PURO - {base} =====")
            print(f"Limite de tempo: {time_limit} segundos")

            tempos, pesos, due_date, n_tarefas, n_maquinas = carregar_mip(caminho)

            print(f"Instância: {base}")
            print(f"Número de tarefas: {n_tarefas}")
            print(f"Número de máquinas: {n_maquinas}")
            print(f"Due date: {due_date}")

            inicio = time.time()

            resultados = resolver_mip_puro(
                tempos,
                pesos,
                due_date,
                time_limit=time_limit,
            )

            fim = time.time()

            print("\n===== RESULTADOS =====")
            print(f"FO = {resultados['objetivo']:.4f}")
            print(f"Cmax = {resultados['cmax']:.4f}")
            print(f"Atraso ponderado = {resultados['atraso_ponderado']:.4f}")
            print(f"GAP = {resultados['gap']:.6f}")
            print(f"Tempo Gurobi = {resultados['tempo']:.4f} s")
            print(f"Tempo total = {fim - inicio:.4f} s")

            print("\nSequência por máquina:")
            for k, tarefas_maquina in enumerate(resultados["solucao"], start=1):
                sequencia = [item["tarefa"] for item in tarefas_maquina]
                print(f"Máquina {k}: {sequencia}")

            plotar_gantt(
                resultados["solucao"],
                due_date,
                nome_arquivo=f"gantt_MIP_{base}_{time_limit}s.png",
            )

    return saida_txt


def rodar_gvns(nome_arquivo):
    base = nome_base(nome_arquivo)
    caminho = os.path.join(DATA_DIR, nome_arquivo)

    max_iter = CONFIG_GVNS["max_iter"]
    n_runs = CONFIG_GVNS["n_runs"]

    saida_txt = os.path.join(
        RESULTS_DIR,
        f"GVNSRVND_{base}_{max_iter}iter.txt"
    )

    saida_csv = os.path.join(
        RESULTS_DIR,
        f"GVNSRVND_{base}_{max_iter}iter_execucoes.csv"
    )

    with open(saida_txt, "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            print(f"===== GVNS + RVND - {base} =====")
            print(f"Número de execuções: {n_runs}")
            print(f"Máximo de iterações: {max_iter}")
            print(f"Max. sem melhora global: {CONFIG_GVNS['max_no_improve_global']}")
            print(f"Max. sem melhora local: {CONFIG_GVNS['max_no_improve_local']}")

            pt, we, due_date, n_tasks, n_machines = load_instance_from_excel(caminho)

            print(f"\nInstância: {base}")
            print(f"Número de tarefas: {n_tasks}")
            print(f"Número de máquinas: {n_machines}")
            print(f"Due date: {due_date}")
            print("FO = Cmax + soma ponderada dos atrasos")

            evaluator = build_tc1_evaluator(we, pt, due_date)
            tasks = build_tasks_order(pt, we)

            inicio = time.time()

            summary = run_multiple_times(
                evaluator,
                tasks,
                n_machines,
                n_runs=n_runs,
                max_iter=max_iter,
                seed_base=CONFIG_GVNS["seed_base"],
                max_no_improve_global=CONFIG_GVNS["max_no_improve_global"],
                max_no_improve_local=CONFIG_GVNS["max_no_improve_local"],
            )

            fim = time.time()

            valores = [item["value"] for item in summary["all_results"]]

            print("\n===== EXECUÇÕES =====")
            for item in summary["all_results"]:
                print(
                    f"Execução {item['run']} | "
                    f"Seed {item['seed']} | "
                    f"FO final = {item['value']:.4f}"
                )

            print("\n===== ESTATÍSTICAS =====")
            print(f"Melhor = {np.min(valores):.4f}")
            print(f"Média = {np.mean(valores):.4f}")
            print(f"Desvio padrão amostral = {np.std(valores, ddof=1):.4f}")
            print(f"Pior = {np.max(valores):.4f}")
            print(f"Tempo total = {fim - inicio:.4f} s")

            best_solution = summary["best_solution"]

            cmax, atraso_ponderado, fo_total = evaluate_objectives(
                best_solution,
                we,
                pt,
                due_date,
            )

            print("\n===== MELHOR SOLUÇÃO =====")
            print(f"FO total = {fo_total:.4f}")
            print(f"Cmax = {cmax:.4f}")
            print(f"Atraso ponderado = {atraso_ponderado:.4f}")

            print("\nSequência por máquina:")
            for k, machine_tasks in enumerate(best_solution, start=1):
                tarefas = [task + 1 for task in machine_tasks]
                print(f"Máquina {k}: {tarefas}")

            plot_convergence(
                summary,
                f"Curvas de convergência - GVNS + RVND - {base}",
                filename=f"convergencia_GVNSRVND_{base}_{max_iter}iter.png",
            )

            plot_best_schedule(
                best_solution,
                f"Melhor solução GVNS + RVND - {base}",
                pt,
                we,
                due_date,
                filename=f"gantt_GVNSRVND_{base}_{max_iter}iter.png",
            )

    linhas = []

    for item in summary["all_results"]:
        linhas.append({
            "instancia": base,
            "run": item["run"],
            "seed": item["seed"],
            "fo_final": item["value"],
            "max_iter": max_iter,
        })

    pd.DataFrame(linhas).to_csv(saida_csv, index=False)

    return saida_txt, saida_csv


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)

    print("Iniciando experimentos estendidos...")

    for instancia in INSTANCIAS:
        print(f"\nRodando MIP para {instancia}...")
        txt_mip = rodar_mip(instancia)
        print(f"Resultado MIP salvo em: {txt_mip}")

        print(f"Rodando GVNS + RVND para {instancia}...")
        txt_gvns, csv_gvns = rodar_gvns(instancia)
        print(f"Resultado GVNS salvo em: {txt_gvns}")
        print(f"CSV GVNS salvo em: {csv_gvns}")

    print("\nExperimentos finalizados.")


if __name__ == "__main__":
    main()