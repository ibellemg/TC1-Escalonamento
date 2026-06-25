import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "i5x25.xlsx")
IMG_DIR = os.path.join(BASE_DIR, "img")


def carregar_instancia(caminho):
    df = pd.read_excel(caminho)
    df = df.dropna(how="all").reset_index(drop=True)

    # Se a planilha nova já veio com colunas Tarefa, M1, ..., Peso
    if "Tarefa" in df.columns and "M1" in df.columns:
        pass
    else:
        # Compatibilidade com a planilha original
        df.columns = ["Tarefa", "M1", "M2", "M3", "M4", "M5", "Peso"]

    due_row = df[df["Tarefa"].astype(str).str.strip().str.lower() == "duedate"]
    if due_row.empty:
        raise ValueError("Não foi encontrada a linha DueDate.")

    due_date = float(due_row.iloc[0]["M1"])

    df = df[df["Tarefa"].astype(str).str.strip().str.lower() != "duedate"].copy()
    df = df[df["Tarefa"].notna()].copy()
    df["Tarefa"] = df["Tarefa"].astype(int)
    df = df.sort_values("Tarefa").reset_index(drop=True)

    machine_cols = [col for col in df.columns if str(col).startswith("M")]
    tempos = df[machine_cols].astype(float).to_numpy()
    pesos = df["Peso"].astype(float).to_numpy()

    n_tarefas = tempos.shape[0]
    n_maquinas = tempos.shape[1]

    return tempos, pesos, due_date, n_tarefas, n_maquinas


def resolver_mip_puro(tempos, pesos, due_date, time_limit=300):
    n_tarefas, n_maquinas = tempos.shape

    tarefas = range(n_tarefas)
    maquinas = range(n_maquinas)

    M = float(np.sum(np.max(tempos, axis=1)))

    modelo = gp.Model("MIP_Puro_Escalonamento")

    x = modelo.addVars(tarefas, maquinas, vtype=GRB.BINARY, name="x")
    y = modelo.addVars(tarefas, tarefas, maquinas, vtype=GRB.BINARY, name="y")

    S = modelo.addVars(tarefas, lb=0, vtype=GRB.CONTINUOUS, name="S")
    C = modelo.addVars(tarefas, lb=0, vtype=GRB.CONTINUOUS, name="C")
    T = modelo.addVars(tarefas, lb=0, vtype=GRB.CONTINUOUS, name="T")
    Cmax = modelo.addVar(lb=0, vtype=GRB.CONTINUOUS, name="Cmax")

    # Cada tarefa em exatamente uma máquina
    for j in tarefas:
        modelo.addConstr(gp.quicksum(x[j, k] for k in maquinas) == 1)

    # Definição do tempo de conclusão
    for j in tarefas:
        for k in maquinas:
            modelo.addConstr(C[j] >= S[j] + tempos[j, k] - M * (1 - x[j, k]))

    # Não sobreposição entre tarefas na mesma máquina
    for k in maquinas:
        for i in tarefas:
            for j in tarefas:
                if i != j:
                    modelo.addConstr(
                        S[j] >= S[i] + tempos[i, k] - M * (1 - y[i, j, k])
                    )

        for i in tarefas:
            for j in tarefas:
                if i < j:
                    modelo.addConstr(y[i, j, k] + y[j, i, k] <= x[i, k])
                    modelo.addConstr(y[i, j, k] + y[j, i, k] <= x[j, k])
                    modelo.addConstr(y[i, j, k] + y[j, i, k] >= x[i, k] + x[j, k] - 1)

    # Makespan e atraso
    for j in tarefas:
        modelo.addConstr(Cmax >= C[j])
        modelo.addConstr(T[j] >= C[j] - due_date)

    # Função objetivo: Cmax + soma ponderada dos atrasos
    modelo.setObjective(
        Cmax + gp.quicksum(pesos[j] * T[j] for j in tarefas),
        GRB.MINIMIZE
    )

    modelo.Params.TimeLimit = time_limit
    modelo.Params.MIPGap = 0.01
    modelo.Params.MIPFocus = 1
    modelo.Params.Heuristics = 0.8
    modelo.Params.Cuts = 3
    modelo.Params.Presolve = 2
    modelo.Params.Symmetry = 2
    modelo.Params.Threads = 8

    modelo.optimize()

    if modelo.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        raise RuntimeError("O modelo não encontrou solução viável.")

    solucao = [[] for _ in maquinas]

    for j in tarefas:
        for k in maquinas:
            if x[j, k].X > 0.5:
                solucao[k].append({
                    "tarefa": j + 1,
                    "inicio": S[j].X,
                    "fim": C[j].X,
                    "duracao": tempos[j, k],
                    "peso": pesos[j],
                    "atraso": T[j].X
                })

    for k in maquinas:
        solucao[k].sort(key=lambda item: item["inicio"])

    resultados = {
        "objetivo": modelo.ObjVal,
        "cmax": Cmax.X,
        "atraso_ponderado": sum(pesos[j] * T[j].X for j in tarefas),
        "gap": modelo.MIPGap,
        "tempo": modelo.Runtime,
        "solucao": solucao
    }

    return resultados


def plotar_gantt(solucao, due_date, nome_arquivo="gantt_mip_puro.png"):
    os.makedirs(IMG_DIR, exist_ok=True)

    plt.figure(figsize=(12, 6))

    for k, tarefas_maquina in enumerate(solucao):
        for item in tarefas_maquina:
            plt.barh(
                y=k + 1,
                width=item["duracao"],
                left=item["inicio"],
                edgecolor="black"
            )
            plt.text(
                item["inicio"] + item["duracao"] / 2,
                k + 1,
                f"J{item['tarefa']}",
                ha="center",
                va="center",
                fontsize=8
            )

    plt.axvline(due_date, linestyle="--", label=f"Due date = {due_date}")
    plt.xlabel("Tempo")
    plt.ylabel("Máquina")
    plt.yticks(range(1, len(solucao) + 1), [f"M{k}" for k in range(1, len(solucao) + 1)])
    plt.title("Melhor solução encontrada pelo MIP puro")
    plt.grid(True, axis="x", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    caminho = os.path.join(IMG_DIR, nome_arquivo)
    plt.savefig(caminho, dpi=300)
    plt.close()

    print(f"Gráfico salvo em: {caminho}")


def imprimir_resultados(resultados):
    print("\n===== RESULTADOS - MIP PURO =====")
    print(f"Função objetivo: {resultados['objetivo']:.4f}")
    print(f"Cmax: {resultados['cmax']:.4f}")
    print(f"Atraso ponderado: {resultados['atraso_ponderado']:.4f}")
    print(f"GAP: {resultados['gap']:.6f}")
    print(f"Tempo de execução: {resultados['tempo']:.4f} segundos")

    print("\nSequência por máquina:")
    for k, tarefas_maquina in enumerate(resultados["solucao"], start=1):
        sequencia = [item["tarefa"] for item in tarefas_maquina]
        print(f"Máquina {k}: {sequencia}")


def main():
    tempos, pesos, due_date, n_tarefas, n_maquinas = carregar_instancia(DATA_PATH)

    print("Instância carregada.")
    print(f"Número de tarefas: {n_tarefas}")
    print(f"Número de máquinas: {n_maquinas}")
    print(f"Due date: {due_date}")

    resultados = resolver_mip_puro(tempos, pesos, due_date)
    imprimir_resultados(resultados)
    plotar_gantt(resultados["solucao"], due_date)


if __name__ == "__main__":
    main()