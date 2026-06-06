import os
import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


def gerar_instancia(
    n_tarefas,
    n_maquinas=5,
    seed=42,
    alpha_due_date=0.20,
):
    rng = np.random.default_rng(seed)

    # --------------------------------------------------
    # Dificuldade intrínseca das tarefas
    # --------------------------------------------------
    dificuldade_tarefa = rng.integers(2, 11, size=n_tarefas)

    # --------------------------------------------------
    # Eficiência relativa das máquinas
    # --------------------------------------------------
    eficiencia_maquina = rng.uniform(0.8, 1.3, size=n_maquinas)

    tempos = np.zeros((n_tarefas, n_maquinas), dtype=int)

    for j in range(n_tarefas):
        for k in range(n_maquinas):
            ruido = rng.normal(1.0, 0.15)

            valor = (
                dificuldade_tarefa[j]
                * eficiencia_maquina[k]
                * ruido
            )

            tempos[j, k] = max(1, int(round(valor)))

    # --------------------------------------------------
    # Pesos das tarefas
    # --------------------------------------------------
    pesos = rng.integers(1, 11, size=n_tarefas)

    # --------------------------------------------------
    # Due date
    # --------------------------------------------------
    carga_media = np.sum(np.mean(tempos, axis=1)) / n_maquinas

    due_date = max(
        1,
        int(round(alpha_due_date * carga_media))
    )

    # --------------------------------------------------
    # Monta dataframe
    # --------------------------------------------------
    colunas = (
        ["Tarefa"]
        + [f"M{k+1}" for k in range(n_maquinas)]
        + ["Peso"]
    )

    dados = []

    for j in range(n_tarefas):
        linha = [j + 1]
        linha.extend(list(tempos[j, :]))
        linha.append(int(pesos[j]))
        dados.append(linha)

    dados.append(
        ["DueDate"]
        + [due_date]
        + [None] * (n_maquinas - 1)
        + [None]
    )

    df = pd.DataFrame(dados, columns=colunas)

    nome_arquivo = f"i{n_maquinas}x{n_tarefas}.xlsx"
    caminho = os.path.join(DATA_DIR, nome_arquivo)

    df.to_excel(caminho, index=False)

    print("=" * 60)
    print(f"Instância gerada: {nome_arquivo}")
    print(f"Tarefas: {n_tarefas}")
    print(f"Máquinas: {n_maquinas}")
    print(f"Carga média: {carga_media:.2f}")
    print(f"Due date: {due_date}")
    print(f"Razão d/carga = {due_date / carga_media:.2f}")
    print(f"Arquivo salvo em: {caminho}")
    print("=" * 60)
    print()


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    gerar_instancia(
        n_tarefas=50,
        n_maquinas=5,
        seed=50,
        alpha_due_date=0.20,
    )

    gerar_instancia(
        n_tarefas=100,
        n_maquinas=5,
        seed=100,
        alpha_due_date=0.20,
    )


if __name__ == "__main__":
    main()