import pandas as pd
import numpy as np
import time
import copy
import gurobipy as gp

# 2. Configurar as credenciais da licença WLS nas variáveis de ambiente do sistema
os.environ['GRB_WLSACCESSID'] = 
os.environ['GRB_WLSSECRET']   = 
os.environ['GRB_LICENSEID']   = 

# ============================================================
# 1. CARREGAMENTO DOS DADOS — INSTÂNCIA FLEXÍVEL
# ============================================================
# Troque o arquivo para avaliar diferentes cenários.
# O código detecta automaticamente o número de máquinas (colunas M1, M2, ...)
# e o número de tarefas (linhas na aba 'Dados').
caminho_arquivo = 'i5x1000.xlsx'

df       = pd.read_excel(caminho_arquivo, sheet_name='Dados')
df_param = pd.read_excel(caminho_arquivo, sheet_name='Parametros')
DD       = float(df_param.loc[df_param['Parametro'] == 'DD', 'Valor'].values[0])

# Detecta máquinas dinamicamente (colunas chamadas M1, M2, M3, ...)
maquinas    = sorted([int(c[1:]) for c in df.columns if c.startswith('M') and c[1:].isdigit()])
num_tarefas = len(df)
num_maquinas = len(maquinas)

p, w = {}, {}
for _, row in df.iterrows():
    j    = int(row['Tarefa'])
    w[j] = float(row['Peso'])
    for i in maquinas:
        p[i, j] = float(row[f'M{i}'])

print(f"Instância: {num_tarefas} tarefas | {num_maquinas} máquinas | DD = {DD}")


# ============================================================
# 2. FUNÇÃO DE AVALIAÇÃO (FO = Cmax + Σ wj·Tj)
# ============================================================
def avaliar_solucao(solucao):
    C_j                   = {}
    total_atraso_ponderado = 0.0
    Cmax                  = 0.0

    for i in maquinas:
        tempo = 0.0
        for j in solucao[i]:
            tempo  += p[i, j]
            C_j[j]  = tempo
            total_atraso_ponderado += w[j] * max(0.0, tempo - DD)
        Cmax = max(Cmax, tempo)

    return Cmax + total_atraso_ponderado, Cmax


# ============================================================
# 3. HEURÍSTICA CONSTRUTIVA (balanceamento de carga + prioridade por peso)
# ============================================================
def gerar_solucao_inicial():
    """
    Ordena as tarefas por peso decrescente (processa as mais penalizadas primeiro)
    e atribui cada uma à máquina com menor carga acumulada naquele momento.
    Isso produz um ponto de partida melhor que o simples mínimo de p[i,j].
    """
    sol   = {i: [] for i in maquinas}
    carga = {i: 0.0 for i in maquinas}

    for j in sorted(range(1, num_tarefas + 1), key=lambda j: w[j], reverse=True):
        m_min = min(maquinas, key=lambda m: carga[m] + p[m, j])
        sol[m_min].append(j)
        carga[m_min] += p[m_min, j]

    return sol


# ============================================================
# 4. OPERADORES DE VIZINHANÇA
# ============================================================
def vizinhanca_shift(solucao):
    """Move uma tarefa aleatória de uma máquina para outra posição (mesma ou diferente)."""
    nova = copy.deepcopy(solucao)
    origem = [m for m in maquinas if nova[m]]
    if not origem:
        return nova

    m_o   = np.random.choice(origem)
    idx_j = np.random.randint(len(nova[m_o]))
    j     = nova[m_o].pop(idx_j)

    m_d   = np.random.choice(maquinas)
    idx_d = np.random.randint(len(nova[m_d]) + 1)
    nova[m_d].insert(idx_d, j)
    return nova


def vizinhanca_swap_inter(solucao):
    """Troca duas tarefas entre máquinas diferentes."""
    nova = copy.deepcopy(solucao)
    com_tarefa = [m for m in maquinas if nova[m]]
    if len(com_tarefa) < 2:
        return nova

    m1, m2 = np.random.choice(com_tarefa, size=2, replace=False)
    idx1   = np.random.randint(len(nova[m1]))
    idx2   = np.random.randint(len(nova[m2]))
    nova[m1][idx1], nova[m2][idx2] = nova[m2][idx2], nova[m1][idx1]
    return nova


def vizinhanca_swap_intra(solucao):
    """Troca a posição de duas tarefas dentro da mesma máquina."""
    nova   = copy.deepcopy(solucao)
    validas = [m for m in maquinas if len(nova[m]) >= 2]
    if not validas:
        return nova

    m      = np.random.choice(validas)
    i1, i2 = np.random.choice(len(nova[m]), size=2, replace=False)
    nova[m][i1], nova[m][i2] = nova[m][i2], nova[m][i1]
    return nova


VIZINHANCAS = [vizinhanca_shift, vizinhanca_swap_inter, vizinhanca_swap_intra]


# ============================================================
# 5. RVND CORRETO — esgota cada vizinhança antes de removê-la
# ============================================================
def RVND(solucao, fo, max_sem_melhora=300):
    """
    Randomized Variable Neighborhood Descent (clássico).

    Lógica por rodada:
      - Sorteia aleatoriamente uma vizinhança da lista ativa.
      - Explora via amostragem aleatória até passar 'max_sem_melhora'
        iterações CONSECUTIVAS sem melhora (critério de esgotamento).
      - Se pelo menos UMA melhora foi encontrada nessa rodada:
            reinicia a lista completa de vizinhanças.
      - Caso contrário:
            remove a vizinhança (ela está esgotada localmente).
      - Para quando a lista fica vazia.
    """
    ativas = list(VIZINHANCAS)

    while ativas:
        vf        = np.random.choice(ativas)
        sem_melh  = 0
        melhorou  = False

        while sem_melh < max_sem_melhora:
            cand_sol   = vf(solucao)
            fo_cand, _ = avaliar_solucao(cand_sol)

            if fo_cand < fo:
                solucao  = cand_sol
                fo       = fo_cand
                melhorou = True
                sem_melh = 0          # continua explorando essa vizinhança
            else:
                sem_melh += 1

        if melhorou:
            ativas = list(VIZINHANCAS)   # melhora encontrada → reseta lista
        else:
            ativas.remove(vf)            # vizinhança esgotada → remove

    return solucao, fo


# ============================================================
# 6. IDENTIFICAÇÃO DA SUB-REGIÃO CRÍTICA (gargalo real)
# ============================================================
def identificar_maquina_critica(solucao):
    """
    Retorna a máquina com maior contribuição combinada para a FO:
        contrib(m) = tempo_total(m) + Σ wj·max(0, Cj − DD)
    Identifica o verdadeiro gargalo — tanto em Cmax quanto em atraso ponderado.
    """
    contrib = {}
    for m in maquinas:
        tempo = 0.0
        soma  = 0.0
        for j in solucao[m]:
            tempo += p[m, j]
            soma  += w[j] * max(0.0, tempo - DD)
        contrib[m] = tempo + soma
    return max(contrib, key=contrib.get)


# ============================================================
# 7. FIX-AND-OPTIMIZE COM GUROBI (janela deslizante na sub-região crítica)
# ============================================================
import gurobipy as gp

def intensificar_fix_and_optimize(solucao, fo, time_limit=5, max_janela=12):
    m_alvo = identificar_maquina_critica(solucao)
    seq_m = list(solucao[m_alvo])
    n_seq = len(seq_m)

    if n_seq < 3:
        return solucao, fo

    # --- Seleção da janela (idêntica à original) ---
    if n_seq > max_janela:
        inicio = np.random.randint(0, n_seq - max_janela + 1)
        antes  = seq_m[:inicio]
        janela = seq_m[inicio: inicio + max_janela]
        depois = seq_m[inicio + max_janela:]
    else:
        antes  = []
        janela = seq_m
        depois = []

    t_inicio = sum(p[m_alvo, j] for j in antes)
    M_big = t_inicio + sum(p[m_alvo, j] for j in janela) + 1

    # --- Modelo Gurobi (puro) ---
    model = gp.Model("FixOpt")
    model.setParam('OutputFlag', 0)          # msg=False
    model.setParam('TimeLimit', time_limit)

    # Variáveis
    y = {}
    for j in janela:
        for k in janela:
            if j != k:
                y[j, k] = model.addVar(vtype=gp.GRB.BINARY, name=f"y_{j}_{k}")

    C = model.addVars(janela, lb=0, name="C")
    T = model.addVars(janela, lb=0, name="T")
    Cmax_loc = model.addVar(lb=0, name="Cmax")

    # Objetivo
    model.setObjective(Cmax_loc + gp.quicksum(w[j] * T[j] for j in janela), gp.GRB.MINIMIZE)

    # Restrições
    for j in janela:
        model.addConstr(Cmax_loc >= C[j])
        model.addConstr(T[j] >= C[j] - DD)
        model.addConstr(C[j] >= t_inicio + p[m_alvo, j])

    for j in janela:
        for k in janela:
            if j != k:
                model.addConstr(C[k] >= C[j] + p[m_alvo, k] - M_big * (1 - y[j, k]))

    for j in janela:
        for k in janela:
            if j < k:
                model.addConstr(y[j, k] + y[k, j] == 1)

    model.optimize()

    if model.Status in (gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT, gp.GRB.SUBOPTIMAL):
        bloco_otimizado = sorted(janela, key=lambda j: C[j].X)
        nova_sol = copy.deepcopy(solucao)
        nova_sol[m_alvo] = antes + bloco_otimizado + depois

        fo_nova, _ = avaliar_solucao(nova_sol)
        if fo_nova < fo:
            return nova_sol, fo_nova

    return solucao, fo

# ============================================================
# 8. FRAMEWORK PRINCIPAL: GVNS + RVND + Fix-and-Optimize
# ============================================================
def executar_matheuristica(max_iter=20, k_max=3, rvnd_sem_melhora=300, mip_time=5):
    """
    Parâmetros recomendados por tamanho de instância:
      - Pequena (≤30 tarefas):  max_iter=25, rvnd_sem_melhora=400
      - Média  (≤80 tarefas):  max_iter=20, rvnd_sem_melhora=300
      - Grande (>80 tarefas):  max_iter=15, rvnd_sem_melhora=200, mip_time=8
    """
    sep = "=" * 52
    print(sep)
    print(f"  GVNS + RVND + Fix-and-Optimize  |  Gurobi")
    print(f"  {num_tarefas} tarefas | {num_maquinas} máquinas | DD = {DD}")
    print(sep)
    t0 = time.time()

    # Solução inicial via heurística construtiva
    sol_melhor        = gerar_solucao_inicial()
    fo_melhor, cm_ini = avaliar_solucao(sol_melhor)
    print(f"FO inicial (construtiva): {fo_melhor:.2f}  |  Cmax: {cm_ini:.2f}\n")

    for it in range(1, max_iter + 1):
        k = 1
        while k <= k_max:

            # ── Perturbação (k movimentos compostos) ──
            sol_p = copy.deepcopy(sol_melhor)
            for _ in range(k):
                sol_p = vizinhanca_shift(sol_p)
                sol_p = vizinhanca_swap_inter(sol_p)
            fo_p, _ = avaliar_solucao(sol_p)

            # ── Busca local: RVND correto ──
            sol_r, fo_r = RVND(sol_p, fo_p, max_sem_melhora=rvnd_sem_melhora)

            # ── Intensificação exata: Fix-and-Optimize com Gurobi ──
            sol_f, fo_f = intensificar_fix_and_optimize(sol_r, fo_r, time_limit=mip_time)

            if fo_f < fo_melhor:
                sol_melhor = sol_f
                fo_melhor  = fo_f
                print(f"  [it {it:02d} | k={k}] ✓ FO = {fo_melhor:.2f}")
                k = 1          # volta ao início do GVNS (melhora global)
            else:
                k += 1         # aumenta intensidade da perturbação

    tempo = time.time() - t0
    fo_final, cmax_final = avaliar_solucao(sol_melhor)

    print(f"\n{sep}")
    print(f"  RESULTADO FINAL")
    print(sep)
    print(f"  Tempo total    : {tempo:.2f} s")
    print(f"  Makespan (Cmax): {cmax_final:.2f}")
    print(f"  FO final (Z)   : {fo_final:.2f}")
    print(f"\n  Escalonamento por máquina:")
    for m in maquinas:
        print(f"    M{m}: {sol_melhor[m]}")
    print(sep)

    return sol_melhor, fo_final


# ============================================================
# 9. EXECUÇÃO
# ============================================================
if __name__ == "__main__":
    melhor_escala, fo = executar_matheuristica(
        max_iter         = 20,
        k_max            = 3,
        rvnd_sem_melhora = 300,
        mip_time         = 5,
    )