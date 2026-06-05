import os
import matplotlib.pyplot as plt
from evaluation import evaluate_makespan, evaluate_weighted_tardiness


def ensure_img_dir():
    os.makedirs("img", exist_ok=True)


def plot_convergence(results_summary, title, filename="convergencia.png"):
    ensure_img_dir()
    plt.figure(figsize=(8, 5))

    for item in results_summary["all_results"]:
        plt.plot(item["history"], label=f"Execução {item['run']}")

    plt.xlabel("Iterações")
    plt.ylabel("Valor da função objetivo")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join("img", filename), dpi=200)
    plt.close()


def plot_best_schedule(solution, title, pt, we, due_date, filename="schedule.png"):
    ensure_img_dir()
    makespan = evaluate_makespan(solution, we, pt)
    weighted_tardiness = evaluate_weighted_tardiness(solution, we, pt, due_date)

    plt.figure(figsize=(10, 5))

    for k, machine_tasks in enumerate(solution):
        current_time = 0.0
        for task in machine_tasks:
            duration = pt[task, k]
            plt.barh(
                y=k + 1,
                width=duration,
                left=current_time,
                edgecolor="black",
            )
            plt.text(
                current_time + duration / 2,
                k + 1,
                str(task + 1),
                ha="center",
                va="center",
                fontsize=8,
            )
            current_time += duration

    plt.axvline(due_date, linestyle="--", label=f"Due date = {due_date}")
    plt.xlabel("Tempo")
    plt.ylabel("Máquina")
    plt.title(title)
    plt.yticks(range(1, len(solution) + 1), [f"M{k}" for k in range(1, len(solution) + 1)])
    plt.legend()
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join("img", filename), dpi=200)
    plt.close()

    print("Resumo da melhor solução:")
    print(f"Makespan = {makespan:.2f}")
    print(f"Atraso ponderado = {weighted_tardiness:.2f}")
    print(solution_to_string(solution))


def print_summary_table(summary, objective_name):
    print(f"\n===== Resultados para {objective_name} =====")

    for item in summary["all_results"]:
        print(f"Execução {item['run']}: valor final = {item['value']:.4f}")

    print(f"\nmin = {summary['min']:.4f}")
    print(f"std = {summary['std']:.4f}")
    print(f"max = {summary['max']:.4f}")


def solution_to_string(solution):
    lines = []

    for k, tasks in enumerate(solution):
        human_tasks = [task + 1 for task in tasks]
        lines.append(f"Máquina {k + 1}: {human_tasks}")

    return "\n".join(lines)


def plot_pareto_frontier(points, title="Fronteira de Pareto", filename="pareto.png"):
    ensure_img_dir()

    if not points:
        print("Nenhum ponto para plotar.")
        return

    f1_values = [point["f1"] if isinstance(point, dict) else point[0] for point in points]
    f2_values = [point["f2"] if isinstance(point, dict) else point[1] for point in points]

    ordered = sorted(zip(f1_values, f2_values), key=lambda x: (x[0], x[1]))
    f1_values = [p[0] for p in ordered]
    f2_values = [p[1] for p in ordered]

    plt.figure(figsize=(8, 5))
    plt.plot(f1_values, f2_values, "o-", label="Soluções não-dominadas")
    plt.xlabel("f1 (Makespan)")
    plt.ylabel("f2 (Atraso Ponderado)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join("img", filename), dpi=200)
    plt.close()


def plot_two_frontiers(weighted_pareto, epsilon_pareto, filename="pareto_comparativo.png"):
    ensure_img_dir()
    plt.figure(figsize=(8, 5))

    if weighted_pareto:
        weighted = sorted(weighted_pareto, key=lambda p: (p["f1"], p["f2"]))
        plt.plot(
            [p["f1"] for p in weighted],
            [p["f2"] for p in weighted],
            "o-",
            label="Soma Ponderada",
        )

    if epsilon_pareto:
        epsilon = sorted(epsilon_pareto, key=lambda p: (p["f1"], p["f2"]))
        plt.plot(
            [p["f1"] for p in epsilon],
            [p["f2"] for p in epsilon],
            "s-",
            label="Epsilon-restrito",
        )

    plt.xlabel("f1 (Makespan)")
    plt.ylabel("f2 (Atraso Ponderado)")
    plt.title("Comparação das fronteiras estimadas")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join("img", filename), dpi=200)
    plt.close()


def print_pareto_points(points, title):
    print(f"\n===== {title} =====")

    if not points:
        print("Nenhuma solução não-dominada encontrada.")
        return

    for i, point in enumerate(points, start=1):
        print(
            f"{i:02d} | f1={point['f1']:.2f} | f2={point['f2']:.2f} | "
            f"método={point['method']} | parâmetro={point['parameter']}"
        )


def _frontiers_items(frontiers_by_run):
    """Aceita fronteiras em dict {run: pontos} ou lista [pontos_run1, ...]."""
    if hasattr(frontiers_by_run, "items"):
        return sorted(frontiers_by_run.items())
    return [(idx + 1, points) for idx, points in enumerate(frontiers_by_run)]


def plot_frontiers_by_run(frontiers_by_run, title, filename):
    """Plota, na mesma figura, uma fronteira não-dominada para cada execução."""
    ensure_img_dir()

    plt.figure(figsize=(8, 5))
    plotted = False

    for run, points in _frontiers_items(frontiers_by_run):
        if not points:
            continue

        ordered = sorted(points, key=lambda p: (p["f1"], p["f2"]))
        plt.plot(
            [p["f1"] for p in ordered],
            [p["f2"] for p in ordered],
            "o-",
            label=f"Execução {run}",
        )
        plotted = True

    plt.xlabel("f1 (Makespan)")
    plt.ylabel("f2 (Atraso Ponderado)")
    plt.title(title)
    if plotted:
        plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join("img", filename), dpi=200)
    plt.close()


def print_frontiers_by_run(frontiers_by_run, title):
    print(f"\n===== {title} =====")

    for run, points in _frontiers_items(frontiers_by_run):
        print(f"Execução {run}: {len(points)} solução(ões) não-dominada(s)")
        for point in points:
            print(f"  f1={point['f1']:.2f} | f2={point['f2']:.2f} | parâmetro={point['parameter']}")
