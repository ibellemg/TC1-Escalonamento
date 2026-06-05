import gurobipy as gp

print("Versão do Gurobi:", gp.gurobi.version())

env = gp.Env()
print("Licença OK")