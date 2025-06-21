
# ------------- Guardar Resultados ----------------

# ----- Importar datos funciones de RL
from hola2 import run_experiment
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
# ---------------- Ejecutar múltiples configuraciones ----------------

experiments = [
    {"alpha": 0.3, "gamma": 0.9, "decay": 0.995, "label": "base"},
    {"alpha": 0.1, "gamma": 0.9, "decay": 0.995, "label": "A"},
    {"alpha": 0.5, "gamma": 0.95, "decay": 0.99,  "label": "B"},
    {"alpha": 0.3, "gamma": 0.8, "decay": 1.0,    "label": "C"},  # epsilon constante
    {"alpha": 0.3, "gamma": 0.9, "decay": 1.0,    "label": "D", "eps_start": 1.0},  # solo exploración
]


for exp in experiments:
    print(f"--- Running experiment {exp['label']} ---")
    run_experiment(
        alpha_val=exp["alpha"],
        gamma_val=exp["gamma"],
        epsilon_decay_val=exp["decay"],
        eps_start=exp.get("eps_start", 0.5),
        label=exp["label"]
    )
labels = [exp["label"] for exp in experiments]

# ---------------- Comparar resultados de todos los experimentos ----------------

##--- TABLA RESUMEN DE EXPERIMENTOS

tabla = []

for label in labels:
    df = pd.read_csv(f"results_{label}.csv")
    promedio = df["Reward"].mean()
    max_r = df["Reward"].max()
    min_r = df["Reward"].min()
    std_r = df["Reward"].std()
    tabla.append([label, round(promedio, 2), max_r, min_r, round(std_r, 2)])

print(tabulate(tabla, headers=["Experimento", "Reward Promedio", "Máx", "Mín", "Std"]))



# -------- GRAFICO POR CADA EXPERIMENTO
# Lista de labels de tus experimentos
labels = ["base", "A", "B", "C", "D"]

for label in labels:
    df = pd.read_csv(f"results_{label}.csv")
    plt.figure(figsize=(8, 5))
    plt.plot(df["Episode"], df["Reward"], color='blue')
    plt.title(f"Recompensa por Episodio - Experimento {label}")
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa total")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"grafico_reward_exp_{label}.png")
    plt.show()

# --- GRAFICO PARA TODOS LOS EXPERIMENTOS
plt.figure(figsize=(10, 6))

for label in labels:
    df = pd.read_csv(f"results_{label}.csv")
    plt.plot(df["Episode"], df["Reward"], label=f"Exp {label}")

plt.title("Comparación de Recompensa por Episodio")
plt.xlabel("Episodio")
plt.ylabel("Recompensa total")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("grafico_comparativo_experimentos.png")
plt.show()

#--- GRAFICO DEL MEJOR EXPERIMENTO
mejor_label = None
mejor_promedio = float('-inf')

for label in labels:
    df = pd.read_csv(f"results_{label}.csv")
    promedio = df["Reward"].mean()
    if promedio > mejor_promedio:
        mejor_promedio = promedio
        mejor_label = label

# Mostrar gráfico del mejor
df_best = pd.read_csv(f"results_{mejor_label}.csv")
plt.figure(figsize=(8, 5))
plt.plot(df_best["Episode"], df_best["Reward"], color='green')
plt.title(f"Mejor Experimento: {mejor_label} (Recompensa promedio = {mejor_promedio:.2f})")
plt.xlabel("Episodio")
plt.ylabel("Recompensa total")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"grafico_mejor_experimento_{mejor_label}.png")
plt.show()
