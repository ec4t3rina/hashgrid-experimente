import json, pandas as pd, matplotlib.pyplot as plt, os

def plot():
    if not os.path.exists("dataset_25.json"): return
    with open("dataset_25.json", "r") as f:
        df = pd.DataFrame(json.load(f))
    
    # Media scorurilor pe fiecare tick
    evo = df.groupby('tick')['score'].mean()

    plt.figure(figsize=(12, 6))
    plt.plot(evo.index, evo.values, color='#2ecc71', linewidth=2, marker='.', markersize=4)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
    plt.title("Evoluția Graduală a Încrederii în Rețeaua de 25 Noduri")
    plt.xlabel("Timp (Ticks)")
    plt.ylabel("Rating Mediu al Interacțiunilor")
    plt.grid(True, alpha=0.2)
    plt.savefig("evolutie_25_noduri.png")
    print("📈 Graficul a fost salvat: evolutie_25_noduri.png")
    plt.show()

if __name__ == "__main__":
    plot()