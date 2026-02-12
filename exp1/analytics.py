import json
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import os

def run_analytics(json_path="dataset_antrenare.json"):
    # 1. Încărcăm datele
    if not os.path.exists(json_path):
        print(f"❌ Nu am găsit {json_path}. Rulează agentul pentru a colecta date!")
        return

    with open(json_path, "r") as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    # Redenumim coloanele dacă JSON-ul tău are formatul nou (source_id, target_id)
    if 'source_id' in df.columns:
        df = df.rename(columns={'source_id': 'source_node', 'target_id': 'target_node', 'score': 'rating_received'})

    # --- VIZUALIZARE 1: TOPOLOGIA GRAFULUI ---
    plt.figure(figsize=(12, 8))
    G = nx.Graph()
    
    # Grupăm interacțiunile pentru a face media scorurilor între aceleași două noduri
    edges = df.groupby(['source_node', 'target_node'])['rating_received'].mean().reset_index()

    for _, row in edges.iterrows():
        src, dst = row['source_node'][:8], row['target_node'][:8]
        G.add_edge(src, dst, weight=row['rating_received'])

    pos = nx.spring_layout(G, k=0.5)
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color='skyblue', edgecolors='black')
    
    # Liniile: cu cât ratingul e mai mare, cu atât sunt mai groase
    weights = [G[u][v]['weight'] * 6 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=weights, edge_color='gray', alpha=0.6)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')

    plt.title("Hashgrid Topology (Grosime linie = Rating Mediu)")
    plt.savefig("hashgrid_topology.png")
    print("✅ Imagine salvată: hashgrid_topology.png")

    # --- VIZUALIZARE 2: DISTRIBUȚIA SCORURILOR (Histograma) ---
    plt.figure(figsize=(10, 6))
    plt.hist(df['rating_received'], bins=20, color='salmon', edgecolor='black', alpha=0.7)
    
    plt.axvline(df['rating_received'].mean(), color='blue', linestyle='dashed', linewidth=2, label=f"Media: {df['rating_received'].mean():.2f}")
    
    plt.title("Distribuția Scorurilor în Rețea")
    plt.xlabel("Rating (0.0 - 1.0)")
    plt.ylabel("Frecvență (Număr Mesaje)")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.savefig("score_distribution.png")
    print("✅ Imagine salvată: score_distribution.png")
    
    # Afișăm ambele ferestre
    plt.show()

if __name__ == "__main__":
    run_analytics()