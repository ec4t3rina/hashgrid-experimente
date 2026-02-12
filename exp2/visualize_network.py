import json, networkx as nx, matplotlib.pyplot as plt, pandas as pd

def draw_trust():
    with open("dataset_25.json", "r") as f:
        df = pd.DataFrame(json.load(f))
    
    # Luăm doar ultimele interacțiuni (cele mai mature)
    recent = df.tail(100)
    G = nx.Graph()
    
    for _, row in recent.iterrows():
        G.add_edge(row['source_id'][:4], row['target_id'][:4], weight=row['score'])

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)
    
    # Desenăm muchiile: grosimea depinde de scor
    weights = [G[u][v]['weight'] * 3 for u, v in G.edges()]
    
    nx.draw(G, pos, with_labels=True, node_size=800, node_color="skyblue", 
            width=weights, edge_color="gray", alpha=0.7)
    
    plt.title("Harta Interacțiunilor de Top (Grosime = Încredere)")
    plt.savefig("network_trust.png")
    print("🕸️ Harta rețelei salvată: network_trust.png")
    plt.show()

if __name__ == "__main__":
    draw_trust()