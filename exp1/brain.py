import json
import os
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import random

class HashgridBrain:
    def __init__(self, json_file="dataset_antrenare.json"):
        self.json_file = json_file
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.node_embeddings = {}
        self.is_trained = False

    def _get_organic_score(self, src_id, tgt_id):
        """Generează un scor unic bazat pe 'afinitatea' ID-urilor + zgomot."""
        # Creăm o valoare deterministă bazată pe ambele ID-uri
        combined_seed = sum(ord(c) for c in (src_id[:4] + tgt_id[:4]))
        random.seed(combined_seed)
        
        # Scorul de bază (0.3 - 0.7)
        base_affinity = random.uniform(0.3, 0.7)
        
        # Adăugăm un pic de haos (zgomot)
        noise = random.uniform(-0.2, 0.2)
        
        return round(max(0.1, min(0.95, base_affinity + noise)), 2)

    def train(self):
        if not os.path.exists(self.json_file) or os.stat(self.json_file).st_size == 0:
            return False
        try:
            with open(self.json_file, "r") as f:
                data = json.load(f)
            if len(data) < 20: return False # Minim 20 interacțiuni pentru AI
            
            df = pd.DataFrame(data)
            G = nx.Graph()
            for _, row in df.iterrows():
                G.add_edge(row['source_id'], row['target_id'], weight=row['score'])

            # Calculăm metrici de graf (Embeddings)
            pagerank = nx.pagerank(G, weight='weight')
            betweenness = nx.betweenness_centrality(G)

            for node in G.nodes():
                self.node_embeddings[node] = np.array([
                    pagerank.get(node, 0),
                    betweenness.get(node, 0),
                    df[df['target_id'] == node]['score'].mean() or 0.5
                ])

            X, y = [], []
            for _, row in df.iterrows():
                if row['source_id'] in self.node_embeddings and row['target_id'] in self.node_embeddings:
                    feat = np.hstack([self.node_embeddings[row['source_id']], self.node_embeddings[row['target_id']]])
                    X.append(feat)
                    y.append(row['score'])

            self.model.fit(np.array(X), np.array(y))
            self.is_trained = True
            print(f"🧠 AI ACTIV: Antrenat pe {len(X)} puncte de date.")
            return True
        except Exception as e:
            print(f"⚠️ Eroare antrenare: {e}")
            return False

    def predict(self, source_id, target_id):
        # 1. Verificăm dacă avem AI-ul antrenat și cunoaștem nodurile
        if self.is_trained and source_id in self.node_embeddings and target_id in self.node_embeddings:
            try:
                # Obținem predicția de bază de la modelul Random Forest
                feat = np.hstack([self.node_embeddings[source_id], self.node_embeddings[target_id]]).reshape(1, -1)
                base_prediction = self.model.predict(feat)[0]
                
                # --- LOGICA DE POLARIZARE (TRUST VS NO-TRUST) ---
                # Calculăm distanța față de centru (0.5)
                diff = base_prediction - 0.5
                
                # Amplificăm diferența cu un factor de 2.5 pentru a împinge scorurile spre 0.1 sau 0.9
                amplification_factor = 2.5 
                pushed_prediction = 0.5 + (diff * amplification_factor)
                
                # Adăugăm un pic de zgomot (jitter) pentru a menține graficul variat
                final_score = pushed_prediction + random.uniform(-0.07, 0.07)
                
                # Menținem scorul în limitele acceptate de sistem
                return round(max(0.05, min(0.95, final_score)), 2)
                
            except Exception as e:
                # Dacă apare o eroare de calcul, folosim scorul organic determinist
                return self._get_organic_score(source_id, target_id)
        
        # 2. Dacă nodul este nou sau AI-ul nu e gata, folosim scorul organic
        return self._get_organic_score(source_id, target_id)