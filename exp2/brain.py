import json, os, networkx as nx, numpy as np, pandas as pd, random
from sklearn.ensemble import RandomForestRegressor

class HashgridBrain:
    def __init__(self, json_file="dataset_25.json"):
        self.json_file = json_file
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.node_embeddings = {}
        self.is_trained = False

    def _get_organic_score(self, src_id, tgt_id):
        # Scor de bază random dar stabil pe perechi
        random.seed(sum(ord(c) for c in (src_id[:3] + tgt_id[:3])))
        return round(random.uniform(0.4, 0.7), 2)

    def train(self):
        if not os.path.exists(self.json_file) or os.stat(self.json_file).st_size == 0:
            return False
        try:
            with open(self.json_file, "r") as f:
                data = json.load(f)
            if len(data) < 10: return False # Prag mic pentru antrenare rapidă
            
            df = pd.DataFrame(data)
            G = nx.Graph()
            for _, row in df.iterrows():
                G.add_edge(row['source_id'], row['target_id'], weight=row['score'])

            pr = nx.pagerank(G, weight='weight')
            dc = nx.degree_centrality(G)

            for node in G.nodes():
                # Embedding: [PageRank, Centralitate, Media scorurilor primite]
                avg_s = df[df['target_id'] == node]['score'].mean() or 0.5
                self.node_embeddings[node] = np.array([pr.get(node, 0), dc.get(node, 0), avg_s])

            X, y = [], []
            for _, row in df.iterrows():
                if row['source_id'] in self.node_embeddings and row['target_id'] in self.node_embeddings:
                    feat = np.hstack([self.node_embeddings[row['source_id']], self.node_embeddings[row['target_id']]])
                    X.append(feat)
                    y.append(row['score'])

            if len(X) > 5:
                self.model.fit(np.array(X), np.array(y))
                self.is_trained = True
                return True
        except: return False

    def predict(self, src_id, tgt_id):
        if self.is_trained and src_id in self.node_embeddings and tgt_id in self.node_embeddings:
            feat = np.hstack([self.node_embeddings[src_id], self.node_embeddings[tgt_id]]).reshape(1, -1)
            base_pred = self.model.predict(feat)[0]
            
            # FACTOR MIC (1.5) = CREȘTERE LENTĂ/GRADUALĂ
            pushed = 0.5 + (base_pred - 0.5) * 1.5
            noise = random.uniform(-0.05, 0.05)
            return round(max(0.05, min(0.95, pushed + noise)), 2)
        
        return self._get_organic_score(src_id, tgt_id)