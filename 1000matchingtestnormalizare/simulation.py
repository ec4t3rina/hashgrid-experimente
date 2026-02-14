import asyncio
import numpy as np
import json
import time
import wandb
import random
from hashgrid import Hashgrid, Message

# CONFIGURARE EXPERIMENT MASIV
API_KEY = "hg_3e40b9898ca54f808a40c25d85157b6cd3190e14ab90c5e0"
NUM_NODES = 1000 
VECTOR_SIZE = 10
MAX_CONCURRENT_REQUESTS = 50 

# INITIALIZARE W&B
wandb.init(
    project="hashgrid-normalized-test",
    entity="hashgrid-experimente",
    name=f"1k-nodes-norm-agg100-{int(time.time())}",
    config={
        "nodes": NUM_NODES,
        "vector_size": VECTOR_SIZE,
        "agg_window": 100,
        "normalization": "theoretical_max"
    }
)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class HashgridLab:
    def __init__(self):
        self.nodes_registry = {} 
        self.best_possible_scores = {} 
        self.semaphore = None
        self.tick_buffer = []

    def precompute_best_matches(self):
        """Calculează plafonul teoretic pentru 1000 de noduri (O(n^2))."""
        print(f"🔍 Pas 2: Calculăm potențialul maxim pentru {NUM_NODES} noduri...")
        start_time = time.time()
        node_ids = list(self.nodes_registry.keys())
        
        # Extragem vectorii pentru calcul vectorial mai rapid
        all_is = np.array([self.nodes_registry[nid]['is'] for nid in node_ids])
        
        for i, nid_a in enumerate(node_ids):
            vec_a_seek = self.nodes_registry[nid_a]['seek']
            
            # Calculăm dot product cu toți ceilalți folosind numpy (mult mai rapid)
            dots = np.dot(all_is, vec_a_seek)
            # Excludem auto-potrivirea (setăm indexul i la -inf)
            dots[i] = -np.inf
            
            max_dot = np.max(dots)
            self.best_possible_scores[nid_a] = sigmoid(max_dot)
            
            if i % 100 == 0:
                print(f"   Progres calcul: {i}/{NUM_NODES} noduri...")

        print(f"✅ Plafon stabilit în {time.time() - start_time:.2f}s.")

    async def create_node_safe(self, grid, name, v_is, v_seek):
        async with self.semaphore: 
            try:
                node = await asyncio.wait_for(grid.create_node(name=name, message=json.dumps(v_is), capacity=1), timeout=15.0)
                return node, v_is, v_seek
            except Exception:
                return None

    async def process_single_node(self, node_id, data):
        async with self.semaphore:
            try:
                node = data['obj']
                messages = await asyncio.wait_for(node.recv(), timeout=10.0)
                if not messages: return []
                
                replies = []
                normalized_scores = []
                for msg in messages:
                    peer_is = np.array(json.loads(msg.message))
                    raw_score = sigmoid(np.dot(peer_is, data['seek']))
                    
                    max_p = self.best_possible_scores[node_id]
                    # Normalizarea față de plafonul teoretic
                    norm_score = raw_score / max_p if max_p > 0 else 0
                    normalized_scores.append(norm_score)
                    
                    replies.append(Message(peer_id=msg.peer_id, round=msg.round, message=json.dumps(data['is'].tolist()), score=raw_score))
                
                if replies: await asyncio.wait_for(node.send(replies), timeout=10.0)
                return normalized_scores
            except: return []

    async def run(self):
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        grid = await Hashgrid.connect(api_key=API_KEY)
        
        print(f"--- Pas 1: Creare {NUM_NODES} noduri ---")
        tasks = [self.create_node_safe(grid, f"n1k_{i}", np.random.uniform(-1,1,VECTOR_SIZE).tolist(), np.random.uniform(-1,1,VECTOR_SIZE).tolist()) for i in range(NUM_NODES)]
        results = await asyncio.gather(*tasks)
        
        for r in results:
            if r:
                node, v_is, v_seek = r
                self.nodes_registry[node.node_id] = {'is': np.array(v_is), 'seek': np.array(v_seek), 'obj': node}
        
        self.precompute_best_matches()
        print(f"🚀 Pas 3: Pornim simularea. Plotare agregată la 100 de tick-uri.")

        async for tick in grid.listen(poll_interval=1.0):
            p_tasks = [self.process_single_node(nid, d) for nid, d in self.nodes_registry.items()]
            all_res = await asyncio.gather(*p_tasks)
            scores = [s for sub in all_res for s in sub]
            
            if scores:
                avg = np.mean(scores)
                self.tick_buffer.append(avg)
                
                # Logăm scorul brut la fiecare tick în wandb
                wandb.log({"norm_score_tick": avg, "tick": tick, "active_pairs": len(scores)})
                
                if len(self.tick_buffer) >= 100:
                    agg = np.mean(self.tick_buffer)
                    wandb.log({"agg_100_score": agg})
                    print(f"📊 [PLOT 100] Eficiență medie (ultimele 100 ticks): {agg*100:.2f}%")
                    self.tick_buffer = []
                
                if tick % 10 == 0:
                    print(f"📈 [TICK {tick}] Eficiență instantanee: {avg*100:.2f}%")
            else:
                print(f"⏳ Tick {tick}: Se caută conexiuni...")
                wandb.log({"tick": tick, "active_pairs": 0})

if __name__ == "__main__":
    try:
        asyncio.run(HashgridLab().run())
    except KeyboardInterrupt:
        print("\nOprit manual.")
        wandb.finish()