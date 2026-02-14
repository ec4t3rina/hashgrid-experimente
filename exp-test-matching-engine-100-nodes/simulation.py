import asyncio
import numpy as np
import json
import time
from hashgrid import Hashgrid, Message, HashgridAPIError

# CONFIG
API_KEY = "hg_3e40b9898ca54f808a40c25d85157b6cd3190e14ab90c5e0"
NUM_NODES = 100
VECTOR_SIZE = 10
DATA_FILE = "simulation_results.json"
MAX_CONCURRENT_REQUESTS = 100 

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class HashgridLab:
    def __init__(self):
        self.nodes_registry = {} 
        self.history = []
        self.semaphore = None # Îl lăsăm None aici

    async def create_node_safe(self, grid, name, v_is, v_seek):
        async with self.semaphore: # Acum va folosi semaforul creat în run()
            try:
                node = await grid.create_node(name=name, message=json.dumps(v_is), capacity=1)
                return node, v_is, v_seek
            except Exception as e:
                return None

    async def process_single_node(self, node_id, data):
        async with self.semaphore:
            try:
                node = data['obj']
                messages = await node.recv()
                
                if not messages:
                    return []

                replies = []
                scores = []
                for msg in messages:
                    peer_is = np.array(json.loads(msg.message))
                    score = float(sigmoid(np.dot(peer_is, data['seek'])))
                    scores.append(score)
                    
                    replies.append(Message(
                        peer_id=msg.peer_id,
                        round=msg.round,
                        message=json.dumps(data['is'].tolist()),
                        score=score
                    ))
                
                if replies:
                    await node.send(replies)
                return scores
            except Exception:
                return []

    async def run(self):
        # INITIALIZARE SEMAFOR AICI (în interiorul loop-ului activ)
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        
        grid = await Hashgrid.connect(api_key=API_KEY)
        print(f"--- Conectat la Grid: {grid.name} | Tick: {grid.tick} ---")

        # 1. Creare Noduri
        print(f"Creăm {NUM_NODES} noduri...")
        create_tasks = []
        for i in range(NUM_NODES):
            v_is = np.random.uniform(-1, 1, VECTOR_SIZE).tolist()
            v_seek = np.random.uniform(-1, 1, VECTOR_SIZE).tolist()
            create_tasks.append(self.create_node_safe(grid, f"lab_node_{i}", v_is, v_seek))
        
        created_results = await asyncio.gather(*create_tasks)
        for res in created_results:
            if res:
                node, v_is, v_seek = res
                self.nodes_registry[node.node_id] = {
                    'is': np.array(v_is),
                    'seek': np.array(v_seek),
                    'obj': node
                }
        
        print(f"--- Înregistrate {len(self.nodes_registry)} noduri active ---")

        # 2. Loop de Simulare
        async for tick in grid.listen(poll_interval=0.5):
            start_time = time.time()
            process_tasks = [
                self.process_single_node(node_id, data) 
                for node_id, data in self.nodes_registry.items()
            ]
            
            all_tick_results = await asyncio.gather(*process_tasks)
            tick_scores = [score for sublist in all_tick_results for score in sublist]

            if tick_scores:
                avg = np.mean(tick_scores)
                self.history.append({"tick": tick, "avg_score": avg, "count": len(tick_scores)})
                elapsed = time.time() - start_time
                print(f"[TICK {tick}] Scor Mediu: {avg:.4f} | Interacțiuni: {len(tick_scores)} | {elapsed:.2f}s")
                
                with open(DATA_FILE, "w") as f:
                    json.dump(self.history, f)

if __name__ == "__main__":
    lab = HashgridLab()
    asyncio.run(lab.run())