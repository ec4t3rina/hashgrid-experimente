import asyncio, json, os, random
from hashgrid import Hashgrid, Message
from brain import HashgridBrain

# CONFIGURARE
API_KEY = "hg_3e40b9898ca54f808a40c25d85157b6cd3190e14ab90c5e0"
JSON_FILE = "dataset_25.json"
PREFIX = "new_"

def save_log(entry):
    data = []
    if os.path.exists(JSON_FILE) and os.stat(JSON_FILE).st_size > 0:
        with open(JSON_FILE, "r") as f:
            try: 
                data = json.load(f)
            except: 
                data = []
    data.append(entry)
    with open(JSON_FILE, "w") as f:
        json.dump(data, f, indent=4)

async def main():
    brain = HashgridBrain(JSON_FILE)
    grid = await Hashgrid.connect(api_key=API_KEY)
    
    # Identificăm nodurile care ne aparțin
    all_nodes = [node async for node in grid.nodes()]
    my_nodes = [n for n in all_nodes if n.name.startswith(PREFIX)]
    
    print(f"🚀 Pornim experimentul cu {len(my_nodes)} noduri.")
    print(f"🔄 Antrenare setată la fiecare 3 pași reali.")

    steps = 0
    async for tick in grid.listen(poll_interval=1.0):
        steps += 1
        
        # 1. ANTRENAMENT BAZAT PE PAȘI (nu pe numărul tick-ului)
        if steps % 3 == 0:
            if brain.train(): 
                print(f"✅ [Step {steps}] Reantrenare reușită (Tick: {tick})")
            else:
                print(f"⏳ [Step {steps}] Colectăm date în continuare...")

        for node in my_nodes:
            try:
                # 2. INIȚIERE MESAJE (20% șansă per nod)
                # Trimitem mesaje către "Elite" (primele 5 noduri) pentru a clădi ierarhia
                if random.random() < 0.2:
                    target = random.choice(my_nodes[:5])
                    if target.node_id != node.node_id:
                        await node.send([Message(peer_id=target.node_id, message="ping", round=tick, score=0.5)])
                        await asyncio.sleep(0.02) # Mică pauză anti-spam

                # 3. PROCESARE MESAJE PRIMITE
                messages = await node.recv()
                if not messages: continue

                replies = []
                for msg in messages:
                    # AI-ul prezice ratingul
                    score = brain.predict(node.node_id, msg.peer_id)
                    
                    # LOGICA DE FILTRARE: Răspundem doar la conexiunile de calitate
                    # Păstrăm o mică șansă (10%) de explorare pentru nodurile noi
                    if score > 0.55 or random.random() < 0.1:
                        save_log({
                            "tick": tick, 
                            "step": steps,
                            "source_id": node.node_id, 
                            "target_id": msg.peer_id, 
                            "score": score
                        })
                        replies.append(Message(peer_id=msg.peer_id, message="ack", round=tick, score=score))

                if replies:
                    await node.send(replies)
                    print(f"💬 {node.name} a procesat {len(replies)} interacțiuni la pasul {steps}")
                    await asyncio.sleep(0.02) # Pauză între apelurile API

            except Exception as e:
                # Prindem erorile de JSON sau Timeout și continuăm
                print(f"⚠️ Eroare temporară la {node.name}: {e}")
                continue

if __name__ == "__main__":
    asyncio.run(main())