import asyncio
import json
import os
import logging
from hashgrid import Hashgrid, Message
from brain import HashgridBrain

# CONFIGURARE
API_KEY = "hg_3e40b9898ca54f808a40c25d85157b6cd3190e14ab90c5e0"
JSON_FILE = "dataset_antrenare.json"

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

def save_to_json(entry):
    data = []
    if os.path.exists(JSON_FILE) and os.stat(JSON_FILE).st_size > 0:
        with open(JSON_FILE, "r") as f:
            try: data = json.load(f)
            except: data = []
    data.append(entry)
    with open(JSON_FILE, "w") as f:
        json.dump(data, f, indent=4)

async def main():
    brain = HashgridBrain(JSON_FILE)
    is_ai_ready = brain.train()
    
    print(f"🚀 Pornire Agent | Mod: {'INTELIGENT' if is_ai_ready else 'EXPLORARE (Scoruri Organice)'}")
    
    grid = await Hashgrid.connect(api_key=API_KEY)
    # Detectăm toate cele 12 noduri
    my_nodes = [n async for n in grid.nodes()]
    print(f"✅ Am detectat {len(my_nodes)} noduri pe grid.")

    async for tick in grid.listen(poll_interval=1.0):
        print(f"🔔 Tick {tick} | Verific activitate...")
        
        for node in my_nodes:
            messages = await node.recv()
            if not messages: continue

            replies = []
            for msg in messages:
                # Obținem scorul diversificat din brain
                score = brain.predict(node.node_id, msg.peer_id)
                
                print(f"📩 {node.name} -> {msg.peer_id[:8]} | Rating: {score}")

                save_to_json({
                    "tick": tick, 
                    "source_id": node.node_id, 
                    "target_id": msg.peer_id, 
                    "score": score
                })
                
                replies.append(Message(
                    peer_id=msg.peer_id, 
                    message=f"ACK from {node.name}", 
                    round=tick, 
                    score=score
                ))

            if replies:
                await node.send(replies)

if __name__ == "__main__":
    asyncio.run(main())
