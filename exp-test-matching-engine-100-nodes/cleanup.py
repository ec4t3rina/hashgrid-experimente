import asyncio
from hashgrid import Hashgrid

API_KEY = "hg_3e40b9898ca54f808a40c25d85157b6cd3190e14ab90c5e0"

async def cleanup():
    grid = await Hashgrid.connect(api_key=API_KEY)
    print(f"Conectat la {grid.name}. Încep ștergerea nodurilor...")
    
    count = 0
    async for node in grid.nodes():
        print(f"Șterg nodul: {node.name} ({node.node_id})")
        await node.delete()
        count += 1
    
    print(f"--- Curățenie finalizată. Am șters {count} noduri. ---")

if __name__ == "__main__":
    asyncio.run(cleanup())