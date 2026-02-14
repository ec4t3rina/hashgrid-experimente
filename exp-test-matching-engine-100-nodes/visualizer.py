import json
import matplotlib.pyplot as plt
import time

def live_plot():
    plt.ion()
    fig, ax = plt.subplots()
    
    while True:
        try:
            with open("simulation_results.json", "r") as f:
                data = json.load(f)
            
            ticks = [d['tick'] for d in data]
            scores = [d['avg_score'] for d in data]
            
            ax.clear()
            ax.plot(ticks, scores, 'r-o', label='Scor Mediu Matching')
            ax.set_xlabel('Tick')
            ax.set_ylabel('Similitudine (0-1)')
            ax.set_title('Test Performanță Matching Engine Hashgrid')
            ax.legend()
            ax.grid(True)
            plt.pause(10)
        except Exception:
            time.sleep(5)

if __name__ == "__main__":
    live_plot()