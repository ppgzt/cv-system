from datetime import datetime
import sys, os

from infra.profiling.agents import CPUMonitor, RAMMonitor, GPUMonitor
from domain.pipelines import SingleStreamStrategy, BatchStreamStrategy, MASStrategy

def main(
        pid: str, strategy: str, herd_size: int, passage_time: int, arrival_time: int, fselection_time: float, fselection_window:float):
    
    if strategy == 'single':
        pipeline = SingleStreamStrategy(
            pid=pid,
            herd_size=herd_size, 
            arrival_time=arrival_time, 
            passage_time=passage_time,
            fselection_time=fselection_time,
            fselection_window=fselection_window, 
        )
    elif strategy == 'batch':
        pipeline = BatchStreamStrategy(
            pid=pid,
            herd_size=herd_size, 
            passage_time=passage_time,
            arrival_time=arrival_time, 
            fselection_time=fselection_time,
            fselection_window=fselection_window, 
        )
    else:
        pipeline = MASStrategy(
            pid=pid,
            mode='batch' if 'batch' in strategy else 'single',
            herd_size=herd_size, 
            passage_time=passage_time,
            arrival_time=arrival_time, 
            fselection_time=fselection_time,
            fselection_window=fselection_window, 
        )
    
    pipeline.run()

if __name__ == "__main__":
    strategy = sys.argv[1]
    pid = f'{strategy}_{datetime.now().isoformat()}'

    cpu_monitor = CPUMonitor(pid=pid)
    ram_monitor = RAMMonitor(pid=pid)
    #gpu_monitor = GPUMonitor(pid=pid)

    try:
        os.mkdir(f'infra/reports/{pid}')
    except FileExistsError:
        pass

    if 'mas' not in strategy:
        cpu_monitor.start()
        ram_monitor.start()
        # gpu_monitor.start()
    
    try:
        # Main program logic
        main(
            pid=pid, strategy = strategy, herd_size = int(sys.argv[2]), passage_time = int(sys.argv[3]), arrival_time = int(sys.argv[4]), fselection_time = float(sys.argv[5]), fselection_window=float(sys.argv[6]),
        )

    finally:
        # Ensure the monitoring thread is stopped when done or on error
        if 'mas' not in strategy:
            cpu_monitor.stop()
            cpu_monitor.join() # Wait for the thread to finish

            ram_monitor.stop()
            ram_monitor.join()

        # gpu_monitor.stop()
        # gpu_monitor.join()        