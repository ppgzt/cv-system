import threading, time, psutil, csv, subprocess
from datetime import datetime

class CPUMonitor(threading.Thread):
    def __init__(self, pid: str):
        super().__init__()
        self.pid = pid
        self.running = True
        self.data = []
        self.daemon = True # Allows the main program to exit even if thread is running

    def run(self):
        # The first call returns 0.0, so we ignore it
        psutil.cpu_percent(percpu=True)
        while self.running:
            # Call with an interval, which blocks the thread for that duration 
            # and updates the measurement internally
            cpu_percent = psutil.cpu_percent(percpu=True, interval=1)
            self.data.append([datetime.now().isoformat()] + cpu_percent)
            print(f'CPU: {cpu_percent}')

    def stop(self):
        self.running = False

        # Criar cabeçalho do CSV
        num_cores = len(self.data[0]) - 1
        cabecalho = ["timestamp"] + [f"cpu_core_{i}" for i in range(num_cores)]

        # Salvar CSV
        with open(f"infra/reports/{self.pid}/cpu.csv", mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(cabecalho)
            writer.writerows(self.data)

class RAMMonitor(threading.Thread):
    def __init__(self, pid: str):
        super().__init__()
        self.pid = pid
        self.running = True
        self.data = []
        self.daemon = True # Allows the main program to exit even if thread is running

    def run(self):
        while self.running:
            # Call with an interval, which blocks the thread for that duration 
            # and updates the measurement internally
            mem = psutil.virtual_memory()
            
            line = [
                datetime.now().isoformat(),
                getattr(mem, 'total', 0),
                getattr(mem, 'available', 0),
                getattr(mem, 'used', 0),
                getattr(mem, 'percent', 0),
                getattr(mem, 'free', 0),
                getattr(mem, 'active', 0),
                getattr(mem, 'inactive', 0),
                getattr(mem, 'buffers', 0),
                getattr(mem, 'cached', 0)
            ]
            
            self.data.append(line)
            print(f'RAM: {mem}')
            
            # wait 1 sec
            time.sleep(1)

    def stop(self):
        self.running = False

        # Cabeçalho CSV
        header = [
            "timestamp",
            "total",
            "available",
            "used",
            "percent",
            "free",
            "active",
            "inactive",
            "buffers",
            "cached"
        ]

        # Salvar arquivo
        with open(f"infra/reports/{self.pid}/mem.csv", mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(self.data)


class GPUMonitor(threading.Thread):
    def __init__(self, pid: str):
        super().__init__()        
        self.pid = pid
        self.running = True
        self.data = []
        self.daemon = True # Allows the main program to exit even if thread is running

    def run(self):
        while self.running:
            gpu_usage = self.__get_gpu_usage()
            gpu_memor = self.__get_gpu_memor()
            gpu_volts = self.__get_gpu_volts()
            gpu_tempr = self.__get_gpu_tempr()
            
            linha = [
                datetime.now().isoformat(),
                gpu_usage,
                gpu_memor,
                gpu_volts,
                gpu_tempr,
            ]
            self.data.append(linha)    
            
            print(f'GPU: {linha}')            
            
            # wait 3 secs, pq Videocore4 GPU tende a não ser tão utilizado!
            time.sleep(3)

    def stop(self):
        self.running = False

        # Cabeçalho CSV
        header = [
            "timestamp",
            "gpu_usage",
            "gpu_memor",
            "gpu_volts",
            "gpu_tempr",
        ]

        # Salvar CSV
        with open(f"infra/reports/{self.pid}/gpu.csv", mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(self.data)

    def __get_gpu_usage(self):
        # vcgencmd does not provide a simple percentage, 
        # but provides freq/voltage info.
        cmd = "vcgencmd measure_clock v3d"
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        return result.stdout.strip()

    def __get_gpu_memor(self):
        # Shows allocated shared memory
        cmd = "vcgencmd get_mem gpu"
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        return result.stdout.strip()
    
    def __get_gpu_volts(self):
        # Shows allocated shared memory
        cmd = "vcgencmd measure_volts"
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        return result.stdout.strip()

    def __get_gpu_tempr(self):
        # Shows allocated shared memory
        cmd = "vcgencmd measure_temp"
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        return result.stdout.strip()