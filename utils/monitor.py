import threading
import time

class RealTimeMonitor(threading.Thread):
    def __init__(self, robot_env, interval=0.05):
        super().__init__()
        self.env = robot_env
        self.interval = interval
        self.stop_event = threading.Event()
        
        # Extended data storage to analyze the "Idle Tail"
        self.data = {
            "time": [], 
            "power": [], 
            "speed": [], 
            "cpu_pwr": [], 
            "gpu_pwr": [], 
            "total_j": 0.0
        }

    def run(self):
        start_t = time.time()
        while not self.stop_event.is_set():
            t = time.time() - start_t
            
            # Capture system state
            v = self.env.get_speed()
            # robot_env.get_total_power() should already include board + motors
            p_total = self.env.get_robot_power() 
            
            # Log data
            self.data["time"].append(t)
            self.data["speed"].append(v)
            self.data["power"].append(p_total)
            
            # Optional: Log sub-components if your board supports it
            # self.data["gpu_pwr"].append(self.env.board.get_gpu_power())
            
            # Energy Integration: P * dt
            self.data["total_j"] += p_total * self.interval
            
            time.sleep(self.interval)

    def stop(self):
        """Called only after BOTH LLM and Robot are confirmed finished."""
        self.stop_event.set()
        return self.data