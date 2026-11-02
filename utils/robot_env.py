import time
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from jetson_orin_nx import Jetson_Orin_NX

class RobotEnv:
    def __init__(self, cfg, is_sim=False):
        self.board = Jetson_Orin_NX(cfg) 
        self.speed = 0.5
        self.fc_list = np.full(8, self.board.cpu_freqs[0])
        self.fg =   self.board.gpu_freqs[0]      
        self.acceleration = 0.5
        self.is_sim = is_sim

    def set_speed(self, v):
        """Sets the target velocity for the robot."""
        self.speed = max(0.0, v)
        print(f"[Robot] speed set to {self.speed} m/s")

    def get_speed(self):
        """Simulates real-time speed with basic physics (acceleration)."""
        return self.speed

    def get_mechanical_power(self):
        """Calculates instantaneous mechanical power based on current speed."""
        v = self.get_speed()
        return 1.14005 * v + 0.10893 * v ** 3 + 1.1625
    
    def set_board_cpu_dvfs(self, cpu, fc)->None:
        self.fc_list[cpu] = fc

        if self.is_sim == False:
            self.board.set_cpu_freq(cpu, self.fc_list[cpu])
    
    def set_board_gpu_dvfs(self, fg)->None:
        self.fg = fg
        if self.is_sim == False:
            self.board.set_gpu_freq(self.fg)
    
    def get_computational_power(self):
        if self.is_sim:
            P_static = 4.5
            alpha = 6.3
            beta = 4.2
            # ---- CPU power (per-core accumulation) ----
            P_cpu = 0.0
            for fc in self.fc_list:
                P_cpu += alpha * (fc / self.board.cpu_freqs[-1])

            # ---- GPU power ----
            P_gpu = beta * (self.fg / self.board.gpu_freqs[-1]) 

            return (P_static + P_cpu + P_gpu)

        else:
            return self.board.read_power()

    def get_robot_power(self):
        """Combines Computing (Board) and Mechanical (Motors) power."""
        p_comp = self.get_computational_power()
        p_mech = self.get_mechanical_power()
        return (p_comp + p_mech)