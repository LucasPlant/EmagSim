import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, 
                             QVBoxLayout, QWidget, QHBoxLayout)
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import sys
import time
from matplotlib.animation import FuncAnimation
from simulation import *

class Animator:
    """Class in charge of animating the simulation data using matplotlib"""

    def __init__(self, simulation):
        self.simulation: Simulation = simulation

    def initialize_plot(self):
        # Matplotlib Figure and Canvas
        self.fig = plt.figure(figsize=(10, 7))
        self.ax = self.fig.add_subplot(111, projection='3d')

        # 3D line plot
        self.lines = []
        for data in self.simulation.get_data():
            self.lines.append(self.ax.plot(data.x, data.y, data.voltage_V, linewidth=3)[0])
        
        # Plot styling
        self.ax.set_xlim(0, 120)
        self.ax.set_ylim(-20, 20)
        self.ax.set_zlim(-6, 6)
        
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Voltage')
        self.ax.set_title('Transmission Line Voltage Propagation')

    def update(self):
        self.simulation.step()
        new_data = self.simulation.get_data()
    
        for line, data in zip(self.lines, new_data):
            line.set_data_3d(data.x, data.y, data.voltage_V)

        return self.lines

    def reset():
        pass

    
class GUI(QMainWindow):
    """Class that implements the GUI implementation"""

    def __init__(self, animator: Animator):
        super().__init__()

        self.animator: Animator = animator

        self.setWindowTitle('3D Transmission Line Simulator')
        self.setGeometry(100, 100, 1200, 800)

        # Central widget setup
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Matplotlib Figure and Canvas
        self.animator.initialize_plot()
        self.canvas = FigureCanvasQTAgg(self.animator.fig)
        main_layout.addWidget(self.canvas)

        # Control buttons
        btn_layout = QHBoxLayout()
        play_btn = QPushButton('Play')
        pause_btn = QPushButton('Pause')
        reset_btn = QPushButton('Reset')

        play_btn.clicked.connect(self.start_simulation)
        pause_btn.clicked.connect(self.pause_simulation)
        reset_btn.clicked.connect(self.reset_simulation)

        btn_layout.addWidget(play_btn)
        btn_layout.addWidget(pause_btn)
        btn_layout.addWidget(reset_btn)
        main_layout.addLayout(btn_layout)

        # Animation timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_canvas)

    def update_canvas(self):
        self.animator.update()
        self.canvas.draw()

    def start_simulation(self):
        """Start the simulation"""
        self.timer.start(50)  # Update every 50 ms

    def pause_simulation(self):
        """Pause the simulation"""
        self.timer.stop()

    def reset_simulation(self):
        """Reset the simulation to initial state"""
        pass


def main():
    app = QApplication(sys.argv)
    simulator = TlineSim()
    animator = Animator(simulator)
    gui = GUI(animator)
    gui.show()
    sys.exit(app.exec_())
    plt.show()

if __name__ == '__main__':
    main()