from components import *
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, 
                             QVBoxLayout, QWidget, QHBoxLayout)
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import sys
import time
from matplotlib.animation import FuncAnimation

"""
TODO:
Clean up the transmit method for transmission lines and add fanout capability
Add in the json capability
Design the gui and allow it to change values through passing data
Reset method
phaser implementation

long term add in a rk4 simulation
"""

@dataclass
class ComponentWrapper:
    component: Component
    x: np.ndarray
    y: np.ndarray

@dataclass
class TLineData:
    x: np.ndarray
    y: np.ndarray
    voltage_V: np.ndarray
    current_A:np.ndarray

class Simulation:
    """
    A class to run the simulation
    """

    # A list of the components in this simulation connected with their 3d mapping for easy graphing
    components: ComponentWrapper = None

    def __init__(self, delta_t_S):
        self.sim_info = SimInfo(0, delta_t_S)

    def step(self):
        self.components: ComponentWrapper
        """Steps the simulation by 1 time step"""
        # Run the transmission part of the simulation
        for wrapped in self.components:
            component: Component = wrapped.component
            component.run_transmissions()

        # Run the update part of the simulation
        for wrapped in self.components:
            component: Component = wrapped.component
            component.update_component()

        # Update the sim info
        self.sim_info.timeStep += 1

    def get_data(self) -> list[TLineData]:
        """Returns the data from the simulation will be the x, y and the tline data"""
        tline_datas: list[TLineData] = []
        for wrapped in self.components:
            component: Component = wrapped.component
            if isinstance(component, TransmissionLine):
                voltage = component._f_voltage_V + component._b_voltage_V
                current = voltage / component.impedance_ohms
                data = TLineData(wrapped.x, wrapped.y, voltage, current)
                tline_datas.append(data)
        return tline_datas
    

class TestingSim(Simulation):
    """A sample simulation to do testing future simulations should be gathered by a json file"""

    def __init__(self):
        super().__init__(1)

        source = ACVoltageSource(50, self.sim_info, 5, 0.1, 0)
        tline = TransmissionLine(100, self.sim_info, 60)
        load = ResistiveLoad(1000, self.sim_info)

        source.connect(ConnectionType.CASCADE, (tline, lambda value: tline.receive_transmission(TlinePorts.FORWARD, value)))
        tline.connect(
            (ConnectionType.CASCADE, ConnectionType.CASCADE),
            (load, load.receive_transmission),
            (source, source.receive_transmission)
        )

        source_wrapped = ComponentWrapper(source, None, None)
        tline_wrapped = ComponentWrapper(tline, np.linspace(20, 60, 60), np.zeros(60))
        load_wrapped = ComponentWrapper(load, None, None)

        self.components = [source_wrapped, tline_wrapped, load_wrapped]


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
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(-1, 1)
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
    simulator = TestingSim()
    animator = Animator(simulator)
    gui = GUI(animator)
    gui.show()
    sys.exit(app.exec_())
    plt.show()

if __name__ == '__main__':
    main()