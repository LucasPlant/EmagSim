import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
    QLabel,
    QDoubleSpinBox,
)
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import sys
import time
from matplotlib.animation import FuncAnimation
from simulation import *

class NumericLabel(QLabel):
    """Helper method to have an updating label with a number"""
    def __init__(self, label: str, number: complex):
        super().__init__(label)
        self._label = label + ": "
        self.setValue(number)

    def setValue(self, value: complex):
        self._value = value
        self.setText(self._label + str(self._value))


class Animator:
    """Class in charge of animating the simulation data using matplotlib"""

    def __init__(self, simulation):
        self.simulation: Simulation = simulation

    def initialize_plot(self):
        # Matplotlib Figure and Canvas
        self.fig = plt.figure(figsize=(10, 7))
        self.ax = self.fig.add_subplot(111, projection="3d")

        # 3D line plot
        self.lines = []
        for data in self.simulation.get_data():
            self.lines.append(
                self.ax.plot(data.x, data.y, data.voltage_V, linewidth=3)[0]
            )

        # Plot styling
        self.ax.set_xlim(0, 120)
        self.ax.set_ylim(-20, 20)
        self.ax.set_zlim(-6, 6)

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Voltage")
        self.ax.set_title("Transmission Line Voltage Propagation")

    def update(self):
        self.simulation.step()
        new_data = self.simulation.get_data()

        for line, data in zip(self.lines, new_data):
            line.set_data_3d(data.x, data.y, data.voltage_V)

        return self.lines

    def reset():
        pass


class TLineSimPage(QMainWindow):
    """Class that implements the GUI implementation"""

    def __init__(self, animator: Animator):
        # TODO May want to split this up to make it more readable
        super().__init__()

        self.animator: Animator = animator

        self.setWindowTitle("3D Transmission Line Simulator")
        self.setGeometry(100, 100, 1200, 800)

        # Central widget setup
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Matplotlib Figure and Canvas
        self.animator.initialize_plot()
        self.canvas = FigureCanvasQTAgg(self.animator.fig)
        main_layout.addWidget(self.canvas)

        # main buttons
        simulation_controls = QHBoxLayout()
        play_btn = QPushButton("Play")
        step_button = QPushButton("Step")
        pause_btn = QPushButton("Pause")
        reset_btn = QPushButton("Reset")

        play_btn.clicked.connect(self.start_simulation)
        step_button.clicked.connect(self.step_simulation)
        pause_btn.clicked.connect(self.pause_simulation)
        reset_btn.clicked.connect(self.reset_simulation)

        simulation_controls.addWidget(play_btn)
        simulation_controls.addWidget(step_button)
        simulation_controls.addWidget(pause_btn)
        simulation_controls.addWidget(reset_btn)
        main_layout.addLayout(simulation_controls)

        def make_callback(func):
            """Helper function to assure that lambda is created by value"""
            return lambda value: func(value)            

        # Add the component specific controls
        for (
            component_name,
            interface,
        ) in self.animator.simulation.get_gui_interfaces().items():
            component_field = QHBoxLayout()
            component_field.addWidget(QLabel(component_name))

            # Add in the input widgets
            if "input" in interface:
                for input_name, input_info in interface["input"].items():
                    input = QDoubleSpinBox()
                    if "range" in input_info:
                        input.setRange(*input_info["range"])
                    else:
                        input.setRange(
                            -sys.float_info.max, sys.float_info.max
                        )  # Set range to float max/min
                    input.setValue(
                        input_info["default"]
                    )  # All inputs defined in the interface should have a default
                    # Set the change to a 10th of the initial value
                    input.setSingleStep(
                        (1 if input_info["default"] == 0 else input_info["default"])
                        / 10
                    )
                    # Set the action
                    input.valueChanged.connect(make_callback(input_info["callback"]))

                    # add label and add input field
                    component_field.addWidget(QLabel(input_name))
                    component_field.addWidget(input)

            # Add the output widgets
            if "output" in interface:
                for output_name, output_info in interface["output"].items():
                    output = NumericLabel(output_name, output_info["func"]())
                    interface["output"][output_name]["field"] = output
                    component_field.addWidget(output)

            main_layout.addLayout(component_field)

        # Animation timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_canvas)

    def update_canvas(self):
        self.animator.update()
        self.canvas.draw()
        for (
            component_name,
            interface,
        ) in self.animator.simulation.get_gui_interfaces().items():
            if "output" in interface:
                for _, info in interface["output"].items():
                    info["field"].setValue(info["func"]())

    def start_simulation(self):
        """Start the simulation"""
        self.timer.start(50)  # Update every 50 ms

    def step_simulation(self):
        """Step the simulation a single time step"""
        self.update_canvas()

    def pause_simulation(self):
        """Pause the simulation"""
        self.timer.stop()

    def reset_simulation(self):
        """Reset the simulation to initial state"""
        pass


def main():
    app = QApplication(sys.argv)
    simulator = ParallelFanOutSim()
    animator = Animator(simulator)
    gui = TLineSimPage(animator)
    gui.show()
    sys.exit(app.exec_())
    plt.show()


if __name__ == "__main__":
    main()
