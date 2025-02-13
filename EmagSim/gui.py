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
    QComboBox,
)
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import sys
import os
import time
from matplotlib.animation import FuncAnimation
from simulation import *
import json


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
        self.ax.autoscale()
        # self.ax.set_xlim(0, 120)
        # self.ax.set_ylim(-20, 20)
        # TODO have a way to intelligently set this
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

    def __init__(self, config_path):
        super().__init__()

        self.config_path = config_path

        self.setWindowTitle("3D Transmission Line Simulator")
        self.setGeometry(100, 100, 1200, 800)

        # Central widget setup
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QVBoxLayout(central_widget)

        self.make_simulation_controls()
        self.initialize_simulation(config_path)
        self.create_simulation()
        self.make_component_specific_controls()

        # Add widgets and layouts to the GUI
        self.main_layout.addWidget(self.canvas)
        self.main_layout.addWidget(self.simulation_controls)
        self.main_layout.addWidget(self.component_controls)

        # Animation frame timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_canvas)

    def initialize_simulation(self, config_path):
        # parse json
        with open(config_path, "r") as config_file:
            config_dict = json.load(config_file)

        # Fetch individual configuration dictionaries
        self.sim = config_dict["sim"]
        components_info_with_options: dict[str, list[dict]] = config_dict["components"]
        self.connections = config_dict["connections"]

        # this dict will map component names too a dict containing 3 values
        # "widget": QComboBox, "values": the possible values (dict), index: the selected index
        self.dropdown_menus: dict[str, dict] = {}
        # make options objects
        for name, options_list in components_info_with_options.items():
            combo_box = QComboBox()
            for component_option_info in options_list:
                combo_box.addItem(component_option_info["type"])

            combo_box.setCurrentText(options_list[0]["type"])

            menu_values = {"widget": combo_box, "values": options_list, "index": 0}
            combo_box.currentIndexChanged.connect(
                lambda idx, name=name: self.component_selection_callback(name, idx)
            )
            self.dropdown_menus[name] = menu_values

    def component_selection_callback(self, name, id):
        """Callback used to change the values of the component selectors"""
        self.dropdown_menus[name]["index"] = id

    def create_simulation(self):
        components_info = {
            name: info["values"][info["index"]]
            for name, info in self.dropdown_menus.items()
        }

        simulator = Simulation.from_dicts(self.sim, components_info, self.connections)

        self.animator = Animator(simulator)

        # Matplotlib Figure and Canvas
        self.animator.initialize_plot()
        self.canvas = FigureCanvasQTAgg(self.animator.fig)

    def make_simulation_controls(self):
        # General buttons for controlling the simulation
        simulation_controls_layout = QHBoxLayout()
        self.simulation_controls = QWidget()
        play_btn = QPushButton(text="Play", parent=self.simulation_controls)
        step_button = QPushButton(text="Step", parent=self.simulation_controls)
        pause_btn = QPushButton(text="Pause", parent=self.simulation_controls)
        reset_btn = QPushButton(text="Reset", parent=self.simulation_controls)

        play_btn.clicked.connect(self.start_simulation)
        step_button.clicked.connect(self.step_simulation)
        pause_btn.clicked.connect(self.pause_simulation)
        reset_btn.clicked.connect(self.restart_simulation)

        simulation_controls_layout.addWidget(play_btn)
        simulation_controls_layout.addWidget(step_button)
        simulation_controls_layout.addWidget(pause_btn)
        simulation_controls_layout.addWidget(reset_btn)

        self.simulation_controls.setLayout(simulation_controls_layout)

    def make_component_specific_controls(self):
        def make_callback(func):
            """Helper function to assure that lambda is created by value"""
            return lambda value: func(value)

        self.component_controls = QWidget()
        component_controls_layout = QVBoxLayout(self.component_controls)

        # Add the component specific controls
        for (
            component_name,
            interface,
        ) in self.animator.simulation.get_gui_interfaces().items():
            component_field = QHBoxLayout(self.component_controls)

            component_field.addWidget(
                self.dropdown_menus[component_name]["widget"]
                if component_name in self.dropdown_menus
                else QLabel(component_name, parent=self.component_controls)
            )

            # Add in the input widgets
            if "input" in interface:
                for input_name, input_info in interface["input"].items():
                    input = QDoubleSpinBox(parent=self.component_controls)
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
                    component_field.addWidget(QLabel(input_name, parent=self.component_controls))
                    component_field.addWidget(input)

            # Add the output widgets
            if "output" in interface:
                for output_name, output_info in interface["output"].items():
                    output = NumericLabel(output_name, output_info["func"]())
                    interface["output"][output_name]["field"] = output
                    component_field.addWidget(output)

            component_controls_layout.addLayout(component_field)
        
        self.component_controls.setLayout(component_controls_layout)

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

    def restart_simulation(self):
        """Reset the simulation to initial state"""
        # Store old widgets and layouts to mark for deletion
        old_canvas = self.canvas
        old_component_controls = self.component_controls

        # Create new widgets and layouts
        self.create_simulation()
        self.make_component_specific_controls()

        # Replace the layouts
        self.main_layout.replaceWidget(old_canvas, self.canvas)
        self.main_layout.replaceWidget(old_component_controls, self.component_controls)

        old_canvas.deleteLater()
        old_component_controls.deleteLater()

def main():
    app = QApplication(sys.argv)
    print(os.getcwd())
    gui = TLineSimPage("EmagSim/tline_configs/single_t_line.json")
    gui.show()
    sys.exit(app.exec_())
    plt.show()


if __name__ == "__main__":
    main()
