from components import *
from dataclasses import dataclass
import numpy as np


@dataclass
class ComponentWrapper:
    name: str
    component: Component
    x: np.ndarray
    y: np.ndarray


@dataclass
class TLineData:
    x: np.ndarray
    y: np.ndarray
    voltage_V: np.ndarray
    current_A: np.ndarray


class Simulation:
    """
    A base class to define the functionality of a simulation
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

    def get_gui_interfaces(self) -> dict:
        """
        Returns a list of interfaces
        TODO fill out documentation here
        """

        interfaces = {}
        for wrapped_component in self.components:
            if wrapped_component.component.get_gui_interface():
                interfaces[wrapped_component.name] = (
                    wrapped_component.component.get_gui_interface()
                )

        return interfaces


# Define some sample simulations
class TlineSim(Simulation):
    """A sample simulation for a simple t-line and source connected to a load"""

    def __init__(self):
        super().__init__(1)

        source = VoltageSource(50, self.sim_info, 3)
        tline = TransmissionLine(100, self.sim_info, 60)
        load = ResistiveLoad(100000, self.sim_info)

        source.connect(
            (
                ConnectionType.CASCADE,
                Connection(
                    tline,
                    lambda value: tline.receive_transmission(TlinePorts.BACK, value),
                ),
            )
        )
        tline.connect(
            front=(ConnectionType.CASCADE, Connection(load, load.receive_transmission)),
            back=(
                ConnectionType.CASCADE,
                Connection(source, source.receive_transmission),
            ),
        )

        source_wrapped = ComponentWrapper("source", source, None, None)
        tline_wrapped = ComponentWrapper(
            "t-line", tline, np.linspace(0, 60, 60), np.zeros(60)
        )
        load_wrapped = ComponentWrapper("load", load, None, None)

        self.components = [source_wrapped, tline_wrapped, load_wrapped]


class CascadeSim(Simulation):

    def __init__(self):
        super().__init__(1)

        source = VoltageSource(50, self.sim_info, 6)
        tline1 = TransmissionLine(50, self.sim_info, 60)
        tline2 = TransmissionLine(100, self.sim_info, 60)
        load = ResistiveLoad(100000, self.sim_info)

        source.connect(
            (
                ConnectionType.CASCADE,
                Connection(
                    tline1,
                    lambda value: tline1.receive_transmission(TlinePorts.BACK, value),
                ),
            )
        )
        tline1.connect(
            front=(
                ConnectionType.CASCADE,
                Connection(
                    tline2,
                    lambda value: tline2.receive_transmission(TlinePorts.BACK, value),
                ),
            ),
            back=(
                ConnectionType.CASCADE,
                Connection(source, source.receive_transmission),
            ),
        )

        tline2.connect(
            front=(ConnectionType.CASCADE, Connection(load, load.receive_transmission)),
            back=(
                ConnectionType.CASCADE,
                Connection(
                    tline1,
                    lambda value: tline1.receive_transmission(TlinePorts.FRONT, value),
                ),
            ),
        )

        source_wrapped = ComponentWrapper("source", source, None, None)
        tline1_wrapped = ComponentWrapper(
            "t-line1", tline1, np.linspace(0, 50, 60), np.zeros(60)
        )
        tline2_wrapped = ComponentWrapper(
            "tline-2", tline2, np.linspace(50, 100, 60), np.zeros(60)
        )
        load_wrapped = ComponentWrapper("load", load, None, None)

        self.components = [source_wrapped, tline1_wrapped, tline2_wrapped, load_wrapped]


class ParallelFanOutSim(Simulation):
    """A sample simulation to do testing future simulations should be gathered by a json file"""

    def __init__(self):
        super().__init__(1)

        source = ACVoltageSource(50, self.sim_info, 5, 0.1, 0)
        # source = VoltageSource(50, self.sim_info, 3)

        tline1 = TransmissionLine(100, self.sim_info, 60)
        tline2 = TransmissionLine(100, self.sim_info, 60)
        tline3 = TransmissionLine(100, self.sim_info, 60)

        load1 = ResistiveLoad(100000, self.sim_info)
        load2 = ResistiveLoad(100000, self.sim_info)

        source.connect(
            (
                ConnectionType.CASCADE,
                Connection(
                    tline1,
                    lambda value: tline1.receive_transmission(TlinePorts.BACK, value),
                ),
            ),
        )
        tline1.connect(
            front=(
                ConnectionType.PARALLEL,
                [
                    Connection(
                        tline2,
                        lambda value: tline2.receive_transmission(
                            TlinePorts.BACK, value
                        ),
                    ),
                    Connection(
                        tline3,
                        lambda value: tline3.receive_transmission(
                            TlinePorts.BACK, value
                        ),
                    ),
                ],
            ),
            back=(
                ConnectionType.CASCADE,
                Connection(source, source.receive_transmission),
            ),
        )
        tline2.connect(
            front=(
                ConnectionType.CASCADE,
                Connection(load1, load1.receive_transmission),
            ),
            back=(
                ConnectionType.PARALLEL,
                [
                    Connection(
                        tline1,
                        lambda value: tline1.receive_transmission(
                            TlinePorts.FRONT, value
                        ),
                    ),
                    Connection(
                        tline3,
                        lambda value: tline3.receive_transmission(
                            TlinePorts.BACK, value
                        ),
                    ),
                ],
            ),
        )
        tline3.connect(
            front=(
                ConnectionType.CASCADE,
                Connection(load2, load2.receive_transmission),
            ),
            back=(
                ConnectionType.PARALLEL,
                [
                    Connection(
                        tline1,
                        lambda value: tline1.receive_transmission(
                            TlinePorts.FRONT, value
                        ),
                    ),
                    Connection(
                        tline2,
                        lambda value: tline2.receive_transmission(
                            TlinePorts.BACK, value
                        ),
                    ),
                ],
            ),
        )

        source_wrapped = ComponentWrapper("source", source, None, None)
        tline1_wrapped = ComponentWrapper(
            "tline1", tline1, np.linspace(0, 60, 60), np.zeros(60)
        )
        tline2_wrapped = ComponentWrapper(
            "tline2",
            tline2,
            np.concat((np.full(10, 60), np.linspace(60, 110, 50))),
            np.concat((np.linspace(0, 10, 10), np.full(50, 10))),
        )
        tline3_wrapped = ComponentWrapper(
            "tline3",
            tline3,
            np.concat((np.full(10, 60), np.linspace(60, 110, 50))),
            np.concat((np.linspace(0, -10, 10), np.full(50, -10))),
        )
        load1_wrapped = ComponentWrapper("load1", load1, None, None)
        load2_wrapped = ComponentWrapper("load2", load2, None, None)

        self.components = [
            source_wrapped,
            tline1_wrapped,
            tline2_wrapped,
            tline3_wrapped,
            load1_wrapped,
            load2_wrapped,
        ]
