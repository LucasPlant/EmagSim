from components import *
from dataclasses import dataclass
import numpy as np
import json
import sys
from queue import Queue


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
    components: list[Component] = None

    def __init__(self, delta_t_S, components=None):
        self.sim_info = SimInfo(0, delta_t_S)
        self.components = components
        self.components_by_name = None

    def step(self):
        self.components: list[Component]
        """Steps the simulation by 1 time step"""
        # Run the transmission part of the simulation
        for component in self.components:
            component.run_transmissions()

        # Run the update part of the simulation
        for component in self.components:
            component.update_component()

        # Update the sim info
        self.sim_info.timeStep += 1

    def get_data(self) -> list[TLineData]:
        """Returns the data from the simulation will be the x, y and the tline data"""
        tline_datas: list[TLineData] = []
        for component in self.components:
            if isinstance(component, TransmissionLine):
                voltage = component._f_voltage_V + component._b_voltage_V
                current = voltage / component.impedance_ohms
                data = TLineData(component.x, component.y, voltage, current)
                tline_datas.append(data)
        return tline_datas

    def get_gui_interfaces(self) -> dict:
        """
        Returns a of the interfaces used to change and observe components
        """

        interfaces = {}
        for component in self.components:
            if component.get_gui_interface():
                interfaces[component.name] = component.get_gui_interface()

        return interfaces

    def __repr__(self):
        debug_string = "===================================\n"
        debug_string += "Simulation\n\n"
        if self.components_by_name is not None:
            debug_string += "components by name\n"
            debug_string += json.dumps(
                {
                    name: (
                        repr(component) if hasattr(component, "__repr__") else component
                    )
                    for name, component in self.components_by_name.items()
                },
                indent=2,
            )
            debug_string += "\n\n"
        if self.components is not None:
            debug_string += "components\n"
            debug_string += str(
                [
                    repr(component) if hasattr(component, "__repr__") else component
                    for component in self.components
                ]
            )
            debug_string += "\n\n"

        return debug_string

    # =============================================================
    # Methods for constructing simulations easily

    @staticmethod
    def from_dicts(sim: dict, components_info: dict, connections: list):
        """
        This can initialize a simulation using dictionaries containing the information on the simulation
        TODO full documentation on the dictionary structure
        ARGS:
        sim: dict containing information on the simulation
        components_info: a dict mapping components names to what type they are and the args used to make them
        connections: a list containing the connection information
        """
        simulation = Simulation(sim["delta_t_s"])
        # print(repr(simulation))

        simulation._construct_components(components_info)
        # print(repr(simulation))

        simulation._connect_components(connections)
        # print(repr(simulation))

        simulation._make_coordinate_mappings(
            simulation.components_by_name[sim["start"]]
        )
        # print(repr(simulation))

        simulation._make_components_list()
        # print(repr(simulation))

        return simulation

    def _make_components_list(self):
        """Turns the components by name dictionary into components lists"""
        self.components = [
            component for _, component in self.components_by_name.items()
        ]

    def _construct_components(self, components_info: dict) -> dict[str, Component]:
        """
        this method constructs the component objects from the component info

        ARGS:
        components_info: a dictionary mapping component names to their type and arguments
        """
        self.components_by_name = {}

        for component_name, construction_info in components_info.items():
            kwargs = construction_info["args"]
            # These args are not included in the dictionary and should be passed to the component
            kwargs["sim_info"] = self.sim_info
            kwargs["name"] = component_name

            # Construct the components by finding the object
            self.components_by_name[component_name] = getattr(
                sys.modules[__name__], construction_info["type"]
            )(**construction_info["args"])

    def _connect_components(self, connections: list[dict]):
        """
        Call the connect methods of all the components to connect them together from a
        list containing information on all of the connections
            [
                {
                    "type": (cascade, series, )
                    "connections": {name: port}
                }
            ]
        """
        # Store the kwargs that are used for the connect method of each component
        connection_kwargs = {name: {} for name in self.components_by_name.keys()}

        # A connection set is the set of connections that must be made to connect multiple connections logically
        # IE A must be connected to B and B must be connected to A
        for connection_set in connections:
            con_type = ConnectionType[connection_set["type"].upper()]
            connected_components = connection_set["components"]
            # Get all the connection objects for the given connection set
            connection_objects = Simulation._form_connection_objects(
                connected_components, self.components_by_name
            )

            # Loop through the components in this connection set to generate their args
            for component_name, port in connected_components.items():
                # Form the list or single connection for the connector
                con_objs_excluding_curr = [
                    connection_obj
                    for connection_name, connection_obj in connection_objects.items()
                    if connection_name is not component_name
                ]

                # In cascade case there are no other components and a list is not expected
                if con_type == ConnectionType.CASCADE:
                    con_objs_excluding_curr = con_objs_excluding_curr[0]

                # Add the args
                connection_kwargs[component_name][
                    "connection" if port is None else port
                ] = (con_type, con_objs_excluding_curr)

        # Go through and execute all of the connection methods
        for component_name, component in self.components_by_name.items():
            component.connect(**connection_kwargs[component_name])

    @staticmethod
    def _form_connection_objects(
        connected_components: dict[str, str], all_components: dict[str, Component]
    ) -> dict[str, Connection]:
        """
        Forms a dictionary mapping connection names to their corresponding connection object
        
        ARGS:
        connected_components: a dictionary mapping the component name to the port it should be connected to
        {name: port} for single port components the port should be None or null in json format
        all_components: a reference to a dictionary that maps the names to the component
        this is the components_by_name dictionary
        """
        # List of connection objects to return
        connection_objs = {}

        # loop through all of the components
        for component_name, component_port in connected_components.items():
            # Fetch the component object from the mapping of names to objects
            component = all_components[component_name]

            # Form the connection object
            if component_port is None:
                connection_obj = Connection(component, component.receive_transmission)
            else:
                connection_obj = Connection(
                    component,
                    lambda value: component.receive_transmission(
                        TlinePorts[component_port.upper()], value
                    ),
                )

            connection_objs[component_name] = connection_obj

        return connection_objs

    def _make_coordinate_mappings(self, start: TransmissionLine):
        """
        Traverse the graph formed by the components and map the transmission lines to their real world components

        ARGS:
        start: the transmission line that we will start the traversal
        """
        # Visited list to prevent double visiting
        # hashing of these objects may not be the best practice because they are mutable
        # This should be fine however because the hash will be the memory address
        # and the data structure only survives for the length of the method
        visited: set[TransmissionLine] = set()
        # Queue to store nodes to visit for our graph traversal
        # we will store the coordinates that the tline starts with
        traversal_queue: Queue[TransmissionLine] = Queue()
        traversal_queue.put(((0, 0), start))

        # BFS style traversal of the whole graph
        while not traversal_queue.empty():
            # Pop from Queue
            tup = traversal_queue.get_nowait()
            starting_coords: tuple[float, float] = tup[0]
            current_transmission_line: TransmissionLine = tup[1]
            visited.add(current_transmission_line)

            # if the popped component isnt a transmission line move on
            if not isinstance(current_transmission_line, TransmissionLine):
                continue

            shape = current_transmission_line.shape
            num_points = current_transmission_line.resolution
            # Calculate the number of points going in x dir and number going in y dir
            num_points_in_x = int(np.ceil(num_points * shape[0] / sum(shape)))
            num_points_in_y = int(np.floor(num_points * shape[1] / sum(shape)))
            # Calculate the coordinates that we will end at
            ending_coords = (
                starting_coords[0]
                + (current_transmission_line.length_m * shape[0] / sum(shape)),
                starting_coords[1]
                + (current_transmission_line.length_m * shape[1] / sum(shape)),
            )

            # We will move in the y dir first
            # Form x coordinates
            x_cords = np.concat(
                (
                    np.full(num_points_in_y, starting_coords[0]),
                    np.linspace(
                        starting_coords[0],
                        ending_coords[0],
                        num_points_in_x,
                    ),
                )
            )
            # Form Y coordinates
            y_cords = np.concat(
                (
                    np.linspace(
                        starting_coords[1],
                        ending_coords[1],
                        num_points_in_y,
                    ),
                    np.full(num_points_in_x, ending_coords[1]),
                )
            )

            # Set the tlines coordinate mapping
            current_transmission_line.set_x_y_mappings(x_cords, y_cords)

            # Add the next items from the front and from the back if they havent been visited yet
            for cords, next_tlines in zip(
                [ending_coords, starting_coords],
                [
                    current_transmission_line._front_connections,
                    current_transmission_line._back_connections,
                ],
            ):
                if type(next_tlines) is list:
                    for connection in next_tlines:
                        next_tline = connection.component
                        if next_tline not in visited:
                            traversal_queue.put((cords, next_tline))
                else:
                    next_tline = next_tlines.component
                    if next_tline not in visited:
                        traversal_queue.put((cords, next_tline))


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
