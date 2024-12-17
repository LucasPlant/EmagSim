import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Union

class TlinePorts(Enum):
    """Enum to define the ports that we can have"""
    FORWARD = 0
    BACKWARD = 1

class ConnectionType(Enum):
    """Enum to hold the connection types"""
    CASCADE = 0
    SERIES = 1
    PARALLEL = 2

@dataclass
class WaveValue:
    """
    Dataclass to hold voltage and current values
    these can be complex to accommodate for the the phaser simulation
    """
    voltage_V: complex
    current_A: complex

@dataclass
class SimInfo:
    """
    A class to hold onto info about the simulation for reference by all of the objects
    """

    # The current time step that we are on
    timeStep: int
    # The the amount of continuous time that passes with each time step
    delta_t_S: float

    def simulation_time_s(self):
        return self.timeStep * self.delta_t_S


class Component:
    """
    A base class to define a component in a circuit
    All components should have a impedance and a receive transmission method
    """

    def __init__(self, impedance_ohms: complex, sim_info: SimInfo):
        """
        Initialize parameters and internal variables

        Args:
        impedance: the impedance of the transmission line or circuit component when looking into it
            For time domain simulations this should be a real number
            When simulating using phasers this can be a complex number using numpys complex number implementation
        """
        self.impedance_ohms = impedance_ohms
        self._sim_info = sim_info


    def connect(self, type: ConnectionType, forward: Union[tuple, list]):
        """
        Method to initialize the connections with other components

        Args:
        type: the type of connection
            for now this will only support purely series or purely parallel connections
        forward: Either a tuple (Cascaded) or a list of tuples (others) containing the connected component and the function used to reference it
        """
        pass

    def receive_transmission(self, value: WaveValue):
        """
        The method to receive a wave transmitted from another component
        Args:
        value: A tuple containing a voltage and a current (voltage, current)
        """
        pass

    def run_transmissions(self):
        """
        Calculate the waves that this component will transmit to other components
        Call the corresponding transmit methods
        """
        pass

    def update_component(self):
        """
        Run the calculations necessary to update the simulation of the given component
        """
        pass


class TransmissionLine(Component):
    """
    An extension of component representing a transmission line
    A transmission line will have 2 ports forward and backward
    """

    def __init__(self, impedance_ohms, sim_info, length_m):
        """
        Constructor for the Tline
        Args:
            impedance: the impedance of the Tline
            velocity: the propagation velocity of the Tline
            length: an integer length representing how many chunks we will use to represent the tline
        """
        super().__init__(impedance_ohms, sim_info)

        # Arrays that represent the voltages along the transmission line will shift these arrays to represent the wave propagation
        self._f_voltage_V = np.zeros(length_m)
        self._b_voltage_V = np.zeros(length_m)

        # Initialize the received values
        self._received_f: WaveValue = WaveValue(0, 0)
        self._received_b: WaveValue = WaveValue(0, 0)

        # Initialize forward and backwards reflection coefficients to be updated by the run transmissions
        self._reflection_coef_f = 0
        self._reflection_coef_b = 0

    def connect(self, types: tuple[ConnectionType, ConnectionType], forward, backward):
        """
        Method to initialize the connections with other components

        Args:
        types: a tuple containing the types of connections (forward, backward)
        forward: Either a tuple (Cascaded) or a list of tuples (others) containing the connected component and the function used to reference it
        backward: Either a tuple (Cascaded) or a list of tuples (others) containing the connected component and the function used to reference it
        """
        self._forward_type = types[0]
        if self._forward_type  is ConnectionType.CASCADE and type(forward) is not tuple:
            raise ValueError("For Cascaded connection forward must be a tuple")
        if self._forward_type  in [ConnectionType.SERIES, ConnectionType.PARALLEL] and type(forward) is not list:
            raise ValueError("For Series and Parallel connection forward must be a list")
        
        self._backward_type = types[1]
        if self._backward_type  is ConnectionType.CASCADE and type(backward) is not tuple:
            raise ValueError("For Cascaded connection backward must be a tuple")
        if self._backward_type in [ConnectionType.SERIES, ConnectionType.PARALLEL] and type(backward) is not list:
            raise ValueError("For Series and Parallel connection backward must be a list")
        
        self._forward_connections = forward
        self._backward_connections = backward

    def receive_transmission(self, port: TlinePorts, value: WaveValue):
        # Store the sums to be injected into our tline when we run the simulation portion
        if port is TlinePorts.FORWARD:
            self._received_f.voltage_V += value.voltage_V
            self._received_f.current_A += value.current_A
        elif port is TlinePorts.BACKWARD:
            self._received_b.voltage_V += value.voltage_V
            self._received_b.current_A += value.current_A
        
    def run_transmissions(self):
        # TODO use functions to re use code between forward and backwards

        # Transmit forward
        if self._forward_type is ConnectionType.CASCADE:
            # Simplest form when there is a 1 to 1 connection
            forward_component: Component = self._forward_connections[0]
            transmission_func = self._forward_connections[1]
            
            # tau = 1 + gamma | gamma = (ZL - Z0) / (ZL + Z0)
            self._reflection_coef_f = (forward_component.impedance_ohms - self.impedance_ohms) / (forward_component.impedance_ohms + self.impedance_ohms)
            transmission_coef = 1 + self._reflection_coef_f

            transmitted_voltage_V = transmission_coef * self._f_voltage_V[-1]
            transmitted_current_A = transmitted_voltage_V / forward_component.impedance_ohms

            transmission_func(WaveValue(transmitted_voltage_V, transmitted_current_A))
        else:
            raise NotImplementedError
        
        # elif self._forward_type is ConnectionType.PARALLEL:
        #     # Find the impedance of the load via parallel addition
        #     sum_1dz = 0.0
        #     for component, _ in self._forward_connections:
        #         # Hopefully this doesnt cause performance issues
        #         # We must do this every step as some components may change their impedance
        #         sum_1dz += component.impedance_ohms
        #     load_impedance_ohms = 1.0 / sum_1dz

        #     # In a parallel fan our every node is transmitted the same voltage
        #     transmission_coef = (1 + (load_impedance_ohms - self.impedance_ohms) / (load_impedance_ohms + self.impedance_ohms))
        #     transmitted_voltage = self._f_voltage_V[-1] * transmission_coef

        #     for component, transmission_func in self._forward_connections:
        #         # Send the transmissions
        #         transmission_func(WaveValue(self._f_voltage_V[-1]))
        # elif self._forward_type is ConnectionType.SERIES:
            
        # Transmit backward
        if self._backward_type is ConnectionType.CASCADE:
            # Simplest form when there is a 1 to 1 connection
            backward_component: Component = self._backward_connections[0]
            transmission_func = self._backward_connections[1]
            
            # tau = 1 + gamma | gamma = (ZL - Z0) / (ZL + Z0)
            self._reflection_coef_b = (backward_component.impedance_ohms - self.impedance_ohms) / (backward_component.impedance_ohms + self.impedance_ohms)
            transmission_coef = 1 + self._reflection_coef_b

            transmitted_voltage_V = transmission_coef * self._b_voltage_V
            transmitted_current_A = transmitted_voltage_V / backward_component.impedance_ohms

            transmission_func(WaveValue(transmitted_voltage_V, transmitted_current_A))
        else:
            raise NotImplementedError
        
    def update_component(self):
        # Roll arrays calculate reflections and 
        reflected_from_f = self._f_voltage_V[-1] * self._reflection_coef_f
        reflected_from_b = self._b_voltage_V[0] * self._reflection_coef_b

        self._f_voltage_V = np.roll(self._f_voltage_V, 1)
        self._b_voltage_V = np.roll(self._b_voltage_V, -1)

        self._f_voltage_V[0] = reflected_from_b + self._received_f.voltage_V
        self._b_voltage_V[-1] = reflected_from_f + self._received_b.voltage_V

        # Reset the variable to keep track of reflections
        self._received_f = WaveValue(0, 0)
        self._received_b = WaveValue(0, 0)


class TransmissionLineFromLC(TransmissionLine):
    """A wrapper around transmission lines that allows you to define them using inductance and capacitance"""


class FunctionGenerator(Component):
    """A component Representing a voltage source"""

    def __init__(self, impedance_ohms, sim_info):
        super().__init__(impedance_ohms, sim_info)

    def voltage_func():
        """Define a function for the voltage output"""
        return 0

    def connect(self, connection_type: ConnectionType, forward):
        self.type = connection_type
        if connection_type is ConnectionType.CASCADE and type(forward) is not tuple:
            raise ValueError("For Cascaded connection forward must be a tuple")
        if connection_type  in [ConnectionType.SERIES, ConnectionType.PARALLEL] and type(forward) is not list:
            raise ValueError("For Series and Parallel connection forward must be a list")
        
        self._connections = forward

    def receive_transmission(self, value: WaveValue):
        pass

    def run_transmissions(self):
        if self.type is ConnectionType.CASCADE:
            # Simplest form when there is a 1 to 1 connection
            component: Component = self._connections[0]
            transmission_func = self._connections[1]
            
            # Resistor divider
            transmission_coef = component.impedance_ohms / (self.impedance_ohms + component.impedance_ohms)

            transmitted_voltage_V = transmission_coef * self.voltage_func()
            transmitted_current_A = transmitted_voltage_V / component.impedance_ohms

            transmission_func(WaveValue(transmitted_voltage_V, transmitted_current_A))
        else:
            raise NotImplementedError

    def update_component(self):
        pass


class VoltageSource(FunctionGenerator):
    """A voltage source that generates an un changing voltage"""

    def __init__(self, impedance_ohms, sim_info, voltage_V):
        super().__init__(impedance_ohms, sim_info)

        self.voltage_V = voltage_V
    
    def voltage_func(self):
        return self.voltage_V
    

class ACVoltageSource(FunctionGenerator):
    """A voltage source that generates an AC signal"""

    def __init__(self, impedance_ohms, sim_info, amplitude_V, frequency_HZ, phase_deg):
        super().__init__(impedance_ohms, sim_info)

        self.amplitude_V = amplitude_V
        self.frequency_HZ =frequency_HZ
        self.phase_deg =phase_deg

    def voltage_func(self):
        return self.amplitude_V * np.sin((2 * np.pi * self.frequency_HZ * self._sim_info.simulation_time_s()) + self.phase_deg)
    

class ResistiveLoad(Component):
    """A component representing a resistor. It does nothing besides having an impedance"""

    def __init__(self, impedance_ohms, sim_info):
        super().__init__(impedance_ohms, sim_info)

    def connect(self, connection_type: ConnectionType, forward):
        pass

    def receive_transmission(self, value: WaveValue):
        pass

    def run_transmissions(self):
        pass

    def update_component(self):
        pass