import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Union


class TlinePorts(Enum):
    """Enum to define the ports that we can have"""

    FRONT = 0
    BACK = 1


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


@dataclass
class Connection:
    """
    A dataclass to represent a reference to an object
    This is paired with a callable on how to send data to that object"""

    component: "Component"
    func: callable


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

    def connect(
        self, connection: tuple[ConnectionType, Union[Connection, list[Connection]]]
    ):
        """
        Method to initialize the connections with other components

        Args:
        connection: a tuple containing the connection type and either a Connection object (cascade) or a list of Connection objects (series and parallel)
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

    # Helper methods that contain the "Scary Math" of T-Lines
    @staticmethod
    def _cascade(z_0_ohms: complex, z_load_ohms: complex) -> tuple[complex, complex]:
        """
        Defines the math behind a cascaded junction or a 1 to 1 connection
        Args:
        Z_0_ohms: The impedance of the current device
        Z_load_ohms: The impedance of the load

        Returns:
        (gamma, tau)
        gamma: The reflection coefficient of the connection
        tau: The transmission coefficient for the connection
        """
        gamma_voltage = (z_load_ohms - z_0_ohms) / (z_load_ohms + z_0_ohms)
        # gamma_current = -1 * gamma_voltage

        tau_voltage = 1 + gamma_voltage
        # tau_current =

        return gamma_voltage, tau_voltage

    @staticmethod
    def _parallel(
        z_0_ohms: complex, z_load_ohms: list[complex]
    ) -> tuple[complex, complex]:
        """
        Defines the math behind a parallel junction
        Args:
        Z_0_ohms: The impedance of the current device
        Z_load_ohms: A list of the impedances of the loads

        Returns:
        (gamma, tau)
        gamma: the reflection coefficient of the connection
        tau: the transmission coefficient of the connection
            In a parallel connection the transmission coefficient is the same across all loads
        """

        # Calculate the parallel impedances
        reciprocal_sum = sum(1 / z for z in z_load_ohms)
        parallel_z = 1 / reciprocal_sum

        gamma_voltage = (parallel_z - z_0_ohms) / (parallel_z + z_0_ohms)

        tau_voltage = 1 + gamma_voltage

        return gamma_voltage, tau_voltage

    @staticmethod
    def _series(
        z_0_ohms: complex, z_load_ohms: list[complex]
    ) -> tuple[complex, list[complex]]:
        """
        Defines the math behind a series junction
        Args:
        Z_0_ohms: The impedance of the current device
        Z_load_ohms: A list of the impedances of the loads

        Returns:
        (gamma, tau)
        gamma: the reflection coefficient of the connection
        tau: a list of the transmission coefficients for each individual load
        """
        z_series = sum(z_load_ohms)
        gamma_voltage = (z_series - z_0_ohms) / (z_series + z_0_ohms)

        tau_total = 1 + gamma_voltage

        # This is essentially a resistor divider over the transmitted voltage
        tau_voltage = [tau_total * (z / z_series) for z in z_load_ohms]

        return gamma_voltage, tau_voltage

    @staticmethod
    def _generic_transmit(
        connection_type: ConnectionType,
        z_0_ohms: complex,
        connections: Union[Connection, list[Connection]],
        incoming_voltage_V: complex,
    ) -> complex:
        """
        A helper method to handle the transmission logic to re use code between transmission forward and backward

        Args:
        connection_type: The type of connection (Cascade, Parallel, or Series)
        z_0_ohms: the impedance of the t-line
        connections: A single connection (Cascade) or a list of connections
        incoming_voltage_V: the voltage coming into the connection to be transmitted

        Returns:
        gamma: the reflection coeficient passed down for the mathematical methods
        """
        if connection_type is ConnectionType.CASCADE:
            # Compute coefficients
            reflection_coef, transmission_coef = Component._cascade(
                z_0_ohms, connections.component.impedance_ohms
            )

            # Use coefficients to calculate transmissions
            transmitted_voltage_V = incoming_voltage_V * transmission_coef
            transmitted_current_A = (
                transmitted_voltage_V / connections.component.impedance_ohms
            )

            connections.func(WaveValue(transmitted_voltage_V, transmitted_current_A))

        elif connection_type is ConnectionType.PARALLEL:
            # Compute coefficients
            reflection_coef, transmission_coef = Component._parallel(
                z_0_ohms,
                [connection.component.impedance_ohms for connection in connections],
            )

            transmitted_voltage_V = incoming_voltage_V * transmission_coef

            for connection in connections:
                transmitted_current_A = (
                    transmitted_voltage_V / connection.component.impedance_ohms
                )
                connection.func(WaveValue(transmitted_voltage_V, transmitted_current_A))

        elif connection_type is ConnectionType.SERIES:
            # Compute coefficients
            reflection_coef, transmission_coefficients = Component._series(
                z_0_ohms,
                [connection.component.impedance_ohms for connection in connections],
            )

            for connection, transmission_coef in zip(
                connections, transmission_coefficients
            ):
                transmitted_voltage_V = transmission_coef * incoming_voltage_V
                transmitted_current_A = (
                    transmitted_voltage_V / connection.component.impedance_ohms
                )
                connection.func(WaveValue(transmitted_voltage_V, transmitted_current_A))

        else:
            raise NotImplementedError

        return reflection_coef


class TransmissionLine(Component):
    """
    An extension of component representing a transmission line
    A transmission line will have 2 ports forward and backward
    The wave will be represented by a 1D numpy array
    This wave will be rotated forward and backward to represent the wave propagation

                    (Forward going wave)
                >>>>>>>>>>forward>>>>>>>>>>>>>
    Back end                                      Front End
                <<<<<<<<<<backward<<<<<<<<<<<<
                    (Backward going wave)
    """

    def __init__(self, impedance_ohms: complex, sim_info: SimInfo, length: int):
        """
        Constructor for the Tline
        Args:
            impedance: the impedance of the Tline
            velocity: the propagation velocity of the Tline
            length: an integer length representing how many chunks we will use to represent the tline
        """
        super().__init__(impedance_ohms, sim_info)

        # Arrays that represent the voltages along the transmission line will shift these arrays to represent the wave propagation
        self._f_voltage_V = np.zeros(length)
        self._b_voltage_V = np.zeros(length)

        # Initialize the received values
        self._received_front: WaveValue = WaveValue(0, 0)
        self._received_back: WaveValue = WaveValue(0, 0)

        # Initialize forward and backwards reflection coefficients to be updated by the run transmissions
        self._reflection_coef_f = 0
        self._reflection_coef_b = 0

    def connect(
        self,
        front: tuple[ConnectionType, Union[Connection, list[Connection]]],
        back: tuple[ConnectionType, Union[Connection, list[Connection]]],
    ):
        """
        Method to initialize the connections with other components

        Args:
        front: A tuple containing a connection type and either a Connection object or List of Connection objects
            represents the connections to the front side of a t-line
        back: A tuple containing a connection type and either a Connection object or List of Connection objects
            represents the connections to the back side of a t-line

        See above for a crude drawing of a transmission line
        """
        self._front_type, self._front_connections = front
        self._back_type, self._back_connections = back

        # Sanity check the inputs
        # Forward
        if (
            self._front_type is ConnectionType.CASCADE
            and type(self._front_connections) is not Connection
        ):
            raise ValueError("For Cascaded connection forward must be a tuple")
        if (
            self._front_type in [ConnectionType.SERIES, ConnectionType.PARALLEL]
            and type(self._front_connections) is not list
        ):
            raise ValueError(
                "For Series and Parallel connection forward must be a list"
            )

        # Backward
        if (
            self._back_type is ConnectionType.CASCADE
            and type(self._back_connections) is not Connection
        ):
            raise ValueError("For Cascaded connection backward must be a tuple")
        if (
            self._back_type in [ConnectionType.SERIES, ConnectionType.PARALLEL]
            and type(self._back_connections) is not list
        ):
            raise ValueError(
                "For Series and Parallel connection backward must be a list"
            )

    def receive_transmission(self, port: TlinePorts, value: WaveValue):
        # Store the sums to be injected into our tline when we run the simulation portion
        if port is TlinePorts.FRONT:
            self._received_front.voltage_V += value.voltage_V
            self._received_front.current_A += value.current_A
        elif port is TlinePorts.BACK:
            self._received_back.voltage_V += value.voltage_V
            self._received_back.current_A += value.current_A

    def run_transmissions(self):
        # Transmit forward
        self._reflection_coef_f = Component._generic_transmit(
            self._front_type,
            self.impedance_ohms,
            self._front_connections,
            self._f_voltage_V[-1],
        )
        # Transmit backward
        self._reflection_coef_b = Component._generic_transmit(
            self._back_type,
            self.impedance_ohms,
            self._back_connections,
            self._b_voltage_V[0],
        )

    def update_component(self):
        # Roll arrays calculate reflections and
        reflected_from_f = self._f_voltage_V[-1] * self._reflection_coef_f
        reflected_from_b = self._b_voltage_V[0] * self._reflection_coef_b

        self._f_voltage_V = np.roll(self._f_voltage_V, 1)
        self._b_voltage_V = np.roll(self._b_voltage_V, -1)

        self._f_voltage_V[0] = reflected_from_b + self._received_back.voltage_V
        self._b_voltage_V[-1] = reflected_from_f + self._received_front.voltage_V

        # Reset the variable to keep track of reflections
        self._received_front = WaveValue(0, 0)
        self._received_back = WaveValue(0, 0)


class TransmissionLineFromLC(TransmissionLine):
    """A wrapper around transmission lines that allows you to define them using inductance and capacitance"""


class FunctionGenerator(Component):
    """A component Representing a voltage source"""

    def __init__(self, impedance_ohms: complex, sim_info: SimInfo):
        super().__init__(impedance_ohms, sim_info)

    def voltage_func(self) -> complex:
        """Define a function for the voltage output"""
        return 0

    def connect(
        self,
        connection: tuple[ConnectionType, Union[Connection, list[Connection]]],
    ):
        self._connection_type, self._connections = connection

        # Sanity check the inputs
        if (
            self._connection_type is ConnectionType.CASCADE
            and type(self._connections) is not Connection
        ):
            raise ValueError("For Cascaded connection forward must be a tuple")
        if (
            self._connection_type in [ConnectionType.SERIES, ConnectionType.PARALLEL]
            and type(self._connections) is not list
        ):
            raise ValueError(
                "For Series and Parallel connection forward must be a list"
            )

    def run_transmissions(self):
        if self._connection_type is ConnectionType.CASCADE:
            # Use coefficients to calculate transmissions
            transmitted_voltage_V = self.voltage_func() * (
                self._connections.component.impedance_ohms
                / (self.impedance_ohms + self._connections.component.impedance_ohms)
            )
            transmitted_current_A = (
                transmitted_voltage_V / self._connections.component.impedance_ohms
            )

            self._connections.func(
                WaveValue(transmitted_voltage_V, transmitted_current_A)
            )

        elif self._connection_type is ConnectionType.PARALLEL:
            # Compute coefficients
            reciprocal_sum = sum(
                1 / connection.component.impedance_ohms
                for connection in self._connections
            )
            parallel_z = 1 / reciprocal_sum

            transmitted_voltage_V = self.voltage_func() * (
                parallel_z / (self.impedance_ohms + parallel_z)
            )

            for connection in self._connections:
                transmitted_current_A = (
                    transmitted_voltage_V / connection.component.impedance_ohms
                )
                connection.func(WaveValue(transmitted_voltage_V, transmitted_current_A))

        elif self._connection_type is ConnectionType.SERIES:
            # Compute coefficients
            impedances = [
                connection.component.impedance_ohms for connection in self._connections
            ]
            total_impedance = sum(impedances) + self.impedance_ohms

            for connection, impedance in zip(self._connections, impedances):
                transmitted_voltage_V = self.voltage_func() * (
                    impedance / (total_impedance + impedance)
                )
                transmitted_current_A = transmitted_voltage_V / impedance
                connection.func(WaveValue(transmitted_voltage_V, transmitted_current_A))

        else:
            raise NotImplementedError


class VoltageSource(FunctionGenerator):
    """
    A voltage source that generates an un changing voltage
    This should only be used in the time domain
    """

    def __init__(self, impedance_ohms: float, sim_info: SimInfo, voltage_V: float):
        super().__init__(impedance_ohms, sim_info)

        self.voltage_V = voltage_V

    def voltage_func(self):
        return self.voltage_V


class ACVoltageSource(FunctionGenerator):
    """
    A voltage source that generates an AC signal
    This should only be used in the time domain
    """

    def __init__(
        self,
        impedance_ohms: float,
        sim_info: SimInfo,
        amplitude_V: float,
        frequency_HZ: float,
        phase_deg: float,
    ):
        super().__init__(impedance_ohms, sim_info)

        self.amplitude_V = amplitude_V
        self.frequency_HZ = frequency_HZ
        self.phase_deg = phase_deg

    def voltage_func(self):
        return self.amplitude_V * np.sin(
            (2 * np.pi * self.frequency_HZ * self._sim_info.simulation_time_s())
            + self.phase_deg
        )


class ResistiveLoad(Component):
    """
    A component representing a resistor. It does nothing besides having an impedance
    In frequency domain simulations this can have an arbitrary impedance
    """

    def __init__(self, impedance_ohms: complex, sim_info: SimInfo):
        super().__init__(impedance_ohms, sim_info)
