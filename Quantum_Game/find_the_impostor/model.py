#imports

import qiskit
import numpy as np
from qiskit.visualization import plot_histogram


# model

class Model:
    simulator = qiskit.Aer.get_backend('qasm_simulator')

    def draw_current_circuit(self):
        print(self.add_circuit.draw())

    # initialize 2 qubits
    def __init__(self):
        # TODO initialize with more friendly state vectors?
        self.circuit = qiskit.QuantumCircuit(2, 2)
        self.state1 = self.circuit.initialize(qiskit.quantum_info.random_statevector(2).data, 0)
        self.state2 = self.circuit.initialize(qiskit.quantum_info.random_statevector(2).data, 1)
        self.add_circuit = qiskit.QuantumCircuit(2)

    # Measure qubits and return state with max probability: ex. [0,1]
    def measureState(self):
        self.circuit.measure([0, 1], [1, 0])
        job = qiskit.execute(self.circuit, self.simulator, shots=1)  # 1 shot to keep it luck dependent?
        result = job.result()
        count = result.get_counts()
        # max_value = max(result.values())
        # return [k for k,v in count.items() if v==1][0]
        return [int(list(count)[0][0]), int(list(count)[0][1])]

    # Return a probability coefficient of specific state
    # state is an array of size 2 which contains 0 or 1
    # such as [0,1], [0,0], [1,0], [1,1]
    def getProbabilityOf(self, state):
        # to get state vector of qubit
        backend = qiskit.Aer.get_backend('statevector_simulator')
        result = qiskit.execute(self.circuit, backend).result()
        out_state = result.get_statevector()
        if state == [0, 0]:
            return out_state[0]
        elif state == [0, 1]:
            return out_state[1]
        elif state == [1, 0]:
            return out_state[2]
        else:
            return out_state[3]

    # Add a gate to the end of the circuit (at specified qubit)
    def add_unitary(self, name, qubit_no):
        if name == "H" or name == "h":
            self.add_circuit.h(qubit_no)
        elif name == "X" or name == "x":
            self.add_circuit.x(qubit_no)
        elif name == "Y" or name == "y":
            self.add_circuit.y(qubit_no)
        elif name == "Z" or name == "z":
            self.add_circuit.z(qubit_no)

        self.circuit += self.add_circuit

    def add_r_gate(self, parameter, qubit_no):
        self.add_circuit.rz(parameter, qubit_no)

        self.circuit += self.add_circuit

    def add_cnot(self, control_qubit_no, target_qubit_no):
        self.add_circuit.cx(control_qubit_no, target_qubit_no)

        self.circuit += self.add_circuit