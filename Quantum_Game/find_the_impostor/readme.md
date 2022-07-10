# Quantum Games

A fun way to learn about quantum circuits.

## Motivation

The science of quantum computing has recently experienced tremendous growth and continues to draw people of all ages. Students were interested in quantum computing at an ever-earlier age after first hearing about it. However, young students have a harder time comprehending the mechanism or approach of quantum algorithms than classical ones due to the complexity of the field. In order to provide young pupils with an interactive platform for implementing circuits, the Quantum game was created. To replicate four vents with their four states in superposition, the game needs two qubits.
The likelihood of discovering the impostor in the corresponding vent is indicated and represented by the amplitude of each state. In this game, the players are essentially asked to create the circuit themselves in order to maximise the amplitude of any one of the states, i.e., to maximise the probability of finding the impostor in one specific vent of their choice in order to hit the impostor with ease. Players could develop better quantum circuit implementation skills during the game by thinking of ways to increase amplitude.


## How to Play?
1. To start with, the player is given the option to use a Quantum Knife or a Classical Knife. The Quantum Knife will give the amplitude of the state that represents the corresponding "vent". The Classical Knife is equivalent to a measurement, through which the player will terminate the game and find out if the impostor is in the corresponding "vent".

2. If the players chose the Quantum Knife, they will then have the option to add a gate to the circuit to manipulate the amplitude of a state (i.e. probability of finding the impostor in the "vent") of their choice.

3. If the gate needs additional information (Ex. H, CNOT), the players will then be asked to input the information

4. A circuit will be shown, and the players will be asked again for their knife option.

5. The game ends when the players use the Classical knife to hit one "vent" of their choice. The quibts will be measured and the resulting state will be used to compare with the "vent" that the players choose. If the results match, a winning message will be shown; otherwise, the players will unfortunately loose the game.



## What is Quantum in this?

This game differs from the classical one, in this the players could manipulate the probability of finding the impostor in one vent of their choice. In that way, the players could potentially find the impostor every time with a good strategy. In addition, the impostor is also described in a superposition state, which can be a good way to visualize the concept of superposition.



## Details 
Used Qiskit and NumPy. To run the code, just clone the repo and install qiskit and run main.py



