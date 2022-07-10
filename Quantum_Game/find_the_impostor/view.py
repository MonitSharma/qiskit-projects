import time

class View:

    BOARD_STRING = "|------------|" \
                   "\n|  1  |   2  |" \
                   "\n|------------|" \
                   "\n|  3  |   4  |" \
                   "\n-------------|"

    def __init__(self, argument=1):
        pass

    def display_initial_message(self, username):
            self.display_board()
            time.sleep(1)
            print("Player Name : " + username)
            time.sleep(1)
            print("\nThere's someone sus out there...")
            time.sleep(1)
            print("It exists here, there and everywhere ... at once")
            time.sleep(1)
            print("We bestow upon you the Quantum Knife!!!")
            time.sleep(1)
            print("Use it and end the terror of the Quantum-impostor!")

    def accept_move(self):
        x = input("What are you going to do? \n Enter c for classical hit and q for quantum hit followed by the Panel number? ")
        return x

    def final_message(self, username, win):
        if win:
            print("Wow ...")
            time.sleep(3)
            print("The Quantum-Impostor is really gone ...")
            time.sleep(3)
            print("You did it " + username)
            print("Congratulations!!!")

        else:
            print("Alas...")
            print("The quantum impostor lives on ... it is here, there and everywhere ... at once")

    def display_quantum_message(self, hint, hole):
        print("New Information!!!")
        time.sleep(3)
        print("Probability amplitude of impostor being in vent " + hole + " is ", hint)
        time.sleep(3)

    def display_board(self):
        print(View.BOARD_STRING)

    def accept_quantum_gate(self):
        result = []
        gate = input("What quantum gate do you wish to add to the circuit? ")
        result.append(gate)

        if gate == 'r' or gate == 'R':
            angle = input("Enter phase ")
            result.append(angle)

        elif gate == 'cx' or gate == 'CX':
            control_qubit = input("Enter control qubit ")
            result.append(control_qubit)

        else:
          result.append("b")
        target_qubit = input("Enter target qubit ")
        result.append(target_qubit)

        return result
