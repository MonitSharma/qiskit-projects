import time
import math
from view import *
from model import *

isGameOver = False
winner = False

def play_game(username, view, model):
    view.display_initial_message(username)
    while not isGameOver:
        move = view.accept_move()
        execute(model, move)
    view.final_message(username, winner)
#    exit(0)

def execute(model, move):
    global isGameOver
    if move == 'show circuit':
        model.draw_current_circuit()
        return
    type_hammer, hole = move[0], move[1]
    if type_hammer == 'c':
        print("Classical Hit!!!")
        isGameOver = True
        winner = classical_smash(model, hole)
    else:
        print('Quantum Hit!!!')
        quantum_smash(model, hole)

def classical_smash(model, hole):
    measured_hole = model.measureState()
    return (measured_hole==[0,0] and hole==1) or (measured_hole==[0,1] and hole==2) or (measured_hole==[1,0] and hole==3) or (measured_hole==[1,1] and hole==4)

def translate_hole(hole):
  if hole==1:
    return [0,0]
  if hole==2:
    return [0,1]
  if hole==3:
    return [1,0]
  if hole==4:
    return [1,1]

def quantum_smash(model, hole):
    #hint is a number or a complex number
    hint = model.getProbabilityOf(translate_hole(hole))
    view.display_quantum_message(hint, hole)
    gate_details = view.accept_quantum_gate()

    if gate_details[0] == 'r' or gate_details[0] == 'R':
        model.add_r_gate(math.radians(float(gate_details[1])), int(gate_details[2]))

    if gate_details[0] == 'cx' or gate_details[0] == 'CX':
        model.add_cnot(int(gate_details[1]), int(gate_details[2]))

    else:
        model.add_unitary(gate_details[0], int(gate_details[2]))

    model.draw_current_circuit()

def initialize_game():
    username = input("What is your name? ")
    view = View()
    model = Model()
    return username, view, model

if __name__ == '__main__':
    x = input("Do you want to play Quantum Game?")
    if x == 'Yes' or x=="yes" or x=="y":
        username, view, model = initialize_game()
        play_game(username, view, model)
    else:
        exit(0)


