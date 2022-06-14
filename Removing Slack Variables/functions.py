#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 10:44:00 2022

@author: alejomonbar
"""
import numpy as np
import matplotlib.pyplot as plt
from docplex.mp.model import Model

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.problems.constraint import ConstraintSense

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_optimization.algorithms import CplexOptimizer

from qiskit import Aer

from qiskit_optimization.translators import from_docplex_mp

import itertools


def BinPacking(num_items, num_bins, weights, max_weight, simplification=False):
    # Construct model using docplex
    mdl = Model("BinPacking")

    y = mdl.binary_var_list(num_bins, name="y") # list of variables that represent the bins
    x =  mdl.binary_var_matrix(num_items, num_bins, "x") # variables that represent the items on the specific bin

    objective = mdl.sum(y)

    mdl.minimize(objective)

    for i in range(num_items):
        # First set of constraints: the items must be in any bin
        mdl.add_constraint(mdl.sum(x[i, j] for j in range(num_bins)) == 1)

    for j in range(num_bins):
        # Second set of constraints: weight constraints
        mdl.add_constraint(mdl.sum(weights[i] * x[i, j] for i in range(num_items)) <= max_weight * y[j])

    # Load quadratic program from docplex model
    qp = from_docplex_mp(mdl)
    if simplification:
        l = int(np.ceil(np.sum(weights)/max_weight))
        qp = qp.substitute_variables({f"y_{_}":1 for _ in range(l)}) # First simplification 
        qp = qp.substitute_variables({"x_0_0":1}) # Assign the first item into the first bin
        qp = qp.substitute_variables({f"x_0_{_}":0 for _ in range(1, num_bins)}) # as the first item is in the first 
                                                                                #bin it couldn't be in the other bins
    qubo = QuadraticProgramToQubo().convert(qp)# Create a converter from quadratic program to qubo representation
    return qubo, qp

def BinPackingNewApproach(num_items, num_bins, weights, max_weight, alpha=0.01, simplification=False):
    # Construct model using docplex
    mdl = Model("BinPackingNewApproach")

    y = mdl.binary_var_list(num_bins, name="y") # list of variables that represent the bins
    x =  mdl.binary_var_matrix(num_items, num_bins, "x") # variables that represent the items on the specific bin

    objective = mdl.sum(y)
    
    # PENALIZATION
    penalization = 0
    for j in range(num_bins):
        t = max_weight * y[j] - mdl.sum(weights[i] * x[i, j] for i in range(num_items))
        penalization += 10*(t**2 - t)
    mdl.minimize(objective + alpha * penalization)

    for i in range(num_items):
        # First set of constraints: the items must be in any bin
        mdl.add_constraint(mdl.sum(x[i, j] for j in range(num_bins)) == 1)

    # Load quadratic program from docplex model
    qp = from_docplex_mp(mdl)
    if simplification:
        l = int(np.ceil(np.sum(weights)/max_weight))
        qp = qp.substitute_variables({f"y_{_}":1 for _ in range(l)}) # First simplification 
        qp = qp.substitute_variables({"x_0_0":1}) # Assign the first item into the first bin
        qp = qp.substitute_variables({f"x_0_{_}":0 for _ in range(1, num_bins)}) # as the first item is in the first 
                                                                                #bin it couldn't be in the other bins
    qubo = QuadraticProgramToQubo().convert(qp)# Create a converter from quadratic program to qubo representation
    return qubo

def Knapsack(weights, values, max_weight):
    mdl = Model("Knapsack")
    
    num_items = len(weights)
    x = mdl.binary_var_list(num_items, name="x")
    
    objective = mdl.sum([x[i]*values[i] for i in range(num_items)])
    mdl.maximize(objective)
    
    mdl.add_constraint(mdl.sum(weights[i] * x[i] for i in range(num_items)) <= max_weight)
    # Converting to QUBO
    qp = from_docplex_mp(mdl)
    qubo = QuadraticProgramToQubo().convert(qp)
    return qubo
    
def KnapsackNewApproach(values, weights, max_weight, alpha=0.01):
    mdl = Model("KnapsackNewApproach")
    print(len(values))
    x = {i: mdl.binary_var(name=f"x_{i}") for i in range(len(values))}
    
    # PENALIZATION
    penalization = 0
    for j in range(len(values)):
        t = max_weight - mdl.sum(weights[i] * x[i] for i in range(len(weights)))
        penalization += -t + t**2 / 2
    mdl.maximize(mdl.sum(values[i] * x[i] for i in x) - alpha *(penalization))

    qp = from_docplex_mp(mdl) #bin it couldn't be in the other bins
    qubo = QuadraticProgramToQubo().convert(qp)# Create a converter from quadratic program to qubo representation
    return qubo

def interpret(results, weights, max_weight, num_items, num_bins, simplify=False):
    """
    Save the results as a list of list where each sublist represent a bin
    and the sublist elements represent the items weights
    
    Args:
    results: results of the optimization
    weights (list): weights of the items
    max_weight (int): Max weight of a bin
    num_items: (int) number of items
    num_bins: (int) number of bins
    """
    if simplify:
        l = int(np.ceil(np.sum(weights)/max_weight))
        bins = l * [1] + list(results[:num_bins - l])
        items = results[num_bins - l: (num_bins - l) + num_bins * (num_items - 1)].reshape(num_items - 1, num_bins)
        items_in_bins = [[i+1 for i in range(num_items-1) if bins[j] and items[i, j]] for j in range(num_bins)]
        items_in_bins[0].append(0)
    else:
        bins = results[:num_bins]
        items = results[num_bins:(num_bins + 1) * num_items].reshape((num_items, num_bins))
        items_in_bins = [[i for i in range(num_items) if bins[j] and items[i, j]] for j in range(num_bins)]
    return items_in_bins

def get_figure(items_in_bins, weights, max_weight, title=None):
    """Get plot of the solution of the Bin Packing Problem.

    Args:
        result : The calculated result of the problem

    Returns:
        fig: A plot of the solution, where x and y represent the bins and
        sum of the weights respectively.
    """
    colors = plt.cm.get_cmap("jet", len(weights))
    num_bins = len(items_in_bins)
    fig, axes = plt.subplots()
    for _, bin_i in enumerate(items_in_bins):
        sum_items = 0
        for item in bin_i:
            axes.bar(_, weights[item], bottom=sum_items, label=f"Item {item}", color=colors(item))
            sum_items += weights[item]
    axes.hlines(max_weight, -0.5, num_bins - 0.5, linestyle="--", color="tab:red", label="Max Weight")
    axes.set_xticks(np.arange(num_bins))
    axes.set_xlabel("Bin")
    axes.set_ylabel("Weight")
    axes.legend()
    if title:
        axes.set_title(title)
    return fig

def qaoa_circuit(qubo: QuadraticProgram, p: int = 1):
    """
    Given a QUBO instance and the number of layers p, constructs the corresponding parameterized QAOA circuit with p layers.
    Args:
        qubo: The quadratic program instance
        p: The number of layers in the QAOA circuit
    Returns:
        The parameterized QAOA circuit
    """
    size = len(qubo.variables)
    qubo_matrix = qubo.objective.quadratic.to_array(symmetric=True)
    qubo_linearity = qubo.objective.linear.to_array()

    #Prepare the quantum and classical registers
    qaoa_circuit = QuantumCircuit(size,size)
    #Apply the initial layer of Hadamard gates to all qubits
    qaoa_circuit.h(range(size))

    #Create the parameters to be used in the circuit
    gammas = ParameterVector('gamma', p)
    betas = ParameterVector('beta', p)

    #Outer loop to create each layer
    for i in range(p):
        
        #Apply R_Z rotational gates from cost layer
        for j in range(size):
            qaoa_circuit.rz((qubo_linearity[j] + np.sum(qubo_matrix[j]))*gammas[i], j)
        #Apply R_ZZ rotational gates for entangled qubit rotations from cost layer
        for j in range(size-1):
            for k in range(j+1,size):
                qaoa_circuit.cx(k,j)
                qaoa_circuit.rz(qubo_matrix[j,k]*gammas[i], j)
                qaoa_circuit.cx(k,j)
#                     qaoa_circuit.cp(0.5*qubo_matrix[j,k]*gammas[i], j, k)
                        
        # Apply single qubit X - rotations with angle 2*beta_i to all qubits
        qaoa_circuit.rx(2*betas[i],range(size))
    qaoa_circuit.measure(range(size), range(size))
    return qaoa_circuit

def cost_func(parameters, circuit, objective, n=10, backend=Aer.get_backend("qasm_simulator")):
    """
    Return a cost function that depends of the QAOA circuit 

    Parameters
    ----------
    parameters : list
        alpha and beta values of the QAOA circuit.
    circuit : QuantumCircuit
        Qiskit quantum circuit of the QAOA.
    objective : QuadraticProgram
        Objective function of the QuadraticProgram
    n : int, optional
        number of strings from the quantum circuit measurement to be use for the cost. The default is 10.
    backend : Qiskit Backend, optional
        The default is Aer.get_backend("qasm_simulator").

    Returns
    -------
    float
        Cost of the evaluation of n string on the objective function 

    """
    cost = 0
    counts = backend.run(circuit.assign_parameters(parameters=parameters)).result().get_counts()
    counts = {k: v for k, v in sorted(counts.items(), key=lambda item: item[1], reverse=True)}
    samples = 0
    for sample in list(counts.keys())[:n]:
        cost += counts[sample] * objective.evaluate([int(_) for _ in sample])
        samples += counts[sample]
    return cost**2 / samples

def check_best_sol(parameters, circuit, qp, max_weight, n=10, backend=Aer.get_backend("qasm_simulator")):
    """
    Return a cost function that depends of the QAOA circuit 

    Parameters
    ----------
    parameters : list
        alpha and beta values of the QAOA circuit.
    circuit : QuantumCircuit
        Qiskit quantum circuit of the QAOA.
    objective : QuadraticProgram
        Objective function of the QuadraticProgram
    n : int, optional
        number of strings from the quantum circuit measurement to be use for the cost. The default is 10.
    backend : Qiskit Backend, optional
        The default is Aer.get_backend("qasm_simulator").

    Returns
    -------
    float
        Cost of the evaluation of n string on the objective function 

    """
    cost = 0
    counts = backend.run(circuit.assign_parameters(parameters=parameters)).result().get_counts()
    counts = {k: v for k, v in sorted(counts.items(), key=lambda item: item[1], reverse=True)}
    cost_min = np.inf
    best_solution = None
    for sample in list(counts.keys())[:n]:
        sample_list = [int(_) for _ in sample]
        cost = qp.objective.evaluate(sample_list)
        if eval_constrains(qp, sample_list, max_weight) and (cost < cost_min):
            best_solution = sample_list
            cost_min = cost
    if best_solution == None:
        return print("There is not possible solution in the samples analized")
    return best_solution

def new_eq_optimal(qubo_new, qubo_classical):
    """
    From the classical solution and considering that cplex solution is the optimal, we can traslate the optimal
    solution to the QUBO representation based on our approach.
    
    
    """
    num_vars = qubo_new.get_num_vars()
    result_cplex = CplexOptimizer().solve(qubo_classical)
    result_new_ideal = qubo_new.objective.evaluate(result_cplex.x[:num_vars])# Replacing the ideal solution into
                                               #our new approach to see the optimal solution on the new objective
                                               #function
    return result_new_ideal

def eval_constrains(qp, result, max_weight):
    """
    Evaluate if all the restrictions of a quadratic program are satisfied.

    Parameters
    ----------
    qp : QuadraticProgram
        Problem to be solved, here the restrictions are still accessible.
    result : list
        Solution of the QUBO .
    max_weight : int
        It works for Bin Packing problem and is the maximum weight a bin can 
        handled.

    Returns
    -------
    Boolean
        If any of the inequality constraints is not satisfied return False.

    """
    constraints = qp.linear_constraints
    varN = len(qp.variables)
    for const in constraints:
        if const.sense in [ConstraintSense.GE, ConstraintSense.LE]:
            print(const.evaluate(result[:varN]) - const.rhs)
            if (const.evaluate(result[:varN]) - const.rhs) > 0:
                return False
        elif const.sense == ConstraintSense.EQ:
            print(const.evaluate(result[:varN]))
            if const.evaluate(result[:varN]) != 1.0:
                return False
    return True

def mapping_cost(alpha, beta, qubo, n=10, backend=Aer.get_backend("qasm_simulator")):
    """
    Only valid for one step in the QAOA solution of a problem.

    Parameters
    ----------
    alpha : array
        angle alpha of the QAOA algorithm.
    beta : array
        angle beta (mixing anlge) of the QAOA algorithm.
    qubo : Quadratic Unconstrained binary optimization
    n : int
        number of solutions taken from the measurement of the circuit
    backend: Qiskit backend
        Backend used to simulated QAOA
    
    Returns
    -------
    Cost: squared array

    """
    circuit = qaoa_circuit(qubo)
    n1 = len(alpha)
    n2 = len(beta)
    
    cost = []
    for parameters in itertools.product(alpha, beta):
        cost.append(cost_func(parameters, circuit, qubo.objective, n, backend))
    cost = np.array(cost).reshape(n1,n2)
    return cost


