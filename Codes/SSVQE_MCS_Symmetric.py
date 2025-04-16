import pandas as pd
import numpy as np
from numpy.random import randn
from scipy.linalg import circulant
import qiskit
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import Sampler, QiskitRuntimeService, Estimator, Options, Session
from qiskit_ibm_runtime.options import SimulatorOptions, EstimatorOptions, ExecutionOptions, SamplerOptions
from qiskit_algorithms import VQD
from qiskit_algorithms.optimizers import SPSA, COBYLA, IMFIL, SLSQP, BOBYQA, SNOBFIT, ADAM, Optimizer, Minimizer, OptimizerResult
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_ibm_runtime.fake_provider import FakeBogotaV2
from qiskit.circuit.library import  EfficientSU2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from __future__ import annotations
import logging
import warnings
from time import time
from collections.abc import Callable, Sequence
from utils import SSVQE



class SSVQE_MCS_Symmetric():
    def __init__(self, number_of_masses, eigenvalues, estimator, input_states, weights, optimizer_iterations, realizations, backend, reps, epsilon, x0=None):

        """
        Initialize the Quantum MCS SSVQE class.

        Parameters:
        - degrees of freedom: Degrees of freedom that will define the symmetric matrix.
        - eigenvalues: Number of eigenvalues to compute.
        - estimator: Quantum Estimator.
        - sampler: Quantum Sampler.
        - fidelity: State Fidelity computation strategy
        - input_states: List of quantum circuits as initial states.
        - weights: Weights (hyperparameters) for each input state.
        - optimizer_iterations: Maximum iterations for the optimizer.
        - realizations: Number of Monte Carlo simulations.
        - backend: Quantum backend used for execution. Both Fake backend, or IBM Quantum Hardware can be used 
        - reps: Number of repetitions in the ansatz.
        - epsilon: Random factor for uncertainty modeling (default: 0.15).
        - x0: Initial point for VQD optimization (default: None).
        """

        self.number_of_masses = number_of_masses
        self.realizations = realizations
        self.estimator = estimator
        self.eigenvalues = eigenvalues
        self.optimizer_iterations = optimizer_iterations
        self.input_states = input_states
        self.weights = weights
        self.backend = backend
        self.reps = reps
        self.epsilon = epsilon  # Perturbation factor for random uncertainty
        self.x0 = x0 if x0 is not None else np.zeros(66) #if x0 not provided, choose all zero initial point for 32-depth EfficientSU2 ansatz

        self.result_numpy_kf = []
        self.results = np.zeros((self.realizations, 4))


    """Example configuration of two-qubit orthogonal initial states and parameterized ansatz.

    The following code defines four orthogonal initial states for a two-qubit system.
    Each state is represented as a quantum circuit with specific X-gate operations applied to
    achieve the desired orthogonality. Additionally, corresponding weights are assigned to each
    state, which may be used in weighted optimization or state preparation.

    Example:
        input_states = [QuantumCircuit(2), QuantumCircuit(2), QuantumCircuit(2), QuantumCircuit(2)]
        input_states[1].x(0)  # Apply X gate on qubit 0
        input_states[2].x(1)  # Apply X gate on qubit 1
        input_states[3].x(0)  
        input_states[3].x(1)  # Apply X gate on both qubits

        weights = [40, 20, 10]  # Weights of each state
    """

    """Example configuration of a quantum Estimator, Sampler, state fidelity computation, and execution options.

    This setup demonstrates the initialization of a quantum backend, runtime service, and various execution 
    parameters for quantum computing experiments. The options define resilience levels and optimization levels 
    to control the fidelity and efficiency of computations. 

    Additionally, an Estimator and Sampler are instantiated, both of which are essential components for 
    variational quantum algorithms. The fidelity computation utilizes the ComputeUncompute method.

    Example:
        backend_fake = FakeBogotaV2()  # Simulated backend
        service = QiskitRuntimeService()  # Quantum runtime service instance

        # General execution options
        options = Options()
        options.resilience_level = 2  # Set error mitigation level
        options.optimization_level = 1  # Define circuit optimization level

        # Sampler-specific options
        sampler_options = Options()
        sampler_options.resilience_level = 1
        sampler_options.optimization_level = 1

        # Simulator options
        simulator = SimulatorOptions()
        options.simulator.seed_simulator = 150  # Set fixed seed for reproducibility

        # Instantiate quantum computing primitives
        estimator = Estimator(options=options, backend=backend)
        sampler = Sampler(options=sampler_options, backend=backend)
        fidelity = ComputeUncompute(sampler)  # Fidelity computation strategy
    """
    @staticmethod
    def k_new(x,epsilon):
        n = len(x)
        rdn = randn(n)
        x_new = np.zeros((n,))
        for i in range(n):
            x_new[i,] = x[i,]*(1+epsilon*rdn[i])

        return x_new
    
    @staticmethod
    def trid_matrix(k):
        n = len(k)    
        main_diag = np.zeros(n)
        main_diag[:-1] = k[:-1] + k[1:]
        main_diag[-1] = k[-1]
        off_diag = -k[1:]
        kf = np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)

        return kf
    
    @staticmethod
    def matrix_iteration_Clough_Penzien(Kf, M, tolerance=1e-4, max_iterations=100, rand_initialization_matrix_iteration = None, trial_vector = None):
        """
        Matrix iteration method by Clough and Penzien to acquire the smallest eigenvalue of M^{-1} * Kf . This method can be used in the tuning of the hyperaparameters of the Variational Algorithms
        
        Parameters:
        - Kf: Stiffness matrix 
        - M: Mass matrix 
        - tolerance: Convergence tolerance (default 1e-4)
        - max_iterations: Maximum number of iterations (default 100)
        
        Returns:
        - l_m: Smallest eigenvalue 
        - phi: Corresponding eigenvector 
        """
        
        n = Kf.shape[0]  # Size of the system
        if trial_vector:
            phi = trial_vector

        # If one wants to start with a random initial mode shape
        if rand_initialization_matrix_iteration == 'Yes':
            phi = np.random.rand(n)  
        
        #else start with all ones trial vector   
        else:
            phi = np.ones((n,))
        
        phi = np.random.rand(n) 
        phi /= np.linalg.norm(phi)  # Normalize
        for _ in range(max_iterations):
            phi_new = solve(Kf, M @ phi)
            phi_new /= np.linalg.norm(phi_new)
            #Rayleigh quotient
            l_m = (phi_new.T @ Kf @ phi_new) / (phi_new.T @ M @ phi_new)

            if np.linalg.norm(phi_new - phi) < tolerance:
                return l_m, phi_new
            phi = phi_new  

        raise ValueError(f"Matrix Iteration Method did not converge within {max_iterations} iterations.")

    def quantum_preparation(self):
        """
        Prepare the quantum ansatz and input states by transforming them into
        Intermediate Scalable Architecture (ISA) circuits.
        """

        ansatz = EfficientSU2(2, su2_gates=["ry"], entanglement='circular', reps=self.reps)  # Default ansatz for two-qubit input matrix (4x4)
        passmanager = generate_preset_pass_manager(optimization_level=1, target=self.backend.target) #Different forms of transpilation can be applied to improve efficiency and/or accuracy

        # Convert abstract circuits to quantum-appropriate ISA circuits
        self.isa_ansatz = passmanager.run(ansatz)
        self.isa_input_states = [passmanager.run(state) for state in self.input_states]

    def quantum_MCS_circulant(self):
        """
        Perform VQD Monte Carlo Simulation (MCS) similar to the last example of the paper; in a symmetric circulant matrix
        The same code is applicable to non-circulant matrices with proper modification of the input matrix

        Returns:
        - results: Eigenvalues from the quantum VQD MCS.
        - result_real: True eigenvalues computed via classical diagonalization.
        """

        x = np.ones((self.number_of_masses,))

        for i in range(self.realizations):
            x_rand = SSVQE_MCS_Symmetric.k_new(x,self.epsilon)
            kf = SSVQE_MCS_Symmetric.trid_matrix(x_rand)

            # Convert matrix into quantum observable format
            observable = SparsePauliOp.from_operator(kf)

            # Ensure ISA layout is defined
            try:
                isa_observable = observable.apply_layout(self.isa_ansatz.layout)
            except AttributeError:
                raise RuntimeError("Quantum preparation must be executed first.")

            # Initialize SSVQE algorithm
            ssvqe_instance = SSVQE(k=self.eigenvalues, weight_vector = self.weights, estimator=self.estimator, initial_point=self.x0,
                                optimizer=COBYLA(maxiter=self.optimizer_iterations),
                                ansatz=self.isa_ansatz, initial_states = self.isa_input_states
                                )

            # Compute eigenvalues
            result_ssvqe = ssvqe_instance.compute_eigenvalues(operator=isa_observable)
            result_numpy_kf = np.linalg.eigvals(kf)

            # Store results
            for j, eigval in enumerate(result_ssvqe.eigenvalues):
                self.results[i, j] = eigval

            # Ensure unique entries before appending
            self.result_numpy_kf.append(np.array(result_numpy_kf))

        return self.results, self.result_numpy_kf


    def results_to_csv(self):
        df = pd.DataFrame(self.results)
        df.to_csv('SSVQE_MCS_symmetric.csv')

        df = pd.DataFrame(self.result_numpy_kf)
        df.to_csv('Numpy-SSVQE_MCS_SSVQE_symmetric.csv')
