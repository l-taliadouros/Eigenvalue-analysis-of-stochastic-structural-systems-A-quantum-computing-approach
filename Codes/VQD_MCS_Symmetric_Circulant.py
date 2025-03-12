#
# This framework applies the Variational Quantum Deflation (VQD) from Qiskit.
#
# Qiskit is an open-source quantum computing framework developed by IBM and licensed under Apache 2.0.
# The VQD implementation used here is part of Qiskit and is licensed under Apache 2.0.
#
# (C) Copyright IBM 2017-2024. Licensed under Apache 2.0 (https://www.apache.org/licenses/LICENSE-2.0).
#
# Permission is granted to use this code for research and educational purposes, provided that proper citation is given.
# When using this code, the conlusions drawn from its results or its modifications, please cite the paper 'Eigenvalue analysis of stochastic structural systems: A quantum computing approach'
#link to the paper:
# Leonidas Taliadouros, Stochastic Engineering Dynamics Lab, Columbia University (2025).
# "" [ ].





import pandas as pd
import numpy as np
from numpy.random import randn
from scipy.linalg import circulant, solve
import qiskit
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import Sampler, QiskitRuntimeService, Estimator, Options, Session
from qiskit_ibm_runtime.options import SimulatorOptions, EstimatorOptions, ExecutionOptions, SamplerOptions
from qiskit_algorithms import VQD
from qiskit_algorithms.optimizers import SPSA, COBYLA, IMFIL, SLSQP, BOBYQA, SNOBFIT, ADAM
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_ibm_runtime.fake_provider import FakeBogotaV2
from qiskit.circuit.library import  EfficientSU2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager



class VQD_MCS_Symmetric_Circulant():
    def __init__(self, N, u, eigenvalues, estimator, sampler, fidelity, optimizer_iterations, realizations, backend, reps, epsilon, x0=None):

        """
        Initialize the Quantum MCS VQD class.

        Parameters:
        - circular_vector: Vector defining the circulant matrix.
        - eigenvalues: Number of eigenvalues to compute.
        - estimator: Quantum Estimator.
        - sampler: Quantum Sampler.
        - fidelity: State Fidelity computation strategy
        - optimizer_iterations: Maximum iterations for the optimizer.
        - realizations: Number of Monte Carlo simulations.
        - backend: Quantum backend used for execution. Both Fake backend, or IBM Quantum Hardware can be used 
        - reps: Number of repetitions in the ansatz.
        - epsilon: Random factor for uncertainty modeling.
        - x0: Initial point for VQD optimization (default: None).
        """

        self.N = N
        self.u = u
        self.realizations = realizations
        self.estimator = estimator
        self.eigenvalues = eigenvalues
        self.optimizer_iterations = optimizer_iterations
        self.sampler = sampler
        self.fidelity = fidelity
        self.backend = backend
        self.reps = reps
        self.epsilon = epsilon 
        self.x0 = x0 if x0 is not None else np.zeros(42) #if x0 not provided, choose all zero initial point for 20-depth EfficientSU2 ansatz

        self.result_numpy_kf = []
        self.results = np.zeros((self.realizations, 4))


    

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
        
        n = Kf.shape[0]  
        if trial_vector:
            phi = trial_vector

        # If one wants to start with a random initial mode shape
        if rand_initialization_matrix_iteration == 'Yes':
            phi = np.random.rand(n)  
        
        #else start with all ones trial vector   
        else:
            phi = np.ones((n,))
        
        phi = np.random.rand(n) 
        phi /= np.linalg.norm(phi)  
        for _ in range(max_iterations):
            phi_new = solve(Kf, M @ phi)
            phi_new /= np.linalg.norm(phi_new)
            #Rayleigh quotient
            l_m = (phi_new.T @ Kf @ phi_new) / (phi_new.T @ M @ phi_new)

            if np.linalg.norm(phi_new - phi) < tolerance:
                return l_m, phi_new
            phi = phi_new  

        raise ValueError(f"Matrix Iteration Method did not converge within {max_iterations} iterations.")
    
    @staticmethod
    def random_circulant_symmetric(N, u, epsilon):
        """
        This function can be modified to produce easily a different type of symmetric circulant matrix within the VQD MCS circulant
        """
        rdn = np.random.randn()  
        circ_element = (u * (1 + epsilon * rdn))**2  
        circ_vector = np.zeros(N)
        circ_vector[[0, -1, 1]] = [1 + 2 * circ_element, -circ_element, -circ_element]

        return circulant(circ_vector)

    def quantum_preparation(self):
        """
        Prepare the quantum ansatz and input states by transforming them into
        Intermediate Scalable Architecture (ISA) circuits.
        """

        ansatz = EfficientSU2(2, su2_gates=["ry"], entanglement='circular', reps=self.reps)  # Default ansatz for two-qubit input matrix (4x4)
        passmanager = generate_preset_pass_manager(optimization_level=1, target=self.backend.target) #Different forms of transpilation can be applied to improve efficiency and/or accuracy

        # Convert abstract circuits to quantum-appropriate ISA circuits
        self.isa_ansatz = passmanager.run(ansatz)

    def quantum_MCS_circulant(self):
        """
        Perform VQD Monte Carlo Simulation (MCS) similar to the last example of the paper by taliadouros et al. in a symmetric circulant matrix
        The same code is applicable to non-circulant matrices with proper modification of the input matrix

        Returns:
        - results: Eigenvalues from the quantum VQD MCS.
        - result_real: True eigenvalues computed via classical diagonalization.
        """

        betas = [33, 33, 33, 33]  # Hyperparameters 
        
        counts = []
        values = []
        steps = []

        def callback(eval_count, params, value, meta, step):
            counts.append(eval_count)
            values.append(value)
            steps.append(step)

        for i in range(self.realizations):
            
            random_circulant_matrix = VQD_MCS_Symmetric_Circulant.random_circulant_symmetric(self.N, self.u, self.epsilon)
            print(random_circulant_matrix)
            print(np.linalg.eigvals(random_circulant_matrix))
            # Convert matrix into quantum observable format
            observable = SparsePauliOp.from_operator(random_circulant_matrix)

            # Ensure ISA layout is defined
            try:
                isa_observable = observable.apply_layout(self.isa_ansatz.layout)
            except AttributeError:
                raise RuntimeError("Quantum preparation must be executed first.")

        
            # Initialize VQD algorithm
            vqd = VQD(
                ansatz=self.isa_ansatz,
                estimator=self.estimator,
                fidelity=self.fidelity,
                optimizer=COBYLA(maxiter=self.optimizer_iterations),
                k=self.eigenvalues,
                initial_point=self.x0,
                betas=betas,
                callback=callback
            )

            # Compute eigenvalues
            result = vqd.compute_eigenvalues(operator=isa_observable)
            result_numpy_kf = np.linalg.eigvals(random_circulant_matrix)
            print(result)
            
            # Store results
            for j, eigval in enumerate(result.eigenvalues):
                self.results[i, j] = eigval

            # Ensure unique entries before appending
            self.result_numpy_kf.append(np.array(result_numpy_kf))

        return self.results, self.result_numpy_kf


    def results_to_csv(self):
        df = pd.DataFrame(self.results)
        df.to_csv('VQD_MCS_taliadouros_et_al.csv')

        df = pd.DataFrame(self.result_numpy_kf)
        df.to_csv('Numpy-VQD_MCS_taliadouros_et_al.csv')
