import numpy as np
import matplotlib.pyplot as plt

def analyze_truss(force_angle = 240,  roller_angle = 30, force_magnitude=2000.0):

    force_node = 'D'
    fixed_nodes = ['A', 'B']

    materials = {
        'steel': {'E': 30e6},
        'aluminum': {'E': 11e6}
    }

    element_properties = {
        1: {'material': 'steel', 'diameter': 0.5},
        2: {'material': 'steel', 'diameter': 0.5},
        3: {'material': 'aluminum', 'diameter': 0.4},
        4: {'material': 'aluminum', 'diameter': 0.4},
        5: {'material': 'steel', 'diameter': 0.5}
    }

    elements = {
        1: ['A', 'B'],
        2: ['A', 'C'],
        3: ['A', 'D'],
        4: ['B', 'C'],
        5: ['C', 'D']
    }

    nodes = {
        'A': np.array([0, 0]),
        'B': np.array([0, 10]),
        'C': np.array([12, 6]),
        'D': np.array([12, 0]),
    }

    
    roller_angle = np.deg2rad(roller_angle)
    force_angle = np.deg2rad(force_angle)

    external_forces = {
        force_node : np.array([np.cos(force_angle)* force_magnitude, np.sin(force_angle)*force_magnitude])  
    }


    element_lengths = {}
    element_angles = {}
    for element, (node1, node2) in elements.items():
        delta = nodes[node2] - nodes[node1]
        element_lengths[element] = np.linalg.norm(delta)
        element_angles[element] = np.arctan2(delta[1], delta[0])


    # Calculate element stiffness matrices
    element_stiffness_matrices = {}
    for element, (node1, node2) in elements.items():
        L = element_lengths[element]
        theta = element_angles[element]
        E = materials[element_properties[element]['material']]['E']
        A = np.pi * (element_properties[element]['diameter'] / 2) ** 2
        
        # Handle numerical precision issues
        if np.isclose(theta, 0):
            c, s = 1.0, 0.0
        elif np.isclose(theta, np.pi/2):
            c, s = 0.0, 1.0
        elif np.isclose(theta, np.pi):
            c, s = -1.0, 0.0
        elif np.isclose(theta, 3*np.pi/2):
            c, s = 0.0, -1.0
        else:
            c = np.cos(theta)
            s = np.sin(theta)

        k = E * A / L * np.array([
            [c**2, c*s, -c**2, -c*s],
            [c*s, s**2, -c*s, -s**2],
            [-c**2, -c*s, c**2, c*s],
            [-c*s, -s**2, c*s, s**2]
        ])
        element_stiffness_matrices[element] = k

    # Assemble global stiffness matrix
    num_nodes = len(nodes)
    degrees_of_freedom = 2 * num_nodes
    K_global = np.zeros((degrees_of_freedom, degrees_of_freedom))

    for element, (node1, node2) in elements.items():
        dof1 = 2 * list(nodes).index(node1)
        dof2 = 2 * list(nodes).index(node2)
        dofs = np.array([dof1, dof1 + 1, dof2, dof2 + 1])
        K_global[np.ix_(dofs, dofs)] += element_stiffness_matrices[element]

    # Apply boundary conditions with roller support
   
    fixed_dofs = [2, 3]  # Fix y-translation and x-translation at node B

    # Calculate the angle perpendicular to the slider
    constraint_angle = roller_angle + np.pi/2 

    # Construct the transformation matrix for the constraint
    T = np.array([
        [np.cos(constraint_angle), np.sin(constraint_angle)],
        [-np.sin(constraint_angle), np.cos(constraint_angle)]
    ])

    # Apply the transformation to the first two rows and columns of K_global
    K_global[:2, :2] = T @ K_global[:2, :2] @ T.T
    free_dofs = np.setdiff1d(np.arange(degrees_of_freedom), fixed_dofs)


    # Create reduced stiffness matrix and force vector
    K_reduced = K_global[np.ix_(free_dofs, free_dofs)]
    F_global = np.zeros(degrees_of_freedom)
    for node, force in external_forces.items():
        dof = 2 * list(nodes).index(node)
        F_global[dof:dof + 2] = force
    F_reduced = F_global[free_dofs]

    # Solve for nodal displacements
    U_reduced = np.linalg.solve(K_reduced, F_reduced)

    # Calculate reaction forces
    U_global = np.zeros(degrees_of_freedom)
    U_global[free_dofs] = U_reduced
    R_global = K_global @ U_global - F_global
    reaction_forces = {node: R_global[2 * i:2 * i + 2] for i, node in enumerate(nodes)}

    # Calculate element stresses
    element_stresses = {}
    for element, (node1, node2) in elements.items():
        L = element_lengths[element]
        theta = element_angles[element]
        E = materials[element_properties[element]['material']]['E']
        dof1 = 2 * list(nodes).index(node1)
        dof2 = 2 * list(nodes).index(node2)
        dofs = np.array([dof1, dof1 + 1, dof2, dof2 + 1])
        u_element = U_global[dofs]
        c = np.cos(theta)
        s = np.sin(theta)
        stress = E / L * np.array([-c, -s, c, s]) @ u_element
        element_stresses[element] = stress


    
    deformed_nodes = {node: coords + U_global[2*i:2*i+2] for i, (node, coords) in enumerate(nodes.items())}

    return U_global, reaction_forces, element_stresses, nodes, deformed_nodes 

