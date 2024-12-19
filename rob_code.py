# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 12:46:56 2024

@author: rober
"""

import numpy as np
import matplotlib.pyplot as plt

def analyze_truss(nodes, elements, materials, element_properties,
                  external_forces, fixed_nodes, roller_angle=0,
                  force_angle=270):
    """
    Analyzes a 2D truss structure under given conditions.

    Args:
        nodes (dict): Node coordinates as {'A': np.array([x, y]), ...}.
        elements (dict): Element connectivity as {1: ['A', 'B'], ...}.
        materials (dict): Material properties as {'steel': {'E': value, 'nu': value}, ...}.
        element_properties (dict): Element properties as {1: {'material': 'steel', 'diameter': value}, ...}.
        external_forces (dict): External forces as {'C': np.array([Fx, Fy]), ...}.
        fixed_nodes (list): List of fixed nodes, e.g., ['A', 'B'].
        roller_angle (float): Angle of the roller support in degrees (default: 0).
        force_angle (float): Angle of the applied external force in degrees (default: 270).

    Returns:
        tuple: Nodal displacements, reaction forces, element stresses, and plot objects.
    """

    # Convert angles to radians
    # roller_angle = np.deg2rad(roller_angle)
    # force_angle = np.deg2rad(force_angle)

    # Calculate element lengths and angles
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
    fixed_dofs = []
    for node in fixed_nodes:
        dof = 2 * list(nodes).index(node)
        fixed_dofs.append(dof)  # Fix only x-translation for roller support
        if node != fixed_nodes[0]:  # Fix both DOFs for other fixed nodes
            fixed_dofs.append(dof + 1)
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

    # Visualize the truss
    fig, ax = plt.subplots()
    for element, (node1, node2) in elements.items():
        x1, y1 = nodes[node1]
        x2, y2 = nodes[node2]
        ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2)

    # Plot undeformed truss
    for node, coords in nodes.items():
        ax.plot(coords[0], coords[1], 'bo', markersize=8)
        ax.text(coords[0] + 0.2, coords[1] + 0.2, node, fontsize=12)

    # Plot deformed truss
    deformed_nodes = {node: coords + U_global[2*i:2*i+2] for i, (node, coords) in enumerate(nodes.items())}
    for element, (node1, node2) in elements.items():
        x1, y1 = deformed_nodes[node1]
        x2, y2 = deformed_nodes[node2]
        ax.plot([x1, x2], [y1, y2], 'r--', linewidth=2)

    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Truss Analysis')

    return U_global, reaction_forces, element_stresses, (fig, ax)

# Example usage with variable conditions
nodes = {
    'A': np.array([0, 0]),
    'B': np.array([0, 10]),
    'C': np.array([12, 6]),
    'D': np.array([12, 0])
}

elements = {
    1: ['A', 'B'],
    2: ['A', 'C'],
    3: ['A', 'D'],
    4: ['B', 'C'],
    5: ['C', 'D']
}

materials = {
    'steel': {'E': 30e6, 'nu': 0.3},  # Steel
    'aluminum': {'E': 11e6, 'nu': 0.33}  # Aluminum
}

element_properties = {
    1: {'material': 'steel', 'diameter': 0.5},
    2: {'material': 'aluminum', 'diameter': 0.4},
    3: {'material': 'steel', 'diameter': 0.5},
    4: {'material': 'aluminum', 'diameter': 0.4},
    5: {'material': 'steel', 'diameter': 0.5}
}

external_forces = {
    'D': np.array([0, -2000])  # Force at node C
}

U, R, S, plots = analyze_truss(nodes, elements, materials, element_properties,
                                external_forces, ['A', 'B'], roller_angle=30, force_angle=135)

# Print results
print("Nodal Displacements:")
for i, node in enumerate(nodes):
    print(f"{node}: {U[2*i:2*i+2]}")
print("\nReaction Forces:")
for node, force in R.items():
    print(f"{node}: {force}")
print("\nElement Stresses:")
for element, stress in S.items():
    print(f"Element {element}: {stress}")

# Save the plot to a file
figure, axis = plots  # Unpack the Matplotlib figure and axis
filename = "truss_analysis_plot.png"  # Desired file name
figure.savefig(filename, dpi=300)  # Save the figure with high resolution (300 DPI)
print(f"\nPlot saved to: {filename}")
plots[0].show()
