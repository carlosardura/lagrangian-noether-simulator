# Lagrangian Mechanics Simulator

This repository provides a modular and scalable simulator based on Python and MATLAB for classical mechanics using the Lagrangian formulation via the **Euler-Lagrange equations**. Designed for the study of particle dynamics, it combines symbolic derivation, numerical integration and pedagogical visualization.

It allows the symbolic calculation of equations of motion for specific potentials of interest, whose numerical solutions are obtained using integration methods with different symplectic properties: **Euler-Cromer**, **Runge-Kutta 4** and **Velocity Verlet**. These methods automatically provide particle trajectories and velocities over a given time interval, allowing a direct comparison of their **short-term and long-term accuracy** and stability in Lagrangian dynamics.

The main goal of the project is to provide an interactive tool for illustrating the connection between a physical system’s symmetries and the corresponding conservation laws, as described by **Noether’s theorem**. By analyzing particle motion under potentials with temporal, translational or rotational invariances, the simulator demonstrates how energy, linear momentum or angular momentum are conserved, providing **visual and numerical evidence** of the theorem.

Both the main script and the integrators module are designed for scalability, allowing future additions such as multiple interacting particles, 3D generalization and the implementation of arbitrary potentials.