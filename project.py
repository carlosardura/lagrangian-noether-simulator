import numpy as np
import sympy as sp
from integrators import eulode_phys, rk4sys_phys, velverlet_phys
import csv
import sys

# ---------- classes and functions ----------

t = sp.symbols('t')
x, y = sp.Function('x')(t), sp.Function('y')(t)
vx, vy = sp.diff(x, t), sp.diff(y, t)
ax, ay = sp.diff(x, t, 2), sp.diff(y, t, 2)

class Particle:
    """
    Represents a particle in 2D space with several physical properties,
    including position, velocity and acceleration arrays, and symbolic
    energy functions.
    """
    def __init__(self, name, mass, V_sym, position=None, velocity=None):
        self.name = name
        self.mass = mass
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)

        self.T_sym = 0.5 * self.mass * (vx**2 + vy**2)
        self.V_sym = V_sym

        self.acceleration = euler_lagrange_equations(self)


def select_particle(p_name):
    """
    Reads "particles.csv" and allows the user to select a particle.
    Returns a dictionary with the mass of the selected particle.
    """
    with open("particles.csv", "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row["name"] == p_name:
                return {
                    "name": row["name"],
                    "mass": float(row["mass"]),
                }
            else:
                raise ValueError("Particle not found")


def select_potential(k=1.0, kx=1.0, ky=2.0, G=1.0, M=1.0, epsilon=1e-6, alpha=1.0):
    """
    Allows the user to select a potential function for the simulation.
    Returns a symbolic potential, with k, G and M equal to 1 for simplicity.
    """
    potentials = {
        "1. Simple Harmonic Oscillator": 0.5 * k * (x**2 + y**2),
        "2. Anisotropic Harmonic Oscillator": 0.5 * (kx * x ** 2 + ky * y ** 2),
        "3. Keplerian": -G * M / sp.sqrt(x**2 + y**2 + epsilon**2),
        "4. Noncentral": alpha * x * y,
    }

    print("Available potentials:")
    for name in potentials:
        print(name)

    while True:
        index = input("Select a potential (1-4): ").lstrip()
        if not index:
            print("Invalid selection. Try again.")
            continue
        for name, func in potentials.items():
            if index[0] == name[0]:
                return func
        print("Invalid selection. Try again.")


def run_simulation(particle, tspan, h):
    """
    Given several integrators, simulates the particle's motion over time
    and each energy inside a time interval. Returns (len(tp), dim) arrays.
    """
    tp_e, pos_e, vel_e = eulode_phys(particle.acceleration, tspan, particle.position, particle.velocity, h)
    tp_rk, pos_rk, vel_rk = rk4sys_phys(particle.acceleration, tspan, particle.position, particle.velocity, h)
    tp_vv, pos_vv, vel_vv = velverlet_phys(particle.acceleration, tspan, particle.position, particle.velocity, h)

    T_e, V_e, E_e = energies_over_time(particle, pos_e, vel_e)
    T_rk, V_rk, E_rk = energies_over_time(particle, pos_rk, vel_rk)
    T_vv, V_vv, E_vv = energies_over_time(particle, pos_vv, vel_vv)

    results = {
        "Euler-Cromer":    {"t": tp_e, "pos": pos_e, "vel": vel_e, "T": T_e, "V": V_e, "E": E_e},
        "RK4":             {"t": tp_rk, "pos": pos_rk, "vel": vel_rk, "T": T_rk, "V": V_rk, "E": E_rk},
        "Velocity-Verlet": {"t": tp_vv, "pos": pos_vv, "vel": vel_vv, "T": T_vv, "V": V_vv, "E": E_vv},
    }

    return results


def euler_lagrange_equations(self):
    """
    Calculates a symbolic lagrangian expression (L=T-V), then calculates the 
    Euler-Lagrange equations for each coordinate and returns the acceleracion 
    array as numeric expressions.
    """
    L_sym = self.T_sym - self.V_sym

    ELx = sp.diff(sp.diff(L_sym, vx), t) - sp.diff(L_sym, x)
    ELy = sp.diff(sp.diff(L_sym, vy), t) - sp.diff(L_sym, y)
    
    acc_x = sp.lambdify((x, y, vx, vy), sp.solve(ELx, ax)[0], "numpy")
    acc_y = sp.lambdify((x, y, vx, vy), sp.solve(ELy, ay)[0], "numpy")

    return np.array([acc_x, acc_y])


def energies_over_time(particle, pos_array, vel_array):
    """
    Turns the symbolic energy functions into numeric expressions, then takes 
    the position and velocity arrays obtained after using the integrators
    and returns the different energies at each point in the given time interval.
    """
    T = sp.lambdify((vx, vy), particle.T_sym, "numpy")
    V = sp.lambdify((x, y), particle.V_sym, "numpy")
    n = len(pos_array)

    T_array = np.zeros(n)
    V_array = np.zeros(n)
    E_array = np.zeros(n)

    T_array = T(vel_array[:,0], vel_array[:,1])
    V_array = V(pos_array[:,0], pos_array[:,1])
    E_array = T_array + V_array

    return T_array, V_array, E_array

# ---------- main function ----------

def main():
    h = float(sys.argv[1])
    if len(sys.argv) == 3:
        t0 = 0.0
        tf = float(sys.argv[2])
    elif len(sys.argv) == 4:
        t0 = float(sys.argv[2])
        tf = float(sys.argv[3])
    else:
        sys.exit("Introduce python project.py h tf or python project.py h t0 tf")
    tspan = (t0, tf)

    p_name = input("Select a particle: ").strip().lower()
    p_data = select_particle(p_name)

    V_sym = select_potential()

    try:
        r0 = list(map(float, input("Introduce r0: ").replace(" ", "").split(",")))
        v0 = list(map(float, input("Introduce v0: ").replace(" ", "").split(",")))
    except ValueError:
        sys.exit("Invalid format.")
    
    if len(r0) != 2 or len(v0) != 2:
        sys.exit("Introduce exactly two values for position and velocity.")

    p = Particle(
        name = p_data["name"],
        mass = p_data["mass"],
        position = r0,
        velocity = v0,
        V_sym = V_sym,
    )

    results = run_simulation(p, tspan, h)


if __name__ == "__main__":
    main()