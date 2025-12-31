import numpy as np
import sympy as sp
from integrators import eulode_phys, rk4sys_phys, velverlet_phys
import matplotlib.pyplot as plt
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
        self.T_sym = 0.5 * self.mass * (sp.Derivative(x, t)**2 + sp.Derivative(y, t)**2)
        self.V_sym = V_sym
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.acceleration, self.conserved = euler_lagrange_equations(self)

def select_particle():
    """
    Reads "particles.csv" and allows the user to select a particle.
    Returns a dictionary with the mass of the selected particle.
    """
    with open("particles.csv", "r") as file:
        reader = csv.DictReader(file)
        particles = list(reader) 

    while True:
        name = input("Select a particle: ").strip().lower()
        for row in particles:
            if row["particle"] == name:
                return row["particle"], float(row["mass"])
        print("Particle not found.")


def select_potential(k=1e-29, kx=1e-29, ky=2e-29, G=1e-11, M=1.0, epsilon=1e-6, alpha=1e-29):
    """
    Allows the user to select a potential function for the simulation.
    Returns a symbolic potential, with small values assigned to the constant coefficientes
    to keep accelerations within a numerically stable range for all particles.
    """
    potentials = {
        "1. Simple Harmonic Oscillator": 0.5 * k * (x**2 + y**2),
        "2. Anisotropic Harmonic Oscillator": 0.5 * (kx * x**2 + ky * y**2),
        "3. Keplerian": -G * M / sp.sqrt(x**2 + y**2 + epsilon**2),
        "4. Noncentral": alpha * x * y,
        "5. Free particle": 0,
    }

    print("Available potentials:")
    for name in potentials:
        print(name)

    while True:
        index = input("Select a potential (1-5): ").lstrip()
        if not index or len(index) > 1:
            print("Invalid selection. Try again.")
            continue
        for name, func in potentials.items():
            if index[0] == name[0]:
                return func
        print("Invalid selection. Try again.")


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


def euler_lagrange_equations(self):
    """
    Calculates a symbolic lagrangian expression (L=T-V), then calculates the 
    Euler-Lagrange equations for each coordinate and returns the acceleracion 
    array as numeric expressions.

    Addionality, analyzes the lagrangian dependence with respect to specific
    values for a further conservation study.
    """
    L_sym = self.T_sym - self.V_sym

    def noether():
        x2, y2, theta = sp.symbols('x y theta')
        L2 = L_sym.subs({x: x2, y: y2})
      # conserved = [E, px, py, Lz]
        conserved = [
            sp.simplify(sp.diff(L2, t)) == 0,
            sp.simplify(sp.diff(L2, x2)) == 0,
            sp.simplify(sp.diff(L2, y2)) == 0,
            sp.simplify(sp.diff(sp.simplify(L2.subs({x2: x2*sp.cos(theta) - y2*sp.sin(theta),
                                                     y2: x2*sp.sin(theta) + y2*sp.cos(theta)})), theta).subs(theta, 0)) == 0
            ]
        return np.array([bool(j) for j in conserved], dtype=bool)

    ELx = sp.diff(sp.diff(L_sym, sp.diff(x,t)), t) - sp.diff(L_sym, x)
    ELy = sp.diff(sp.diff(L_sym, sp.diff(y,t)), t) - sp.diff(L_sym, y)

    acc_x = sp.solve(ELx, ax)[0]
    acc_y = sp.solve(ELy, ay)[0]
    acc = sp.lambdify((x, y, vx, vy), [acc_x, acc_y], "numpy")
    def acceleration(pos, vel):
        return np.array(acc(pos[0], pos[1], vel[0], vel[1]), dtype=float)

    return acceleration, noether


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
    
    px_e, py_e = particle.mass * vel_e[:,0], particle.mass * vel_e[:,1]
    px_rk, py_rk = particle.mass * vel_rk[:,0], particle.mass * vel_rk[:,1]
    px_vv, py_vv = particle.mass * vel_vv[:,0], particle.mass * vel_vv[:,1]

    L_e = particle.mass * (pos_e[:,0]*vel_e[:,1] - pos_e[:,1]*vel_e[:,0])
    L_rk = particle.mass * (pos_rk[:,0]*vel_rk[:,1] - pos_rk[:,1]*vel_rk[:,0])
    L_vv = particle.mass * (pos_vv[:,0]*vel_vv[:,1] - pos_vv[:,1]*vel_vv[:,0])

    results = {
        "Euler-Cromer":    {"t": tp_e, "pos": pos_e, "vel": vel_e, "T": T_e,  
                            "V": V_e,  "E": E_e, "px": px_e, "py": py_e, "Lz": L_e},
        "Runge-Kutta 4":   {"t": tp_rk, "pos": pos_rk, "vel": vel_rk, "T": T_rk,
                            "V": V_rk, "E": E_rk, "px": px_rk, "py": py_rk, "Lz": L_rk},
        "Velocity Verlet": {"t": tp_vv, "pos": pos_vv, "vel": vel_vv, "T": T_vv,
                            "V": V_vv, "E": E_vv, "px": px_vv, "py": py_vv, "Lz": L_vv}
    }

    return results


def graphs(dict, title, xdata, ydata, xcol, ycol, xlabel, ylabel, conserved):
    """
    Generates a generic 2x2 plot template to compare the results obtained from the 
    different numerical integration methods.

    The first three panels show the evolution of the selected quantities for each 
    integration method, while the fourth one is designed for error analysis. If the 
    quantity is conserved according to Noether's theorem, the last panel will show 
    the conservation error with respect to its initial value. Otherwise, it shows the 
    deviation error between methods over time, displaying their absolute error and 
    its mean for a numerical accuracy study.
    """
    methods = list(dict.keys())
    colors = ["#FFD000", "#FF0000", "#222ED5",
              "#AB8901", "#9B0000", "#00058F",
              "#FFE375", "#FF7A7A", "#368BFB",
              "#FF6F08", "#00C500", "#8900D3"]
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()

    try:
        iter(xdata)
    except TypeError:
        xdata = [xdata]
    
    try:
        iter(ydata)
    except TypeError:
        ydata = [ydata]
    
    for i, method in enumerate(methods):
        if len(ydata) == 1:
            x = dict[method][xdata[0]]
            y = dict[method][ydata[0]]
            if x.ndim != 1: x = x[:, xcol]
            if y.ndim != 1: y = y[:, ycol]
            axs[i].plot(x, y, color=colors[i])
        else: 
            for j, (xkey, ykey) in enumerate(zip(xdata, ydata)):
                x = dict[method][xkey]
                y = dict[method][ykey]
                xj = xcol[j] if len(xcol) > 1 else xcol[0]
                yj = ycol[j] if len(ycol) > 1 else ycol[0]
                if x.ndim > 1: x = x[:, xj]
                if y.ndim > 1: y = y[:, yj]
                axs[i].plot(x, y, color=colors[3 * j + i], label=ykey)
        
        axs[i].set_xlabel(xlabel)
        axs[i].set_ylabel(ylabel)
        axs[i].set_title(method)
        axs[i].grid(True)
        if len(ydata) > 1:
            axs[i].legend()


    t = dict[methods[0]]["t"]

    if conserved:
        axs[3].set_title(f"Conservation error in {ydata[0]}")
        for k, method in enumerate(methods):
            y = dict[method][ydata[0]]
            if y.ndim > 1: y = y[:, ycol[0]]
            err = y - y[0]
            axs[3].plot(t, err, color=colors[k], label={method})
    else:
        axs[3].set_title(f"Method-to-method deviation")
        for i in range(len(methods)):
           for j in range(i+1, len(methods)):
                method1 = dict[methods[i]][ydata[0]]
                method2 = dict[methods[j]][ydata[0]]

                if method1.ndim != 1: method1 = method1[:, ycol]
                if method2.ndim != 1: method2 = method2[:, ycol]

                abs_err = np.abs(method1 - method2)
                mean_abs_err = np.mean(abs_err)
                axs[3].plot(t, abs_err, color=colors[i + j + 8], label=rf"{methods[i]} vs {methods[j]}: $\bar{{\epsilon}}$ = {mean_abs_err:.6e}" )

    axs[3].grid(True)
    axs[3].set_xlabel("t")
    axs[3].set_ylabel("Error")
    axs[3].legend()

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

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
        sys.exit("Introduce 'python project.py h tf' or 'python project.py h t0 tf'")
    tspan = (t0, tf)
    
    p_name, p_mass = select_particle()
    V_sym = select_potential()

    while True:
        try:
            r0 = input("Introduce r0 [x0, y0]: ")
            r0 = r0.replace(" ", "").replace("[", "").replace("]", "")
            r0 = list(map(float, r0.split(",")))
            if len(r0) != 2:
                print("Introduce exactly two values.")
                continue
            break
        except ValueError:
            print("Invalid format.")
            continue

    while True:
        try:
            v0 = input("Introduce v0 [vx0, vy0]: ")
            v0 = v0.replace(" ", "").replace("[", "").replace("]", "")
            v0 = list(map(float, v0.split(",")))
            if len(v0) != 2:
                print("Introduce exactly two values.")
                continue
            break
        except ValueError:
            print("Invalid format.")
            continue

    p = Particle(
        name = p_name,
        mass = p_mass,
        V_sym = V_sym,
        position = r0,
        velocity = v0,
    )

    results = run_simulation(p, tspan, h)
    quantities = p.conserved()
    
    titles = ["Trajectory y(x)", "Energies over time", "Phase-space", "Angular momentum over time"]
    Exdata, Eydata = ["t", "t", "t"], ["E", "V", "T"]
    p_xdata, p_ydata, p_xcol, p_ycol = ["pos", "pos"], ["px", "py"], [0, 1], [0, 1] 

    graphs(results, titles[0], ["pos"], ["pos"], [0], [1], "x", "y(x)", False)
    graphs(results, titles[1], Exdata, Eydata, [0], [0], "t", "Energies", quantities[0])
    graphs(results, titles[2], p_xdata, p_ydata, p_xcol, p_ycol, "x, y", "px, py", quantities[1])
    graphs(results, titles[3], ["t"], ["Lz"], [0], [0], "t", "Lz", quantities[3])

if __name__ == "__main__":
    main()