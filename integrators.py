import numpy as np

def eulode_phys(acc, tspan, pos0, vel0, h):
    
    t0, tf = tspan[0], tspan[1]
    if tf <= t0:
        raise ValueError("tspan must be strictly increasing")

    tp = np.arange(t0, tf + h, h) # same rounding problem as in "integrators.m"
    
    if tp[-1] < tf:
        tp = np.append(tp, tf)
    elif tp[-1] > tf:
        tp[-1] = tf
    n_out = len(tp)

    x = np.asarray(pos0, dtype=float).reshape(-1)
    v = np.asarray(vel0, dtype=float).reshape(-1)
    dim = x.size

    pos = np.zeros((n_out, dim))
    vel = np.zeros((n_out, dim))
    pos[0, :] = x
    vel[0, :] = v

    tt = t0

    for i in range(n_out - 1):
        t_end = tp[i + 1]
        while tt < t_end:
            hh = min(h, t_end - tt)
            v = v + hh * acc(x, v)
            x = x + hh * v
            tt += hh
        pos[i + 1, :] = x
        vel[i + 1, :] = v

    return tp, pos, vel


def rk4sys_phys(acc, tspan, pos0, vel0, h):

    t0, tf = tspan[0], tspan[-1]
    if tf <= t0:
        raise ValueError("tspan must be strictly increasing")

    if len(tspan) == 2:
        tp = np.arange(t0, tf + h, h)
        if tp[-1] < tf:
            tp = np.append(tp, tf)
        elif tp[-1] > tf:
            tp[-1] = tf
    else:
        tp = np.asarray(tspan, dtype=float)

    n_out = len(tp)
    x = np.asarray(pos0, dtype=float).reshape(-1)
    v = np.asarray(vel0, dtype=float).reshape(-1)
    dim = x.size

    pos = np.zeros((n_out, dim))
    vel = np.zeros((n_out, dim))
    pos[0, :] = x
    vel[0, :] = v

    tt = t0
    n = 0

    for n in range(n_out - 1):
        t_target = tp[n + 1]
        while tt < t_target:
            hh = min(h, t_target - tt)

            k1x = v
            k1v = acc(x, v)

            k2x = v + hh/2 * k1v
            k2v = acc(x + hh/2 * k1x, v + hh/2 * k1v)

            k3x = v + hh/2 * k2v
            k3v = acc(x + hh/2 * k2x, v + hh/2 * k2v)

            k4x = v + hh * k3v
            k4v = acc(x + hh * k3x, v + hh * k3v)

            x = x + hh/6 * (k1x + 2*k2x + 2*k3x + k4x)
            v = v + hh/6 * (k1v + 2*k2v + 2*k3v + k4v)
            tt += hh

        pos[n+1, :] = x
        vel[n+1, :] = v

    return tp, pos, vel


def velverlet_phys(acc, tspan, pos0, vel0, h):
    t0, tf = tspan[0], tspan[1]
    if tf <= t0:
        raise ValueError("tspan must be strictly increasing")

    tp = np.arange(t0, tf + h, h)
    if tp[-1] < tf:
        tp = np.append(tp, tf)
    elif tp[-1] > tf:
        tp[-1] = tf

    n_out = len(tp)
    x = np.asarray(pos0, dtype=float).reshape(-1)
    v = np.asarray(vel0, dtype=float).reshape(-1)
    dim = x.size

    pos = np.zeros((n_out, dim))
    vel = np.zeros((n_out, dim))
    pos[0, :] = x
    vel[0, :] = v

    tt = t0

    for i in range(n_out - 1):
        t_end = tp[i + 1]
        while tt < t_end:
            hh = min(h, t_end - tt)

            a0 = acc(x, v)
            x_new = x + v*hh + 0.5*a0*hh**2
            a_new = acc(x_new, v)
            v_new = v + 0.5*(a0 + a_new)*hh

            x = x_new
            v = v_new
            tt += hh

        pos[i + 1, :] = x
        vel[i + 1, :] = v

    return tp, pos, vel