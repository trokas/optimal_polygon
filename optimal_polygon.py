import math
import numpy as np


def _angle(u, v, w, d='+'):
    """
    Measures angle between points u, v and w in positive or negative direction

    Args:
        u (list): x and y coordinates
        v (list): x and y coordinates
        w (list): x and y coordinates
        d (str, optional): direction of angle '+' = counterclockwise

    Returns:
        float: angle in radians
    """
    vu = np.arctan2(u[1] - v[1], u[0] - v[0])
    vw = np.arctan2(w[1] - v[1], w[0] - v[0])
    phi = vw - vu
    if phi < 0:
        phi += 2 * np.pi
    if d == '-':
        phi = 2 * np.pi - phi
    return np.round(phi, 6)


def _intersect(A, B, C, D):
    """
    Returns intersection of lines AB and CD

    Args:
        A (list): x and y coordinates
        B (list): x and y coordinates
        C (list): x and y coordinates
        D (list): x and y coordinates

    Returns:
        tuple: x and y coordinates of the intersection
    """
    d = (B[0] - A[0]) * (D[1] - C[1]) - (D[0] - C[0]) * (B[1] - A[1])
    x = ((B[0] * A[1] - A[0] * B[1]) * (D[0] - C[0]) - (D[0] * C[1] - C[0] * D[1]) * (B[0] - A[0])) / d
    y = ((B[0] * A[1] - A[0] * B[1]) * (D[1] - C[1]) - (D[0] * C[1] - C[0] * D[1]) * (B[1] - A[1])) / d
    return (np.round(x, 6), np.round(y, 6))


def optimal_polygon(y, w=0.5, debug=False):
    """
    Constructs optimal polygon and returns pivot points.
    Based on 'An Optimal Algorithm for Approximating a. Piecewise Linear
    Function. HIROSHI IMAI and MASAO Iri.'

    Args:
        y (np.array or list): time series
        w (float, optional): shift used for tunnel

    Returns:
        np.array: pivot points
    """
    # Make sure that we use numpy array
    y = np.array(y)
    x = np.arange(len(y))

    # Initialization
    y = np.round(y, 6)
    p_plus = (x[0], y[0] + w)
    l_plus = (x[0], y[0] + w)
    r_plus = (x[1], y[1] + w)
    s_plus = {(x[0], y[0] + w): (x[1], y[1] + w)}
    t_plus = {(x[1], y[1] + w): (x[0], y[0] + w)}
    p_minus = (x[0], y[0] - w)
    l_minus = (x[0], y[0] - w)
    r_minus = (x[1], y[1] - w)
    s_minus = {(x[0], y[0] - w): (x[1], y[1] - w)}
    t_minus = {(x[1], y[1] - w): (x[0], y[0] - w)}
    q = []
    i = 2

    while i < len(y):
        # Updating CH_plus (convex hull) and CH_minus
        p = (x[i - 1], y[i - 1] + w)
        p_i_plus = (x[i], y[i] + w)
        while (p != p_plus) and _angle(p_i_plus, p, t_plus[p], '+') > np.pi:
            p = t_plus[p]
        s_plus[p] = p_i_plus
        t_plus[p_i_plus] = p

        p = (x[i - 1], y[i - 1] - w)
        p_i_minus = (x[i], y[i] - w)
        while (p != p_minus) and _angle(p_i_minus, p, t_minus[p], '-') > np.pi:
            p = t_minus[p]
        s_minus[p] = p_i_minus
        t_minus[p_i_minus] = p

        # Check if CH_plus and CH_minus intersect
        if _angle(p_i_plus, l_plus, r_minus, '+') < np.pi:
            q.append((_intersect(l_plus, r_minus, p_plus, p_minus), l_plus, r_minus, p_plus, p_minus))
            p_minus = r_minus
            p_plus = _intersect(l_plus, r_minus, (x[i - 1], y[i - 1] + w), p_i_plus)
            s_plus[p_plus] = p_i_plus
            t_plus[p_i_plus] = p_plus
            r_plus = p_i_plus
            r_minus = p_i_minus
            l_plus = p_plus
            l_minus = p_minus
            while _angle(l_minus, r_plus, s_minus[l_minus], '-') < np.pi:
                l_minus = s_minus[l_minus]
        elif _angle(p_i_minus, l_minus, r_plus, '-') < np.pi:
            q.append((_intersect(l_minus, r_plus, p_minus, p_plus), l_minus, r_plus, p_minus, p_plus))
            p_plus = r_plus
            p_minus = _intersect(l_minus, r_plus, (x[i - 1], y[i - 1] - w), p_i_minus)
            s_minus[p_minus] = p_i_minus
            t_minus[p_i_minus] = p_minus
            r_minus = p_i_minus
            r_plus = p_i_plus
            l_minus = p_minus
            l_plus = p_plus
            while _angle(l_plus, r_minus, s_plus[l_plus], '+') < np.pi:
                l_plus = s_plus[l_plus]
        else:
            # Updating the two seperating and supporting lines
            if _angle(p_i_plus, l_minus, r_plus, '+') < np.pi:
                r_plus = p_i_plus
                while _angle(p_i_plus, l_minus, s_minus[l_minus], '+') < np.pi:
                    l_minus = s_minus[l_minus]

            if _angle(p_i_minus, l_plus, r_minus, '-') < np.pi:
                r_minus = p_i_minus
                while _angle(p_i_minus, l_plus, s_plus[l_plus], '-') < np.pi:
                    l_plus = s_plus[l_plus]
        i += 1

    # Add last change point
    a = _intersect(l_plus, r_minus, p_plus, p_minus)
    b = _intersect(l_minus, r_plus, p_minus, p_plus)
    p = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    q.append((p, r_minus, r_plus, p_minus, p_plus))

    end_a = _intersect(p, r_plus, p_i_minus, p_i_plus)
    end_b = _intersect(p, r_minus, p_i_minus, p_i_plus)
    end = ((end_a[0] + end_b[0]) / 2, (end_a[1] + end_b[1]) / 2)
    q.append((end, (None, None), (None, None), p_i_minus, p_i_plus))

    if debug:
        return np.array(q)
    else:
        return np.array([o[0] for o in q])
