from typing import List, Tuple

def cubic_solver(a=0, b=0, c=0, d=0):
    # Finds roots of polynomial ax^3 + 0x^2 + cx + d = 0
    assert b == 0, "cubic solver only works with b=0"
    if a == 0:
        return -d / c
    a, b, c, d = float(a), float(b), float(c), float(d)

    # Cardano's formula
    c /= a
    d /= a
    Q = c / 3
    R = -d / 2
    delta = Q**3 + R**2
    if delta > 0:
        # One root, not sure why the -k case goes here, seems to not work correctly for k=-0.3559
        # TODO.
        C = (R + (delta**0.5))**(1/3)
        out = abs(C - (Q / C))
        return out
    else:
        S: complex = (R + (delta**0.5))**(1/3)
        T: complex = (R - (delta**0.5))**(1/3)
        # out1: complex = S + T
        # out2: complex = -(S + T) / 2 + (S - T) * 1j * (3**0.5) / 2
        out3: complex = -(S + T) / 2 - (S - T) * 1j * (3**0.5) / 2
        return out3.real

def fit_parabola_horizontal_line(points: List[Tuple[float]]) -> Tuple[float, float, float]:
    # Returns A, B, C for which:
    # y = Ax**2 + Bx + C
    x1, x2, x3 = points[0][0], points[1][0], points[2][0]
    y1, y2, y3 = points[0][1], points[1][1], points[2][1]
    denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
    if denom == 0:
        A = 0
        B = (y2 - y1) / (x2 - x1)
        C = (B * -x2) + y2
    else:
        A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
        B = (x3**2 * (y1 - y2) + x2**2 * (y3 - y1) + x1**2 * (y2 - y3)) / denom
        C = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom
    return (A, B, C)

def fit_parabola_vertical_line(points: List[Tuple[float]]) -> Tuple[float, float, float]:
    # Returns A, B, C for which:
    # x = Ay**2 + By + C
    inverted_points = [(y, x) for x,y in points]
    return fit_parabola_horizontal_line(inverted_points)