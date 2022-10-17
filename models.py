from utilities import (
    cubic_solver,
)

class first_order_spherical:
    def __init__(self, k=0.0):
        self.k1 = k

    def forward(self, xd, yd):
        # Expecting coordinates where center is (0,0)
        r2 = xd**2 + yd**2
        xc = xd * (1 + self.k1 * r2)
        yc = yd * (1 + self.k1 * r2)
        return (xc, yc)

    def reverse(self, xc, yc):
        rc = (xc**2 + yc**2)**0.5
        rd = cubic_solver(self.k1, 0, 1, -rc)
        xd = xc * (rd / rc)
        yd = yc * (rd / rc)
        return (xd, yd)

