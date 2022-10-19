from utilities import (
    cubic_solver,
)

# TODO: make sure this works with images with aspect ratio < 1.
class first_order_spherical:
    def __init__(self, k=0.0, aspect_ratio=1.0):
        # aspect_ratio is ratio width:height
        self.k1 = k
        self.aspect = aspect_ratio
        if self.k1 <= 0:
            # Scale according to r^2 at the corners
            self.rescale_factor =  1 / (1 + self.k1 * ((0.5 * self.aspect)**2 + 0.5**2))
        elif self.k1 > 0:
            # Scale according to r^2 at the top and bottom
            self.rescale_factor = 1 / (1 + self.k1 * ((0.5)**2 + (0 * self.aspect)**2))


    def forward(self, xd, yd):
        # Expecting coordinates where center is (0,0), one of the corners is (1,1).
        xd *= self.aspect
        r2 = xd**2 + yd**2
        xc = xd * (1 + self.k1 * r2) * self.rescale_factor
        yc = yd * (1 + self.k1 * r2) * self.rescale_factor
        xc /= self.aspect
        return (xc, yc)

    def reverse(self, xc, yc):
        xc, yc = xc * self.aspect / self.rescale_factor, yc / self.rescale_factor
        rc = (xc**2 + yc**2)**0.5
        rd = cubic_solver(self.k1, 0, 1, -rc)
        xd = xc * (rd / rc)
        yd = yc * (rd / rc)
        xd /= self.aspect
        return (xd, yd)

class parabola:
    def __init__(self, A, B, C):
        self.a = A
        self.b = B
        self.c = C

    def forward(self, x: float) -> float:
        return self.a * (x**2) + self.b * x + self.c

    def estimate_k(self) -> float:
        # return -self.a / self.c
        return -self.a / (self.c * (3 * self.a * self.c + 3 * (self.b**2) + 1))




