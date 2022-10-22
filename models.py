from solvers import (
    cubic_solver,
)

# TODO: make sure this works with images with aspect ratio < 1.
class first_order_spherical:
    def __init__(self, k=0.0, aspect_ratio=1.0, rescale=True):
        # aspect_ratio is ratio width:height
        self.k1 = k
        self.aspect = aspect_ratio
        # if self.k1 <= 0:
        #     # Scale according to r^2 at the top and bottom
        #     r2 = (0 * self.aspect)**2 + (0.5)**2
        # elif self.k1 > 0:
        #     # Scale according to r^2 at the corners
        #     r2 = (0.5 * self.aspect)**2 + 0.5**2

        # r2 = cubic_solver(self.k1, 0, 1, -(r2**0.5))**2
        # self.rescale_factor =  1/(1 + self.k1 * (r2))
        # if not rescale:
        self.rescale_factor = 1.0
        if rescale:
            self._rescale_calculator()


    def forward(self, xd, yd):
        # Expecting coordinates where center is (0,0), one of the corners is (0.5,0.5).
        xd, yd = xd * self.aspect, yd
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

    def _rescale_calculator(self):
        p1 = self.forward(0.5, 0.5)
        p2 = self.forward(0.0, 0.5)
        r1 = ((0.5 * self.aspect)**2 + 0.5**2)**0.5 / ((p1[0] * self.aspect)**2 + p1[1]**2)**0.5
        r2 = ((0.0 * self.aspect)**2 + 0.5**2)**0.5 / ((p2[0] * self.aspect)**2 + p2[1]**2)**0.5
        self.rescale_factor = min(r1, r2)
        print(f"selected rescale factor {self.rescale_factor}")

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




