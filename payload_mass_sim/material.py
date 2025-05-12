class Material:
    def __init__(self, E, rho, nu, G, k_s, ys):
        self.E = E
        self.rho = rho
        self.nu = nu
        self.G = G
        self.k_s = k_s  # shear correction factor
        self.ys = ys # yield stress
    
    @classmethod
    def aluminum(cls):
        E = 70e9; nu = 0.3; rho = 2.7e3
        G = E / 2.0 / (1+nu)

        # Shear correction factor:
        k_s = 5/6
        ys = 11e6
        return cls(E, rho, nu, G, k_s, ys)