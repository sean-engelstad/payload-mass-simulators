class Material:
    def __init__(self, E, rho, nu, G):
        self.E = E
        self.rho = rho
        self.nu = nu
        self.G = G
    
    @classmethod
    def aluminum(cls):
        E = 70e9; nu = 0.3; rho = 2.7e3
        G = E / 2.0 / (1+nu)
        return cls(E, rho, nu, G)