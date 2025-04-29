import numpy as np

def f(x,y):
    return 0.5 * x**2 + 3.0 * y**2 + 0.356 * x * y

x = 1.0
y = 1.0

# now get first derivs
h = 1e-30
fx = np.imag(f(x+h*1j,y)-f(x,y)) / h
fy = np.imag(f(x,y+h*1j)-f(x,y))/ h
print(f"{fx=}\n{fy=}")

# now get second derivs
# has mouch lower truncation error than finite diff
# see https://ancs.eng.buffalo.edu/pdf/ancs_papers/2008/complex_step08.pdf
h = 1e-5
fxx = 2.0 / h**2 * (f(x,y) - np.real(f(x+1j*h,y)))
fyy = 2.0 / h**2 * (f(x,y) - np.real(f(x,y+1j*h)))
fxy = 1.0/h**2 * (f(x,y) - np.real(f(x+1j*h,y+1j*h))) - 0.5 * (fxx + fyy)
print(f"{fxx=}\n{fyy=}\n{fxy=}")

# versus finite diff second derivs
fx_FD = 1.0 / h**2 * (f(x+h,y) - 2 * f(x,y) + f(x-h,y))
print(F"{fx_FD=}")