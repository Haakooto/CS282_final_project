import matplotlib.pyplot as plt
import matplotlib
import numpy as np

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 22}
matplotlib.rc('font', **font)
plt.rcParams["figure.figsize"] = (10,6)

x = lambda n, b=0.5: np.linspace(-b, b, n)
y = lambda n, b=2: np.random.uniform(-b, b, n)
p = lambda n, b1=0.5, b2=2: x(n, b1) * y(n, b2)
d = plt.hist(p(10000), bins=30, density=True)
plt.xlabel("x-value")
plt.ylabel("Relative frequency")
plt.title("Distribution of points along x-axis")
plt.savefig("figures/point_dist_sig")