import numpy as np
import pso

def beale(x):
	X = x[0]
	Y = x[1]
	f = (1.5 - X + (X*Y))**2
	f += (2.25 - X + (X*Y**2))**2
	f += (2.625 - X + (X*Y**3))**2
	return f

def easom(x):
	X = x[0]
	Y = x[1]
	f = -1*np.cos(X)*np.cos(Y)*np.exp(-((X-np.pi)**2) - ((Y-np.pi)**2))
	return f

print(easom(np.array([np.pi, np.pi])))
p = pso.PSO(npop=50)
best = p.optimize(easom, (2,), n_iter=100, min_pos = np.array([-5.0,-5.0]), max_pos= np.array([5.0,5.0]))
print(best)
print(easom(best))