import numpy as np

class Particle:
	def __init__(self,pos,phi_1,phi_2,damp=0.99):
		self.loc = pos
		self.pb = pos
		self.velocity = np.zeros(pos.shape)
		self.damp = damp
		phi = phi_1+phi_2
		d = (phi*phi) - (4*phi)
		d = d**(.5)
		chi = 2/abs(2-phi-d)
		self.c1 = phi_1*chi
		self.c2 = phi_2*chi
		self.w = chi

	def move(self, gb,func, vel_max):
		self.velocity = self.velocity*self.w
		self.velocity += np.random.uniform()*(self.pb - self.loc)*self.c1
		self.velocity += np.random.uniform()*(gb - self.loc)*self.c2
		vel_min = -1 * vel_max
		self.velocity = np.maximum(np.minimum(self.velocity,vel_max),vel_min)
		self.loc += self.velocity
		self.w = self.w * self.damp
		if(func(self.pb) > func(self.loc)):
			self.pb = self.loc

class PSO:
	def __init__(self,npop,phi_1=2.8,phi_2=1.3):
		self.population = npop
		self.phi_1 = phi_1
		self.phi_2 = phi_2
		
	def optimize(self,func,input_shape,n_iter,min_pos,max_pos,vel_max=0.05):
		self.swarm = []
		self.gb = np.random.uniform(low=min_pos,high=max_pos,size=(input_shape))
		for i in range(self.population):
			self.swarm.append(Particle(np.random.uniform(low=min_pos, high=max_pos, size=input_shape),self.phi_1,self.phi_2))
			x = func(self.swarm[i].loc)
			if(x < func(self.gb)):
				self.gb = self.swarm[i].loc
		vel = vel_max*(max_pos-min_pos)

		for i in range(n_iter):
			for j in self.swarm:
				j.move(self.gb,func,vel)
				if(func(j.loc) < func(self.gb)):
					self.gb = j.loc

		return self.gb


