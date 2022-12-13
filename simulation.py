import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import random
import time


n = 100
k = 9.0e+10
e = 1.6e-19
m = 1.5e-31
interval = 1.0e-7
ran = 1.0e-2

fig,ax = plt.subplots()
ax.set_xlim(-1.2*ran,1.2*ran)
ax.set_ylim(-1.2*ran,1.2*ran)
ax.set_aspect("equal")

objs = []
objj = []

for i in range(n):
	exec("a"+ str(i) +",=ax.plot([],[],color = \"blue\",marker = \".\")")
	exec("objs.append(a"+str(i)+")")




class calc:
	def __init__(self):
		self.p = np.array([[random.uniform(-1.0*ran,1.0*ran),random.uniform(-1.0*ran,1.0*ran)] for i in range(n)],dtype="float32")
		self.v = np.array([[1e-10,1e-10] for i in range(n)],dtype="float32")
		self.a = np.array([[1e-10,1e-10] for i in range(n)],dtype="float32")
	
	

	def acc(self):
		for l,i in enumerate(self.p,0):
			q = self.p -i
			w = np.linalg.norm(q,axis = 1)
			r = -q*k*e*e / ((w[:,np.newaxis] ** 3) *m) 
			t = np.nan_to_num(r,nan=0)
			self.a[l] = sum(t)
			 


	def velocit(self):
		#global interval
		self.v = self.a * interval + self.v
		self.v = np.where((self.p > 100) & (self.p < -100) , self.v, -self.v)
		self.v = self.v*0.001
		#average = np.linalg.norm(self.v)
		#interval = 1.0e-12/average	

	
	def pos(self):
		self.p = self.v * interval +self.p
		self.p = np.where(self.p > -100, self.p, -1.0*ran)
		self.p = np.where(self.p < 100 , self.p, 1.0*ran)
		



	def put(self):
		return list(self.p)
		


	def energy(self):
		K = sum((m/2)*(np.linalg.norm(self.v,axis = 1) **2))
		U = 0
		for i in self.p:
			q = self.p -i
			w = np.linalg.norm(q,axis= 1)
			r = k*e*e/w
			u = sum(r[~np.isinf(r)])
			U += u 
		

elc = calc()



def animation_func(frame):
	elc.acc()
	elc.velocit()
	elc.pos()
	
	for i,l in enumerate(elc.put(),0):
		objs[i].set_data(l)
		print(l,"l")	
		
loop = 0			
"""
while True:
	elc.acc()
	elc.velocit()
	elc.pos()
	loop += 1
	print(loop)
	if np.linalg.norm(elc.v) < 1e-5:
		x,y  = [[],[]]
		for i in elc.put():
			x += i[0]
			y += i[1]
		
		plt.scatter(x,y,".")	
		plt.show()
		break
"""





anim = FuncAnimation(fig, animation_func)
plt.show()
