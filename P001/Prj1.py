import numpy as np
from numpy.linalg import inv
from scipy.stats import f ,t
import matplotlib.pyplot as plt
plt.close('all')
# __________________________________________
# Data ----------------------------------
p_ep1 = np.loadtxt('Data/object points epoch1.txt')
p_ep2 = np.loadtxt('Data/object points epoch2.txt')


Qxx_ep1 = np.loadtxt('Data/Q_xx epoch1.txt')
Qxx_ep2 = np.loadtxt('Data/Q_xx epoch2.txt')


sigma = 1.02
n = p_ep1.shape[0]
m = p_ep1.shape[1]-1

# __________________________________________
# 1.1 Preparation

Qx_new1 = Qxx_ep1*np.power(sigma,2)*10**-6 
Qx_new2 = Qxx_ep2*np.power(sigma,2)*10**-6 



x_hat01 = np.zeros((n*m,1))
x_hat01[0:n*m:2,:] = np.reshape(p_ep1[:,1],(n,1))
x_hat01[1:n*m:2,:] = np.reshape(p_ep1[:,2],(n,1))
x_hat02 = np.zeros((n*m,1))
x_hat02[0:n*m:2,:] = np.reshape(p_ep2[:,1],(n,1))
x_hat02[1:n*m:2,:] = np.reshape(p_ep2[:,2],(n,1))

# __________________________________________
# 1.2 Calculation of translation 
F = np.zeros((m,n*m))
F[0,0:n*m:2] = 1
F[1,1:n*m:2] = 1

x_hatSp01 = (1/n)*np.dot(F,x_hat01)
x_hatSp02 = (1/n)*np.dot(F,x_hat02)

Qx_Sp01 = np.power(1/n,2)*np.dot(np.dot(F,Qx_new1),np.transpose(F))
Qx_Sp02 = np.power(1/n,2)*np.dot(np.dot(F,Qx_new2),np.transpose(F))

DeltaX_Sp = x_hatSp02 - x_hatSp01
Qxx_DeltaSp = Qx_Sp01 + Qx_Sp02

plt.figure()
plt.imshow(Qxx_DeltaSp)
plt.colorbar()
plt.title('Q\u0394x\u0394x epoch1',weight = 'bold')


# __________________________________________
# 1.3 Significance test
F_hat = np.dot(np.dot(np.transpose(DeltaX_Sp),inv(Qxx_DeltaSp)),DeltaX_Sp)
Test_Tr_F = 2*f.ppf(0.95, 2, 244)

# __________________________________________
# 2.1 Preparation ----------------------------------
G = np.eye(n*m) - (1/n)*np.dot(np.transpose(F),F)

ksi1 = np.dot(G,x_hat01)
ksi2 = np.dot(G,x_hat02) 

Qkk_1 = np.dot(np.dot(G,Qx_new1),np.transpose(G))
Qkk_2 = np.dot(np.dot(G,Qx_new2),np.transpose(G))

#___________________________________________
# 2.2 Calculation of rotation
et1 = ksi1[0:m*n:2]
ks1 = ksi1[1:m*n:2]
t1 = np.arctan2(et1,ks1)

et2 = ksi2[0:m*n:2]
ks2 = ksi2[1:m*n:2]
t2 = np.arctan2(et2,ks2)

#___________________________________________
# 2.2 Calculation of rotation
w = t2-t1

H1 = np.zeros((n,n*m))
H2 = np.zeros((n,n*m))
for i in range(0,n,1):
    
    H1[i,2*i:2*i+2] = np.reshape((np.array([ksi1[2*i+1]/(ksi1[2*i+1]**2 + ksi1[2*i]**2) , -ksi1[2*i]/(ksi1[2*i+1]**2 + ksi1[2*i]**2)])),(2,))
    H2[i,2*i:2*i+2] = np.reshape((np.array([ksi2[2*i+1]/(ksi2[2*i+1]**2 + ksi2[2*i]**2) , -ksi2[2*i]/(ksi2[2*i+1]**2 + ksi2[2*i]**2)])),(2,))
    
Qt1 = np.dot(np.dot(H1,Qkk_1),np.transpose(H1))
Qt2 = np.dot(np.dot(H2,Qkk_2),np.transpose(H2))

Qw = Qt1 + Qt2

et = np.ones((1,n))
w0 = (1/n)*np.dot(et,w)

sigma_w0 = np.sqrt((1/n)**2*np.dot(np.dot(et,Qw),np.transpose(et)))

#____________________________________________
# 1.3 Significance test
T_hat = abs(w0)/sigma_w0
tf = t.ppf(0.95,244)
