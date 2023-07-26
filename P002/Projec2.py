import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.stats import f ,t

plt.close('all')

p_ep1 = np.loadtxt('Data/Koord1.dat')
p_ep2 = np.loadtxt('Data/Koord2.dat')

plt.figure()
plt.plot(p_ep1[:,1],p_ep1[:,2],'o',markersize = 8 , markeredgewidth=2)
plt.plot(p_ep1[:,1],p_ep1[:,2],'+',markersize = 8 , markeredgewidth=2)
plt.legend(['Epoch1','Epoch2'])
for i in range(0,len(p_ep1)):
    plt.text(p_ep1[i,1],p_ep1[i,2]-3, int(p_ep1[i,0]), ha='center', va='bottom')
plt.grid()
plt.xlabel('x[m]')
plt.ylabel('y[m]')
plt.title(r'Coordiante of points', fontsize=14)



Qxx1 = np.loadtxt('Data/Qxx1.dat')
Qxx2 = np.loadtxt('Data/Qxx2.dat')

Qxx1 = Qxx1 + np.transpose(Qxx1) - Qxx1*np.eye(2*len(p_ep1))
Qxx2 = Qxx2 + np.transpose(Qxx2) - Qxx2*np.eye(2*len(p_ep1))

s01 = 1.08*10**-3 # A posteriori standard deviation for both epochs
s02 = 1.01*10**-3

Qxx1 = Qxx1*s01**2
Qxx2 = Qxx2*s02**2

plt.figure()
plt.subplot(121)
plt.imshow(Qxx1)
plt.title(r'$Q_{xx_{1}}$', fontsize=14)
plt.colorbar()

plt.subplot(122)
plt.imshow(Qxx2)
plt.title(r'$Q_{xx_{2}}$', fontsize=14)
plt.colorbar()

df1 = 146 # Degree of freedom in adjustment of both epochs
df2 = 146

n = len(p_ep1)
xhat_01 = np.zeros((2*n,1))
xhat_01[0:2*n:2,0] = p_ep1[:,1]
xhat_01[1:2*n:2,0] = p_ep1[:,2]

xhat_02 = np.zeros((2*n,1))
xhat_02[0:2*n:2,0] = p_ep2[:,1]
xhat_02[1:2*n:2,0] = p_ep2[:,2]

# ---------------Pointwise Analysis------------
Qdd = Qxx1 + Qxx2

plt.figure()
plt.imshow(Qdd)
plt.title(r'$Q_{dd}$', fontsize=14)
plt.colorbar()

Ti = np.zeros((n,1))
for i in range(0,n):
    xy1_i = np.array([p_ep1[i,1],p_ep1[i,2]])
    xy2_i = np.array([p_ep2[i,1],p_ep2[i,2]])
    
    di = xy2_i - xy1_i
    
    Qdidi = Qdd[2*i:2*i+2,2*i:2*i+2] 
    
    Ti[i,0] = np.dot(np.dot(np.transpose(di) ,inv(Qdidi)),di)

Test_Val_Shift = 2*f.ppf(0.95, 2, df1 + df2)
Test_Shift = Ti < Test_Val_Shift 

# -------------Global Analysis---------------

t_903_902 = np.arctan2(p_ep1[2,2]-p_ep1[1,2],p_ep1[2,1]-p_ep1[1,1])
epsilon = +t_903_902-np.pi/2


Rz = np.array([[np.cos(epsilon) , np.sin(epsilon)],[-np.sin(epsilon) , np.cos(epsilon)]])
R = np.zeros((2*n,2*n))
for i in range(0,n):
    R[2*i:2*i+2,2*i:2*i+2] = Rz

xrot_1 = np.dot(R,xhat_01)
xrot_2 = np.dot(R,xhat_02)

xrot_1[0:2*n:2,0] = -xrot_1[0:2*n:2,0]
xrot_2[0:2*n:2,0] = -xrot_2[0:2*n:2,0]
Qxrxr1 = np.dot(np.dot(R,Qxx1),np.transpose(R))
Qxrxr2 = np.dot(np.dot(R,Qxx2),np.transpose(R))

plt.figure()
plt.imshow(Qxrxr1)
plt.title(r'$Q_{xrot_1 xrot_1}$', fontsize=14)
plt.colorbar()
plt.figure()
plt.imshow(Qxx2)
plt.title(r'$Q_{xrot_1 xrot_1}$', fontsize=14)
plt.colorbar()


plt.figure()
plt.plot(xrot_1[1:2*n:2,0],xrot_1[0:2*n:2,0],'o',markersize = 8 , markeredgewidth=2)
plt.plot(xrot_2[1:2*n:2,0],xrot_2[0:2*n:2,0],'+',markersize = 8 , markeredgewidth=2)
plt.axis('equal')
plt.grid()
plt.xlabel('y[m]')
plt.ylabel('x[m]')
plt.title(r'Rotated Coordiante of points', fontsize=14)

F1 = np.array([[0, 0, 1, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 1, 0, 0, 0, 0]])

x_mu = np.dot(0.5*F1,xrot_1)

F = np.array([[1, 0, 1, 0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]])
G1 = np.dot(0.5*np.transpose(F),F1)
G = np.eye(2*n) - G1

kisi1 =np.dot(G,xrot_1)
kisi2 =xrot_2 - np.dot(G1,xrot_1)


plt.figure()
plt.plot(kisi1[1:2*n:2,0],kisi1[0:2*n:2,0],'o',markersize = 8 , markeredgewidth=2)
plt.plot(kisi2[1:2*n:2,0],kisi2[0:2*n:2,0],'+',markersize = 8 , markeredgewidth=2)
plt.axis('equal')
plt.xlabel('y[m]')
plt.ylabel('x[m]')
plt.title(r'Rotated and shifted Coordiante of points', fontsize=14)
plt.grid()
plt.legend(['Epoch1','Epoch2'])

Qkk1 = np.dot(np.dot(G,Qxrxr1),np.transpose(G))
Qkk2 = Qxrxr2 + np.dot(np.dot(G1,Qxrxr1),np.transpose(G1))

plt.figure()
plt.subplot(121)
plt.imshow(Qkk1)
plt.title(r'$Q_{\xi_1 \xi_1}$', fontsize=14)
plt.colorbar()
plt.subplot(122)
plt.imshow(Qkk2)
plt.title(r'$Q_{\xi_2 \xi_2}$', fontsize=14)
plt.colorbar()


dkisi = kisi2 - kisi1

H = np.zeros((n, 2 * n))
for i in range(n):
    H[i, 2 * i+1] = 1
    
deta = np.dot(H,dkisi)
Qdeta_deta = np.dot(np.dot(H,Qkk1 + Qkk2),np.transpose(H))

A = np.hstack((np.ones((n, 1)), np.reshape(-kisi1[::2,0]**2,(n,1))))
p = np.dot(np.dot(np.dot(inv(np.dot(np.dot(np.transpose(A),inv(Qdeta_deta)),A)),np.transpose(A)),inv(Qdeta_deta)),deta)
Qpp =inv(np.dot(np.dot(np.transpose(A),inv(Qdeta_deta)),A))

T = np.dot(np.dot(np.transpose(p),inv(Qpp)),p)
Test_Val_p = 2*f.ppf(0.95, 2, 3)
Test_p = T < Test_Val_Shift 

x = np.arange(-60,60,1)
A1 = np.hstack((np.ones((len(x), 1)), np.reshape(x**2,(len(x),1))))
deta1 = np.dot(A1,p)

plt.figure()
plt.subplot(122)
plt.imshow(Qpp)
plt.title(r'$Q_{p p}$', fontsize=14)
plt.colorbar()
plt.subplot(121)
plt.plot(x,deta1)
plt.grid()
plt.xlabel(r'$\xi$')
plt.ylabel(r'$\eta$')
plt.title('Deformation', fontsize=14)
