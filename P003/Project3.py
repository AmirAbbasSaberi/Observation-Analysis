import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.stats import f ,t

plt.close('all')

xy = np.loadtxt('Data/Coordinate.txt')
Qxx = np.loadtxt('Data/Qxx.txt')

s0 = 1.02*10**-3
Qxx_new = s0**2*Qxx

n,m = np.shape(xy)
ep1 = xy[0:int(n/2), :]
ep2 = xy[int(n/2): , :]



plt.figure()

plt.plot(ep1[:,1],ep1[:,2],'o',markersize = 8 , markeredgewidth=2)
plt.axis('equal')
plt.grid('on')
plt.xlabel('North[m]')
plt.ylabel('East[m]')
plt.plot(ep2[:,1],ep2[:,2],'+',markersize = 8 , markeredgewidth=2)
for i in range(0,len(ep1)):
    plt.text(ep1[i,1]+12,ep1[i,2]-6, int(ep1[i,0]), ha='center', va='bottom')
plt.title('Points in two epochs')


plt.figure()

plt.plot(ep1[[0,1],1],ep1[[0,1],2],color = 'black')
plt.plot(ep1[[0,2],1],ep1[[0,2],2],color = 'black')
plt.plot(ep1[[1,2],1],ep1[[1,2],2],color = 'black')
plt.plot(ep1[[1,3],1],ep1[[1,3],2],color = 'black')
plt.plot(ep1[[2,3],1],ep1[[2,3],2],color = 'black')

plt.plot(ep1[:,1],ep1[:,2],'o',markersize = 8 , markeredgewidth=2)
plt.axis('equal')
plt.grid('on')
plt.xlabel('North[m]')
plt.ylabel('East[m]')
plt.plot(ep2[:,1],ep2[:,2],'+',markersize = 8 , markeredgewidth=2)
for i in range(0,len(ep1)):
    plt.text(ep1[i,1]+12,ep1[i,2]-6, int(ep1[i,0]), ha='center', va='bottom')
plt.title('Points in two epochs')


xy1 = np.zeros((n,1))
xy1[0::2,0] = ep1[:,1] 
xy1[1::2,0] = ep1[:,2] 

xy2 = np.zeros((n,1))
xy2[0::2,0] = ep2[:,1] 
xy2[1::2,0] = ep2[:,2] 

tr = np.zeros((int(n/2)*4 , int(n/2)*6))
for i in range(0,n):
    tr[2*i,2*i + i] = 1
    tr[2*i+1,2*i+1 + i] = 1
    
Qxx_nn = np.dot(tr,np.dot(Qxx_new,np.transpose(tr)))

Qxx1 = Qxx_nn[0:n,0:n]
Qxx2 = Qxx_nn[n:,n:]

n = int(n/2)
m = m - 2
F = np.zeros((m,n*m))
F[0,0:n*m:2] = 1
F[1,1:n*m:2] = 1

xySp01 = (1/n)*np.dot(F,xy1)
xySp02 = (1/n)*np.dot(F,xy2)

# yx_bar = xy1 - xySp01

xysp1 = np.zeros((2*n,1))
xysp1[0::2,0] =  xySp01[0]
xysp1[1::2,0] =  xySp01[1]

xy_bar = xy1 - xysp1
u = xy2 - xy1

H123 = np.zeros((2*3,6))
for i in range(0,n-1):
    H123[2*i, 0: ] = [xy_bar[2*i,0] , xy_bar[2*i+1,0] , 0 , xy_bar[2*i+1,0] , 1 , 0]

    H123[2*i+1, 0: ] = [0 , xy_bar[2*i,0] , xy_bar[2*i+1,0] , -xy_bar[2*i,0] , 0 , 1]
p123 = np.dot(inv(H123),u[0:2*3,0])
Qpp123 = inv(H123)*(Qxx1[0:6,0:6] + Qxx2[0:6,0:6])*np.transpose(inv(H123))


H234 = np.zeros((2*3,6))
for i in range(1,n):
    H234[2*(i-1), 0: ] = [xy_bar[2*i,0] , xy_bar[2*i+1,0] , 0 , xy_bar[2*i+1,0] , 1 , 0]

    H234[2*(i-1)+1, 0: ] = [0 , xy_bar[2*i,0] , xy_bar[2*i+1,0] , -xy_bar[2*i,0] , 0 , 1]
p234 = np.dot(inv(H234),u[2:,0])
Qpp234 = inv(H234)*(Qxx1[0:6,0:6] + Qxx2[0:6,0:6])*np.transpose(inv(H234))

plt.figure()
plt.subplot(121)
plt.imshow(Qpp123)
plt.colorbar()
plt.title(r'$Qxx_{123}$')
plt.subplot(122)
plt.imshow(Qpp234)
plt.colorbar()
plt.title(r'$Qxx_{234}$')

t12 = np.arctan2(ep1[1,2] - ep1[0,2],ep1[1,1] - ep1[0,1])
t23 = np.arctan2(ep1[2,2] - ep1[1,2],ep1[2,1] - ep1[1,1])
t13 = np.arctan2(ep1[2,2] - ep1[0,2],ep1[2,1] - ep1[0,1])
e12 = p123[0]*(np.cos(t12))**2 + p123[1]*np.sin(2*t12) + p123[2]*(np.sin(t12))**2
e23 = p123[0]*(np.cos(t23))**2 + p123[1]*np.sin(2*t23) + p123[2]*(np.sin(t23))**2
e13 = p123[0]*(np.cos(t13))**2 + p123[1]*np.sin(2*t13) + p123[2]*(np.sin(t13))**2
gamma = 2*p123[1]

ee123 = np.sqrt((p123[0] - p123[2])**2 + 4*p123[1]**2)
e123_1 = 0.5*(p123[0] + p123[2] + ee123)
e123_2 = 0.5*(p123[0] + p123[2] - ee123)
theta123 = 0.5*(np.arctan2(2*p123[1] , p123[0] - p123[1]))

ee234 = np.sqrt((p234[0] - p234[2])**2 + 4*p234[1]**2)
e234_1 = 0.5*(p234[0] + p234[2] + ee234)
e234_2 = 0.5*(p234[0] + p234[2] - ee234)
theta234 = 0.5*(np.arctan2(2*p234[1] , p234[0] - p123[1]))

si = np.linspace(0,2*np.pi,101)

x = e123_1*np.cos(si)
y = e123_2*np.sin(si)

F123 = np.array([[1,0,1,0,1,0,0,0],[0,1,0,1,0,1,0,0]])
x123 = x*np.cos(theta123) - y*np.sin(theta123)
y123 = x*np.sin(theta123) + y*np.cos(theta123)

xsp123 = np.dot(F123,xy1)/3





F234 = np.array([[0,0,1,0,1,0,1,0],[0,0,0,1,0,1,0,1]])
x234 = x*np.cos(theta234) - y*np.sin(theta234)
y234 = x*np.sin(theta234) + y*np.cos(theta234)

xsp234 = np.dot(F234,xy1)/3


plt.figure()
k = 100000

plt.plot(ep1[[0,1],1],ep1[[0,1],2],color = 'black')
plt.plot(ep1[[0,2],1],ep1[[0,2],2],color = 'black')
plt.plot(ep1[[1,2],1],ep1[[1,2],2],color = 'black')
plt.plot(ep1[[1,3],1],ep1[[1,3],2],color = 'black')
plt.plot(ep1[[2,3],1],ep1[[2,3],2],color = 'black')

plt.plot(ep1[:,1],ep1[:,2],'o',markersize = 8 , markeredgewidth=2)
plt.axis('equal')
plt.grid('on')
plt.xlabel('North[m]')
plt.ylabel('East[m]')
plt.plot(ep2[:,1],ep2[:,2],'+',markersize = 8 , markeredgewidth=2)

plt.plot(k*x234 + xsp234[0],k*y234 + xsp234[1],'r')
plt.plot(k*x123 + xsp123[0],k*y123 + xsp123[1],'g')

plt.plot(xsp234[0],xsp234[1],'r.')
plt.plot(xsp123[0],xsp123[1],'g.')

for i in range(0,len(ep1)):
    plt.text(ep1[i,1]+12,ep1[i,2]-6, int(ep1[i,0]), ha='center', va='bottom')
plt.title('Strain Ellipse')
# Qx_Sp01 = np.power(1/n,2)*np.dot(np.dot(F,Qx_new1),np.transpose(F))
# Qx_Sp02 = np.power(1/n,2)*np.dot(np.dot(F,Qx_new2),np.transpose(F))



