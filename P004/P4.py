import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.stats import f ,t,chi2

plt.close('all')

data = np.loadtxt('File.txt')

p = [1 , 6]

j = np.where(data[:,2] == p[0])
p_data = np.reshape(data[j,:],(5,9))

t = p_data[:,1]
dt = np.diff(t)

# _________________ Inital Data _______________
x0 = p_data[0,3:6]
x = p_data[:,3:6]

vx = np.diff(x[:,0])
vy = np.diff(x[:,1])
vz = np.diff(x[:,2])

v0x = np.array([vx[0] , vy[0] , vz[0]])
var0 = np.power(p_data[0,6:]*10**-3,2)
a0 = np.array([0 , 0 , 0])

yp = np.concatenate((x0,v0x,a0),axis=0)


syy11 = np.diag(var0)
syy12 = np.zeros((3,3))
syy13 = np.zeros((3,3))

syy21 = np.zeros((3,3))
syy22 = 10**-4*np.eye(3)
syy23 = np.zeros((3,3))

syy31 = np.zeros((3,3))
syy33 = 10**-4*np.eye(3)
syy32 = np.zeros((3,3))

Syy_p1 = np.concatenate((syy11,syy12,syy13),axis=1)
Syy_p2 = np.concatenate((syy21,syy22,syy23),axis=1)
Syy_p3 = np.concatenate((syy31,syy32,syy33),axis=1)

Syy_p = np.concatenate((Syy_p1,Syy_p2,Syy_p3),axis=0)

Yp = [yp]
Yn = []
Sp = [Syy_p]
Sn = []
T = []
I = []
#_______________ Algorithm _________________
for i in range(0,len(dt)):
    # print(i)
    Dt = dt[i]
    
    phi11 = np.eye(3)
    phi12 = np.eye(3)*Dt
    phi13 = np.eye(3)*0.5*Dt**2
    
    phi21 = np.zeros((3,3))
    phi22 = np.eye(3)
    phi23 = np.eye(3)*Dt
    
    phi31 = np.zeros((3,3))
    phi32 = np.zeros((3,3))
    phi33 = np.eye(3)
    
    Phi1 = np.concatenate((phi11,phi12,phi13),axis=1)
    Phi2 = np.concatenate((phi21,phi22,phi23),axis=1)
    Phi3 = np.concatenate((phi31,phi32,phi33),axis=1)

    Phi_i = np.concatenate((Phi1,Phi2,Phi3),axis=0)

    Wdt_i = np.concatenate((0.5*Dt**2*np.eye(3),Dt*np.eye(3),np.eye(3)),axis=0)
    Sww = 10**-4*np.eye(3)
    
    
    # Prediction ---------------------------------------
    Syy_pi = Sp[i]
    yp_i = Yp[i]
    
    yn_i1 = np.dot(Phi_i,yp_i)
    Syy_ni1 = np.dot(np.dot(Phi_i,Syy_pi),np.transpose(Phi_i)) + np.dot(np.dot(Wdt_i,Sww),np.transpose(Wdt_i))
    
    
    # Innovation ----------------------------------------
    l_i1 = p_data[i+1,3:6]
    Sll_i1 = np.power(np.diag(p_data[i+1,6:]*10**-3),2)  
    A = np.concatenate((np.eye(3),np.zeros((3,3)),np.zeros((3,3))),axis=1)
    i_i1 = l_i1 - np.dot(A,yn_i1)
    Sii_i1 = Sll_i1 + np.dot(np.dot(A,Syy_ni1),np.transpose(A)) 
    
    
    # Filtering -----------------------------------------
    K_i1 = np.dot(np.dot(Syy_ni1,np.transpose(A)),inv(Sii_i1))
    yp_i1 = yn_i1 + np.dot(K_i1,i_i1)
    Syy_pi1 = np.dot((np.eye(9) - np.dot(K_i1,A)),Syy_ni1)
    
    Yp.append(yp_i1)
    Yn.append(yn_i1)
    Sp.append(Syy_pi1)
    Sp.append(Syy_ni1)
    
    # Test of significance of innovation and filtered state (velocity and acceleration) ------------
    r = i_i1
    Srr = Sii_i1
    Tr = np.dot(np.transpose(r),np.dot(Srr,r))
    
    v = yp_i1[3:6]
    Svv = Syy_pi1[3:6,3:6]
    Tv = np.dot(np.transpose(v),np.dot(Svv,v))
    
    a = yp_i1[6:]
    Saa = Syy_pi1[6:,6:]
    Ta = np.dot(np.transpose(a),np.dot(Saa,a))
    
    T.append(np.array([Tr , Tv , Ta]) < chi2.ppf(0.95, df=3))
    I.append(i_i1)
    if sum(np.array([Tr , Tv , Ta]) < chi2.ppf(0.95, df=3)) == 3:
        print('Test Accepted')
    
    

x = np.zeros((5,1))
y = np.zeros((5,1))
z = np.zeros((5,1)) 
v = np.zeros((5,1)) 
a = np.zeros((5,1))    
for i in range(0,len(Yp)):
    y1 = Yp[i]
    x[i] = y1[0]
    y[i] = y1[1]
    z[i] = y1[2]
    v[i] = np.sqrt(sum(np.power(y1[3:6],2)))
    a[i] = np.sqrt(sum(np.power(y1[6:],2)))

xp = np.zeros((4,1))
yp = np.zeros((4,1))
zp = np.zeros((4,1))
ap = np.zeros((4,1))
vp = np.zeros((4,1))  
for i in range(0,len(Yn)):
    y1 = Yn[i]
    xp[i] = y1[0]
    yp[i] = y1[1]
    zp[i] = y1[2]
    vp[i] = np.sqrt(sum(np.power(y1[3:6],2)))
    ap[i] = np.sqrt(sum(np.power(y1[6:],2)))
    
    
plt.figure()
plt.subplot(121)
plt.plot(t[1:],xp,'-.bs')
plt.plot(t,x,'-go')
plt.plot(t,p_data[:,3],'--r*')
plt.grid('on')
plt.xlabel(r'$t_{(year)}$')
plt.ylabel(r'$x_{(m)}$')
plt.legend(['Predicted','Filtered','Real'])
plt.title('Esting Kalman Filter and Real data')
plt.subplot(122)
plt.plot(t,x,'-o')
plt.plot(t,p_data[:,3],'--r*')
plt.grid('on')
plt.xlabel(r'$t_{(year)}$')
plt.ylabel(r'$x_{(m)}$')
plt.legend(['Filtered','Real'])
plt.title('Esting Kalman Filter and Real data')

plt.figure()
plt.subplot(121)
plt.plot(t[1:],yp,'-.bs')
plt.plot(t,y,'-go')
plt.plot(t,p_data[:,4],'--r*')
plt.grid('on')
plt.xlabel(r'$t_{(year)}$')
plt.ylabel(r'$y_{(m)}$')
plt.legend(['Predicted','Filtered','Real'])
plt.title('Northing Kalman Filter and Real data')
plt.subplot(122)
plt.plot(t,y,'-o')
plt.grid('on')
plt.xlabel(r'$t_{(year)}$')
plt.ylabel(r'$y_{(m)}$')
plt.title('Northing Kalman Filter and Real data')
plt.plot(t,p_data[:,4],'--r*')
plt.legend(['Filtered','Real'])

plt.figure()
plt.subplot(121)
plt.plot(t[1:],zp,'-.bs')
plt.plot(t,z,'-go')
plt.plot(t,p_data[:,5],'--r*')
plt.grid('on')
plt.xlabel(r'$t_{(year)}$')
plt.ylabel(r'$z_{(m)}$')
plt.legend(['Predicted','Filtered','Real'])
plt.title('Zenith Kalman Filter and Real data')
plt.subplot(122)    
plt.plot(t,z,'-o')
plt.plot(t,p_data[:,5],'--r*')
plt.grid('on')
plt.xlabel(r'$t_{(year)}$')
plt.ylabel(r'$z_{(m)}$')
plt.legend(['Filtered','Real'])
plt.title('Zenith Kalman Filter and Real data')
    
    
plt.figure()
plt.subplot(121)
plt.plot(t[1:],vp,'-.bs')
plt.plot(t,v,'-go')
plt.grid('on')
plt.xlabel(r'$t_{(year)}$')
plt.ylabel(r'$x_{(m/y)}$')
plt.legend(['Predicted','Filtered'])
plt.title('Velocity Kalman Filter and predicted data')

plt.subplot(122)
plt.plot(t[1:],ap,'-.bs')
plt.plot(t,a,'-go')
plt.grid('on')
plt.xlabel(r'$t_{(year)}$')
plt.ylabel(r'$a_{(m/y^2)}$')
plt.legend(['Predicted','Filtered'])
plt.title('Acceleration Kalman Filter and predicted data')
