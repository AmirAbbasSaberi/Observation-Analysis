import numpy as np
from numpy.linalg import inv
from scipy.stats import f ,t
import matplotlib.pyplot as plt
from numpy import linalg as LA

plt.close('all')

# __________________________________________
# Data ----------------------------------
p_ep1 = np.loadtxt('Data/object points epoch1.txt')
p_ep2 = np.loadtxt('Data/object points epoch2.txt')


Qxx_ep1 = np.loadtxt('Data/Q_xx epoch1.txt')
Qxx_ep2 = np.loadtxt('Data/Q_xx epoch2.txt')

sigma = 1.02
Qx_new1 = Qxx_ep1*np.power(sigma,2)*10**-6 
Qx_new2 = Qxx_ep2*np.power(sigma,2)*10**-6 


n = p_ep1.shape[0]
plt.Figure()
for i in range(0,n,1):
    
    mu = p_ep1[i,1:]
    cov = Qx_new1[2*i:2*i+2,2*i:2*i+2]*30000000
    x, y = np.random.multivariate_normal(mu, cov, 500).T
    plt.plot(x,y,'.')
   
    
    eigval , eigvec = LA.eig(cov)
    
    max_eigval = np.max(eigval)
    min_eigval = np.min(eigval)
    
    i = np.where(eigval == max_eigval)
    max_eigvec = eigvec[:,i]
    angle = np.arctan2(max_eigvec[1],max_eigvec[0]) + np.pi/2
    
    chisquare_val = 2.4477
    
    a=chisquare_val*np.sqrt(max_eigval)
    b=chisquare_val*np.sqrt(min_eigval)
    
    
    theta_grid = np.arange(0.0, 2*np.pi,np.pi/100)
    ellipse_x_r  = a*np.cos( theta_grid )
    ellipse_y_r  = b*np.sin( theta_grid )
    
    R = np.zeros((2,2)) 
    R[0,0] , R[1,0] ,R[0,1] , R[1,1] = np.cos(angle) , np.sin(angle) , -np.sin(angle) , np.cos(angle)
    
    i= len(theta_grid)
    ell_r = np.zeros((2,i))
    ell_r[0,:] = ellipse_x_r
    ell_r[1,:] = ellipse_y_r
    r_ell = np.dot(np.transpose( ell_r),R)
    
    plt.plot(r_ell[:,1]+mu[0],r_ell[:,0]+mu[1])
    
plt.grid()
plt.axis('equal')
plt.title('Ellipse Error of Coordinate Epoch1', fontsize = 20,weight = 'bold')
plt.xlabel('East')
plt.xlabel('North')

plt.figure()
plt.subplot(121)
plt.imshow(Qx_new1)
plt.colorbar()
plt.title('Qxx epoch1', fontsize = 20,weight = 'bold')

plt.subplot(122)
plt.imshow(Qx_new2)
plt.colorbar()
plt.title('Qxx epoch2', fontsize = 20,weight = 'bold')