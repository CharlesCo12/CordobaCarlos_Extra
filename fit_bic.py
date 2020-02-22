import numpy as np
import matplotlib.pyplot as plt
data = np.loadtxt('MagneticRatios.txt')
x_obs = data[:,0]
y_obs = data[:,1]
sigma_y = 0.01

def model_A(x, params):
    y = params[0]*np.sin(params[1]*x)/x**2
    return y

def model_B(x, params):
    y = params[0]*np.sin(params[1]*x)/x**params[2]
    return y


def logprior_model_A(params):
    if (params[0]>0.4 or params[0]<0 or params[1]>2 or params[1]<0):
        return -np.inf
    else:
        return 0
    
def logprior_model_B(params):
    if (params[0]>0.4 or params[0]<0 or params[1]>2.5 or params[1]<0 or params[2]<1 or params[2]>5):
        return -np.inf
    else:
        return 0
def loglike(x_obs, y_obs, sigma_y_obs, model, params):
    l = np.sum(-0.5*(y_obs -model(x_obs, params))**2/sigma_y_obs**2)
    return l
N=20000
sigma_A=0.4/np.sqrt(N)
sigma_n=4/np.sqrt(N)
sigma_w=1/np.sqrt(N)

params_1=[[np.random.random()*0.4, np.random.random()]]
params_2=[[np.random.random()*0.4, np.random.random(), 4*np.random.random()+1]]

loglike_values_1 = []
loglike_values_2 = []

for i in range(1, N):
    current_params_1 = params_1[i-1]
    next_params_1 = np.array(current_params_1) + np.array([np.random.normal(scale= sigma_A),np.random.normal(scale = sigma_w)])
        
    loglike_current_1 = loglike(x_obs, y_obs, sigma_y, model_A, current_params_1) + logprior_model_A(current_params_1)
    loglike_next_1 = loglike(x_obs, y_obs, sigma_y, model_A, next_params_1) + logprior_model_A(next_params_1)
    
    current_params_2 = params_2[i-1]
    next_params_2 = np.array(current_params_2) + np.array([np.random.normal(scale=sigma_A),np.random.normal(scale=sigma_w),np.random.normal(scale=sigma_n)])
        
    loglike_current_2 = loglike(x_obs, y_obs, sigma_y, model_B, current_params_2) + logprior_model_B(current_params_2)
    loglike_next_2 = loglike(x_obs, y_obs, sigma_y, model_B, next_params_2) + logprior_model_B(next_params_2)
            
    r = np.min([np.exp(loglike_next_1 - loglike_current_1), 1.0])
    r1 = np.min([np.exp(loglike_next_2 - loglike_current_2), 1.0])
    alpha = np.random.random()

    if (alpha<r and alpha<r1):
        params_2.append(next_params_2)
        loglike_values_2.append(loglike_next_2)
        params_1.append(next_params_1)
        loglike_values_1.append(loglike_next_1)
    elif(alpha<r and alpha>=r1):
        params_1.append(next_params_1)
        loglike_values_1.append(loglike_next_1)
        params_2.append(current_params_2)
        loglike_values_2.append(loglike_current_2)
    elif(alpha>=r and alpha<r1):
        params_2.append(next_params_2)
        loglike_values_2.append(loglike_next_2)
        params_1.append(current_params_1)
        loglike_values_1.append(loglike_current_1)
    else:
        params_1.append(current_params_1)
        loglike_values_1.append(loglike_current_1)
        params_2.append(current_params_2)
        loglike_values_2.append(loglike_current_2)        
#Se toman la última mitad de los datos
params_1 = params_1[N//2:]
loglike_values_1 = loglike_values_1[N//2:]

params_2 = params_2[N//2:]
loglike_values_2 = loglike_values_2[N//2:]

#Cálculo de los BIC
max_loglike_1 = np.max(loglike_values_1)
n_dim_1 = 2
n_points_1 = len(y_obs)
BIC_1=2.0*(-max_loglike_1 + 0.5*n_dim_1*np.log(n_points_1))
max_loglike_2 = np.max(loglike_values_2)
n_dim_2 = 3
n_points_2 = len(y_obs)
BIC_2=2.0*(-max_loglike_2 + 0.5*n_dim_2*np.log(n_points_2))

#Gráficas
b1=np.mean(params_1,axis=0)[0]
std1=np.std(params_1,axis=0)[0]
b2=np.mean(params_1,axis=0)[1]
std2=np.std(params_1,axis=0)[1]
plt.figure(figsize=(12,8))
plt.subplot(221)
plt.title(r'$\beta_{} = {:.2f} \pm {:.2f}$'.format(0,b1,std1))
plt.hist(np.array(params_1)[:,0],log=True)
plt.subplot(222)
plt.title(r'$\beta_{} = {:.2f} \pm {:.2f}$'.format(1,b2,std2))
plt.hist(np.array(params_1)[:,1],log=True,color='green')
plt.subplot(223)
plt.title('BIC = {:.2f}'.format(BIC_1))
plt.scatter(x_obs,y_obs,s=0.8,label='Datos')
plt.errorbar(x_obs,y_obs,yerr=sigma_y,fmt='o')
x1=np.linspace(0,10,100)
plt.plot(x1,model_A(x1,np.mean(params_1,axis=0)),label='fit')
plt.ylim(np.min(y_obs)-0.01,np.max(y_obs)+0.01)
plt.legend(loc=0.0)
plt.savefig('potencia_fija.png')
plt.close()

b_1=np.mean(params_2,axis=0)[0]
std_1=np.std(params_2,axis=0)[0]
b_2=np.mean(params_2,axis=0)[1]
std_2=np.std(params_2,axis=0)[1]
b_3=np.mean(params_2,axis=0)[2]
std_3=np.std(params_2,axis=0)[2]

plt.figure(figsize=(12,8))
plt.subplot(221)
plt.title(r'$\beta_{} = {:.3f} \pm {:.3f}$'.format(0,b_1,std_1))
plt.hist(np.array(params_2)[:,0],log=True)
plt.subplot(222)
plt.title(r'$\beta_{} = {:.3f} \pm {:.3f}$'.format(1,b_2,std_2))
plt.hist(np.array(params_2)[:,1],log=True,color='green')
plt.subplot(223)
plt.title(r'$\beta_{} = {:.3f} \pm {:.3f}$'.format(2,b_3,std_3))
plt.hist(np.array(params_2)[:,2],log=True,color='orange')
plt.subplot(224)
plt.title('BIC = {:.2f}'.format(BIC_2))
plt.scatter(x_obs,y_obs,s=0.8,label='Datos')
plt.errorbar(x_obs,y_obs,yerr=sigma_y,fmt='o')
x1=np.linspace(0,10,100)
plt.plot(x1,model_B(x1,np.mean(params_2,axis=0)),label='fit')
plt.ylim(np.min(y_obs)-0.01,np.max(y_obs)+0.01)
plt.legend(loc=0.0)
plt.savefig('potencia_variable.png')
plt.close()