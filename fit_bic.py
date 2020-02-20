import numpy as np
import matplotlib.pylab as plt

def model_1(x, param):
    return param[0]*np.sin(param[1]*x)/(x**2)

def model_2(x, param):
    return param[0]*np.sin(param[1]*x)/(x**param[2])

def loglike(x_obs, y_obs, sigma_y_obs, model, params):
    l = np.sum(-0.5*(y_obs - model(x_obs, params))**2/sigma_y_obs**2)
    return l

def run_mcmc(x_obs, y_obs, sigma_y_obs, 
             model, n_params=2, n_iterations=20000, scale=0.1):
    
    params = np.ones([n_iterations, n_params])
    loglike_values = np.ones(n_iterations)*3.0
    loglike_values[0] = loglike(x_obs, y_obs, sigma_y_obs, model, params[0,:])
    for i in range(1, n_iterations):
        current_params = params[i-1,:]
        next_params = current_params + np.random.normal(scale=scale, size=n_params)
        
        loglike_current = loglike(x_obs, y_obs, sigma_y_obs, model, current_params) 
        loglike_next = loglike(x_obs, y_obs, sigma_y_obs, model, next_params) 
            
        r = np.min([np.exp(loglike_next - loglike_current), 1.0])
        alpha = np.random.random()

        if alpha < r:
            params[i,:] = next_params
            loglike_values[i] = loglike_next
        else:
            params[i,:] = current_params
            loglike_values[i] = loglike_current
        
    params = params[n_iterations//2:,:]
    loglike_values = loglike_values[n_iterations//2:]
    return {'params':params, 'x_obs':x_obs, 'y_obs':y_obs, 'loglike_values':loglike_values}

def BIC(params_model):
    max_loglike = np.max(params_model['loglike_values'])
    n_dim = np.shape(params_model['params'])[1]
    n_points = len(params_model['y_obs'])
    return 2.0*(-max_loglike + 0.5*n_dim*np.log(n_points))

def plot_model(params_model, model, model_name):
    n_dim = np.shape(params_model['params'])[1]
    n_points = len(params_model['y_obs'])
    
    plt.figure(figsize = (4*(n_dim//2+1),6))
    for i in range(n_dim):
        plt.subplot(2, n_dim//2+1, i+1)
        plt.hist(params_model['params'][:,i], density=True, bins=30)
        plt.title(r"$\beta_{}= {:.2f}\pm {:.2f}$".format(i,np.mean(params_model['params'][:,i]), np.std(params_model['params'][:,i])))
        plt.xlabel(r"$\beta_{}$".format(i))
        
    plt.subplot(2,n_dim//2+1, i+2)
    best = np.mean(params_model['params'], axis=0)
    x = params_model['x_obs']
    x_model = np.linspace(x.min(), x.max(), 100)
    y_model = model(x_model, best)
    plt.plot(x_model, y_model)
    plt.errorbar(x, y, yerr=sigma_y, fmt='o')
    plt.title("BIC={:.2f}".format(BIC(params_model)))
    
    plt.subplots_adjust(hspace=0.5)
    plt.savefig(model_name+".png")

# Lee los datos
data = np.loadtxt('MagneticRatios.txt')
x = data[:,0]
y = data[:,1]
sigma_y = 0.1

# ajusta los modelos
params_model_1 = run_mcmc(x, y, sigma_y, model_1, scale=0.05, n_iterations=1000000)
params_model_2 = run_mcmc(x, y, sigma_y, model_2, n_params=3, scale=0.05, n_iterations=1000000)

# hace las graficas
plot_model(params_model_1, model_1, 'model_1')
plot_model(params_model_2, model_2, 'model_2')