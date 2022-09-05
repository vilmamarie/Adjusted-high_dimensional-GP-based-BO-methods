import GPy
# import matlab.engine
import numpy as np
from pyDOE import lhs
import functions
from REMBO import EI
import timeit

def dim_sampling(low_dim, X, bx_size):
    if len(X.shape)==1:
        X=X.reshape((1, X.shape[0]))
    n=X.shape[0]
    high_dim=X.shape[1]
    low_obs=np.zeros((n,low_dim))
    high_to_low=np.zeros(high_dim,dtype=int)
    sign=np.random.choice([-1,1],high_dim)
    for i in range(high_dim):
        high_to_low[i]=np.random.choice(range(low_dim))
        low_obs[:,high_to_low[i]]=X[:,i]*sign[i]+ low_obs[:,high_to_low[i]]

    for i in range(n):
        for j in range(low_dim):
            if low_obs[i][j] > bx_size: low_obs[i][j] = bx_size
            elif low_obs[i][j] < -bx_size: low_obs[i][j] = -bx_size
    return low_obs, high_to_low, sign

def back_projection(low_obs, high_to_low, sign, bx_size):
    if len(low_obs.shape)==1:
        low_obs=low_obs.reshape((1, low_obs.shape[0]))
    n=low_obs.shape[0]
    high_dim=high_to_low.shape[0]
    low_dim=low_obs.shape[1]
    high_obs=np.zeros((n,high_dim))
    scale=1

    for i in range(high_dim):
        high_obs[:,i]=sign[i]*(low_obs[:,high_to_low[i]]*scale)
    for i in range(n):
        for j in range(high_dim):
            if high_obs[i][j] > bx_size: high_obs[i][j] = bx_size
            elif high_obs[i][j] < -bx_size: high_obs[i][j] = -bx_size
    return high_obs

def projection(high_obs, high_to_low, sign, bx_size, low_dim):
    if len(high_obs.shape)==1:
        high_obs=high_obs.reshape((1, high_obs.shape[0]))
    n=high_obs.shape[0]
    low_obs=np.zeros((n,low_dim))
    scale=1
    for i in range(low_dim):
        htl_idx = np.where(high_to_low == i)[0][0]
        low_obs[:, i] = high_obs[:, htl_idx]/(sign[htl_idx]*scale)
    for i in range(n):
        for j in range(low_dim):
            if low_obs[i][j] > bx_size: low_obs[i][j] = bx_size
            elif low_obs[i][j] < -bx_size: low_obs[i][j] = -bx_size
    return low_obs

def RunMain(low_dim=2, high_dim=25, initial_n=20, total_itr=100, func_type='Branin',
            s=None, active_var=None, ARD=False, variance=1., length_scale=None, box_size=None,
            high_to_low=None, sign=None, hyper_opt_interval=20, noise_var=0, param_ranges=None, output_path=None,afn=EI):
    """

    :param high_dim: the dimension of high dimensional search space
    :param low_dim: The effective dimension of the algorithm.
    :param initial_n: the number of initial points
    :param total_itr: the number of iterations of algorithm. The total
        number of test function evaluations is initial_n + total_itr
    :param func_type: the name of test function
    :param s: initial points
    :param active_var: a vector with the size of greater or equal to
        the number of active variables of test function. The values of
        vector are integers less than high_dim value.
    :param ARD: if TRUE, kernel is isomorphic
    :param variance: signal variance of the kernel
    :param length_scale: length scale values of the kernel
    :param box_size: this variable indicates the search space [-box_size, box_size]^d
    :param high_to_low: a vector with D elements. each element can have a value from {0,..,d-1}
    :param sign: a vector with D elements. each element is either +1 or -1.
    :param hyper_opt_interval: the number of iterations between two consecutive
        hyper parameters optimizations
    :param noise_var: noise variance of the test functions
    :return: a tuple of best values of each iteration, all observed points, and
        corresponding test function values of observed points
    """

    if active_var is None:
        active_var= np.arange(high_dim)
    if box_size is None:
        box_size=1
    if high_to_low is None:
        high_to_low=np.random.choice(range(low_dim), high_dim)
    if sign is None:
        sign = np.random.choice([-1, 1], high_dim)

    #Specifying the type of objective function
    if func_type=='Branin':
        test_func = functions.Branin(active_var, noise_var=noise_var, param_ranges=param_ranges)
    elif func_type=='Rosenbrock':
        test_func = functions.Rosenbrock(active_var, noise_var=noise_var, param_ranges=param_ranges)
    elif func_type=='Hartmann6':
        test_func = functions.Hartmann6(active_var, noise_var=noise_var, param_ranges=param_ranges)
    elif func_type == 'Hartmann3':
        test_func = functions.Hartmann3(active_var, noise_var=noise_var, param_ranges=param_ranges)
    elif func_type == 'StybTang':
        test_func = functions.StybTang(active_var, noise_var=noise_var, param_ranges=param_ranges)
    elif func_type == 'Ackley':
        test_func = functions.Ackley(active_var, noise_var=noise_var, param_ranges=param_ranges)
    elif func_type == 'Griewank':
        test_func = functions.Griewank(active_var, noise_var=noise_var, param_ranges=param_ranges)
    elif func_type == 'Levy':
        test_func = functions.Levy(active_var, noise_var=noise_var, param_ranges=param_ranges)
    elif func_type == 'Levy13':
        test_func = functions.Levy13(active_var, noise_var=noise_var, param_ranges=param_ranges)
    elif func_type == 'Schwefel':
        test_func = functions.Schwefel(active_var, noise_var=noise_var, param_ranges=param_ranges)
    elif func_type == 'PredaySimmobility':
        test_func = functions.PredaySimmobility(active_var, noise_var=noise_var, param_ranges=param_ranges, home_dir=output_path)
    else:
        TypeError('The input for func_type variable is invalid, which is', func_type)
        return

    best_results = np.zeros([1, total_itr + initial_n])
    elapsed=np.zeros([1, total_itr + initial_n])

    # Creating the initial points. The shape of s is nxD
    f_s = None
    f_s_true = None
    if s is None:
        s=lhs(low_dim, initial_n) * 2 * box_size - box_size
        f_s_true = test_func.evaluate_true(back_projection(s, high_to_low, sign, box_size))
        f_s = test_func.evaluate(back_projection(s,high_to_low,sign,box_size),f_s_true)
    else:
        if np.shape(s)[1] > low_dim:
            ## Downscale to the box size (-1,1)
            if np.shape(s)[1] == high_dim:
                downscale_fn = getattr(test_func, "downscale_domain", None)
                if callable(downscale_fn):
                    s = test_func.downscale_domain(s)
                f_s_true = test_func.evaluate_true(s)
                f_s = test_func.evaluate(s,f_s_true)

            else:
                f_s = np.copy(s[:,high_dim]).reshape(s.shape[0], 1)
                f_s_true = np.copy(s[:,high_dim]).reshape(s.shape[0], 1)
                s = s[:,active_var]
                downscale_fn = getattr(test_func, "downscale_domain", None)
                if callable(downscale_fn):
                    s = test_func.downscale_domain(s)
            s = projection(s, high_to_low, sign, box_size, low_dim)

    for i in range(initial_n):
        best_results[0,i]=np.max(f_s_true[0:i+1]) # np.max -> np.min (temp reverted back to original)

    # Building and fitting a new GP model
    kern = GPy.kern.Matern52(input_dim=low_dim, ARD=ARD, variance=variance, lengthscale=length_scale)
    m = GPy.models.GPRegression(s, f_s, kernel=kern)
    m.likelihood.variance = 1e-3

    # Main loop
    for i in range(total_itr):

        start = timeit.default_timer()

        # Updating GP model
        m.set_XY(s, f_s)
        if (i+initial_n<=25 and i % 5 == 0) or (i+initial_n>25 and i % hyper_opt_interval == 0):
            try:
                m.optimize()
            except Exception as inst:
                print("[ERROR]", inst)

        # Maximizing acquisition function
        D = lhs(low_dim, 2000) * 2 * box_size - box_size
        mu, var = m.predict(D)

        ei_d = afn(len(D), max(f_s), mu, var) #max -> min (temp reverted back to original)
        index = np.argmax(ei_d)

        # Adding the new point to our sample
        s = np.append(s, [D[index]], axis=0)
        new_high_point=back_projection(D[index], high_to_low, sign, box_size)
        f_s_eval = test_func.evaluate_true(new_high_point)
        f_s_true = np.append(f_s_true, f_s_eval, axis=0)
        f_s = np.append(f_s, test_func.evaluate(new_high_point, f_s_eval), axis=0)

        stop = timeit.default_timer()
        best_results[0, i + initial_n] = np.max(f_s_true) # np.max -> np.min (temp reverted back to original)
        elapsed[0, i + initial_n]=stop-start

    # if func_type == 'WalkerSpeed':
    #     eng.quit()
    high_s = back_projection(s, high_to_low, sign, box_size)
    scale_fn = getattr(test_func, "scale_domain", None)
    if callable(scale_fn):
        high_s = test_func.scale_domain(high_s)
    return best_results, elapsed, s, f_s, f_s_true, high_s

if __name__=='__main__':
    res, time, s, f_s, f_s_true, _= RunMain(low_dim=8, high_dim=25, initial_n=20, total_itr=50, ARD=True, noise_var=1)
    print(res,time)
