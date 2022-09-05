import GPy
# import matlab.engine
import numpy as np
import math
from pyDOE import lhs
from scipy.stats import norm
import functions
import projection_matrix
import projections
import kernel_inputs
import timeit

def EI(D_size,f_max,mu,var):
    """
    :param D_size: number of points for which EI function will be calculated
    :param f_max: the best value found for the test function so far
    :param mu: a vector of predicted values for mean of the test function
        corresponding to the points
    :param var: a vector of predicted values for variance of the test function
        corresponding to the points
    :return: a vector of EI values of the points
    """
    ei=np.zeros((D_size,1))
    std_dev=np.sqrt(var)
    for i in range(D_size):
        if var[i]!=0:
            z= (mu[i] - f_max) / std_dev[i]
            ei[i]= (mu[i]-f_max) * norm.cdf(z) + std_dev[i] * norm.pdf(z)
    return ei

def PI(D_size,f_max,mu,var):
    std_dev = np.sqrt(np.absolute(var))
    pi = [norm.cdf((mu[i] - f_max)/std_dev[i]) for i in range(D_size)]
    return pi

def RunRembo(low_dim=2, high_dim=20, initial_n=20, total_itr=100, func_type='Branin',
             matrix_type='simple', kern_inp_type='Y', A_input=None, s=None, active_var=None,
             hyper_opt_interval=20, ARD=False, variance=1., length_scale=None, box_size=None,
             noise_var=0, param_ranges=None, output_path=None,afn=EI):
    """"

    :param low_dim: the dimension of low dimensional search space
    :param high_dim: the dimension of high dimensional search space
    :param initial_n: the number of initial points
    :param total_itr: the number of iterations of algorithm. The total
        number of test function evaluations is initial_n + total_itr
    :param func_type: the name of test function
    :param matrix_type: the type of projection matrix
    :param kern_inp_type: the type of projection. Projected points
        are used as the input of kernel
    :param A_input: a projection matrix with iid gaussian elements.
        The size of matrix is low_dim * high_dim
    :param s: initial points
    :param active_var: a vector with the size of greater or equal to
        the number of active variables of test function. The values of
        vector are integers less than high_dim value.
    :param hyper_opt_interval: the number of iterations between two consecutive
        hyper parameters optimizations
    :param ARD: if TRUE, kernel is isomorphic
    :param variance: signal variance of the kernel
    :param length_scale: length scale values of the kernel
    :param box_size: this variable indicates the search space [-box_size, box_size]^d
    :param noise_var: noise variance of the test functions
    :return: a tuple of best values of each iteration, all observed points, and
        corresponding test function values of observed points
    """

    if active_var is None:
        active_var = np.arange(high_dim)
    if box_size is None:
        box_size = math.sqrt(low_dim)
    if hyper_opt_interval is None:
        hyper_opt_interval = 10

    #Specifying the type of objective function
    if func_type == 'Branin':
        test_func = functions.Branin(active_var, noise_var=noise_var, param_ranges=param_ranges)
    elif func_type == 'Rosenbrock':
        test_func = functions.Rosenbrock(active_var, noise_var=noise_var, param_ranges=param_ranges)
    elif func_type == 'Hartmann6':
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
        test_func = functions.PredaySimmobility(active_var, noise_var=noise_var, param_ranges=param_ranges,
                                                home_dir=output_path)
    else:
        TypeError('The input for func_type variable is invalid, which is', func_type)
        return

    #Specifying the type of embedding matrix
    if matrix_type=='simple':
        matrix=projection_matrix.SimpleGaussian(low_dim, high_dim)
    elif matrix_type=='normal':
        matrix= projection_matrix.Normalized(low_dim, high_dim)
    elif matrix_type=='orthogonal':
        matrix = projection_matrix.Orthogonalized(low_dim, high_dim)
    else:
        TypeError('The input for matrix_type variable is invalid, which is', matrix_type)
        return

    # Generating matrix A
    if A_input is not None:
        matrix.A = A_input

    A = matrix.evaluate()

    #Specifying the input type of kernel
    if kern_inp_type=='Y':
        kern_inp = kernel_inputs.InputY(A)
        input_dim=low_dim
    elif kern_inp_type=='X':
        kern_inp = kernel_inputs.InputX(A)
        input_dim = high_dim
    elif kern_inp_type == 'psi':
        kern_inp = kernel_inputs.InputPsi(A)
        input_dim = high_dim
    else:
        TypeError('The input for kern_inp_type variable is invalid, which is', kern_inp_type)
        return

    #Specifying the convex projection
    cnv_prj=projections.ConvexProjection(A)

    best_results = np.zeros([1, total_itr + initial_n])
    elapsed = np.zeros([1, total_itr + initial_n])
    f_s = None
    f_s_true = None

    if s is None:
        s = lhs(low_dim, initial_n) * 2.0 * box_size - box_size
        f_s = test_func.evaluate(cnv_prj.evaluate(s))
        f_s_true = test_func.evaluate_true(cnv_prj.evaluate(s))
    else:
        ### THIS IS NOT WORKING AS IT SHUOLD! THE FIRST PROJECTION IS DIFFERENT DUE TO FLOAT PRECISSION
        s = s[:, active_var]
        if np.shape(s)[1] > low_dim:
            ## Downscale to the box size (0,1) ###(-1,1)
            downscale_fn = getattr(test_func, "normalize_domain", None)
            if not callable(downscale_fn):
                NotImplementedError('The downscaling/normalizing function is not implemented: ', "normalize_domain")
                return

            if np.shape(s)[1] == high_dim:
                print(np.min(downscale_fn(s)))
                print(np.max(downscale_fn(s)))
                s = downscale_fn(s) * 2.0 * box_size - box_size
                f_s_true = test_func.evaluate_true(s)
                f_s = test_func.evaluate(s, f_s_true)
            else:
                print(np.min(downscale_fn(s)))
                print(np.max(downscale_fn(s)))
                f_s = np.copy(s[:, high_dim]).reshape(s.shape[0], 1)
                f_s_true = np.copy(s[:, high_dim]).reshape(s.shape[0], 1)
                s = downscale_fn(s) * 2.0 * box_size - box_size

    # s_projected = cnv_prj.project(s)  # projection(s, high_to_low, sign, box_size, low_dim)
    # s_evaluated = cnv_prj.evaluate(s_projected)
    # s_evaluated_projected = cnv_prj.project(s_evaluated)
    # s_evaluated_projected_evaluated = cnv_prj.evaluate(s_evaluated_projected)
    # s_evaluated_projected_evaluated_projected = cnv_prj.project(s_evaluated_projected_evaluated)
    # print("====================")
    #
    #
    # print(s[1, :])
    # print(s_evaluated[1, :])
    # print(s_evaluated_projected_evaluated[1, :])
    #
    # print(s_projected[1, :])
    # print(s_evaluated_projected[1, :])
    # print(s_evaluated_projected_evaluated_projected[1, :])
    #
    # print(str(np.min(s)) + "/" + str(np.max(s)))
    # print(str(np.min(s_evaluated)) + "/" + str(np.max(s_evaluated)))
    # print(str(np.min(s_evaluated_projected_evaluated)) + "/" + str(np.max(s_evaluated_projected_evaluated)))
    #
    # print(str(np.min(s_projected)) + "/" + str(np.max(s_projected)))
    # print(str(np.min(s_evaluated_projected)) + "/" + str(np.max(s_evaluated_projected)))
    # print(str(np.min(s_evaluated_projected_evaluated_projected)) + "/" + str(np.max(s_evaluated_projected_evaluated_projected)))
    #
    # print("&&&&&&&&&&&&&&&&&&&&&&")

    # ccc1 = lhs(4, initial_n)
    # ccc = ccc1 * 2 * box_size - box_size
    #
    # print(np.min(ccc1))
    # print(np.max(ccc1))
    # print(np.min(ccc))
    # print(np.max(ccc))
    # prj_ccc = cnv_prj.evaluate(ccc)
    # print(np.min(prj_ccc))
    # print(np.max(prj_ccc))
    #
    # print(ccc[1, :])
    # print(prj_ccc[1, :])
    # print(cnv_prj.project(prj_ccc)[1, :])
    # print(cnv_prj.evaluate(cnv_prj.project(prj_ccc))[1,:])

    for i in range(initial_n):
        best_results[0,i]=np.max(f_s_true[0:i+1])

    # Generating GP model
    k = GPy.kern.Matern52(input_dim=input_dim, ARD=ARD, variance=variance, lengthscale=length_scale)
    m = GPy.models.GPRegression(kern_inp.evaluate(s), f_s, kernel=k)
    m.likelihood.variance = 1e-6

    # Main loop of the algorithm
    for i in range(total_itr):

        start = timeit.default_timer()
        # Updating GP model
        m.set_XY(kern_inp.evaluate(s),f_s)
        if (i+initial_n<=25 and i % 5 == 0) or (i+initial_n>25 and i % hyper_opt_interval == 0):
            m.optimize()

        # finding the next point for sampling
        D = lhs(low_dim, 2000) * 2.0 * box_size - box_size
        mu, var = m.predict(kern_inp.evaluate(D))
        ei_d = afn(len(D), max(f_s), mu, var)
        index = np.argmax(ei_d)
        s = np.append(s, [D[index]], axis=0)
        f_s = np.append(f_s, test_func.evaluate(cnv_prj.evaluate([D[index]])), axis=0)
        f_s_true = np.append(f_s_true, test_func.evaluate_true(cnv_prj.evaluate([D[index]])), axis=0)

        #Collecting data
        stop = timeit.default_timer()
        best_results[0,i + initial_n]=np.max(f_s_true)
        elapsed[0, i + initial_n] = stop - start

    # if func_type == 'WalkerSpeed':
    #     eng.quit()

    return best_results, elapsed, s, f_s, f_s_true, cnv_prj.evaluate(s)

if __name__=='__main__':
    res,_, s, f_s, fs_true, high_s =RunRembo(low_dim=2, high_dim=100, func_type='StybTang', initial_n=10,
                                             total_itr=50, kern_inp_type='Y', ARD=True, noise_var=1)


