import argparse
import functools
import os.path
import pickle
import sys
import time
from datetime import datetime
from random import seed
import inspect

EXPERIMENT_DIR = '/Users/vladimir/Workspaces/PyCharm/HyperSphere/Experiments'
CONSOLE_PRINT = False
import torch
from torch.autograd import Variable
import torch.multiprocessing as multiprocessing

if os.path.realpath(__file__).rsplit('/', 3)[0] not in sys.path:
	sys.path.append(os.path.realpath(__file__).rsplit('/', 3)[0])

from HyperSphere.BO.acquisition.acquisition_maximization import suggest, optimization_candidates, optimization_init_points, deepcopy_inference, expected_improvement, probability_improvement
from HyperSphere.GP.models.gp_regression import GPRegression
from HyperSphere.test_functions.benchmarks import *

# Kernels
from HyperSphere.GP.kernels.modules.matern52 import Matern52
from HyperSphere.GP.kernels.modules.radialization import RadializationKernel

# Inferences
from HyperSphere.GP.inference.inference import Inference
from HyperSphere.BO.shadow_inference.inference_sphere_satellite import ShadowInference as satellite_ShadowInference
from HyperSphere.BO.shadow_inference.inference_sphere_origin import ShadowInference as origin_ShadowInference
from HyperSphere.BO.shadow_inference.inference_sphere_origin_satellite import ShadowInference as both_ShadowInference

# feature_map
from HyperSphere.feature_map.modules.kumaraswamy import Kumaraswamy

# boundary conditions
from HyperSphere.feature_map.functionals import sphere_bound


def BO(geometry=None, n_eval=200, path=None, func=None, ndim=None, boundary=False, ard=False, origin=False, warping=False, parallel=False, global_constants_dir=None, output_path=None, acq_function=expected_improvement):
	assert (path is None) != (func is None)

	if path is None:
		assert (func.dim == 0) != (ndim is None)
		assert geometry is not None

		if func.__name__ == "PredaySimmobility":
			func.__defaults__ = (None, EXPERIMENT_DIR)

		if ndim is None:
			ndim = func.dim

		# Load the pickle with lower and upper bounds, if exists
		lower_param_bounds = -40
		upper_param_bounds = 40
		if global_constants_dir is not None and os.path.exists(global_constants_dir + '/space_range_def'):
			with open(global_constants_dir + '/space_range_def', 'rb') as pickle_file:
				param_bounds = pickle.load(pickle_file)
				lower_param_bounds = torch.from_numpy(param_bounds[0]).float()
				upper_param_bounds = torch.from_numpy(param_bounds[1]).float()

		exp_conf_str = geometry
		if geometry == 'sphere':
			assert not ard
			exp_conf_str += 'warping' if warping else ''
			radius_input_map = Kumaraswamy(ndim=1, max_input=ndim ** 0.5) if warping else None
			model = GPRegression(kernel=RadializationKernel(max_power=3, search_radius=ndim ** 0.5, radius_input_map=radius_input_map))
			inference_method = None
			if origin and boundary:
				inference_method = both_ShadowInference
				exp_conf_str += 'both'
			elif origin:
				inference_method = origin_ShadowInference
				exp_conf_str += 'origin'
			elif boundary:
				inference_method = satellite_ShadowInference
				exp_conf_str += 'boundary'
			else:
				inference_method = Inference
				exp_conf_str += 'none'
			bnd = sphere_bound(ndim ** 0.5)
		elif geometry == 'cube':
			assert not origin
			exp_conf_str += ('ard' if ard else '') + ('boundary' if boundary else '')
			model = GPRegression(kernel=Matern52(ndim=ndim, ard=ard))
			inference_method = satellite_ShadowInference if boundary else Inference
			bnd = (-1, 1)

		if not os.path.isdir(EXPERIMENT_DIR):
			raise ValueError('In file : ' + os.path.realpath(__file__) + '\nEXPERIMENT_DIR variable is not properly assigned. Please check it.')
		dir_list = [elm for elm in os.listdir(EXPERIMENT_DIR) if os.path.isdir(os.path.join(EXPERIMENT_DIR, elm))]
		folder_name = func.__name__ + '_D' + str(ndim) + '_' + exp_conf_str + '_' + datetime.now().strftime('%Y%m%d-%H:%M:%S:%f')
		os.makedirs(os.path.join(EXPERIMENT_DIR, folder_name))
		logfile_dir = os.path.join(EXPERIMENT_DIR, folder_name, 'log')
		os.makedirs(logfile_dir)
		model_filename = os.path.join(EXPERIMENT_DIR, folder_name, 'model.pt')
		data_config_filename = os.path.join(EXPERIMENT_DIR, folder_name, 'data_config.pkl')

		## TODO: set the x_input from file, if exists
		x_input = None
		output = None
		if global_constants_dir is not None and os.path.exists(global_constants_dir + '/init_sample'):

			with open(global_constants_dir + '/init_sample', 'rb') as pickle_file:
				x_input = torch.from_numpy(pickle.load(pickle_file)).float()

			if x_input.size(1) > ndim:
				output = x_input[:, -1]
				x_input = x_input[:, :-1]

				origin_mask = torch.sum(x_input ** 2, 1) == 0
				origin_mask_tensor = torch.sum(origin_mask).detach()
				n_origin = origin_mask_tensor.item() if len(origin_mask_tensor.shape) == 0 else origin_mask_tensor[0]

				n_best_initial_ind = torch.min(output, 0)[1]
				if n_origin == 0:
					x_input = torch.stack((torch.zeros(ndim), x_input[n_best_initial_ind]), dim=0)
					output = torch.cat(( torch.reshape(func(x_input[0]), (1,1)), torch.reshape(output[n_best_initial_ind], (1, 1)) ))
				else:
					x_input = torch.stack((x_input[origin_mask, :], x_input[n_best_initial_ind]), dim=0)
					output = torch.cat(
						(torch.reshape(output[origin_mask], (1, 1)), torch.reshape(output[n_best_initial_ind], (1, 1))))
			else:
				x_input = torch.stack([torch.zeros(ndim), torch.FloatTensor(ndim).uniform_(-1, 1)])
		else:
			x_input = torch.stack([torch.zeros(ndim), torch.FloatTensor(ndim).uniform_(-1, 1)]).float()
		x_input = Variable(x_input)

		if func.func_name == 'stochastic_depth_resnet_cifar100':
			special_init_point = torch.FloatTensor([-0.88672996375809265, -0.83845553984377363, -0.80082455589209434, -0.76868080609344613, -0.74002860499719103, -0.71384507914214379, -0.6895229479156415, -0.66666666534211871, -0.64500158781765049, -0.62432778870160499, -0.60449429448743319, -0.58538383736427368, -0.56690311453886821, -0.54897644926147593, -0.53154137077618735, -0.51454570980003023, -0.49794520561122835, -0.4817019618876005, -0.46578329447738975, -0.45016063464220946, -0.43480887900991927, -0.41970588594137237, -0.40483184457290511, -0.39016909932337462, -0.37570168000845294, -0.36141512736958714, -0.34729635533386094, -0.33333334161175654, -0.31951507564952675, -0.30583136944490208, -0.29227292909996905, -0.27883100126437665, -0.26549747264739709, -0.25226475894331168, -0.23912574658399377, -0.22607369983030123, -0.2131023835975443, -0.20020577167418563, -0.18737817967669568, -0.1746141913340078, -0.16190858934371632, -0.14925649319813961, -0.13665309066289877, -0.12409378040195429, -0.11157411163518405, -0.099089726169870107, -0.086636502479268351, -0.074210299199806373, -0.061807101474520065, -0.049422967019945307, -0.037054013082912562, -0.024696364163967699, -0.012346298973719083, 0])
			x_input = torch.cat([x_input, Variable(special_init_point)])
		n_init_eval = x_input.size(0)

		if output is None:
			output = Variable(torch.zeros(n_init_eval, 1))
			for i in range(n_init_eval):
				output[i] = func(x_input[i])

		time_list = [time.time()] * n_init_eval
		elapse_list = [0] * n_init_eval
		pred_mean_list = [0] * n_init_eval
		pred_std_list = [0] * n_init_eval
		pred_var_list = [0] * n_init_eval
		pred_stdmax_list = [1] * n_init_eval
		pred_varmax_list = [1] * n_init_eval
		reference_list = [output.data.squeeze()[0]] * n_init_eval
		refind_list = [1] * n_init_eval
		dist_to_ref_list = [0] * n_init_eval
		sample_info_list = [(10, 0, 10)] * n_init_eval

		if parallel is True:
			model.share_memory()

		inference = inference_method((x_input, output), model)
		inference.init_parameters()
		inference.sampling(n_sample=1, n_burnin=99, n_thin=1)
	else:
		if not os.path.exists(path):
			path = os.path.join(EXPERIMENT_DIR, path)
		logfile_dir = os.path.join(path, 'log')
		model_filename = os.path.join(path, 'model.pt')
		data_config_filename = os.path.join(path, 'data_config.pkl')

		model = torch.load(model_filename)
		data_config_file = open(data_config_filename, 'r')
		for key, value in pickle.load(data_config_file).iteritems():
			if key != 'logfile_dir':
				exec(key + '=value')
		data_config_file.close()

	ignored_variable_names = ['n_eval', 'path', 'i', 'key', 'value', 'logfile_dir', 'n_init_eval',
	                          'data_config_file', 'dir_list', 'folder_name', 'model_filename', 'data_config_filename',
	                          'kernel', 'model', 'inference', 'parallel', 'pool', 'param_bounds', 'pickle_file']
	stored_variable_names = set(locals().keys()).difference(set(ignored_variable_names))

	if path is None:
		torch.save(model, model_filename)
		stored_variable = dict()
		for key in stored_variable_names:
			stored_variable[key] = locals()[key]
		f = open(data_config_filename, 'w')
		pickle.dump(stored_variable, f)
		f.close()

	print('Experiment based on data in %s' % os.path.split(model_filename)[0])

	# multiprocessing conflicts with pytorch linear algebra operation
	pool = True if parallel else None #multiprocessing.Pool(N_INIT) if parallel else None

	for _ in range(n_eval):
		start_time = time.time()
		x_input_size = x_input.size(0)
		logfile = open(os.path.join(logfile_dir, str(x_input.size(0) + 1).zfill(4) + '.out'), 'w')
		#logfile = open(os.path.join(logfile_dir, 'iterative_output.out'), 'w')
		inference = inference_method((x_input, output), model)

		reference, ref_ind = torch.min(output, 0)
		reference = reference.data.squeeze()
		reference = reference.item() if len(reference.size()) == 0 else reference[0]
		gp_hyper_params = inference.sampling(n_sample=10, n_burnin=0, n_thin=1)
		inferences = deepcopy_inference(inference, gp_hyper_params)

		x0_cand = optimization_candidates(x_input, output, lower_param_bounds, upper_param_bounds, global_constants_dir=global_constants_dir)
		x0, sample_info = optimization_init_points(x0_cand, reference=reference, inferences=inferences, global_constants_dir=global_constants_dir)
		next_x_point, pred_mean, pred_std, pred_var, pred_stdmax, pred_varmax = suggest(x0=x0, reference=reference, acquisition_function=acq_function, inferences=inferences, bounds=bnd, pool=pool, global_constants_dir=global_constants_dir)

		time_list.append(time.time())
		elapse_list.append(time_list[-1] - time_list[-2])
		pred_mean_list.append(pred_mean.squeeze().item())
		pred_std_list.append(pred_std.squeeze().item())
		pred_var_list.append(pred_var.squeeze().item())
		pred_stdmax_list.append(pred_stdmax.squeeze().item())
		pred_varmax_list.append(pred_varmax.squeeze().item())
		reference_list.append(reference)
		refind_list.append(ref_ind.data.squeeze().item() + 1)
		dist_to_ref_list.append(torch.sum((next_x_point - x_input[ref_ind]).data ** 2) ** 0.5)
		sample_info_list.append(sample_info)

		x_input = torch.cat([x_input, next_x_point], 0)
		output = torch.cat([output, torch.reshape(func(x_input[-1]), (1, 1))])

		min_ind = torch.min(output, 0)[1]
		min_loc = x_input[min_ind]
		min_val = output[min_ind]
		dist_to_suggest = torch.sum((x_input - x_input[-1]).data ** 2, 1) ** 0.5
		dist_to_min = torch.sum((x_input - min_loc).data ** 2, 1) ** 0.5
		out_of_box = torch.sum((torch.abs(x_input.data) > 1), 1)
		if CONSOLE_PRINT is True:
			print('')
		for i in range(x_input.size(0)):
			time_str = time.strftime('%H:%M:%S', time.gmtime(time_list[i])) + '(' + time.strftime('%H:%M:%S', time.gmtime(elapse_list[i])) + ')  '
			data_str = ('%3d-th : %+12.4f(R:%8.4f[%4d]/ref:[%3d]%8.4f), sample([%2d] best:%2d/worst:%2d), '
			            'mean : %+.4E, std : %.4E(%5.4f), var : %.4E(%5.4f), '
			            '2ownMIN : %8.4f, 2curMIN : %8.4f, 2new : %8.4f' %
			            (i + 1, output.data.squeeze()[i], torch.sum(x_input.data[i] ** 2) ** 0.5, out_of_box[i], refind_list[i], reference_list[i],
			             sample_info_list[i][2], sample_info_list[i][0], sample_info_list[i][1],
			             pred_mean_list[i], pred_std_list[i], pred_std_list[i] / pred_stdmax_list[i], pred_var_list[i], pred_var_list[i] / pred_varmax_list[i],
			             dist_to_ref_list[i], dist_to_min[i], dist_to_suggest[i]))
			min_str = '  <========= MIN' if i == min_ind.data.squeeze().item() else ''
			if CONSOLE_PRINT is True:
				print(time_str + data_str + min_str)
			logfile.writelines(time_str + data_str + min_str + '\n')
		logfile.close()

		if os.path.exists(os.path.join(logfile_dir, str(x_input_size).zfill(4) + '.out')):
			os.remove(os.path.join(logfile_dir, str(x_input_size).zfill(4) + '.out'))

		torch.save(model, model_filename)
		stored_variable = dict()
		for key in stored_variable_names:
			stored_variable[key] = locals()[key]
		f = open(data_config_filename, 'w')
		pickle.dump(stored_variable, f)
		f.close()

	if output_path is not None:
		fileObject = open(output_path, 'wb')
		pickle.dump([x_input, output, min_ind, dist_to_suggest, dist_to_min, out_of_box], fileObject)
		fileObject.close()

	# if parallel:
	# 	pool.close()

	if CONSOLE_PRINT is True:
		print('Experiment based on data in %s' % os.path.split(model_filename)[0])

	return os.path.split(model_filename)[0]

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Bayesian Optimization runner')
	parser.add_argument('-g', '--geometry', dest='geometry', help='cube/sphere')
	parser.add_argument('-e', '--n_eval', dest='n_eval', type=int, default=1)
	parser.add_argument('-p', '--path', dest='path')
	parser.add_argument('-d', '--dim', dest='ndim', type=int)
	parser.add_argument('-f', '--func', dest='func_name')
	parser.add_argument('--boundary', dest='boundary', action='store_true', default=False)
	parser.add_argument('--origin', dest='origin', action='store_true', default=False)
	parser.add_argument('--ard', dest='ard', action='store_true', default=False)
	parser.add_argument('--warping', dest='warping', action='store_true', default=False)
	parser.add_argument('--parallel', dest='parallel', action='store_true', default=False)

	parser.add_argument('-x', '--exp_dir', dest='exp_dir')
	parser.add_argument('-c', '--constants_dir', dest='global_constants_dir')
	parser.add_argument('-r', '--random_seed', dest='random_seed')
	parser.add_argument('-q', '--acquisition_function', dest='acquisition_function')

	args = parser.parse_args()
	# if args.n_eval == 0:
	# 	args.n_eval = 3 if args.path is None else 1
	assert (args.path is None) != (args.func_name is None)
	args_dict = vars(args)
	if args.func_name is not None:
		exec 'func=' + args.func_name
		args_dict['func'] = func
	del args_dict['func_name']
	if args.path is None:
		assert (func.dim == 0) != (args.ndim is None)
		assert args.geometry is not None
		if args.geometry == 'sphere':
			assert not args.ard
		elif args.geometry == 'cube':
			assert not args.origin
			assert not args.warping

	if args.exp_dir is not None:
		EXPERIMENT_DIR = args.exp_dir
		args.output_path = EXPERIMENT_DIR + "/method_output"
		del args_dict['exp_dir']

	if args.random_seed is not None:
		seed(a=int(args.random_seed))
	del args_dict['random_seed']

	if args.acquisition_function is not None:
		if args.acquisition_function == "PI":
			args.acq_function = probability_improvement
		else:
			args.acq_function = expected_improvement
		del args_dict['acquisition_function']

	print(BO(**vars(args)))
