import math
import numpy as np
import subprocess
import os
import csv


# import matlab.engine
# import torch
# from BOCK_benchmarks.mnist_weight import mnist_weight

# All functions are defined in such a way that have global maximums, THIS IS CHANGED FOR BRANIN FUNC (- in front is removed!!!)
# if a function originally has a minimum, the final objective value is multiplied by -1

class TestFunction:
    def evaluate(self, x, true_eval=None):
        pass


class Rosenbrock(TestFunction):
    def __init__(self, act_var, noise_var=0, param_ranges=None):
        if param_ranges is None:
            self.range = np.array([[-2, 2],
                                   [-2, 2]])
        else:
            self.range = np.transpose(param_ranges)

        self.act_var = act_var
        self.var = noise_var

    def scale_domain(self, x):
        # Scaling the domain
        x_copy = np.copy(x)
        if len(x_copy.shape) == 1:
            x_copy = x_copy.reshape((1, x_copy.shape[0]))
        for i in range(len(self.range)):
            x_copy[:, i] = x_copy[:, i] * (self.range[i, 1] - self.range[i, 0]) / 2 + (
                    self.range[i, 1] + self.range[i, 0]) / 2
        return x_copy

    def evaluate_true(self, x):
        # Calculating the output
        scaled_x = self.scale_domain(x)
        f = [[0]]
        f[0] = [-(math.pow(1 - i[self.act_var[0]], 2) + 100 * math.pow(
            i[self.act_var[1]] - math.pow(i[self.act_var[0]], 2), 2)) for i in scaled_x]
        f = np.transpose(f)
        return f

    def evaluate(self, x, true_eval=None):
        scaled_x = self.scale_domain(x)
        n = len(scaled_x)
        if true_eval is not None:
            return true_eval + np.random.normal(0, self.var, (n, 1))
        return self.evaluate_true(x) + np.random.normal(0, self.var, (n, 1))


class Branin(TestFunction):
    def __init__(self, act_var, noise_var=0, param_ranges=None):
        if param_ranges is None:
            self.range = np.array([[-5, 10],
                                   [0, 15]])
        else:
            self.range = np.transpose(param_ranges)

        self.act_var = act_var
        self.var = noise_var

    def scale_domain(self, x):
        # Scaling the domain
        x_copy = np.copy(x)
        if len(x_copy.shape) == 1:
            x_copy = x_copy.reshape((1, x_copy.shape[0]))
        for i in range(len(self.range)):
            x_copy[:, i] = x_copy[:, i] * (self.range[i, 1] - self.range[i, 0]) / 2 + (
                    self.range[i, 1] + self.range[i, 0]) / 2
        return x_copy

    def evaluate_true(self, x):
        scaled_x = self.scale_domain(x)
        # Calculating the output
        f = [[0]]
        f[0] = [-((i[self.act_var[1]] - (5.1 / (4 * math.pi ** 2)) * i[self.act_var[0]] ** 2 + i[
            self.act_var[0]] * 5 / math.pi - 6) ** 2 + 10 * (
                          1 - 1 / (8 * math.pi)) * np.cos(i[self.act_var[0]]) + 10) for i in scaled_x]
        f = np.transpose(f)
        return f

    def evaluate(self, x, true_eval=None):
        scaled_x = self.scale_domain(x)
        n = len(scaled_x)
        if true_eval is not None:
            return true_eval + np.random.normal(0, self.var, (n, 1))
        return self.evaluate_true(x) + np.random.normal(0, self.var, (n, 1))


class Hartmann6(TestFunction):
    def __init__(self, act_var, noise_var=0, param_ranges=None):
        if param_ranges is None:
            self.range = np.array([[0, 1],
                                   [0, 1],
                                   [0, 1],
                                   [0, 1],
                                   [0, 1],
                                   [0, 1]])
        else:
            self.range = np.transpose(param_ranges)

        self.act_var = act_var
        self.var = noise_var

    def scale_domain(self, x):
        # Scaling the domain
        x_copy = np.copy(x)
        if len(x_copy.shape) == 1:
            x_copy = x_copy.reshape((1, x_copy.shape[0]))
        for i in range(len(self.range)):
            x_copy[:, i] = x_copy[:, i] * (self.range[i, 1] - self.range[i, 0]) / 2 + (
                    self.range[i, 1] + self.range[i, 0]) / 2
        return x_copy

    def evaluate_true(self, x):
        # Calculating the output
        # Created on 08.09.2016
        # @author: Stefan Falkner
        alpha = [1.00, 1.20, 3.00, 3.20]
        A = np.array([[10.00, 3.00, 17.00, 3.50, 1.70, 8.00],
                      [0.05, 10.00, 17.00, 0.10, 8.00, 14.00],
                      [3.00, 3.50, 1.70, 10.00, 17.00, 8.00],
                      [17.00, 8.00, 0.05, 10.00, 0.10, 14.00]])
        P = 0.0001 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                               [2329, 4135, 8307, 3736, 1004, 9991],
                               [2348, 1451, 3522, 2883, 3047, 6650],
                               [4047, 8828, 8732, 5743, 1091, 381]])
        scaled_x = self.scale_domain(x)
        n = len(scaled_x)
        external_sum = np.zeros((n, 1))
        for r in range(n):
            for i in range(4):
                internal_sum = 0
                for j in range(6):
                    internal_sum = internal_sum + A[i, j] * (scaled_x[r, self.act_var[j]] - P[i, j]) ** 2
                external_sum[r] = external_sum[r] + alpha[i] * np.exp(-internal_sum)
        return external_sum

    def evaluate(self, x, true_eval=None):
        scaled_x = self.scale_domain(x)
        n = len(scaled_x)
        if true_eval is not None:
            return true_eval + np.random.normal(0, self.var, (n, 1))
        return self.evaluate_true(x) + np.random.normal(0, self.var, (n, 1))


class StybTang(TestFunction):
    def __init__(self, act_var, noise_var=0, param_ranges=None):
        D = len(act_var)
        a = np.ones((D, 2))

        if param_ranges is None:
            a = a * 5  # [-5, 5]
            a[:, 0] = a[:, 0] * -1
        else:
            a = np.transpose(param_ranges)

        self.range = a
        self.act_var = act_var
        self.var = noise_var

    def scale_domain(self, x):
        # Scaling the domain
        x_copy = np.copy(x)
        if len(x_copy.shape) == 1:
            x_copy = x_copy.reshape((1, x_copy.shape[0]))
        for i in range(len(self.range)):
            x_copy[:, i] = x_copy[:, i] * (self.range[i, 1] - self.range[i, 0]) / 2 + (
                    self.range[i, 1] + self.range[i, 0]) / 2
        return x_copy

    def evaluate_true(self, x):
        # Calculating the output
        scaled_x = self.scale_domain(x)
        f = [-0.5 * np.sum(np.power(scaled_x, 4) - 16 * np.power(scaled_x, 2) + 5 * scaled_x, axis=1)]
        f = np.transpose(f)
        return f

    def evaluate(self, x, true_eval=None):
        scaled_x = self.scale_domain(x)
        n = len(scaled_x)
        if true_eval is not None:
            return true_eval + np.random.normal(0, self.var, (n, 1))
        return self.evaluate_true(x) + np.random.normal(0, self.var, (n, 1))


class Quadratic(TestFunction):
    def __init__(self, act_var=None, noise_var=0):
        self.range = np.array([[-1, 1],
                               [-1, 1]])
        if act_var is None:
            self.act_var = np.arange(self.range.shape[0])
        else:
            self.act_var = act_var
        self.var = noise_var

    def scale_domain(self, x):
        # Scaling the domain
        x_copy = np.copy(x)
        if len(x_copy.shape) == 1:
            x_copy = x_copy.reshape((1, x_copy.shape[0]))
        for i in range(len(self.range)):
            x_copy[:, i] = x_copy[:, i] * (self.range[i, 1] - self.range[i, 0]) / 2 + (
                    self.range[i, 1] + self.range[i, 0]) / 2
        return x_copy

    def evaluate_true(self, x):
        scaled_x = self.scale_domain(x)
        f = [[0]]
        f[0] = [-((i[self.act_var[0]] - 1) ** 2 + (i[self.act_var[1]] - 1) ** 2) for i in scaled_x]
        f = np.transpose(f)
        return f

    def evaluate(self, x, true_eval=None):
        scaled_x = self.scale_domain(x)
        n = len(scaled_x)
        if true_eval is not None:
            return true_eval + np.random.normal(0, self.var, (n, 1))
        return self.evaluate_true(x) + np.random.normal(0, self.var, (n, 1))


class Ackley(TestFunction):
    def __init__(self, act_var=None, noise_var=0, param_ranges=None):
        D = len(act_var)
        a = np.ones((D, 2))

        if param_ranges is None:
            a = a * 40  # [-40, 40]
            a[:, 0] = a[:, 0] * -1
        else:
            a = np.transpose(param_ranges)

        self.range = a
        self.act_var = act_var
        self.var = noise_var

    def scale_domain(self, x):
        # Scaling the domain
        x_copy = np.copy(x)
        if len(x_copy.shape) == 1:
            x_copy = x_copy.reshape((1, x_copy.shape[0]))
        for i in range(len(self.range)):
            x_copy[:, i] = x_copy[:, i] * (self.range[i, 1] - self.range[i, 0]) / 2 + (
                    self.range[i, 1] + self.range[i, 0]) / 2

        return x_copy

    def downscale_domain(self, x):
        # Scaling the domain
        x_copy = np.copy(x)
        if len(x_copy.shape) == 1:
            x_copy = x_copy.reshape((1, x_copy.shape[0]))
        for i in range(len(self.range)):
            x_copy[:, i] = (2.0 * x_copy[:, i] - (self.range[i, 1] + self.range[i, 0])) / (
                    self.range[i, 1] - self.range[i, 0])
        return x_copy

    def normalize_domain(self, x):
        # Scaling the domain
        x_copy = np.copy(x)
        if len(x_copy.shape) == 1:
            x_copy = x_copy.reshape((1, x_copy.shape[0]))
        for i in range(len(self.range)):
            x_copy[:, i] = (x_copy[:, i] - self.range[i, 0]) / (self.range[i, 1] - self.range[i, 0])
        return x_copy

    def evaluate_true(self, x):
        scaled_x = self.scale_domain(x)
        n = float(scaled_x.shape[1])
        t_first = -0.2 * np.sqrt(np.sum(np.power(scaled_x, 2), axis=1) / n)
        t_second = np.sum(np.cos(2.0 * np.pi * scaled_x), axis=1) / n
        f = [-(-20.0 * np.exp(t_first) - np.exp(t_second) + 20.0 + np.exp(1))]
        f = np.transpose(f)
        return f

    def evaluate(self, x, true_eval=None):
        scaled_x = self.scale_domain(x)
        n = len(scaled_x)
        if true_eval is not None:
            return true_eval + np.random.normal(0, self.var, (n, 1))
        return self.evaluate_true(x) + np.random.normal(0, self.var, (n, 1))

class Griewank(TestFunction):
    def __init__(self, act_var=None, noise_var=0, param_ranges=None):
        D = len(act_var)
        a = np.ones((D, 2))
        if param_ranges is None:
            a = a * 600
            a[:, 0] = a[:, 0] * -1
        else:
            a = np.transpose(param_ranges)

        self.range = a
        self.act_var = act_var
        self.var = noise_var

    def scale_domain(self, x):
        # Scaling the domain
        x_copy = np.copy(x)
        if len(x_copy.shape) == 1:
            x_copy = x_copy.reshape((1, x_copy.shape[0]))
        for i in range(len(self.range)):
            x_copy[:, i] = x_copy[:, i] * (self.range[i, 1] - self.range[i, 0]) / 2 + (
                    self.range[i, 1] + self.range[i, 0]) / 2

        return x_copy

    def downscale_domain(self, x):
        # Scaling the domain
        x_copy = np.copy(x)
        if len(x_copy.shape) == 1:
            x_copy = x_copy.reshape((1, x_copy.shape[0]))
        for i in range(len(self.range)):
            x_copy[:, i] = (2.0 * x_copy[:, i] - (self.range[i, 1] + self.range[i, 0])) / (
                    self.range[i, 1] - self.range[i, 0])
        return x_copy

    def evaluate_true(self, x):
        scaled_x = self.scale_domain(x)
        n = float(scaled_x.shape[1])

        ii = np.arange(1, n + 1)
        sum_i = np.sum(np.power(scaled_x, 2) / 4000.0, axis=1)
        prod_i = np.prod(np.cos(scaled_x / np.sqrt(ii)), axis=1)

        f = [-(sum_i - prod_i + 1.0)]
        f = np.transpose(f)
        return f

    def evaluate(self, x, true_eval=None):
        scaled_x = self.scale_domain(x)
        n = len(scaled_x)
        if true_eval is not None:
            return true_eval + np.random.normal(0, self.var, (n, 1))
        return self.evaluate_true(x) + np.random.normal(0, self.var, (n, 1))

class Levy(TestFunction):
    def __init__(self, act_var=None, noise_var=0, param_ranges=None):
        D = len(act_var)
        a = np.ones((D, 2))
        if param_ranges is None:
            a = a * 10
            a[:, 0] = a[:, 0] * -1
        else:
            a = np.transpose(param_ranges)

        self.range = a
        self.act_var = act_var
        self.var = noise_var

    def scale_domain(self, x):
        # Scaling the domain
        x_copy = np.copy(x)
        if len(x_copy.shape) == 1:
            x_copy = x_copy.reshape((1, x_copy.shape[0]))
        for i in range(len(self.range)):
            x_copy[:, i] = x_copy[:, i] * (self.range[i, 1] - self.range[i, 0]) / 2 + (
                    self.range[i, 1] + self.range[i, 0]) / 2

        return x_copy

    def downscale_domain(self, x):
        # Scaling the domain
        x_copy = np.copy(x)
        if len(x_copy.shape) == 1:
            x_copy = x_copy.reshape((1, x_copy.shape[0]))
        for i in range(len(self.range)):
            x_copy[:, i] = (2.0 * x_copy[:, i] - (self.range[i, 1] + self.range[i, 0])) / (
                    self.range[i, 1] - self.range[i, 0])
        return x_copy

    def evaluate_true_old(self, x):
        scaled_x = self.scale_domain(x)
        n = scaled_x.shape[1]

        w = (scaled_x - 1.0) / 4.0 + 1.0
        output = np.sin(math.pi * w[:, :1]) ** 2.0
        for i in range(n - 1):
            output += (w[:, i:i + 1] - 1.0) ** 2 * (1.0 + 10.0 * np.sin(math.pi * w[:, i:i + 1] + 1.0) ** 2.0)
        output += ((w[:, -1:] - 1.0) ** 2 * (1.0 + np.sin(2 * math.pi * w[:, -1:]) ** 2.0))

        f = [-output]
        f = np.transpose(f)
        return f

    def evaluate_true(self, x):
        scaled_x = self.scale_domain(x)
        f = [[0]]
        f[0] = np.zeros(scaled_x.shape[0])
        for i in range(scaled_x.shape[0]):
            w = np.array([1 + (j - 1) / 4 for j in scaled_x[i]])
            w_1 = 1 + (scaled_x[i][self.act_var[0]] - 1) / 4
            w_d = 1 + (scaled_x[i][self.act_var[-1]] - 1) / 4
            w_d_1 = np.delete(w, self.act_var[-1])
            f[0][i] = -1 * ((math.sin(math.pi * w_1)) ** 2 + np.sum(np.power(w_d_1 - 1, 2) *
                                                                    (1 + 10 * np.power(np.sin(math.pi * w_d_1 + 1),
                                                                                       2))) + ((w_d - 1) ** 2) * (
                                        1 + (math.sin(2 * math.pi * w_d)) ** 2))
        f = np.transpose(f)
        return f

    def evaluate(self, x, true_eval=None):
        scaled_x = self.scale_domain(x)
        n = len(scaled_x)
        if true_eval is not None:
            return true_eval + np.random.normal(0, self.var, (n, 1))
        return self.evaluate_true(x) + np.random.normal(0, self.var, (n, 1))

class Schwefel(TestFunction):
    def __init__(self, act_var=None, noise_var=0, param_ranges=None):
        D = len(act_var)
        a = np.ones((D, 2))
        if param_ranges is None:
            a = a * 500
            a[:, 0] = a[:, 0] * -1
        else:
            a = np.transpose(param_ranges)

        self.range = a
        self.act_var = act_var
        self.var = noise_var

    def scale_domain(self, x):
        # Scaling the domain
        x_copy = np.copy(x)
        if len(x_copy.shape) == 1:
            x_copy = x_copy.reshape((1, x_copy.shape[0]))
        for i in range(len(self.range)):
            x_copy[:, i] = x_copy[:, i] * (self.range[i, 1] - self.range[i, 0]) / 2 + (
                    self.range[i, 1] + self.range[i, 0]) / 2

        return x_copy

    def downscale_domain(self, x):
        # Scaling the domain
        x_copy = np.copy(x)
        if len(x_copy.shape) == 1:
            x_copy = x_copy.reshape((1, x_copy.shape[0]))
        for i in range(len(self.range)):
            x_copy[:, i] = (2.0 * x_copy[:, i] - (self.range[i, 1] + self.range[i, 0])) / (
                    self.range[i, 1] - self.range[i, 0])
        return x_copy

    def evaluate_true(self, x):
        scaled_x = self.scale_domain(x)
        n = float(scaled_x.shape[1])

        term_1 = np.sum(scaled_x * np.sin(np.abs(scaled_x) ** 0.5), axis=1)

        f = [-(418.9829 * n - term_1)]
        f = np.transpose(f)
        return f

    def evaluate(self, x, true_eval=None):
        scaled_x = self.scale_domain(x)
        n = len(scaled_x)
        if true_eval is not None:
            return true_eval + np.random.normal(0, self.var, (n, 1))
        return self.evaluate_true(x) + np.random.normal(0, self.var, (n, 1))

class PredaySimmobility(TestFunction):
    def __init__(self, act_var=None, noise_var=0, param_ranges=None, home_dir=None):
        D = len(act_var)
        a = np.ones((D, 2))
        if param_ranges is None:
            a = a * 10  # [-10, 10]
            a[:, 0] = a[:, 0] * -1
        else:
            a = np.transpose(param_ranges)

        self.range = a
        self.act_var = act_var
        self.var = noise_var
        self.home_dir = os.path.dirname(home_dir)

    def scale_domain(self, x):
        # Scaling the domain
        x_copy = np.copy(x)
        if len(x_copy.shape) == 1:
            x_copy = x_copy.reshape((1, x_copy.shape[0]))
        for i in range(len(self.range)):
            x_copy[:, i] = x_copy[:, i] * (self.range[i, 1] - self.range[i, 0]) / 2 + (
                    self.range[i, 1] + self.range[i, 0]) / 2
        return x_copy

    def downscale_domain(self, x):
        # Scaling the domain
        x_copy = np.copy(x)
        if len(x_copy.shape) == 1:
            x_copy = x_copy.reshape((1, x_copy.shape[0]))
        for i in range(len(self.range)):
            x_copy[:, i] = (2.0 * x_copy[:, i] - (self.range[i, 1] + self.range[i, 0])) / (
                    self.range[i, 1] - self.range[i, 0])
        return x_copy

    def evaluate_true(self, x):
        scaled_x = self.scale_domain(x)
        input_set_f = self.home_dir + "/params_for_eval.csv"
        output_set_f = self.home_dir + "/eval_values.csv"

        fileObject = open(input_set_f, 'w')
        writer = csv.writer(fileObject)
        writer.writerows(scaled_x)
        fileObject.close()
        call_stack = subprocess.call(
            ["Rscript", "--vanilla", self.home_dir + "/external.R", self.home_dir, input_set_f, output_set_f])
        # if call_stack != 0 => Error has occurred!
        f = [-np.genfromtxt(output_set_f, delimiter=',')]
        f = np.transpose(f)

        if f.ndim == 1:
            f = f.reshape(1, 1)
        return f

    def evaluate(self, x, true_eval=None):
        n = len(x)
        if true_eval is not None:
            return true_eval + np.random.normal(0, self.var, (n, 1))
        return self.evaluate_true(x) + np.random.normal(0, self.var, (n, 1))


class Levy13(TestFunction):
    def __init__(self, act_var, noise_var=0, param_ranges=None):
        if param_ranges is None:
            self.range = np.array([[-10, 10],
                                   [-10, 10]])
        else:
            self.range = np.transpose(param_ranges)

        self.act_var = act_var
        self.var = noise_var

    def scale_domain(self, x):
        # Scaling the domain
        x_copy = np.copy(x)
        if len(x_copy.shape) == 1:
            x_copy = x_copy.reshape((1, x_copy.shape[0]))
        for i in range(len(self.range)):
            x_copy[:, i] = x_copy[:, i] * (self.range[i, 1] - self.range[i, 0]) / 2 + (
                    self.range[i, 1] + self.range[i, 0]) / 2
        return x_copy

    def evaluate_true(self, x):
        scaled_x = self.scale_domain(x)
        # Calculating the output
        f = [[0]]
        f[0] = [-1 * ((math.sin(3 * math.pi * i[self.act_var[0]])) ** 2 + ((i[self.act_var[0]] - 1) ** 2) * (
                    1 + (math.sin(3 * math.pi * i[self.act_var[1]])) ** 2) + ((i[self.act_var[1]] - 1) ** 2) * (1
                                                                                                                + (
                                                                                                                    math.sin(
                                                                                                                        2 * math.pi *
                                                                                                                        i[
                                                                                                                            self.act_var[
                                                                                                                                1]])) ** 2))
                for i in scaled_x]
        f = np.transpose(f)
        return f

    def evaluate(self, x, true_eval=None):
        scaled_x = self.scale_domain(x)
        n = len(scaled_x)
        if true_eval is not None:
            return true_eval + np.random.normal(0, self.var, (n, 1))
        return self.evaluate_true(x) + np.random.normal(0, self.var, (n, 1))

class Hartmann3(TestFunction):
    def __init__(self, act_var, noise_var=0, param_ranges=None):
        if param_ranges is None:
            self.range = np.array([[0, 1], [0, 1], [0, 1]])
        else:
            self.range = np.transpose(param_ranges)

        self.act_var = act_var
        self.var = noise_var

    def scale_domain(self,x):
        # Scaling the domain
        x_copy = np.copy(x)
        if len(x_copy.shape) == 1:
            x_copy = x_copy.reshape((1, x_copy.shape[0]))
        for i in range(len(self.range)):
            x_copy[:, i] = x_copy[:, i] * (self.range[i, 1] - self.range[i, 0]) / 2 + (
                    self.range[i, 1] + self.range[i, 0]) / 2
        return x_copy

    def evaluate_true(self, x):
        alpha = [1.00, 1.20, 3.00, 3.20]
        matrix_a = np.array([[3.00, 10.00, 30.00],
                            [0.10, 10.00, 35.00],
                            [3.00, 10.00, 30.00],
                            [0.10, 10.00, 35.00]])
        matrix_p = 0.0001 * np.array([[3689, 1170, 2673],
                                      [4699, 4387, 7470],
                                      [1091, 8732, 5547],
                                      [381, 5743, 8828]])
        scaled_x = self.scale_domain(x)
        n = scaled_x.shape[0]
        external_sum = np.zeros((n, 1))
        for r in range(n):
            for i in range(4):
                internal_sum = 0
                for j in range(3):
                    internal_sum = internal_sum + matrix_a[i, j] * (scaled_x[r, self.act_var[j]] - matrix_p[i, j]) ** 2
                external_sum[r] = external_sum[r] + alpha[i] * np.exp(-internal_sum)
        return external_sum

    def evaluate(self, x, true_eval=None):
        scaled_x = self.scale_domain(x)
        n = len(scaled_x)
        if true_eval is not None:
            return true_eval + np.random.normal(0, self.var, (n, 1))
        return self.evaluate_true(x) + np.random.normal(0, self.var, (n, 1))