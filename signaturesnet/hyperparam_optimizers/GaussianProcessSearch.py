import os
import sys
import json
import pandas
import pathlib
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.learning import GaussianProcessRegressor
from skopt import Optimizer
from skopt.learning.gaussian_process.kernels import ConstantKernel, Matern
from skopt.plots import plot_objective, plot_evaluations
from hyperparam_optimizers import load_save

sys.path.append(os.path.dirname(os.path.abspath(__file__))) # add the current directory to the path

session_params = {}

""" 
EXAMPLE OF USE

lrate_range = Real(name="lrate", low=10e-5, high=10e-4)
k_range = Integer(name="k", low=5, high=30)
batch_range = Integer(name="batch", low=104, high=208)

search_space = [lrate_range, k_range, batch_range]

# Function to be executed at every run of the optimizer (n_calls) for each set of parameters values.
def run(lrate, k, batch):
    print("lrate:", lrate)
    print("k:", k)
    print("batch:", batch)

    val = train(lrate, k, batch) # Send the parameters to the function to be evaulated.
    return val

out_file = "out.csv"

# Reuse previous runs in case there are.
if path.exists(out_file) and os.path.getsize(out_file) > 0:
    in_file = out_file 
else:
    in_file = None
    
gp_search = GaussianProcessSearch(search_space = search_space, # Range of the parameters to be tested.
                                  fixed_space = {}, # Parameters that will remain untouched.
                                  evaluator = run,
                                  input_file = in_file, # File with previous runs (if exists).
                                  output_file = out_file)

gp_search.init_session() # Initialize the session variables.

best_params, best_value = gp_search.get_maximum(
    n_calls = 50, # Number of evaluations.
    n_random_starts = 5, # Number of random starts.
    noise = 0.01, # Noise of the data.
    verbose = True,
    plot_results = False)

print("best_params:", best_params)
print("best_value:", best_value)
"""

def load(data_file):
    """
    Load the given data file and returns a dictionary of the values.

    Arguments:
        data_file (str): Path to a file containing data in one of the known formats (json, csv).

    Returns:
        A dictionary with the loaded data.
    """
    _, ext = os.path.splitext(data_file)
    try:
        return known_file_types[ext]['load'](data_file)
    except KeyError:
        raise Exception('Error loading file: type ' + str(ext) + ' is not supported')

def save(data_file, dictionary):
    """
    Save the given dictionary in the given file. Format is determined by data_file extension.

    Arguments:
        data_file (str): Path to a file in which to save the data. Extension is used to determine.
        the format, therefore the path must contain an extension.
        dictionary (dict): Dictionary with the data.

    """
    _, ext = os.path.splitext(data_file)
    try:
        known_file_types[ext]['save'](data_file, dictionary)
    except KeyError:
        raise Exception('Error loading file: type ' + str(ext) + ' is not supported')

def _load_json(data_file):
    with open(data_file, 'r') as file:
        return json.load(file)

def _save_json(data_file, dictionary):
    with open(data_file, 'w') as file:
        json.dump(dictionary, file)

def _load_csv(data_file):
    data_frame = pandas.read_csv(data_file, header=0)
    res = {}
    for k in data_frame:
        res[k] = data_frame[k].tolist()
    return res

def _save_csv(data_file, dictionary):
    data_frame = pandas.DataFrame.from_dict(data=dictionary, orient='columns')
    data_frame.to_csv(data_file, header=True, index=False)

known_file_types = {'.json':{'load': _load_json,
                             'save': _save_json},
                    '.csv':{'load': _load_csv,
                            'save': _save_csv}}

class GaussianProcessSearch:
    def __init__(self, search_space, fixed_space, evaluator, input_file = None, output_file = None):
        """
        Instantiate the GaussianProcessSearch and create the GaussianProcessRegressor.

        Arguments:
            search_space (list): List of skopt.space.Dimension objects (Integer, Real, or Categorical) whose name must match the correspondent variable name in the evaluator function.
            fixed_space (dict): Dictionary of parameters that will be passed by default to theevaluator function. The keys must match the correspondent names in the function.
            evaluator (function): Function of which we want to estimate the maximum. It must take the union of search_space and fixed_space as parameters and return a scalar value.
            input_file (str): Path to the file containing points in the search space and corresponding values that are already known.
            output_file (str): Path to the file where updated results will be stored.
        """
        self.search_space = search_space
        self.fixed_space = fixed_space
        self.evaluator = evaluator
        self.input_file = input_file
        self.output_file = output_file
        self.x_values = []
        self.y_values = []
        self.solutions = []

        if input_file is not None:
            try: # Try to load the data from the input file
                data_dict = load_save.load(data_file=input_file)
                self.x_values, self.y_values = self._extract_values(data_dict)
            except OSError as e:
                raise OSError("Cannot read input file. \n" + str(e))

    @staticmethod
    def _get_gp_regressor(length_scale=1., nu=2.5, noise=0.1):
        """
        Creates the GaussianProcessRegressor model

        Arguments:
            length_scale (Union[float, list]): Length scale of the GP kernel. If float, it is the same for all dimensions, if array each element defines the length scale of the dimension.
            nu (float): Controls the smoothness of the approximation. See https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Matern.html.

        Returns:
            A skopt.learning.GaussianProcessRegressor with the given parameters.
        """
        kernel = ConstantKernel(1.0) * Matern(length_scale=length_scale, nu=nu)
        return GaussianProcessRegressor(kernel=kernel, alpha=noise**2)

    def get_maximum(self, n_calls=10, n_random_starts=5, noise=0.01, verbose=True, plot_results=False):
        """
        Performs Bayesian optimization by iteratively evaluating the given function on points that are likely to be a global maximum.
        After the optimization, the evaluated values are stored in self.x_values and self.y_values and appended to the data file if provided.

        Arguments:
            n_calls (int): Number of iterations.
            n_random_starts (int): Initial random evaluations if no previpus values are provided.
            noise (float): Estimated noise in the data.
            verbose (bool): Whether to print optimization details at each evaluation.
            plot_results (bool): Whether to plot an analysis of the solution.

        Returns:
            A tuple (x, y) with the argmax and max found of the evaluated function.
        """
        x_values = [x for x in self.x_values] if len(self.x_values) > 0 else None
        # Negate y_values because skopt performs minimization instead of maximization
        y_values = [-y for y in self.y_values] if len(self.y_values) > 0 else None
        rand_starts = 2 if len(self.x_values)==0 and n_random_starts==0 else n_random_starts
        res = gp_minimize(func=GaussianProcessSearch.evaluate,
                          dimensions=self.search_space,
                          n_calls=n_calls,
                          n_random_starts=rand_starts,
                          acq_func="EI",
                          acq_optimizer="lbfgs",
                          x0=x_values,
                          y0=y_values,
                          noise=noise,
                          n_jobs=-1,
                          callback=self.__save_res,
                          verbose=verbose)
        if plot_results:
            ax = plot_objective(res)
            plt.show()
            ax = plot_evaluations(res)
            plt.show()

        self.x_values = [[float(val) for val in point] for point in res.x_iters]
        self.y_values = [-val for val in res.func_vals]
        if self.output_file is not None:
            self.save_values()
            try:
                ax = plot_objective(res)
                plt.savefig( self.output_file + "_objective_plot.png")
            except Exception as e:
                print(e)
            try:
                ax = plot_evaluations(res)
                plt.savefig( self.output_file + "_evaluations_plot.png")
            except Exception as e:
                print(e)
        return res.x, -res.fun

    def add_point_value(self, point, value):
        """
        Add a point and the correspondent value to the knowledge.

        Arguments:
            point (Union[list, dict]): List of values correspondent to self.search_space dimensions (in the same order), or dictionary {dimension_name: value} for all the dimensions in self.search_space.
            value (float): Value of the function at the given point.
        """
        p = []
        if isinstance(point, list):
            p = point
        elif isinstance(point, dict):
            for dim in self.search_space:
                p.append(point[dim.name])
        else:
            raise ValueError("Param point of add_point_value must be a list or a dictionary.")
        self.x_values.append(p)
        self.y_values.append(value)

    def get_next_candidate(self, n_points):
        """
        Returns the next candidates for the skopt acquisition function.

        Arguments:
            n_points (int): Number of candidates desired.

        Returns:
            List of points that would be chosen by gp_minimize as next candidate.
        """
        # Negate y_values because skopt performs minimization instead of maximization
        y_values = [-y for y in self.y_values]
        optimizer = Optimizer(dimensions=self.search_space,
                              base_estimator="gp",
                              n_initial_points=len(self.x_values),
                              acq_func="EI")
        
        optimizer.tell(self.x_values, y_values)  # TODO Does this fit the values???
        points = optimizer.ask(n_points=n_points)
        return self._to_dict_list(points)

    def get_random_candidate(self, n_points):
        candidates = []
        for _ in range(n_points):
            candidate = {}
            for elem in self.search_space:
                candidate[str(elem.name)] = elem.rvs(n_samples=1)[0]
            candidates.append(candidate)
        return candidates

    def _to_dict_list(self, points):
        """
        Transform the list of points in a list of dictionaries {dimension_name: value}.

        Arguments:
            points (list): List of lists of value, where for each list, the i-th element corresponds to a value for the i-th dimension of the search space.

        Returns:
            A list of dictionaries, where each dictionary has the search space dimensions as keys and the correspondent value of points, in the self.search_space order.
        """
        def to_dict(point):
            d = {}
            for i, dim in enumerate(self.search_space):
                d[dim.name] = point[i]
            return d
        return [to_dict(p) for p in points]

    def init_session(self):
        """
        Save in session variables. the parameters that will be passed to the evaluation function by default.
        """
        global session_params
        session_params["fixed_space"] = self.fixed_space
        session_params["evaluator"] = self.evaluator
        session_params["dimension_names"] = [dim.name for dim in self.search_space]

    def reset_session(self):
        """
        Reset session variables.
        """
        global session_params
        session_params = {}

    def _extract_values(self, data_dict):
        """
        Extracts the x values and target values from the given data dictionary.

         Arguments:
             data_dict (dict): A dictionaty like: {<param_name>: [list of values]} where all lists have the same length and values at same index belong to the same point. The only exception is data_dict["value"] that must contain a list of float correspondent to the function evaluations in the points.

         Returns:
             A tuple (x_values, y_values) where
                 x_values (list): List of points in the search space.
                 y_values (list): List of known values for the x_values points.
        """
        y_values = data_dict["value"]
        x_values = []
        for i, dimension in enumerate(self.search_space):
            name = dimension.name
            try:
                for j, v in enumerate(data_dict[name]):
                    if i == 0:  # If first dimension, instantiate an array for data point
                        x_values.append([])
                    x_values[j].append(data_dict[name][j])
            except KeyError:
                raise KeyError("Search space expects a " + name + " dimension but loaded data "
                                                                  "does not contain it")
        return x_values, y_values

    def _pack_values(self):
        """
        Packs the known values to a dictionary where keys are dimension names.

        Returns: 
            A dictionary {dimension_name: [dimension_values] for all dimensions, value: [result_values]}
        """
        res_dict = {}
        for i, dimension in enumerate(self.search_space):
            res_dict[dimension.name] = []
            for point in self.x_values:
                res_dict[dimension.name].append(point[i])
        res_dict["value"] = self.y_values
        return res_dict

    def save_values(self):
        """
        Save in the data file the known x_values and y_values.
        """
        data_dict = self._pack_values()
        load_save.save(self.output_file, data_dict)

    @staticmethod
    def _to_key_value(values):
        """
        Transform the given list of values in a key-value dictionary from the search_space names.

        Arguments:
            values (list): List of values of the same length as self.search_space.

        Returns:
            A dictionary key[i]: value[i] where key[i] is the name of the i-th dimension of self.search_space and value[i] is the i-th element of values.
        """
        global session_params
        name_value_dict = {}
        for i, name in enumerate(session_params["dimension_names"]):
            name_value_dict[name] = values[i]
        return name_value_dict

    @staticmethod
    def evaluate(point):
        """
        Evaluate the evaluator function at the given point.

        Arguments:
            point (list): List of values each one corresponding to a dimension of self.search_space.

        Returns:
            The value of self.evaluator at the given point, negated (to be used in minimization).
        """
        global session_params
        evaluator_func = session_params["evaluator"]
        fixed_space = session_params["fixed_space"]
        # Transform the point in a mapping param_name=value
        name_value_dict = GaussianProcessSearch._to_key_value(point)
        args = {**fixed_space, **name_value_dict}
        return -evaluator_func(**args)

    def __save_res(self, res):
        self.solutions.append([res.x, res.fun])
        pathlib.Path("gpro_results/").mkdir(parents=True, exist_ok=True)
        numpy_name = "gpro_results/gpro_points.npy" 
        # np.save(numpy_name, self.solutions)
        self.save_checkpoint(res)

    def save_checkpoint(self, res):
        x_values = [[float(val) for val in point] for point in res.x_iters]
        y_values = [-val for val in res.func_vals]

        res_dict = {}
        for i, dimension in enumerate(self.search_space):
            res_dict[dimension.name] = []
            for point in x_values:
                res_dict[dimension.name].append(point[i])
        res_dict["value"] = y_values

        load_save.save(self.output_file, res_dict)
