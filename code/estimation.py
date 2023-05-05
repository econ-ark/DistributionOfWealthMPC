"""
This is the main file for the cstwMPC project.  It estimates one version of the model
each time it is executed.  The following parameters *must* be defined in the __main__
namespace in order for this file to run correctly:
    
param_name : str
    Which parameter to introduce heterogeneity in (usually DiscFac).
dist_type : str
    Which type of distribution to use (can be 'uniform' or 'lognormal').
do_param_dist : bool
    Do param-dist version if True, param-point if False.
do_lifecycle : bool
    Use lifecycle model if True, perpetual youth if False.
do_agg_shocks : bool
    Whether to solve the FBS aggregate shocks version of the model 
    or use idiosyncratic shocks only.
do_liquid : bool
    Matches liquid assets data when True, net worth data when False.
do_tractable : bool
    Whether to use an extremely simple alternate specification 
    of households' optimization problem.
run_estimation : bool
    Whether to actually estimate the model specified by the other options.
run_sensitivity : [bool]
    Whether to run each of eight sensitivity analyses; currently inoperative.  
    Order: rho, xi_sigma, psi_sigma, mu, urate, mortality, g, R
find_beta_vs_KY : bool
    Whether to computes K/Y ratio for a wide range of beta; should have 
    do_param_dist = False and param_name = 'DiscFac'.
    Currently inoperative.
path_to_models : str
    Absolute path to the location of this file.
    
All of these parameters are set when running this file from one of the do_XXX.py
files in the root directory.
"""

from copy import copy, deepcopy
from time import time

import numpy as np
from HARK.utilities import get_lorenz_shares
from scipy.optimize import minimize, minimize_scalar, root_scalar

from code.agents import AggDoWAgent, AggDoWMarket, DoWAgent, DoWMarket


def get_ky_ratio_difference(
    center, economy, param_name, param_count, spread, dist_type
):
    """
    Finds the difference between simulated and target capital to income ratio in an economy when
    a given parameter has heterogeneity according to some distribution.

    Parameters
    ----------
    economy : CstwMPCMarket
        An object representing the entire economy, containing the various AgentTypes as an attribute.
    param_name : string
        The name of the parameter of interest that varies across the population.
    param_count : int
        The number of different values the parameter of interest will take on.
    center : float
        A measure of centrality for the distribution of the parameter of interest.
    spread : float
        A measure of spread or diffusion for the distribution of the parameter of interest.
    dist_type : string
        The type of distribution to be used.  Can be "lognormal" or "uniform" (can expand).

    Returns
    -------
    diff : float
        Difference between simulated and target capital to income ratio for this economy.
    """
    # Make sure we're not wasting time calculating stuff
    economy.assign_parameters(LorenzBool=False, ManyStatsBool=False)
    # Distribute parameters
    economy.distribute_params(param_name, param_count, center, spread, dist_type)
    economy.solve()
    diff = economy.calc_KY_ratio_difference()
    print(f"get_KY_ratio_difference tried center = {center} and got {diff}")
    return diff


def find_lorenz_distance_at_target_ky(
    spread, economy, param_name, param_count, center_range, dist_type
):
    """
    Finds the sum of squared distances between simulated and target Lorenz points in an economy when
    a given parameter has heterogeneity according to some distribution.  The class of distribution
    and a measure of spread are given as inputs, but the measure of centrality such that the capital
    to income ratio matches the target ratio must be found.

    Parameters
    ----------
    economy : CstwMPCMarket
        An object representing the entire economy, containing the various AgentTypes as an attribute.
    param_name : string
        The name of the parameter of interest that varies across the population.
    param_count : int
        The number of different values the parameter of interest will take on.
    center_range : [float,float]
        Bounding values for a measure of centrality for the distribution of the parameter of interest.
    spread : float
        A measure of spread or diffusion for the distribution of the parameter of interest.
    dist_type : string
        The type of distribution to be used.  Can be "lognormal" or "uniform" (can expand).

    Returns
    -------
    dist : float
        Sum of squared distances between simulated and target Lorenz points for this economy (sqrt).
    """
    # Define the function to search for the correct value of center, then find its zero

    # use more sophisticated discrete distribution with zero mass point at ends of support
    # root finding
    optimal_center = root_scalar(
        get_ky_ratio_difference,
        args=(economy, param_name, param_count, spread, dist_type),
        method="toms748",
        bracket=center_range,
        xtol=10 ** (-6),
    ).root
    economy.center_save = optimal_center

    # Get the sum of squared Lorenz distances given the correct distribution of the parameter
    # Make sure we actually calculate simulated Lorenz points
    economy.assign_parameters(LorenzBool=True)
    # Distribute parameters
    economy.distribute_params(
        param_name, param_count, optimal_center, spread, dist_type
    )
    economy.solve_agents()
    economy.make_history()
    dist = economy.calc_lorenz_distance()
    economy.assign_parameters(LorenzBool=False)
    print(f"find_lorenz_distance_at_target_KY tried spread = {spread} and got {dist}")

    return dist


def get_target_ky_and_find_lorenz_distance(
    x, economy, param_name, param_count, dist_type
):
    center, spread = x
    # Make sure we actually calculate simulated Lorenz points
    economy.assign_parameters(LorenzBool=True, ManyStatsBool=False)
    # Distribute parameters
    economy.distribute_params(param_name, param_count, center, spread, dist_type)
    economy.solve()
    diff = economy.calc_KY_ratio_difference()
    # Get the sum of squared Lorenz distances given the correct distribution of the parameter
    dist = economy.calc_lorenz_distance()
    print(f"get_KY_ratio_difference tried center = {center} and got {diff}")
    print(f"find_lorenz_distance_at_target_KY tried spread = {spread} and got {dist}")

    economy.center_save = center

    return dist + diff**2


def calc_stationary_age_dstn(LivPrb, terminal_period):
    """
    Calculates the steady state proportions of each age given survival probability sequence LivPrb.
    Assumes that agents who die are replaced by a newborn agent with t_age=0.

    Parameters
    ----------
    LivPrb : [float]
        Sequence of survival probabilities in ordinary chronological order.  Has length T_cycle.
    terminal_period : bool
        Indicator for whether a terminal period follows the last period in the cycle (with LivPrb=0).

    Returns
    -------
    AgeDstn : np.array
        Stationary distribution of age.  Stochastic vector with frequencies of each age.
    """
    term_age = len(LivPrb)
    if terminal_period:
        MrkvArray = np.zeros((term_age + 1, term_age + 1))
        top = term_age
    else:
        MrkvArray = np.zeros((term_age, term_age))
        top = term_age - 1

    for t in range(top):
        MrkvArray[t, 0] = 1.0 - LivPrb[t]
        MrkvArray[t, t + 1] = LivPrb[t]
        MrkvArray[t + 1, 0] = 1.0

    w, v = np.linalg.eig(np.transpose(MrkvArray))
    idx = (np.abs(w - 1.0)).argmin()
    x = v[:, idx].astype(float)
    AgeDstn = x / np.sum(x)
    return AgeDstn


###############################################################################
### ACTUAL WORK BEGINS BELOW THIS LINE  #######################################
###############################################################################


def get_spec_name(options):
    # Construct the name of the specification from user options
    if options["param_name"] == "DiscFac":
        param_text = "beta"
    elif options["param_name"] == "CRRA":
        param_text = "rho"
    else:
        param_text = options["param_name"]

    if options["do_lifecycle"]:
        life_text = "LC"
    else:
        life_text = "PY"

    if options["do_param_dist"]:
        model_text = "Dist"
    else:
        model_text = "Point"

    if options["do_liquid"]:
        wealth_text = "Liquid"
    else:
        wealth_text = "NetWorth"

    if options["do_agg_shocks"]:
        shock_text = "Agg"
    else:
        shock_text = "Ind"

    spec_name = life_text + param_text + model_text + shock_text + wealth_text

    return spec_name


def get_param_count(options):
    if options["do_param_dist"]:
        param_count = 7  # Number of discrete beta types in beta-dist
    else:
        param_count = 1  # Just one beta type in beta-point

    return param_count


def get_hark_classes(options):
    if options["do_agg_shocks"]:
        agent_class = AggDoWAgent
        market_class = AggDoWMarket
    else:
        agent_class = DoWAgent
        market_class = DoWMarket

    return agent_class, market_class


def set_targets(options, params):
    # Set targets for K/Y and the Lorenz curve based on the data
    if options["do_liquid"]:
        lorenz_target = np.array([0.0, 0.004, 0.025, 0.117])
        lorenz_data = np.hstack(
            (
                np.array(0.0),
                get_lorenz_shares(
                    params.SCF_wealth,
                    weights=params.SCF_weights,
                    percentiles=np.arange(0.01, 1.0, 0.01).tolist(),
                ),
                np.array(1.0),
            )
        )
        ky_target = 6.60
    else:  # This is hacky until I can find the liquid wealth data and import it
        lorenz_target = get_lorenz_shares(
            params.SCF_wealth,
            weights=params.SCF_weights,
            percentiles=params.percentiles_to_match,
        )
        lorenz_data = np.hstack(
            (
                np.array(0.0),
                get_lorenz_shares(
                    params.SCF_wealth,
                    weights=params.SCF_weights,
                    percentiles=np.arange(0.01, 1.0, 0.01).tolist(),
                ),
                np.array(1.0),
            )
        )
        # lorenz_target = np.array([-0.002, 0.01, 0.053,0.171])
        ky_target = 10.26

    return lorenz_target, lorenz_data, ky_target


def set_population(options, params):
    # Set total number of simulated agents in the population
    if options["do_param_dist"]:
        if options["do_agg_shocks"]:
            population = params.pop_sim_agg_dist
        else:
            population = params.pop_sim_ind_dist
    else:
        if options["do_agg_shocks"]:
            population = params.pop_sim_agg_point
        else:
            population = params.pop_sim_ind_point

    return population


def make_agents(options, params, agent_class, param_count):
    # Make AgentTypes for estimation
    if options["do_lifecycle"]:
        dropout_type = agent_class(**params.init_dropout)
        dropout_type.AgeDstn = calc_stationary_age_dstn(dropout_type.LivPrb, True)
        highschool_type = deepcopy(dropout_type)
        highschool_type.assign_parameters(**params.adj_highschool)
        highschool_type.AgeDstn = calc_stationary_age_dstn(highschool_type.LivPrb, True)
        college_type = deepcopy(dropout_type)
        college_type.assign_parameters(**params.adj_college)
        college_type.AgeDstn = calc_stationary_age_dstn(college_type.LivPrb, True)
        dropout_type.update()
        highschool_type.update()
        college_type.update()
        agent_list = []
        for n in range(param_count):
            agent_list.append(deepcopy(dropout_type))
            agent_list.append(deepcopy(highschool_type))
            agent_list.append(deepcopy(college_type))
    else:
        if options["do_agg_shocks"]:
            perpetualyouth_type = agent_class(**params.init_agg_shocks)
        else:
            perpetualyouth_type = agent_class(**params.init_infinite)
        perpetualyouth_type.AgeDstn = np.array(1.0)
        agent_list = []
        for n in range(param_count):
            agent_list.append(deepcopy(perpetualyouth_type))

    return agent_list


def set_up_economy(options, params, param_count):
    agent_class, market_class = get_hark_classes(options)
    agent_list = make_agents(options, params, agent_class, param_count)

    # Give all the AgentTypes different seeds
    for j, agent in enumerate(agent_list):
        agent.seed = j

    # Make an economy for the consumers to live in
    market_dict = copy(params.init_market)
    market_dict["AggShockBool"] = options["do_agg_shocks"]
    market_dict["Population"] = set_population(options, params)
    economy = market_class(**market_dict)
    economy.agents = agent_list
    (
        economy.LorenzTarget,
        economy.LorenzData,
        economy.KYratioTarget,
    ) = set_targets(options, params)

    if options["do_lifecycle"]:
        economy.assign_parameters(
            PopGroFac=params.PopGroFac,
            TypeWeight=params.TypeWeight_lifecycle,
            T_retire=params.working_T - 1,
            act_T=params.T_sim_LC,
            ignore_periods=params.ignore_periods_LC,
        )
    else:
        economy.assign_parameters(
            PopGroFac=1.0,
            TypeWeight=[1.0],
            act_T=params.T_sim_PY,
            ignore_periods=params.ignore_periods_PY,
        )

    if options["do_agg_shocks"]:
        economy.assign_parameters(**params.aggregate_params)
        economy.update()
        economy.make_AggShkHist()

    return economy


def estimate(options, params):
    spec_name = get_spec_name(options)
    param_count = get_param_count(options)
    economy = set_up_economy(options, params, param_count)

    # Estimate the model as requested
    if options["run_estimation"]:
        print(f"Beginning an estimation with the specification name {spec_name}...")

        # Choose the bounding region for the parameter search
        if options["param_name"] == "CRRA":
            param_range = [0.2, 70.0]
            spread_range = [0.00001, 1.0]
        elif options["param_name"] == "DiscFac":
            param_range = [0.95, 0.995]
            spread_range = [0.006, 0.008]
        else:
            print(f"Parameter range for {options['param_name']} has not been defined!")

        if options["do_param_dist"]:
            # Run the param-dist estimation

            if options["do_combo_estimation"]:
                t_start = time()

                results = minimize(
                    get_target_ky_and_find_lorenz_distance,
                    [0.9867, 0.0067],
                    args=(
                        economy,
                        options["param_name"],
                        param_count,
                        options["dist_type"],
                    ),
                    bounds=[param_range, spread_range],
                )

                t_end = time()

                center_estimate, spread_estimate = results.x

            else:
                t_start = time()
                spread_estimate = (
                    minimize_scalar(
                        find_lorenz_distance_at_target_ky,
                        bounds=spread_range,
                        args=(
                            economy,
                            options["param_name"],
                            param_count,
                            param_range,
                            options["dist_type"],
                        ),
                        tol=1e-4,
                    )
                ).x
                center_estimate = economy.center_save

                t_end = time()
        else:
            # Run the param-point estimation only

            t_start = time()
            center_estimate = root_scalar(
                get_ky_ratio_difference,
                args=(
                    economy,
                    options["param_name"],
                    param_count,
                    0.0,
                    options["dist_type"],
                ),
                method="toms748",
                bracket=param_range,
                xtol=1e-6,
            ).root
            spread_estimate = 0.0
            t_end = time()

        # Display statistics about the estimated model
        economy.assign_parameters(LorenzBool=True, ManyStatsBool=True)

        economy.distribute_params(
            options["param_name"],
            param_count,
            center_estimate,
            spread_estimate,
            options["dist_type"],
        )
        economy.solve()
        economy.calc_lorenz_distance()
        print(
            f"Estimate is center={center_estimate}, spread={spread_estimate}, "
            f"took {t_end - t_start} seconds."
        )

        economy.center_estimate = center_estimate
        economy.spread_estimate = spread_estimate
        economy.show_many_stats(spec_name)
        print(f"These results have been saved to ./code/results/{spec_name}.txt\n\n")

    return economy


class Estimator:
    def __init__(self, options, parameters):
        self.options = options
        self.parameters = parameters

    def estimate(self):
        return estimate(self.options, self.parameters)


if __name__ == "__main__":
    import code.calibration as parameters
    from code.options.all_options import all_options

    basic_options = all_options["UseUniformBetaDist"].copy()
    basic_options.update(all_options["DoStandardWork"])
    estimate(basic_options, parameters)
