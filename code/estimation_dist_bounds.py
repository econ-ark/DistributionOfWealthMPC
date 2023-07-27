from code.agents import AggDoWAgent, AggDoWMarket, DoWAgent, DoWMarket
from code.calibration import init_infinite
from copy import copy, deepcopy
from time import time

import numpy as np
from estimation import (
    get_ky_ratio_difference,
    get_param_count,
    get_spec_name,
    get_target_ky_and_find_lorenz_distance,
    set_up_economy,
)
from HARK.utilities import get_lorenz_shares
from scipy.optimize import minimize, minimize_scalar, root_scalar


def get_R_upper(economy, param_name, param_count, dist_type):
    pass


def get_center_spread(r_lower, r_upper):
    center = (r_lower + r_upper) / 2
    spread = (r_upper - r_lower) / 2

    return center, spread


def get_target_ky_and_find_lorenz_distance_given_R_upper(
    r_lower, r_upper, economy, param_name, param_count, dist_type
):
    x = get_center_spread(r_lower, r_upper)

    return get_target_ky_and_find_lorenz_distance(
        x, economy, param_name, param_count, dist_type
    )


def estimate_r_lower(options, params):
    spec_name = get_spec_name(options)
    param_count = get_param_count(options)
    economy = set_up_economy(options, params, param_count)
    R_upper = get_R_upper(
        economy, options["param_name"], param_count, options["dist_type"]
    )
    epsilon = 1e-8

    # Estimate the model as requested
    if options["run_estimation"]:
        print(f"Beginning an estimation with the specification name {spec_name}...")

        # Choose the bounding region for the parameter search
        if options["param_name"] == "Rfree":
            spread_range = [0.0, R_upper - epsilon]
        else:
            print(f"Parameter range for {options['param_name']} has not been defined!")

        if options["do_param_dist"]:
            # Run the param-dist estimation

            t_start = time()
            r_lower_estimate = (
                minimize_scalar(
                    get_target_ky_and_find_lorenz_distance_given_R_upper,
                    bounds=spread_range,
                    args=(
                        R_upper,
                        economy,
                        options["param_name"],
                        param_count,
                        options["dist_type"],
                    ),
                    tol=1e-4,
                )
            ).x

            t_end = time()

        # Display statistics about the estimated model
        economy.assign_parameters(LorenzBool=True, ManyStatsBool=True)

        center_estimate, spread_estimate = get_center_spread(r_lower_estimate, R_upper)

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
        print(
            f"These results have been saved to ./code/results/dist_bounds/{spec_name}.txt\n\n"
        )

    return economy
