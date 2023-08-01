from time import time

from code.estimation import (
    get_param_count,
    get_spec_name,
    get_target_ky_and_find_lorenz_distance,
    set_up_economy,
    get_ky_ratio_difference
)
from scipy.optimize import minimize_scalar, root_scalar
from HARK.distribution import (
    expected
)


# Don't forget to store the value of R_upper as an attribute
# of the economy, so it can be called later.
def get_R_cusp(economy,options):

    economy.agents[0].Ex_PermShkInv = expected(lambda x: 1 / x, economy.agents[0].PermShkDstn[0])[0]

    if options["do_lifecycle"]:
        R_cusp_LC = ((economy.agents[0].PermGroFac[0] / (economy.agents[0].Ex_PermShkInv * economy.agents[0].LivPrb[0])) ** economy.agents[0].CRRA ) * (1/economy.agents[0].DiscFac)
        return R_cusp_LC
    else:
        R_cusp_PY = ((economy.agents[0].PermGroFac[0] / economy.agents[0].Ex_PermShkInv) ** economy.agents[0].CRRA ) * (1/economy.agents[0].DiscFac)
        return R_cusp_PY

def get_center_spread(r_lower, r_upper):
    center = (r_lower + r_upper) / 2
    spread = (r_upper - r_lower) / 2

    return center, spread


def get_target_ky_and_find_lorenz_distance_given_R_bounds(
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
    R_cusp = get_R_cusp(
        economy, options
    )
    epsilon = 1e-8

    # Estimate the model as requested
    if options["run_estimation"]:
        print(f"Beginning an estimation with the specification name {spec_name}...")

        # Choose the bounding region for the parameter search
        if options["param_name"] == "Rfree":
            spread_range = [0.0, R_cusp - epsilon]
            param_range = [economy.agents[0].Rfree - 0.01, economy.agents[0].Rfree + 0.1]
        else:
            print(f"Parameter range for {options['param_name']} has not been defined!")

        if options["do_param_dist"]:
            # Run the param-dist estimation

            t_start = time()
            r_lower_estimate = (
                minimize_scalar(
                    get_target_ky_and_find_lorenz_distance_given_R_bounds,
                    bounds=spread_range,
                    args=(
                        R_cusp,
                        economy,
                        options["param_name"],
                        param_count,
                        options["dist_type"],
                    ),
                    tol=1e-4,
                )
            ).x

            t_end = time()

            center_estimate, spread_estimate = get_center_spread(r_lower_estimate, R_cusp - epsilon) #should this be (R_upper - epsilon)?

        else:
            # Run the param-point estimation only

            t_start = time()
            center_estimate = root_scalar(
                get_ky_ratio_difference,
                args=(
                    0.0,
                    economy,
                    options["param_name"],
                    param_count,
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

        economy.show_many_stats(spec_name)
        print(
            f"These results have been saved to ./code/results/dist_bounds/{spec_name}.txt\n\n"
        )

         # Store these as attributes for later use
        economy.spec_name = spec_name
        economy.param_count = param_count
        economy.R_cusp = R_cusp
        economy.epsilon = epsilon

    return economy


def estimate_r_upper_given_r_lower(options, params):
    economy = estimate_r_lower(options, params)
    R_lower = economy.optimal_r_lower

    # Estimate the model as requested
    if options["run_estimation"]:
        print(f"Beginning an estimation with the specification name {economy.spec_name}...")

        # Choose the bounding region for the parameter search
        # Not sure if this is correct, but it seems like this part
        # Should incorporate the lower bound found in the previous estimation.
        if options["param_name"] == "Rfree":
            spread_range = [R_lower, economy.R_cusp - economy.epsilon]
        else:
            print(f"Parameter range for {options['param_name']} has not been defined!")

        if options["do_param_dist"]:
            # Run the param-dist estimation

            t_start = time()
            r_upper_estimate = (
                minimize_scalar(
                    get_target_ky_and_find_lorenz_distance_given_R_bounds,
                    bounds=spread_range,
                    args=(
                        R_lower,
                        economy,
                        options["param_name"],
                        economy.param_count,
                        options["dist_type"],
                    ),
                    tol=1e-4,
                )
            ).x

            t_end = time()

            center_estimate, spread_estimate = get_center_spread(R_lower, r_upper_estimate)

        # Display statistics about the estimated model
        economy.assign_parameters(LorenzBool=True, ManyStatsBool=True)

        economy.distribute_params(
            options["param_name"],
            economy.param_count,
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
        economy.show_many_stats(economy.spec_name)
        print(
            f"These results have been saved to ./code/results/dist_bounds/{economy.spec_name}.txt\n\n"
        )

    return economy
