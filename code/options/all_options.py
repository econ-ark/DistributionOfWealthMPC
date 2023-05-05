all_options = {}

"""
This options file specifies the "standard" work options for cstwMPC, estimating the model only.
"""
all_options["DoStandardWork"] = {
    "run_estimation": True,  # Runs the estimation if True
    # Choose which sensitivity analyses to run: rho, xi_sigma, psi_sigma, mu, urate, mortality, g, R
    "run_sensitivity": [False, False, False, False, False, False, False, False],
    # Computes K/Y ratio for a wide range of beta; should have do_beta_dist = False
    "find_beta_vs_KY": False,
    # Uses a "tractable consumer" rather than solving full model when True
    "do_tractable": False,
}

"""
This options file specifies parameter heterogeneity, making the choice in the paper:
uniformly distributed discount factors.
"""
all_options["UseUniformBetaDist"] = {
    "param_name": "DiscFac",  # Which parameter to introduce heterogeneity in
    "dist_type": "uniform",  # Which type of distribution to use
}

"""
This options file establishes the simplest model specification possible: no ex ante
heterogeneity, no aggregate shocks, perpetual youth model, matching net worth.
"""
all_options["SimpleSpecPoint"] = {
    "do_param_dist": False,  # Do param-dist version if True, param-point if False
    "do_lifecycle": False,  # Use lifecycle model if True, perpetual youth if False
    "do_agg_shocks": False,  # Solve the FBS aggregate shocks version of the model
    "do_liquid": False,  # Matches liquid assets data when True, net worth data when False
}

"""
This options file establishes the second simplest model specification possible:
with heterogeneity, no aggregate shocks, perpetual youth model, matching net worth.
"""
all_options["SimpleSpecDist"] = {
    "do_param_dist": True,  # Do param-dist version if True, param-point if False
    "do_lifecycle": False,  # Use lifecycle model if True, perpetual youth if False
    "do_agg_shocks": False,  # Solve the FBS aggregate shocks version of the model
    "do_liquid": False,  # Matches liquid assets data when True, net worth data when False
}


"""
This options file establishes the main beta-point specification in the paper:
with heterogeneity, FBS-style aggregate shocks, perpetual youth model, matching net worth.
"""
all_options["MainSpecPoint"] = {
    "do_param_dist": False,  # Do param-dist version if True, param-point if False
    "do_lifecycle": False,  # Use lifecycle model if True, perpetual youth if False
    "do_agg_shocks": True,  # Solve the FBS aggregate shocks version of the model
    "do_liquid": False,  # Matches liquid assets data when True, net worth data when False
}


"""
This options file establishes the main beta-dist specification in the paper:
with heterogeneity, FBS-style aggregate shocks, perpetual youth model, matching net worth.
"""
all_options["MainSpecDist"] = {
    "do_param_dist": True,  # Do param-dist version if True, param-point if False
    "do_lifecycle": False,  # Use lifecycle model if True, perpetual youth if False
    "do_agg_shocks": True,  # Solve the FBS aggregate shocks version of the model
    "do_liquid": False,  # Matches liquid assets data when True, net worth data when False
}

all_options["LCSpecPoint"] = {
    "do_param_dist": False,  # Do param-dist version if True, param-point if False
    "do_lifecycle": True,  # Use lifecycle model if True, perpetual youth if False
    "do_agg_shocks": True,  # Solve the FBS aggregate shocks version of the model
    "do_liquid": False,  # Matches liquid assets data when True, net worth data when False
}

all_options["LCSpecDist"] = {
    "do_param_dist": True,  # Do param-dist version if True, param-point if False
    "do_lifecycle": True,  # Use lifecycle model if True, perpetual youth if False
    "do_agg_shocks": True,  # Solve the FBS aggregate shocks version of the model
    "do_liquid": False,  # Matches liquid assets data when True, net worth data when False
}
