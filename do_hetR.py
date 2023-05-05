"""
This file will run the absolute minimum amount of work that actually produces
relevant output-- no aggregate shocks, perpetual youth, matching net worth.
Will run both beta-point and beta-dist versions.
"""

import sys

sys.path.append("..")

import code.calibration as parameters
from code.estimation import estimate
from code.options.all_options import all_options

basic_options = all_options["UseUniformBetaDist"].copy()
basic_options["param_name"] = "Rfree"
basic_options.update(all_options["DoStandardWork"])

# Run beta-point model

simple_point = basic_options.copy()
simple_point.update(all_options["SimpleSpecPoint"])

estimate(simple_point, parameters)

# Run beta-dist model

simple_dist = basic_options.copy()
simple_dist["do_combo_estimation"] = False
simple_dist.update(all_options["SimpleSpecDist"])

estimate(simple_dist, parameters)
