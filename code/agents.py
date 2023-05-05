import os
from copy import copy

import matplotlib.pyplot as plt
import numpy as np
from HARK import AgentType, Market
from HARK.ConsumptionSaving.ConsAggShockModel import (
    AggShockConsumerType,
    CobbDouglasEconomy,
)
from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType
from HARK.distribution import DiscreteDistribution, Lognormal, Uniform
from HARK.utilities import calc_subpop_avg, get_lorenz_shares, get_percentiles

here = os.path.dirname(os.path.realpath(__file__))


def mystr(number):
    return f"{number:.3f}"


class CstwMPCAgent(AgentType):  # EstimationAgentClass
    """
    A slight extension of the idiosyncratic consumer type for the cstwMPC model.
    """

    def reset(self):
        self.initialize_sim()
        self.t_age = (
            DiscreteDistribution(
                self.AgeDstn,
                np.arange(self.AgeDstn.size),
                seed=self.RNG.integers(0, 2**31 - 1),
            )
            .draw(self.AgentCount, exact_match=False)
            .astype(int)
        )
        self.t_cycle = copy(self.t_age)
        if hasattr(self, "kGrid"):
            # Start simulation near SS
            self.aLvlNow = self.kInit * np.ones(self.AgentCount)
            self.aNrmNow = self.aLvlNow / self.pLvlNow

    def market_action(self):
        if hasattr(self, "kGrid"):
            self.pLvl = self.pLvlNow / np.mean(self.pLvlNow)
        self.simulate(1)


class DoWAgent(CstwMPCAgent, IndShockConsumerType):
    pass


class AggDoWAgent(CstwMPCAgent, AggShockConsumerType):
    pass


class CstwMPCMarket(Market):  # EstimationMarketClass
    """
    A class for representing the economy in the cstwMPC model.
    """

    def __init__(self, **kwds):
        """
        Make a new instance of CstwMPCMarket.
        """

        reap_vars = ["aLvl", "pLvl", "MPCnow", "TranShk", "EmpNow", "t_age"]
        # Nothing needs to be sent back to agents in the idiosyncratic shocks version
        sow_vars = []
        const_vars = []  # ['LorenzBool','ManyStatsBool']
        track_vars = [
            "MaggNow",
            "AaggNow",
            "KtoYnow",
            "Lorenz",
            "LorenzLong",
            "MPCall",
            "MPCretired",
            "MPCemployed",
            "MPCunemployed",
            "MPCbyIncome",
            "MPCbyWealthRatio",
            "HandToMouthPct",
        ]
        dyn_vars = []  # No dynamics in the idiosyncratic shocks version

        if kwds.get("AggShockBool", False):
            sow_vars = [
                "MaggNow",
                "AaggNow",
                "RfreeNow",
                "wRteNow",
                "PermShkAggNow",
                "TranShkAggNow",
                "KtoLnow",
            ]
            dyn_vars = ["AFunc"]

        super().__init__(
            sow_vars=sow_vars,
            reap_vars=reap_vars,
            const_vars=const_vars,
            track_vars=track_vars,
            dyn_vars=dyn_vars,
        )
        self.assign_parameters(**kwds)
        if self.AggShockBool:
            self.max_loops = 20

        self.center_save = None

        # Save the current file's directory location for writing output:
        self.my_file_path = here

    def solve(self):
        """
        Solves the CstwMPCMarket.
        """
        if self.AggShockBool:
            for agent in self.agents:
                agent.get_economy_data(self)
            Market.solve(self)
        else:
            self.solve_agents()
            self.make_history()

    def reap(self):
        super().reap()

        if "MPCnow" in self.reap_vars:
            harvest = []

            for agent in self.agents:
                harvest.append(agent.MPCnow)

            self.reap_state["MPCnow"] = harvest

        if "t_age" in self.reap_vars:
            harvest = []

            for agent in self.agents:
                harvest.append(agent.t_age)

            self.reap_state["t_age"] = harvest

        if "EmpNow" in self.reap_vars and len(self.reap_state["EmpNow"]) == 0:
            harvest = []

            for agent in self.agents:
                harvest.append(agent.EmpNow)

            self.reap_state["EmpNow"] = harvest

        for var in self.reap_vars:
            harvest = []
            shock = False

            for agent in self.agents:
                if var in agent.shocks:
                    harvest.append(agent.shocks[var])
                    shock = True

            if shock:
                self.reap_state[var] = harvest

    def mill_rule(self, aLvl, pLvl, MPCnow, TranShk, EmpNow, t_age):
        """
        The mill_rule for this class simply calls the method calc_stats.
        """
        self.calc_stats(
            aLvl,
            pLvl,
            MPCnow,
            TranShk,
            EmpNow,
            t_age,
            self.parameters["LorenzBool"],
            self.parameters["ManyStatsBool"],
        )

        if self.AggShockBool:
            return self.calc_R_and_W(aLvl, pLvl)
        else:  # These variables are tracked but not created in no-agg-shocks specifications
            self.MaggNow = 0.0
            self.AaggNow = 0.0

    def calc_stats(
        self,
        aLvlNow,
        pLvlNow,
        MPCnow,
        TranShkNow,
        EmpNow,
        t_age,
        LorenzBool,
        ManyStatsBool,
    ):
        """
        Calculate various statistics about the current population in the economy.

        Parameters
        ----------
        aLvlNow : [np.array]
            Arrays with end-of-period assets, listed by each ConsumerType in self.agents.
        pLvlNow : [np.array]
            Arrays with permanent income levels, listed by each ConsumerType in self.agents.
        MPCnow : [np.array]
            Arrays with marginal propensity to consume, listed by each ConsumerType in self.agents.
        TranShkNow : [np.array]
            Arrays with transitory income shocks, listed by each ConsumerType in self.agents.
        EmpNow : [np.array]
            Arrays with employment states: True if employed, False otherwise.
        t_age : [np.array]
            Arrays with periods elapsed since model entry, listed by each ConsumerType in self.agents.
        LorenzBool: bool
            Indicator for whether the Lorenz target points should be calculated.  Usually False,
            only True when DiscFac has been identified for a particular nabla.
        ManyStatsBool: bool
            Indicator for whether a lot of statistics for tables should be calculated. Usually False,
            only True when parameters have been estimated and we want values for tables.

        Returns
        -------
        None
        """
        # Combine inputs into single arrays
        aLvl = np.hstack(aLvlNow)
        pLvl = np.hstack(pLvlNow)
        age = np.hstack(t_age)
        TranShk = np.hstack(TranShkNow)
        EmpNow = np.hstack(EmpNow)

        # Calculate the capital to income ratio in the economy
        CohortWeight = self.PopGroFac ** (-age)
        CapAgg = np.sum(aLvl * CohortWeight)
        IncAgg = np.sum(pLvl * TranShk * CohortWeight)
        KtoYnow = CapAgg / IncAgg
        self.KtoYnow = KtoYnow

        # Store Lorenz data if requested
        self.LorenzLong = np.nan
        if LorenzBool:
            order = np.argsort(aLvl)
            aLvl = aLvl[order]
            CohortWeight = CohortWeight[order]
            wealth_shares = get_lorenz_shares(
                aLvl,
                weights=CohortWeight,
                percentiles=self.LorenzPercentiles,
                presorted=True,
            )
            self.Lorenz = wealth_shares

            if ManyStatsBool:
                self.LorenzLong = get_lorenz_shares(
                    aLvl,
                    weights=CohortWeight,
                    percentiles=np.arange(0.01, 1.0, 0.01),
                    presorted=True,
                )
        else:
            self.Lorenz = np.nan  # Store nothing if we don't want Lorenz data

        # Calculate a whole bunch of statistics if requested
        if ManyStatsBool:
            # Reshape other inputs
            MPC = np.hstack(MPCnow)

            # Sort other data items if aLvl and CohortWeight were sorted
            if LorenzBool:
                pLvl = pLvl[order]
                MPC = MPC[order]
                TranShk = TranShk[order]
                age = age[order]
                EmpNow = EmpNow[order]
            aNrm = aLvl / pLvl  # Normalized assets (wealth ratio)
            IncLvl = TranShk * pLvl  # Labor income this period

            # Calculate overall population MPC and by subpopulations
            MPCannual = 1.0 - (1.0 - MPC) ** 4
            self.MPCall = np.sum(MPCannual * CohortWeight) / np.sum(CohortWeight)
            employed = EmpNow
            unemployed = np.logical_not(employed)
            if (
                self.T_retire > 0
            ):  # Adjust for the lifecycle model, where agents might be retired instead
                unemployed = np.logical_and(unemployed, age < self.T_retire)
                employed = np.logical_and(employed, age < self.T_retire)
                retired = age >= self.T_retire
            else:
                retired = np.zeros_like(unemployed, dtype=bool)
            self.MPCunemployed = np.sum(
                MPCannual[unemployed] * CohortWeight[unemployed]
            ) / np.sum(CohortWeight[unemployed])
            self.MPCemployed = np.sum(
                MPCannual[employed] * CohortWeight[employed]
            ) / np.sum(CohortWeight[employed])
            self.MPCretired = np.sum(
                MPCannual[retired] * CohortWeight[retired]
            ) / np.sum(CohortWeight[retired])
            self.MPCbyWealthRatio = calc_subpop_avg(
                MPCannual, aNrm, self.cutoffs, CohortWeight
            )
            self.MPCbyIncome = calc_subpop_avg(
                MPCannual, IncLvl, self.cutoffs, CohortWeight
            )

            # Calculate the wealth quintile distribution of "hand to mouth" consumers
            quintile_cuts = get_percentiles(
                aLvl, weights=CohortWeight, percentiles=[0.2, 0.4, 0.6, 0.8]
            )
            wealth_quintiles = np.ones(aLvl.size, dtype=int)
            wealth_quintiles[aLvl > quintile_cuts[0]] = 2
            wealth_quintiles[aLvl > quintile_cuts[1]] = 3
            wealth_quintiles[aLvl > quintile_cuts[2]] = 4
            wealth_quintiles[aLvl > quintile_cuts[3]] = 5
            MPC_cutoff = get_percentiles(
                MPCannual, weights=CohortWeight, percentiles=[2.0 / 3.0]
            )  # Looking at consumers with MPCs in the top 1/3
            these = MPCannual > MPC_cutoff
            in_top_third_MPC = wealth_quintiles[these]
            temp_weights = CohortWeight[these]
            hand_to_mouth_total = np.sum(temp_weights)
            hand_to_mouth_pct = []
            for q in range(1, 6):
                hand_to_mouth_pct.append(
                    np.sum(temp_weights[in_top_third_MPC == q]) / hand_to_mouth_total
                )
            self.HandToMouthPct = np.array(hand_to_mouth_pct)

        else:  # If we don't want these stats, just put empty values in history
            self.MPCall = np.nan
            self.MPCunemployed = np.nan
            self.MPCemployed = np.nan
            self.MPCretired = np.nan
            self.MPCbyWealthRatio = np.nan
            self.MPCbyIncome = np.nan
            self.HandToMouthPct = np.nan

    def distribute_params(self, param_name, param_count, center, spread, dist_type):
        """
        Distributes heterogeneous values of one parameter to the AgentTypes in self.agents.
        Parameters
        ----------
        param_name : string
            Name of the parameter to be assigned.
        param_count : int
            Number of different values the parameter will take on.
        center : float
            A measure of centrality for the distribution of the parameter.
        spread : float
            A measure of spread or diffusion for the distribution of the parameter.
        dist_type : string
            The type of distribution to be used.  Can be "lognormal" or "uniform" (can expand).
        Returns
        -------
        None
        """
        # Get a list of discrete values for the parameter
        if dist_type == "uniform":
            # If uniform, center is middle of distribution, spread is distance to either edge
            param_dist = Uniform(bot=center - spread, top=center + spread).discretize(
                N=param_count
            )
        elif dist_type == "lognormal":
            # If lognormal, center is the mean and spread is the standard deviation (in log)
            tail_N = 3
            param_dist = Lognormal(
                mu=np.log(center) - 0.5 * spread**2,
                sigma=spread,
                tail_N=tail_N,
                tail_bound=[0.0, 0.9],
                tail_order=np.e,
            ).discretize(N=param_count - tail_N)

        # Distribute the parameters to the various types, assigning consecutive types the same
        # value if there are more types than values
        replication_factor = len(self.agents) // param_count
        # Note: the double division is integer division in Python 3, this makes it explicit
        j = 0
        b = 0
        while j < len(self.agents):
            for n in range(replication_factor):
                self.agents[j].assign_parameters(
                    AgentCount=int(
                        self.Population * param_dist.pmv[b] * self.TypeWeight[n]
                    )
                )
                print(param_dist.atoms[0, b])
                self.agents[j].assign_parameters(**{param_name: param_dist.atoms[0, b]})
                j += 1
            b += 1

    def calc_KY_ratio_difference(self):
        """
        Returns the difference between the simulated capital to income ratio and the target ratio.
        Can only be run after solving all AgentTypes and running make_history.
        Parameters
        ----------
        None
        Returns
        -------
        diff : float
            Difference between simulated and target capital to income ratio.
        """
        # Ignore the first X periods to allow economy to stabilize from initial conditions
        KYratioSim = np.mean(np.array(self.history["KtoYnow"])[self.ignore_periods :])
        diff = KYratioSim - self.KYratioTarget

        return diff

    def calc_lorenz_distance(self):
        """
        Returns the sum of squared differences between simulated and target Lorenz points.
        Parameters
        ----------
        None
        Returns
        -------
        dist : float
            Sum of squared distances between simulated and target Lorenz points (sqrt)
        """
        LorenzSim = np.mean(
            np.array(self.history["Lorenz"])[self.ignore_periods :], axis=0
        )
        dist = np.sqrt(np.sum((100 * (LorenzSim - self.LorenzTarget)) ** 2))
        self.LorenzDistance = dist
        return dist

    def show_many_stats(self, spec_name=None):
        """
        Calculates the "many statistics" by averaging histories across simulated periods.  Displays
        the results as text and saves them to files if spec_name is not None.
        Parameters
        ----------
        spec_name : string
            A name or label for the current specification.
        Returns
        -------
        None
        """
        # Calculate MPC overall and by subpopulations
        MPCall = np.mean(self.history["MPCall"][self.ignore_periods :])
        MPCemployed = np.mean(self.history["MPCemployed"][self.ignore_periods :])
        MPCunemployed = np.mean(self.history["MPCunemployed"][self.ignore_periods :])
        MPCretired = np.mean(self.history["MPCretired"][self.ignore_periods :])
        MPCbyIncome = np.mean(
            np.array(self.history["MPCbyIncome"])[self.ignore_periods :, :], axis=0
        )
        MPCbyWealthRatio = np.mean(
            np.array(self.history["MPCbyWealthRatio"])[self.ignore_periods :, :], axis=0
        )
        HandToMouthPct = np.mean(
            np.array(self.history["HandToMouthPct"])[self.ignore_periods :, :], axis=0
        )

        np.hstack(
            (
                np.array(0.0),
                np.mean(
                    np.array(self.history["LorenzLong"])[self.ignore_periods :], axis=0
                ),
                np.array(1.0),
            )
        )
        LorenzAxis = np.arange(101, dtype=float)

        plt.plot(LorenzAxis, self.LorenzData, "-k", linewidth=1.5)
        # TODO: Fix this.
        # plt.plot(LorenzAxis,LorenzSim,'--k',linewidth=1.5)
        plt.xlabel("Income percentile", fontsize=12)
        plt.ylabel("Cumulative wealth share", fontsize=12)
        plt.ylim([-0.02, 1.0])
        # if running from command line, set interactive mode on, and make figure without blocking execution
        if (
            str(type(get_ipython()))
            == "<class 'ipykernel.zmqshell.ZMQInteractiveShell'>"
        ):
            print("Running in interactive shell (Jupyter notebook or spyder)")
            plt.show()
        else:
            print(
                "Running in terminal; do not wait for user to close figure before moving on"
            )
            plt.ioff()
            plt.show(block=False)
            # Give OS time to make the plot (it only draws when main thread is sleeping)
            plt.pause(2)

        # Create a list of strings to concatenate
        results_list = [
            f"Estimate is center={self.center_estimate}, spread={self.spread_estimate}\n",
            f"Lorenz distance is {self.LorenzDistance}\n",
            f"Average MPC for all consumers is {mystr(MPCall)}\n",
            f"Average MPC in the top percentile of W/Y is {mystr(MPCbyWealthRatio[0])}\n",
            f"Average MPC in the top decile of W/Y is {mystr(MPCbyWealthRatio[1])}\n",
            f"Average MPC in the top quintile of W/Y is {mystr(MPCbyWealthRatio[2])}\n",
            f"Average MPC in the second quintile of W/Y is {mystr(MPCbyWealthRatio[3])}\n",
            f"Average MPC in the middle quintile of W/Y is {mystr(MPCbyWealthRatio[4])}\n",
            f"Average MPC in the fourth quintile of W/Y is {mystr(MPCbyWealthRatio[5])}\n",
            f"Average MPC in the bottom quintile of W/Y is {mystr(MPCbyWealthRatio[6])}\n",
            f"Average MPC in the top percentile of y is {mystr(MPCbyIncome[0])}\n",
            f"Average MPC in the top decile of y is {mystr(MPCbyIncome[1])}\n",
            f"Average MPC in the top quintile of y is {mystr(MPCbyIncome[2])}\n",
            f"Average MPC in the second quintile of y is {mystr(MPCbyIncome[3])}\n",
            f"Average MPC in the middle quintile of y is {mystr(MPCbyIncome[4])}\n",
            f"Average MPC in the fourth quintile of y is {mystr(MPCbyIncome[5])}\n",
            f"Average MPC in the bottom quintile of y is {mystr(MPCbyIncome[6])}\n",
            f"Average MPC for the employed is {mystr(MPCemployed)}\n",
            f"Average MPC for the unemployed is {mystr(MPCunemployed)}\n",
            f"Average MPC for the retired is {mystr(MPCretired)}\n",
            "Of the population with the 1/3 highest MPCs...\n",
            f"{mystr(HandToMouthPct[0] * 100)}% are in the bottom wealth quintile,\n",
            f"{mystr(HandToMouthPct[1] * 100)}% are in the second wealth quintile,\n",
            f"{mystr(HandToMouthPct[2] * 100)}% are in the third wealth quintile,\n",
            f"{mystr(HandToMouthPct[3] * 100)}% are in the fourth wealth quintile,\n",
            f"and {mystr(HandToMouthPct[4] * 100)}% are in the top wealth quintile.\n",
        ]

        # Concatenate the list into a single string
        results_string = "".join(results_list)

        print(results_string)

        # Save results to disk
        if spec_name is not None:
            with open(
                self.my_file_path + "/results/" + spec_name + "Results.txt",
                "w",
                encoding="utf-8",
            ) as f:
                f.write(results_string)
                f.close()


class DoWMarket(CstwMPCMarket, Market):
    pass


class AggDoWMarket(CstwMPCMarket, CobbDouglasEconomy):
    pass
