from typing import Sequence

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt
import xarray as xr
from arviz import InferenceData


class BHHierarchicalModel:
    """
    Bayesian Hierarchical Brock-Hommes Model.

    This class implements a Bayesian hierarchical version of the
    Brock-Hommes heterogeneous agent model using PyMC and PyTensor.

    The model estimates trader-specific forecasting parameters with
    hierarchical priors and generates latent price dynamics through
    recursive state transitions.

    :param observed_prices: Observed time series of prices.
    :param n_traders: Number of heterogeneous trader types.
    :param r: Gross risk-free return factor.
    :param sigma2: Variance of returns used in demand calculations.
    :param risk_aversion: Risk aversion coefficient.
    """

    def __init__(
        self,
        observed_prices: Sequence[float],
        n_traders: int = 4,
        r: float = 1.01,
        sigma2: float = 0.25,
        risk_aversion: float = 1.0,
    ) -> None:
        """
        Initialize the Bayesian hierachical Brock-Hommes model.

        :param observed_prices: Observed price time series.
        :param n_traders: Number of heterogeneous trader types.
        :param r: Gross risk-free return factor.
        :param sigma2: Variance of returns used in demand calculations.
        :param risk_aversion: Risk aversion coefficient.
        """
        # convert observed prices into numpy array
        self.x_obs = np.asarray(observed_prices)

        # total number of observations
        self.T = len(observed_prices)

        # store model configuration parameters
        self.n_traders = n_traders
        self.r = r
        self.sigma2 = sigma2
        self.risk_aversion = risk_aversion

    def build(self) -> pm.Model:
        """
        Build the PyMC probabilistic model.

        The model contains:

        * Hierarchical priors for trader parameters
        * Trader-level forecasting coefficients
        * Recursive Brock-Hommes state dynamics
        * Gaussian observation model

        :returns: Constructed PyMC model.
        """
        # create pymc probabilistic model context
        with pm.Model() as model:
            # =====================================================
            # HYPERPRIORS
            # =====================================================

            # population mean for trader trend coefficients
            mu_g = pm.Normal("mu_g", 0, 0.5)

            # population standard deviation for trend coefficients
            sigma_g = pm.HalfNormal("sigma_g", 0.2)

            # population mean for trader bias coefficients
            mu_b = pm.Normal("mu_b", 0, 0.1)

            # population standard deviation for bias coefficients
            sigma_b = pm.HalfNormal("sigma_b", 0.1)

            # =====================================================
            # TRADER PARAMETERS
            # =====================================================

            # trader-specific trend parameters
            g = pm.TruncatedNormal(
                name="g",
                mu=mu_g,
                sigma=sigma_g,
                lower=-1.0,
                upper=1.0,
                shape=self.n_traders,
            )

            # trader-specific bias parameters
            b = pm.Normal(
                name="b",
                mu=mu_b,
                sigma=sigma_b,
                shape=self.n_traders,
            )

            # observation noise standard deviation
            obs_sigma = pm.HalfNormal(
                "obs_sigma",
                0.1,
            )

            # =====================================================
            # INITIAL CONDITIONS
            # =====================================================

            # use first observed value as initial latent state
            x0 = self.x_obs[0]

            # =====================================================
            # STATE TRANSITION FUNCTION
            # =====================================================

            def transition(x_prev: pt.TensorVariable, g: pt.TensorVariable, b: pt.TensorVariable, r: float, sigma2: float, risk_aversion: float) -> pt.TensorVariable:
                """
                Compute one-step Brock-Hommes state transition.

                :param x_prev: Previous latent price state.
                :param g: Trader trend coefficient.
                :param b: Trader bias coefficient.
                :param r: Gross return factor.
                :param sigma2: Variance parameter.
                :param risk_aversion: Risk aversion coefficient.
                :returns: updated latent price state.
                """
                # traders from forecasts based on linear prediction rule
                forecasts = g * x_prev + b

                # compute trader demands using mean-variance framework
                demands = (forecasts - r * x_prev) / (risk_aversion * sigma2)

                # clip extreme demands for numerical stability
                demands = pt.clip(
                    demands,
                    -10,
                    10,
                )

                # aggregate trader demands into next-period price
                x_new = pt.clip(pt.mean(demands) / r, -5, 5)

                return x_new

            # =====================================================
            # SCAN RECURRENCE
            # =====================================================

            # recursively generate latent states over time
            outputs, _ = pytensor.scan(fn=transition, outputs_info=[x0], non_sequences=[g, b, self.r, self.sigma2, self.risk_aversion], n_steps=self.T - 1)

            # concentrate initial state with generated outputs
            x_latent = pt.concatenate(
                [
                    [x0],
                    outputs,
                ]
            )

            # =====================================================
            # OBSERVATION MODEL
            # =====================================================

            # observed prices are noisy realizations
            # of latent brock-hommes prices
            pm.Normal(
                "x_obs",
                mu=x_latent,
                sigma=obs_sigma,
                observed=self.x_obs,
            )

            # save model preference
            self.model = model

        return model

    def fit(self, draws: int = 2000, tune: int = 2000) -> InferenceData:
        """
        Fit the Bayesian model using MCMC sampling.

        :param draws: Number of posterior samples to draw.
        :param tune: Number of tuning iterations.
        :returns: Posterior inference data object.
        """
        # run mcmc sampler inside model context
        with self.model:
            trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=4,
                target_accept=0.95,
                return_inferencedata=True,
            )

        # store fitted posterior samples
        self.trace = trace

        return trace

    def summary(self) -> pd.DataFrame | xr.Dataset:
        """
        Generate posterior summary statistics.

        :returns: summary table of posterior estimates.
        """
        # return arviz posterior summary
        return az.summary(self.trace)
