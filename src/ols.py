# Purpose: A small wrapper around statsmodels' OLS (linear regression). It keeps the results
# in one place and has the checks on the model (VIF, Breusch-Pagan, Jarque-Bera and
# Cook's distance). run_model.py compares it to scikit-learn to make sure the coefficients match.
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor
from statsmodels.stats.stattools import jarque_bera


class OLS:
    """
    Ordinary least squares regression

    Fit it with fit(X, y), then look at the results in the attributes
    (coefs, robust_se, r2, ...) or in coef_table().
    """

    def __init__(self, column_names):
        # the intercept is added as the first term
        self.names = ["Intercept"] + list(column_names)

    @staticmethod
    def _add_intercept(X) -> np.ndarray:
        """Add a column of 1s to the front of X for the intercept"""
        return sm.add_constant(np.asarray(X, dtype=float), prepend=True, has_constant="add")

    def fit(self, X, y) -> "OLS":
        """
        Fit the model

        Args:
            X (data frame): The predictor columns
            y (Series): The target (log of award value)

        Returns:
            The fitted model (itself)
        """
        design = self._add_intercept(X)
        y = np.asarray(y, dtype=float)

        # 1. Fit the model twice, once with the usual standard errors and once with robust (HC3)
        # ones. The robust ones still work if the spread of the errors isn't the same for every row
        normal_fit = sm.OLS(y, design).fit()
        robust_fit = sm.OLS(y, design).fit(cov_type="HC3", use_t=True)
        conf_int = np.asarray(robust_fit.conf_int(alpha=0.05))   # 95% confidence intervals

        # 2. Save everything so we can look at it later
        self.n_rows, self.n_cols = design.shape
        self.coefs = np.asarray(robust_fit.params)
        self.resid = np.asarray(normal_fit.resid)
        self.fitted = np.asarray(normal_fit.fittedvalues)
        self.resid_variance = float(normal_fit.mse_resid)
        self.robust_se = np.asarray(robust_fit.bse)
        self.normal_se = np.asarray(normal_fit.bse)   # the usual SEs
        self.t_stats = np.asarray(robust_fit.tvalues)
        self.p_values = np.asarray(robust_fit.pvalues)
        self.ci_low, self.ci_high = conf_int[:, 0], conf_int[:, 1]
        self.r2 = float(normal_fit.rsquared)
        self.adj_r2 = float(normal_fit.rsquared_adj)

        # leverage = how much each row pulls the fitted line toward itself
        self.leverage = np.asarray(OLSInfluence(normal_fit).hat_matrix_diag)
        self._design_matrix = design
        self._normal_fit = normal_fit
        return self

    def predict(self, X) -> np.ndarray:
        """Predict the target for new rows"""
        return self._add_intercept(X) @ self.coefs

    def coef_table(self) -> pd.DataFrame:
        """Put the coefficients, standard errors, t stats, p-values and intervals in one table"""
        return pd.DataFrame({
            "term": self.names, "coef": self.coefs, "se_hc3": self.robust_se,
            "se_classic": self.normal_se, "t": self.t_stats, "p_value": self.p_values,
            "ci95_low": self.ci_low, "ci95_high": self.ci_high,
        })

    # the checks on the model start here

    def breusch_pagan(self) -> dict:
        """
        Test if the spread of the residuals is the same for every row

        A small p-value means it is not the same (which is why we use robust standard errors).
        """
        lm_stat, lm_p_value, _, _ = het_breuschpagan(self.resid, self._design_matrix)
        return {"lm_stat": float(lm_stat), "df": int(self.n_cols - 1), "p_value": float(lm_p_value)}

    def jarque_bera(self) -> dict:
        """Test if the residuals look normal (skew and kurtosis are also returned)"""
        jb_stat, jb_p_value, skew, kurtosis = jarque_bera(self.resid)
        # statsmodels gives kurtosis where a bell curve is 3, so subtract 3 to make a bell curve 0
        return {"stat": float(jb_stat), "p_value": float(jb_p_value), "skew": float(skew),
                "excess_kurtosis": float(kurtosis - 3)}

    def cooks_distance(self) -> np.ndarray:
        """Cook's distance for each row (bigger = that row changes the model more)"""
        return np.asarray(OLSInfluence(self._normal_fit).cooks_distance[0])

    @staticmethod
    def vif(X: pd.DataFrame) -> pd.Series:
        """
        Variance inflation factor for each column

        Checks if the columns overlap too much with each other (above 10 is a problem).
        """
        varying_columns = X.loc[:, X.std() > 0]   # a column that never changes can't be checked
        design = sm.add_constant(varying_columns.to_numpy(dtype=float), prepend=True,
                                 has_constant="add")
        # column 0 is the intercept, so the first real column is number 1
        vif_values = [variance_inflation_factor(design, i + 1)
                      for i in range(varying_columns.shape[1])]
        return pd.Series(vif_values, index=varying_columns.columns,
                         name="VIF").sort_values(ascending=False)
