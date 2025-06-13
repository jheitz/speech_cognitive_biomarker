from scipy import stats
from scipy.stats import distributions
from scipy._lib._bunch import _make_tuple_bunch
from scipy.stats._binomtest import _binary_search_for_binom_tst as _binary_search
import numpy as np

# this is from scipy version 1.15

def fisher_exact(table, alternative=None, *, method=None):
    """Perform a Fisher exact test on a contingency table.

    For a 2x2 table,
    the null hypothesis is that the true odds ratio of the populations
    underlying the observations is one, and the observations were sampled
    from these populations under a condition: the marginals of the
    resulting table must equal those of the observed table.
    The statistic is the unconditional maximum likelihood estimate of the odds
    ratio, and the p-value is the probability under the null hypothesis of
    obtaining a table at least as extreme as the one that was actually
    observed.

    For other table sizes, or if `method` is provided, the null hypothesis
    is that the rows and columns of the tables have fixed sums and are
    independent; i.e., the table was sampled from a `scipy.stats.random_table`
    distribution with the observed marginals. The statistic is the
    probability mass of this distribution evaluated at `table`, and the
    p-value is the percentage of the population of tables with statistic at
    least as extreme (small) as that of `table`. There is only one alternative
    hypothesis available: the rows and columns are not independent.

    There are other possible choices of statistic and two-sided
    p-value definition associated with Fisher's exact test; please see the
    Notes for more information.

    Parameters
    ----------
    table : array_like of ints
        A contingency table.  Elements must be non-negative integers.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis for 2x2 tables; unused for other
        table sizes.
        The following options are available (default is 'two-sided'):

        * 'two-sided': the odds ratio of the underlying population is not one
        * 'less': the odds ratio of the underlying population is less than one
        * 'greater': the odds ratio of the underlying population is greater
          than one

        See the Notes for more details.

    method : ResamplingMethod, optional
        Defines the method used to compute the p-value.
        If `method` is an instance of `PermutationMethod`/`MonteCarloMethod`,
        the p-value is computed using
        `scipy.stats.permutation_test`/`scipy.stats.monte_carlo_test` with the
        provided configuration options and other appropriate settings.
        Note that if `method` is an instance of `MonteCarloMethod`, the ``rvs``
        attribute must be left unspecified; Monte Carlo samples are always drawn
        using the ``rvs`` method of `scipy.stats.random_table`.
        Otherwise, the p-value is computed as documented in the notes.

        .. versionadded:: 1.15.0

    Returns
    -------
    res : SignificanceResult
        An object containing attributes:

        statistic : float
            For a 2x2 table with default `method`, this is the odds ratio - the
            prior odds ratio not a posterior estimate. In all other cases, this
            is the probability density of obtaining the observed table under the
            null hypothesis of independence with marginals fixed.
        pvalue : float
            The probability under the null hypothesis of obtaining a
            table at least as extreme as the one that was actually observed.

    Raises
    ------
    ValueError
        If `table` is not two-dimensional or has negative entries.

    See Also
    --------
    chi2_contingency : Chi-square test of independence of variables in a
        contingency table.  This can be used as an alternative to
        `fisher_exact` when the numbers in the table are large.
    contingency.odds_ratio : Compute the odds ratio (sample or conditional
        MLE) for a 2x2 contingency table.
    barnard_exact : Barnard's exact test, which is a more powerful alternative
        than Fisher's exact test for 2x2 contingency tables.
    boschloo_exact : Boschloo's exact test, which is a more powerful
        alternative than Fisher's exact test for 2x2 contingency tables.
    :ref:`hypothesis_fisher_exact` : Extended example

    Notes
    -----
    *Null hypothesis and p-values*

    The null hypothesis is that the true odds ratio of the populations
    underlying the observations is one, and the observations were sampled at
    random from these populations under a condition: the marginals of the
    resulting table must equal those of the observed table. Equivalently,
    the null hypothesis is that the input table is from the hypergeometric
    distribution with parameters (as used in `hypergeom`)
    ``M = a + b + c + d``, ``n = a + b`` and ``N = a + c``, where the
    input table is ``[[a, b], [c, d]]``.  This distribution has support
    ``max(0, N + n - M) <= x <= min(N, n)``, or, in terms of the values
    in the input table, ``min(0, a - d) <= x <= a + min(b, c)``.  ``x``
    can be interpreted as the upper-left element of a 2x2 table, so the
    tables in the distribution have form::

        [  x           n - x     ]
        [N - x    M - (n + N) + x]

    For example, if::

        table = [6  2]
                [1  4]

    then the support is ``2 <= x <= 7``, and the tables in the distribution
    are::

        [2 6]   [3 5]   [4 4]   [5 3]   [6 2]  [7 1]
        [5 0]   [4 1]   [3 2]   [2 3]   [1 4]  [0 5]

    The probability of each table is given by the hypergeometric distribution
    ``hypergeom.pmf(x, M, n, N)``.  For this example, these are (rounded to
    three significant digits)::

        x       2      3      4      5       6        7
        p  0.0163  0.163  0.408  0.326  0.0816  0.00466

    These can be computed with::

        >>> import numpy as np
        >>> from scipy.stats import hypergeom
        >>> table = np.array([[6, 2], [1, 4]])
        >>> M = table.sum()
        >>> n = table[0].sum()
        >>> N = table[:, 0].sum()
        >>> start, end = hypergeom.support(M, n, N)
        >>> hypergeom.pmf(np.arange(start, end+1), M, n, N)
        array([0.01631702, 0.16317016, 0.40792541, 0.32634033, 0.08158508,
               0.004662  ])

    The two-sided p-value is the probability that, under the null hypothesis,
    a random table would have a probability equal to or less than the
    probability of the input table.  For our example, the probability of
    the input table (where ``x = 6``) is 0.0816.  The x values where the
    probability does not exceed this are 2, 6 and 7, so the two-sided p-value
    is ``0.0163 + 0.0816 + 0.00466 ~= 0.10256``::

        >>> from scipy.stats import fisher_exact
        >>> res = fisher_exact(table, alternative='two-sided')
        >>> res.pvalue
        0.10256410256410257

    The one-sided p-value for ``alternative='greater'`` is the probability
    that a random table has ``x >= a``, which in our example is ``x >= 6``,
    or ``0.0816 + 0.00466 ~= 0.08626``::

        >>> res = fisher_exact(table, alternative='greater')
        >>> res.pvalue
        0.08624708624708627

    This is equivalent to computing the survival function of the
    distribution at ``x = 5`` (one less than ``x`` from the input table,
    because we want to include the probability of ``x = 6`` in the sum)::

        >>> hypergeom.sf(5, M, n, N)
        0.08624708624708627

    For ``alternative='less'``, the one-sided p-value is the probability
    that a random table has ``x <= a``, (i.e. ``x <= 6`` in our example),
    or ``0.0163 + 0.163 + 0.408 + 0.326 + 0.0816 ~= 0.9949``::

        >>> res = fisher_exact(table, alternative='less')
        >>> res.pvalue
        0.9953379953379957

    This is equivalent to computing the cumulative distribution function
    of the distribution at ``x = 6``:

        >>> hypergeom.cdf(6, M, n, N)
        0.9953379953379957

    *Odds ratio*

    The calculated odds ratio is different from the value computed by the
    R function ``fisher.test``.  This implementation returns the "sample"
    or "unconditional" maximum likelihood estimate, while ``fisher.test``
    in R uses the conditional maximum likelihood estimate.  To compute the
    conditional maximum likelihood estimate of the odds ratio, use
    `scipy.stats.contingency.odds_ratio`.

    References
    ----------
    .. [1] Fisher, Sir Ronald A, "The Design of Experiments:
           Mathematics of a Lady Tasting Tea." ISBN 978-0-486-41151-4, 1935.
    .. [2] "Fisher's exact test",
           https://en.wikipedia.org/wiki/Fisher's_exact_test

    Examples
    --------

    >>> from scipy.stats import fisher_exact
    >>> res = fisher_exact([[8, 2], [1, 5]])
    >>> res.statistic
    20.0
    >>> res.pvalue
    0.034965034965034975

    For tables with shape other than ``(2, 2)``, provide an instance of
    `scipy.stats.MonteCarloMethod` or `scipy.stats.PermutationMethod` for the
    `method` parameter:

    >>> import numpy as np
    >>> from scipy.stats import MonteCarloMethod
    >>> rng = np.random.default_rng(4507195762371367)
    >>> method = MonteCarloMethod(rng=rng)
    >>> fisher_exact([[8, 2, 3], [1, 5, 4]], method=method)
    SignificanceResult(statistic=np.float64(0.005782), pvalue=np.float64(0.0603))

    For a more detailed example, see :ref:`hypothesis_fisher_exact`.
    """
    hypergeom = distributions.hypergeom
    # int32 is not enough for the algorithm
    c = np.asarray(table, dtype=np.int64)
    if not c.ndim == 2:
        raise ValueError("The input `table` must have two dimensions.")

    if np.any(c < 0):
        raise ValueError("All values in `table` must be nonnegative.")

    if not c.shape == (2, 2) or method is not None:
        return _fisher_exact_rxc(c, alternative, method)
    alternative = 'two-sided' if alternative is None else alternative

    if 0 in c.sum(axis=0) or 0 in c.sum(axis=1):
        # If both values in a row or column are zero, the p-value is 1 and
        # the odds ratio is NaN.
        return SignificanceResult(np.nan, 1.0)

    if c[1, 0] > 0 and c[0, 1] > 0:
        oddsratio = c[0, 0] * c[1, 1] / (c[1, 0] * c[0, 1])
    else:
        oddsratio = np.inf

    n1 = c[0, 0] + c[0, 1]
    n2 = c[1, 0] + c[1, 1]
    n = c[0, 0] + c[1, 0]

    def pmf(x):
        return hypergeom.pmf(x, n1 + n2, n1, n)

    if alternative == 'less':
        pvalue = hypergeom.cdf(c[0, 0], n1 + n2, n1, n)
    elif alternative == 'greater':
        # Same formula as the 'less' case, but with the second column.
        pvalue = hypergeom.cdf(c[0, 1], n1 + n2, n1, c[0, 1] + c[1, 1])
    elif alternative == 'two-sided':
        mode = int((n + 1) * (n1 + 1) / (n1 + n2 + 2))
        pexact = hypergeom.pmf(c[0, 0], n1 + n2, n1, n)
        pmode = hypergeom.pmf(mode, n1 + n2, n1, n)

        epsilon = 1e-14
        gamma = 1 + epsilon

        if np.abs(pexact - pmode) / np.maximum(pexact, pmode) <= epsilon:
            return SignificanceResult(oddsratio, 1.)

        elif c[0, 0] < mode:
            plower = hypergeom.cdf(c[0, 0], n1 + n2, n1, n)
            if hypergeom.pmf(n, n1 + n2, n1, n) > pexact * gamma:
                return SignificanceResult(oddsratio, plower)

            guess = _binary_search(lambda x: -pmf(x), -pexact * gamma, mode, n)
            pvalue = plower + hypergeom.sf(guess, n1 + n2, n1, n)
        else:
            pupper = hypergeom.sf(c[0, 0] - 1, n1 + n2, n1, n)
            if hypergeom.pmf(0, n1 + n2, n1, n) > pexact * gamma:
                return SignificanceResult(oddsratio, pupper)

            guess = _binary_search(pmf, pexact * gamma, 0, mode)
            pvalue = pupper + hypergeom.cdf(guess, n1 + n2, n1, n)
    else:
        msg = "`alternative` should be one of {'two-sided', 'less', 'greater'}"
        raise ValueError(msg)

    pvalue = min(pvalue, 1.0)

    return SignificanceResult(oddsratio, pvalue)




def _fisher_exact_rxc(table, alternative, method):
    if alternative is not None:
        message = ('`alternative` must be the default (None) unless '
                  '`table` has shape `(2, 2)` and `method is None`.')
        raise ValueError(message)

    if table.size == 0:
        raise ValueError("`table` must have at least one row and one column.")

    if table.shape[0] == 1 or table.shape[1] == 1 or np.all(table == 0):
        # Only one such table with those marginals
        return SignificanceResult(1.0, 1.0)

    if method is None:
        method = stats.MonteCarloMethod()

    if isinstance(method, stats.PermutationMethod):
        res = _fisher_exact_permutation_method(table, method)
    elif isinstance(method, stats.MonteCarloMethod):
        res = _fisher_exact_monte_carlo_method(table, method)
    else:
        message = (f'`{method=}` not recognized; if provided, `method` must be an '
                   'instance of `PermutationMethod` or `MonteCarloMethod`.')
        raise ValueError(message)

    return SignificanceResult(np.clip(res.statistic, None, 1.0), res.pvalue)



def _fisher_exact_permutation_method(table, method):
    x, y = _untabulate(table)
    colsums = np.sum(table, axis=0)
    rowsums = np.sum(table, axis=1)
    X = stats.random_table(rowsums, colsums)

    # `permutation_test` with `permutation_type='pairings' permutes the order of `x`,
    # which pairs observations in `x` with different observations in `y`.
    def statistic(x):
        # crosstab the resample and compute the statistic
        table = stats.contingency.crosstab(x, y)[1]
        return X.pmf(table)

    # tables with *smaller* probability mass are considered to be more extreme
    return stats.permutation_test((x,), statistic, permutation_type='pairings',
                                  alternative='less', **method._asdict())


def _fisher_exact_monte_carlo_method(table, method):
    method = method._asdict()

    if method.pop('rvs', None) is not None:
        message = ('If the `method` argument of `fisher_exact` is an '
                   'instance of `MonteCarloMethod`, its `rvs` attribute '
                   'must be unspecified. Use the `MonteCarloMethod` `rng` argument '
                   'to control the random state.')
        raise ValueError(message)
    rng = np.random.default_rng(method.pop('rng', None))

    # `random_table.rvs` produces random contingency tables with the given marginals
    # under the null hypothesis of independence
    shape = table.shape
    colsums = np.sum(table, axis=0)
    rowsums = np.sum(table, axis=1)
    totsum = np.sum(table)
    X = stats.random_table(rowsums, colsums, seed=rng)

    def rvs(size):
        n_resamples = size[0]
        return X.rvs(size=n_resamples).reshape(size)

    # axis signals to `monte_carlo_test` that statistic is vectorized, but we know
    # how it will pass the table(s), so we don't need to use `axis` explicitly.
    def statistic(table, axis):
        shape_ = (-1,) + shape if table.size > totsum else shape
        return X.pmf(table.reshape(shape_))

    # tables with *smaller* probability mass are considered to be more extreme
    return stats.monte_carlo_test(table.ravel(), rvs, statistic,
                                  alternative='less', **method)



SignificanceResult = _make_tuple_bunch('SignificanceResult',
                                       ['statistic', 'pvalue'], [])


def _untabulate(table):
    # converts a contingency table to paired samples indicating the
    # correspondence between row and column indices
    r, c = table.shape
    x, y = [], []
    for i in range(r):
        for j in range(c):
            x.append([i] * table[i, j])
            y.append([j] * table[i, j])
    return np.concatenate(x), np.concatenate(y)


