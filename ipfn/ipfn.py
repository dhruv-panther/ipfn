#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import pandas as pd
from itertools import product
import copy

# Optional Polars support
try:
    import polars as pl  # type: ignore
except Exception:  # pragma: no cover - polars optional at runtime
    pl = None


class ipfn(object):

    def __init__(self, original, aggregates, dimensions, weight_col='total',
                 convergence_rate=1e-5, max_iteration=500, verbose=0, rate_tolerance=1e-8, algorithm='optimized'):
        """
        Initialize the ipfn class

        original: numpy darray matrix or dataframe to perform the ipfn on.

        aggregates: list of numpy array or darray or pandas dataframe/series. The aggregates are the same as the marginals.
        They are the target values that we want along one or several axis when aggregating along one or several axes.

        dimensions: list of lists with integers if working with numpy objects, or column names if working with pandas objects.
        Preserved dimensions along which we sum to get the corresponding aggregates.

        convergence_rate: if there are many aggregates/marginal, it could be useful to loosen the convergence criterion.

        max_iteration: Integer. Maximum number of iterations allowed.

        verbose: integer 0, 1 or 2. Each case number includes the outputs of the previous case numbers.
        0: Updated matrix returned.
        1: Flag with the output status (0 for failure and 1 for success).
        2: dataframe with iteration numbers and convergence rate information at all steps.

        rate_tolerance: float value. If above 0.0, like 0.001, the algorithm will stop once the difference between the conv_rate variable of 2 consecutive iterations is below that specified value

        algorithm: 'optimized' (default) uses vectorized numpy implementation; 'legacy' uses the original slice-iteration algorithm (slower). Only affects numpy path.

        For examples, please open the ipfn script or look for help on functions ipfn_np and ipfn_df
        """
        self.original = original
        self.aggregates = aggregates
        self.dimensions = dimensions
        self.weight_col = weight_col
        self.conv_rate = convergence_rate
        self.max_itr = max_iteration
        if verbose not in [0, 1, 2]:
            raise(ValueError(f"wrong verbose input, must be either 0, 1 or 2 but got {verbose}"))
        self.verbose = verbose
        self.rate_tolerance = rate_tolerance
        if algorithm not in ('optimized', 'legacy', 'polars'):
            raise(ValueError(f"algorithm must be 'optimized', 'legacy' or 'polars' but got {algorithm}"))
        self.algorithm = algorithm

    @staticmethod
    def index_axis_elem(dims, axes, elems):
        inc_axis = 0
        idx = ()
        for dim in range(dims):
            if (inc_axis < len(axes)):
                if (dim == axes[inc_axis]):
                    idx += (elems[inc_axis],)
                    inc_axis += 1
                else:
                    idx += (np.s_[:],)
        return idx

    def _ipfn_np_legacy(self, m, aggregates, dimensions, weight_col='total'):
        """
        Legacy numpy implementation retained for benchmarking.
        """

        # Check that the inputs are numpay arrays of floats
        inc = 0
        for aggregate in aggregates:
            if not isinstance(aggregate, np.ndarray):
                aggregate = np.array(aggregate).astype(float)
                aggregates[inc] = aggregate
            elif aggregate.dtype not in [float, float]:
                aggregate = aggregate.astype(float)
                aggregates[inc] = aggregate
            inc += 1
        if not isinstance(m, np.ndarray):
            m = np.array(m)
        elif m.dtype not in [float, float]:
            m = m.astype(float)

        steps = len(aggregates)
        dim = len(m.shape)
        product_elem = []
        tables = [m]
        for inc in range(steps - 1):
            tables.append(np.array(np.zeros(m.shape)))
        original = copy.copy(m)

        for inc in range(steps):
            if inc == (steps - 1):
                table_update = m
                table_current = tables[inc].copy()
            else:
                table_update = tables[inc + 1]
                table_current = tables[inc]
            for dimension in dimensions[inc]:
                product_elem.append(range(m.shape[dimension]))
            for item in product(*product_elem):
                idx = self.index_axis_elem(dim, dimensions[inc], item)
                table_current_slice = table_current[idx]
                mijk = table_current_slice.sum()
                xijk = aggregates[inc]
                xijk = xijk[item]
                if mijk == 0:
                    table_update[idx] = table_current_slice
                else:
                    table_update[idx] = table_current_slice * 1.0 * xijk / mijk
            product_elem = []

        max_conv = 0
        for inc in range(steps):
            for dimension in dimensions[inc]:
                product_elem.append(range(m.shape[dimension]))
            for item in product(*product_elem):
                idx = self.index_axis_elem(dim, dimensions[inc], item)
                ori_ijk = aggregates[inc][item]
                m_slice = m[idx]
                m_ijk = m_slice.sum()
                if abs(m_ijk / ori_ijk - 1) > max_conv:
                    max_conv = abs(m_ijk / ori_ijk - 1)
            product_elem = []

        return m, max_conv

    def ipfn_np(self, m, aggregates, dimensions, weight_col='total'):
        """
        Optimized numpy implementation using vectorized broadcasting.
        """
        # Ensure numpy arrays of floats
        for i, aggregate in enumerate(aggregates):
            if not isinstance(aggregate, np.ndarray):
                aggregates[i] = np.array(aggregate, dtype=float)
            elif aggregate.dtype not in [float, float]:
                aggregates[i] = aggregate.astype(float)
        if not isinstance(m, np.ndarray):
            m = np.array(m, dtype=float)
        elif m.dtype not in [float, float]:
            m = m.astype(float)

        steps = len(aggregates)
        dim = m.ndim

        # Main proportional fitting passes
        for inc in range(steps):
            axes_keep = dimensions[inc]
            # axes to sum (all except axes_keep)
            sum_axes = tuple(ax for ax in range(dim) if ax not in axes_keep)

            # Current marginal sums with dimensions kept for broadcasting
            denom = m.sum(axis=sum_axes, keepdims=True)

            # Target reshaped for broadcasting on original m shape
            target = np.array(aggregates[inc], dtype=float)
            reshape_shape = [1] * dim
            for k, ax in enumerate(axes_keep):
                reshape_shape[ax] = target.shape[k]
            target_b = target.reshape(reshape_shape)

            # Compute ratio into denom buffer to avoid extra allocation
            np.divide(target_b, denom, out=denom, where=denom != 0)
            # When denom == 0, keep ratio 1.0 (no change)
            if np.any(denom == 0):
                denom[denom == 0] = 1.0

            # Scale m in-place
            m *= denom

        # Convergence computation
        max_conv = 0.0
        for inc in range(steps):
            axes_keep = dimensions[inc]
            sum_axes = tuple(ax for ax in range(dim) if ax not in axes_keep)
            current = m.sum(axis=sum_axes)
            target = np.array(aggregates[inc], dtype=float)
            with np.errstate(divide='ignore', invalid='ignore'):
                diff = np.abs(current / target - 1.0)
            if np.any(target == 0):
                # If target==0 and current>0, set inf; if both 0, set 0
                zmask = target == 0
                zval = np.where(current == 0, 0.0, np.inf)
                diff = np.where(zmask, zval, diff)
            # nanmax handles potential NaNs from 0/0
            local_max = np.nanmax(diff)
            if local_max > max_conv:
                max_conv = float(local_max)

        return m, max_conv

    def _ipfn_df_legacy(self, df, aggregates, dimensions, weight_col='total'):
        """
        Legacy pandas implementation retained for benchmarking.
        """
        steps = len(aggregates)
        tables = [df]
        for inc in range(steps - 1):
            tables.append(df.copy())
        original = df.copy()

        # Calculate the new weights for each dimension
        inc = 0
        for features in dimensions:
            if inc == (steps - 1):
                table_update = df
                table_current = tables[inc].copy()
            else:
                table_update = tables[inc + 1]
                table_current = tables[inc]

            tmp = table_current.groupby(features)[weight_col].sum()
            xijk = aggregates[inc]

            feat_l = []
            for feature in features:
                feat_l.append(np.unique(table_current[feature]))
            table_update.set_index(features, inplace=True)
            table_current.set_index(features, inplace=True)

            multi_index_flag = isinstance(table_update.index, pd.MultiIndex)
            if multi_index_flag:
                if not table_update.index.is_monotonic_increasing:
                    table_update.sort_index(inplace=True)
                if not table_current.index.is_monotonic_increasing:
                    table_current.sort_index(inplace=True)

            for feature in product(*feat_l):
                den = tmp.loc[feature]
                # calculate new weight for this iteration

                if not multi_index_flag:
                    msk = table_update.index == feature[0]
                else:
                    msk = feature

                if den == 0:
                    table_update.loc[msk, weight_col] =\
                        table_current.loc[feature, weight_col] *\
                        xijk.loc[feature]
                else:
                    table_update.loc[msk, weight_col] = \
                        table_current.loc[feature, weight_col].astype(float) * \
                        xijk.loc[feature] / den

            table_update.reset_index(inplace=True)
            table_current.reset_index(inplace=True)
            inc += 1

        # Calculate the max convergence rate
        max_conv = 0
        inc = 0
        for features in dimensions:
            tmp = table_update.groupby(features)[weight_col].sum()
            ori_ijk = aggregates[inc]
            temp_conv = max(abs(tmp / ori_ijk - 1))
            if temp_conv > max_conv:
                max_conv = temp_conv
            inc += 1

        return table_update, max_conv

    def ipfn_df(self, df, aggregates, dimensions, weight_col='total'):
        """
        Runs the ipfn method from a dataframe df, aggregates/marginals and the dimension(s) preserved.
        Vectorized implementation to reduce Python-level loops and memory usage.
        """
        steps = len(aggregates)

        # Calculate the new weights for each dimension directly on df
        for inc, features in enumerate(dimensions):
            # Current group sums
            tmp = df.groupby(features, sort=False)[weight_col].sum()
            xijk = aggregates[inc]
            # Align target to current group index if needed
            if isinstance(xijk, (pd.Series, pd.DataFrame)):
                if isinstance(xijk, pd.DataFrame):
                    if xijk.shape[1] != 1:
                        raise(ValueError('Aggregate DataFrame must have a single column'))
                    xijk = xijk.iloc[:, 0]
                xijk_aligned = xijk.reindex(tmp.index)
            else:
                xijk_aligned = pd.Series(np.array(xijk, dtype=float).ravel(), index=tmp.index)

            ratio = pd.Series(1.0, index=tmp.index, dtype=float)
            nonzero = tmp != 0
            ratio.loc[nonzero] = (xijk_aligned.loc[nonzero].astype(float) / tmp.loc[nonzero].astype(float))

            ratio_df = ratio.rename('ipf_ratio').reset_index()
            merged = df.merge(ratio_df, on=features, how='left', sort=False)
            df[weight_col] = merged[weight_col].astype(float) * merged['ipf_ratio'].astype(float)

        # Calculate the max convergence rate
        max_conv = 0
        for inc, features in enumerate(dimensions):
            tmp = df.groupby(features, sort=False)[weight_col].sum()
            ori_ijk = aggregates[inc]
            if isinstance(ori_ijk, pd.DataFrame):
                ori_ijk = ori_ijk.iloc[:, 0]
            if isinstance(ori_ijk, pd.Series):
                ori_ijk = ori_ijk.reindex(tmp.index)
            with np.errstate(divide='ignore', invalid='ignore'):
                temp_conv = np.nanmax(np.abs(tmp.values / np.array(ori_ijk, dtype=float) - 1))
            if temp_conv > max_conv:
                max_conv = temp_conv

        return df, max_conv

    @staticmethod
    def _aggregate_to_pl_df(xijk, features, target_col: str = 'target'):
        """Convert aggregate (pandas Series/DataFrame or numpy) to a Polars DataFrame with given feature columns and target column."""
        if pl is None:
            raise RuntimeError("Polars is not available")
        # pandas Series/DataFrame path
        if isinstance(xijk, pd.DataFrame):
            if xijk.shape[1] != 1:
                raise(ValueError('Aggregate DataFrame must have a single column'))
            xijk = xijk.iloc[:, 0]
        if isinstance(xijk, pd.Series):
            df_pd = xijk.rename(target_col).reset_index()
            # Ensure columns order matches features then target
            # If features do not match, rely on column names after reset_index
            # Rename target to target_col already done
            return pl.DataFrame(df_pd)
        # numpy array path (fallback): build cartesian from features unique values is not available here
        # We only support Series/DataFrame aggregates for DataFrame backend
        arr = np.array(xijk, dtype=float).ravel()
        # Create a single dummy column when features is single and length matches
        if len(features) == 1:
            return pl.DataFrame({features[0]: range(len(arr)), target_col: arr})
        else:
            raise ValueError("When using polars backend with DataFrame inputs, aggregates should be pandas Series/DataFrame")

    def ipfn_pl(self, df_pl, aggregates, dimensions, weight_col='total'):
        """
        Polars-based implementation of IPFN for tabular inputs.
        Accepts a Polars DataFrame and aggregates as pandas Series/DataFrame or Polars DataFrames.
        Returns a Polars DataFrame and the max convergence.
        """
        if pl is None:
            raise RuntimeError("Polars is not installed. Please install polars to use algorithm='polars'.")

        steps = len(aggregates)
        df = df_pl
        # Ensure weight column is float
        if weight_col not in df.columns:
            raise ValueError(f"weight_col '{weight_col}' not found in DataFrame")
        df = df.with_columns(pl.col(weight_col).cast(pl.Float64))

        for inc, features in enumerate(dimensions):
            # Group sums
            grouped = df.group_by(features).agg(pl.col(weight_col).sum().alias('sum_w'))
            # Target as Polars
            xijk = aggregates[inc]
            if isinstance(xijk, pl.DataFrame):
                target_df = xijk.rename({xijk.columns[-1]: 'target'})
            else:
                target_df = self._aggregate_to_pl_df(xijk, features, target_col='target')
            # Join and compute ratio
            ratio_tbl = grouped.join(target_df, on=features, how='left')
            ratio_tbl = ratio_tbl.with_columns(
                pl.when(pl.col('sum_w') != 0)
                  .then(pl.col('target').cast(pl.Float64) / pl.col('sum_w'))
                  .otherwise(1.0)
                  .alias('ipf_ratio')
            ).select(features + ['ipf_ratio'])
            # Apply ratio to weights
            df = df.join(ratio_tbl, on=features, how='left')
            df = df.with_columns((pl.col(weight_col) * pl.col('ipf_ratio')).alias(weight_col))
            df = df.drop('ipf_ratio')

        # Convergence computation
        max_conv = 0.0
        for inc, features in enumerate(dimensions):
            grouped = df.group_by(features).agg(pl.col(weight_col).sum().alias('sum_w'))
            xijk = aggregates[inc]
            if isinstance(xijk, pl.DataFrame):
                target_df = xijk.rename({xijk.columns[-1]: 'target'})
            else:
                target_df = self._aggregate_to_pl_df(xijk, features, target_col='target')
            joined = grouped.join(target_df, on=features, how='left')
            joined = joined.with_columns(
                pl.when(pl.col('target') == 0)
                  .then(pl.when(pl.col('sum_w') == 0).then(0.0).otherwise(float('inf')))
                  .otherwise((pl.col('sum_w') / pl.col('target') - 1.0).abs())
                  .alias('conv_abs')
            )
            local_max = joined.select(pl.max('conv_abs')).item()
            if local_max is not None and float(local_max) > max_conv:
                max_conv = float(local_max)

        return df, max_conv

    def iteration(self):
        """
        Runs the ipfn algorithm. Automatically detects of working with numpy ndarray or (pandas|polars) dataframes.
        """

        i = 0
        conv = np.inf
        old_conv = -np.inf
        conv_list = []
        m = self.original

        used_polars_backend = False

        # Determine backend based on input type and requested algorithm
        if isinstance(m, pd.DataFrame):
            if self.algorithm == 'legacy':
                ipfn_method = self._ipfn_df_legacy
            elif self.algorithm == 'polars':
                if pl is None:
                    raise RuntimeError("algorithm='polars' requested but Polars is not installed")
                # convert pandas to polars
                try:
                    m = pl.from_pandas(m)
                except Exception:
                    m = pl.DataFrame(m)
                ipfn_method = self.ipfn_pl
                used_polars_backend = True
            else:
                ipfn_method = self.ipfn_df
        elif isinstance(m, np.ndarray):
            ipfn_method = self._ipfn_np_legacy if self.algorithm == 'legacy' else self.ipfn_np
            self.original = self.original.astype('float64')
        elif (pl is not None) and isinstance(m, pl.DataFrame):
            ipfn_method = self.ipfn_pl
            used_polars_backend = True
        else:
            raise(ValueError('Data input instance not recognized. The input matrix is not a numpy array or (pandas|polars) DataFrame'))

        while ((i <= self.max_itr and conv > self.conv_rate) and (i <= self.max_itr and abs(conv - old_conv) > self.rate_tolerance)):
            old_conv = conv
            m, conv = ipfn_method(m, self.aggregates, self.dimensions, self.weight_col)
            conv_list.append(conv)
            i += 1
        converged = 1
        if i <= self.max_itr:
            if (not conv > self.conv_rate) & (self.verbose > 1):
                print('ipfn converged: convergence_rate below threshold')
            elif not abs(conv - old_conv) > self.rate_tolerance:
                print('ipfn converged: convergence_rate not updating or below rate_tolerance')
        else:
            print('Maximum iterations reached')
            converged = 0

        # Convert back to pandas DataFrame if we used polars backend but the original was pandas
        if used_polars_backend and (isinstance(self.original, pd.DataFrame)):
            try:
                m = m.to_pandas()
            except Exception:
                # Fallback: construct via rows
                m = pd.DataFrame(m.to_dicts())

        # Handle the verbose
        if self.verbose == 0:
            return m
        elif self.verbose == 1:
            return m, converged
        elif self.verbose == 2:
            return m, converged, pd.DataFrame({'iteration': range(i), 'conv': conv_list}).set_index('iteration')
        else:
            raise(ValueError(f'wrong verbose input, must be either 0, 1 or 2 but got {self.verbose}'))
