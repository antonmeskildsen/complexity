import pandas as pd
import streamlit as st
import altair as alt
from altair import datum

from gen import *

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.datasets import fetch_california_housing, load_diabetes

import sympy
from sympy import lambdify, symbols
from sympy.parsing.sympy_parser import parse_expr


# @st.cache
def gen_s_data(nf, n, domain_min, domain_max, margin, fun_type):
    u_std = stats.uniform(loc=domain_min - margin,
                          scale=domain_max - domain_min + margin * 2)
    if fun_type == 'Noise':
        return nf.sample_data(n, u_std.rvs)  # , u_std
    else:
        return nf.sample_data(n, [u_std.rvs, u_std.rvs])  # , u_std


dataset = fetch_california_housing(data_home='./data')

with st.sidebar:
    "# Settings"
    with st.expander('Advanced:'):
        typ = st.radio('Source type', ['Function', 'Dataset'])

    "## Problem settings"
    if typ == 'Function':
        fun = st.selectbox('Function', ['cos', 'polynomial', 'linear'])

        if fun == 'cos':
            func = np.cos
            domain = (0, 10)
            noise_max = 2.
            "$h(x) = cos(x)$"
        elif fun == 'polynomial':
            func = lambda x: x ** 3 - 5 * x ** 2
            "$h(x) = x^3 - 5x^2$"
            domain = (-1, 6)
            noise_max = 30.
        elif fun == 'linear':
            func = lambda x: -0.8 * x + 2
            "$h(x) = -0.8x + 2$"
            domain = (0, 5)
            noise_max = 2.

        fun_type = st.selectbox('Function type', ['Noise', 'Hidden feature'])
        if fun_type == 'Noise':
            noise_scale = st.slider('Noise scale', 0., noise_max, 0.2)
            nf = NoisyFunction(func, stats.norm(scale=noise_scale).rvs)
        else:
            func_y = np.sin
            func_new = lambda x, y: func(x) + func_y(y)
            nf = HiddenParamFunction(func_new)
    elif typ == 'Dataset':
        feature = st.selectbox('Input feature', dataset.feature_names)

        feature_index = dataset.feature_names.index(feature)
        x_all = dataset.data[:, feature_index]
        y_all = dataset.target
        domain = (x_all.min(), x_all.max())

    samples = st.slider('Samples', 2, 100)
    # samples_per_model = int(st.number_input('Samples per model', 1, 1000, 10))
    samples_per_model = samples

    """
    ## Model settings
    The fitted model is a polynomial of degree $n$.
    """
    order = st.slider('Polynomial order (n)', 0, 15, 1)
    regularization = st.checkbox('Use regularization term?')
    if regularization:
        reg_type = st.radio('Type', ['Lasso (L1)', 'Ridge (L2)'])
        a2 = st.slider('Lambda (10^x)', -20, 2, -2)
        alpha = 10 ** a2
        f"Actual value: $\lambda={alpha}$"
        # alpha = st.number_input(r'Alpha', value=0.1)

        if reg_type == 'Lasso (L1)':
            model = Lasso(alpha)
        elif reg_type == 'Ridge (L2)':
            model = Ridge(alpha)
    else:
        model = LinearRegression()

    """
    ## Data generation
    """
    n_models = int(st.number_input('N models', 1, 100, 10))
    with st.expander('Extra settings'):
        domain_min = st.number_input('Domain minimum', value=domain[0])
        domain_max = st.number_input('Domain maximum', value=domain[1])
        margin = st.number_input('Margin', min_value=0., value=1.)

# Generating data
if typ == 'Function':
    n = samples_per_model * n_models

    x, y = gen_s_data(nf, n, domain_min, domain_max, margin, fun_type)
elif typ == 'Dataset':
    n = samples_per_model * n_models
    idx = np.random.choice(np.arange(len(x_all)), n)
    x = x_all[idx]
    y = y_all[idx]

"""
# Uncertainty in machine learning
This is a companion webpage for lecture 11. It contains playground examples for 
experimenting with uncertainty and using regularisation to control model complexity.
"""

"""
## Problem and data
"""

col1, col2 = st.columns(2)

xs = np.linspace(domain_min, domain_max, num=100)
if typ == 'Function':
    ys = func(xs)

    true_func = pd.DataFrame(np.array([xs, ys]).T, columns=['x', 'y'])
    true_func['Legend'] = 'Original function'

df = gen_point_split_dataframe(x, y, n_models)

with col1:
    selection = alt.selection_multi(fields=['experiment'])
    color = alt.condition(selection,
                          alt.Color('experiment:N', legend=None),
                          alt.value('lightgray'))

    ch = alt.Chart(df).interactive(bind_y=False).mark_point().encode(
        x='x',
        y='y',
        color='experiment:N'
    )

    if typ == 'Function':
        ch_true = alt.Chart(true_func).mark_line().encode(
            x='x',
            y='y',
            color=alt.value('#000000')
        )
        ch += ch_true
    # legend = alt.Chart(df).mark_point().encode(
    #     y=alt.Y('experiment:N', axis=alt.Axis(orient='right')),
    #     color=color
    # ).add_selection(
    #     selection
    # )

    "### Ground truth data"
    st.altair_chart(ch, use_container_width=True)

base_model = make_pipeline(PolynomialFeatures(degree=order), StandardScaler(),
                           model)
ms = multi_model_fit(base_model, x, y, n_models)

xs = np.linspace(domain_min, domain_max, num=100)
yss = multi_model_eval(ms, xs)

model_dataframe = multi_model_eval_dataframe(xs, yss)

with col2:
    ch = alt.Chart(model_dataframe).interactive(bind_y=False).mark_line().encode(
        x='x',
        y='y',
        color='experiment:N'
    )

    "### Fitted models"
    st.altair_chart(ch, use_container_width=True)

"""
## Metrics
"""

mean_y = model_dataframe.groupby('x').mean()
if typ == 'Function':
    mg = pd.merge(mean_y, true_func, on='x')
    bias_sq = ((mg['y_x'] - mg['y_y']) ** 2).mean()
else:
    bias_sq = np.nan

# Merge predicted values with mean predictions mean_y
merged = pd.merge(model_dataframe, mean_y, on='x')
variance = ((merged['y_x'] - merged['y_y']) ** 2).mean()

if typ == 'Function':
    # MSE optimal. Simply the data MSE
    sqe = ((df['y'] - func(df['x'])) ** 2).mean()
else:
    sqe = np.nan
# The measured MSE for the models (mean)
sqes = 0
for i, fitted in enumerate(ms):
    subset = df[df['experiment'] != float(i)]
    sqe2 = ((subset['y'] - fitted.predict(subset['x'].to_numpy().reshape(-1, 1))) ** 2)
    sqes += sqe2.mean()
sqes /= len(ms)

cola, colb, colc, cold = st.columns(4)

with cola:
    st.metric('Bias^2', np.round(bias_sq, 4))

with colb:
    st.metric('Variance', np.round(variance, 4))

with colc:
    st.metric('Noise (optimal MSE)', np.round(sqe, 4))

with cold:
    st.metric('Mean MSE', np.round(sqes, 4))

with st.expander('Metric details'):
    r"""
    **Bias:**
    The bias represents how much the average prediction of all the models deviates from
    the optimal solution (the selected function).

    **Variance:**
    Variance is the model uncertainty. It is measured as the average deviation of a
    specific model from the average model. In the plot below the shaded area visualises
    the standard deviation $\sqrt{Var}$  for each x-value. The full metric is an average 
    over all values.

    **Noise (optimal MSE):**
    This is the lower error bound for the specific problem. In our case, we defined 
    this function first and added noise afterwards but in reality, the problem is 
    already defined and this function is not known. Without knowledge of the underlying 
    model, it is impossible to calculate the actual noise for a given problem.

    **Actual loss:**
    This is the measured test loss for a given problem.
    """

"""
## Analysis
"""

col1, col2 = st.columns(2)

with col1:
    """
    ### Average model performance
    The following plot visualises the average of all the experimental models. This
    is effectively a model ensemble.
    """
    model_dataframe['Legend'] = 'Mean prediction'

    line = alt.Chart(model_dataframe).interactive(bind_y=False).mark_line().encode(
        x='x',
        y='mean(y)',
        color='Legend'
    )

    if typ == 'Function':
        true_func['Legend'] = 'Original'

        line_true = alt.Chart(true_func).mark_line().encode(
            x='x',
            y='y',
            color='Legend'
        )
        line += line_true

    band = alt.Chart(model_dataframe).mark_errorband(extent='stdev').encode(
        x='x',
        y='y',
        color='Legend'
    )

    st.altair_chart(band + line, use_container_width=True)

with col2:
    """
    ### Weights
    This plot visualises the weights for the models
    """

    coeffs = []
    for fitted in ms:
        coeffs.append(fitted.steps[2][1].coef_)
    coeffs = np.abs(np.stack(coeffs)).mean(axis=0)

    coeffs_df = pd.DataFrame(coeffs, columns=['Weight']).reset_index()

    ch = alt.Chart(coeffs_df).mark_bar().encode(
        alt.Y('Weight'),  # , scale=alt.Scale(domain=(10 ** -1, 1000))),
        x='index:N'
    )

    st.altair_chart(ch, use_container_width=True)
    # st.write(base_model.steps[2][1].coef_)
