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

import sympy
from sympy import lambdify, symbols
from sympy.parsing.sympy_parser import parse_expr

with st.sidebar:
    "# Settings"

    "## Function"
    fun = st.selectbox('Function', ['cos', 'polynomial', 'linear'])

    if fun == 'cos':
        func = np.cos
        domain = (0, 10)
    elif fun == 'polynomial':
        func = lambda x: x ** 3 - 5 * x ** 2
        domain = (-1, 6)
    elif fun == 'linear':
        func = lambda x: -0.8 * x + 2
        domain = (0, 5)

    "## Noise"
    noise_scale = st.slider('Noise scale', 0., 2., 0.2)

    "## Samples"
    samples = st.slider('Samples', 2, 100)
    #samples_per_model = int(st.number_input('Samples per model', 1, 1000, 10))
    samples_per_model = samples

    """
    ## Model settings
    The fitted model is a polynomial of degree $n$.
    """
    order = st.slider('Polynomial order (n)', 0, 15, 1)
    regularization = st.checkbox('Use regularization term?')
    if regularization:
        reg_type = st.radio('Type', ['Lasso (L1)', 'Ridge (L2)'])
        a2 = st.slider('Alpha2', -20, 2, -2)
        alpha = 10**a2
        #alpha = st.number_input(r'Alpha', value=0.1)

        if reg_type == 'Lasso (L1)':
            model = Lasso(alpha)
        elif reg_type == 'Ridge (L2)':
            model = Ridge(alpha)
    else:
        model = LinearRegression()

    """
    ## Data generation
    """
    n_models = int(st.number_input('N models', 1, 20, 10))
    with st.expander('Extra settings'):
        domain_min = st.number_input('Domain minimum', value=domain[0])
        domain_max = st.number_input('Domain maximum', value=domain[1])
        margin = st.number_input('Margin', min_value=0., value=1.)

"""
# Uncertainty in machine learning
This is a companion webpage for lecture 11. It contains playground examples for 
experimenting with uncertainty and using regularisation to control model complexity.
"""

"""
## Overfitting

"""

"""

## Problem uncertainty (noise)
This section contains a few interactive tools that let you experiment with how 
uncertainty in the problem domain can affect the generalisation performance (i.e.
overfitting) of the fitted model. 


"""


# @st.cache
def gen_s_data(nf, n, domain_min, domain_max, margin):
    u_std = stats.uniform(loc=domain_min - margin,
                          scale=domain_max - domain_min + margin * 2)
    return nf.sample_data(n, u_std.rvs)  # , u_std


col1, col2 = st.columns(2)

col1.write('### Problem settings')
# func = st.selectbox('Function', [np.cos, np.sin])

# st.slider()

col2.write('### Data generation')

nf = NoisyFunction(func, stats.norm(scale=noise_scale).rvs)
n = samples_per_model * n_models


x, y = gen_s_data(nf, n, domain_min, domain_max, margin)

xs = np.linspace(domain_min, domain_max, num=100)
ys = func(xs)

true_func = pd.DataFrame(np.array([xs, ys]).T, columns=['x', 'y'])

df = gen_point_split_dataframe(x, y, n_models)

ch = alt.Chart(df).interactive().mark_point().encode(
    x='x',
    y='y',
    color='num:N'
)

"### Results"
col1.altair_chart(ch, use_container_width=True)

'### Model settings'
base_model = make_pipeline(PolynomialFeatures(degree=order), StandardScaler(),
                           model)
ms = multi_model_fit(base_model, x, y, n_models)

xs = np.linspace(domain_min, domain_max)
yss = multi_model_eval(ms, xs)

model_dataframe = multi_model_eval_dataframe(xs, yss)

ch = alt.Chart(model_dataframe).interactive().mark_line().encode(
    x='x',
    y='y',
    color='num:N'
)

col2.altair_chart(ch, use_container_width=True)



# @st.cache
def gen_s_data(nf, n, domain_min, domain_max, margin):
    u_std = stats.uniform(loc=domain_min - margin,
                          scale=domain_max - domain_min + margin * 2)
    return nf.sample_data(n, u_std.rvs)  # , u_std


col1, col2 = st.columns(2)

col1.write('### Problem settings')
# func = st.selectbox('Function', [np.cos, np.sin])


noise_scale = col1.number_input('Noise scale', 0., 2., 0.2)

col2.write('### Data generation')
samples_per_model = int(col2.number_input('Samples per model', 1, 100, 10))

nf = NoisyFunction(func, stats.norm(scale=noise_scale).rvs)
n = samples_per_model * n_models

x, y = gen_s_data(nf, n, domain_min, domain_max, margin)

xs = np.linspace(domain_min, domain_max, num=100)
ys = func(xs)

true_func = pd.DataFrame(np.array([xs, ys]).T, columns=['x', 'y'])

df = gen_point_split_dataframe(x, y, n_models)

ch = alt.Chart(df).interactive().mark_point().encode(
    x='x',
    y='y',
    color='num:N'
)

"### Results"
st.altair_chart(ch, use_container_width=True)

'### Model settings'

base_model = make_pipeline(PolynomialFeatures(degree=order), StandardScaler(),
                           model)
ms = multi_model_fit(base_model, x, y, n_models)

xs = np.linspace(domain_min, domain_max)
yss = multi_model_eval(ms, xs)

model_dataframe = multi_model_eval_dataframe(xs, yss)

ch = alt.Chart(model_dataframe).interactive().mark_line().encode(
    x='x',
    y='y',
    color='num:N'
)

st.altair_chart(ch, use_container_width=True)

line = alt.Chart(model_dataframe).interactive().mark_line().encode(
    x='x',
    y='mean(y)'
)

band = alt.Chart(model_dataframe).mark_errorband(extent='ci').encode(
    x='x',
    y='y',
)

st.altair_chart(band + line, use_container_width=True)


"""
## Model complexity (bias/variance)
This section contains tools for experimenting with model complexity and its effect on
bias and variance.
"""

"""
## Regularisation
Finally, we focus on methods for preventing or diminishing the effect of overfitting.
"""
