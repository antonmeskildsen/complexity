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
func = lambda x: 0.5 * x

#st.slider()
noise_scale = col1.slider('Noise scale', 0., 2., 0.2)

col2.write('### Data generation')
samples_per_model = int(col2.number_input('Samples per model', 1, 1000, 10))
n_models = 10
domain_min = 0
domain_max = 5
margin = 1

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
base_model = make_pipeline(PolynomialFeatures(degree=1), StandardScaler(),
                           LinearRegression())
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


"""
## Model complexity (bias/variance)
This section contains tools for experimenting with model complexity and its effect on
bias and variance.
"""

"""
## Regularisation
Finally, we focus on methods for preventing or diminishing the effect of overfitting.
"""


# @st.cache
def gen_s_data(nf, n, domain_min, domain_max, margin):
    u_std = stats.uniform(loc=domain_min - margin,
                          scale=domain_max - domain_min + margin * 2)
    return nf.sample_data(n, u_std.rvs)  # , u_std


col1, col2 = st.columns(2)

col1.write('### Problem settings')
# func = st.selectbox('Function', [np.cos, np.sin])
f2 = col1.text_input('Function (in x)', value='x**2')
x = symbols('x')
f2n = parse_expr(f2)
func = lambdify(x, f2n, "numpy")

noise_scale = col1.number_input('Noise scale', 0., 2., 0.2)

col2.write('### Data generation')
samples_per_model = int(col2.number_input('Samples per model', 1, 100, 10))
n_models = int(col2.number_input('N models', 1, 20, 10))
domain_min = col2.number_input('Domain minimum', value=0.)
domain_max = col2.number_input('Domain maximum', value=5.)
margin = col2.number_input('Margin', min_value=0., value=1.)

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
order = st.slider('Polynomial order', 0, 15, 1)
alpha = st.number_input(r'Alpha $\alpha$', value=0.1)

base_model = make_pipeline(PolynomialFeatures(degree=order), StandardScaler(),
                           Ridge(alpha))
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

if st.checkbox('Old stuff (not used anymore)'):
    """
    ## Probability and distributions
    
    """

    col1, col2 = st.columns(2)

    dist_map = ["Normal", "Beta"]

    dist_name = col1.selectbox('Distribution function', options=dist_map)

    if dist_name == "Normal":
        mean = col1.slider('Mean', min_value=-5., max_value=5., value=0.)
        var = col1.slider('Variance', min_value=0.1, max_value=5., value=1.)
        std = np.sqrt(var)
        dist = stats.norm(mean, std)
    elif dist_name == "Beta":
        alpha = col1.slider('Alpha', min_value=0., max_value=5., value=1.)
        beta = col1.slider('Beta', min_value=0., max_value=5., value=1.)
        dist = stats.beta(alpha, beta)

    xs = np.linspace(*dist.interval(0.99), num=100)
    # dist = stats.norm(loc=mean, scale=std)
    ys = dist.pdf(xs)

    df = pd.DataFrame(np.array([xs, ys]).T, columns=['x', 'y'])

    pdf_chart = alt.Chart(df).interactive(bind_x=True, bind_y=False).mark_line().encode(
        x='x',
        y='y'
    )

    var_chart = alt.Chart(df).mark_area().encode(
        x='x',
        y='y'
    ).transform_filter(
        (datum.x > dist.mean() - dist.std()) & (datum.x < dist.mean() + dist.std())
    )

    dd = pd.DataFrame({
        'Val': [dist.mean()],
        'color': ['black']
    })
    markers = alt.Chart(dd).mark_rule().encode(
        x='Val'
    )

    col2.altair_chart(pdf_chart + var_chart + markers)
