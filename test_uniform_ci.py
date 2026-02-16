#!/usr/bin/env python3
"""Test the uniform confidence interval implementation."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'notebooks'))

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
from pysyncon import Dataprep, Synth
from pysyncon.utils import PlaceboTest
import util
import importlib
importlib.reload(util)

# Test with Maule (Chile) data
print("Loading Chile data...")
df = pd.read_excel('data/chile_data.xlsx')

CHILE_PLOT_YAXIS_LABEL = 'GDP per Capita (Millions of CLP)'

print("Setting up Dataprep for Chile...")
dataprep = Dataprep(
    foo=df,
    predictors=['agropecuario', 'pesca', 'mineria', 'industria_m',
       'electricidad', 'construccion', 'comercio', 'transporte',
       'servicios_financieros', 'vivienda', 'administracion',
       'ensenanza', 'salud', 'entretenimiento', 'hogares_privados',
       'administracion_extraterritorial'],
    predictors_op="mean",
    time_predictors_prior=range(2000, 2009),
    special_predictors=[("pib_per_capita", range(2005, 2009), "mean")],
    dependent="pib_per_capita",
    unit_variable="region",
    time_variable="year",
    treatment_identifier="VII Del Maule",
    controls_identifier=[
        "I De Tarapacá",
        "II De Antofagasta",
        "III De Atacama",
        "IV De Coquimbo",
        "IX De La Araucanía",
        "V De Valparaíso",
        "VI Del Libertador General Bernardo OHiggins",
        "X De Los Lagos",
        "XI Aysén del General Carlos Ibáñez del Campo",
        "XII De Magallanes y de la Antártica Chilena",
        "RMS Región Metropolitana de Santiago",
    ],
    time_optimize_ssr=range(2000, 2009),
)

print("Fitting synthetic control...")
synth = Synth()
synth.fit(dataprep=dataprep, optim_method="Nelder-Mead", optim_initial="ols")

print("Running placebo test (this may take a few minutes)...")
placebo_test = PlaceboTest()
placebo_test.fit(
    dataprep=dataprep,
    scm=synth,
    scm_options={"optim_method": "Nelder-Mead", "optim_initial": "ols"},
)

print("Generating uniform confidence interval plot...")
util.gap_plot_with_uniform_ci(
    synth=synth,
    placebo=placebo_test,
    time_period=range(2000, 2024),
    treatment_time=2010,
    alpha=0.1,  # 90% confidence interval
    mspe_threshold=100,
    divide_by=1000000,
    y_axis_label=CHILE_PLOT_YAXIS_LABEL,
    filename='maule_gap_uniform_ci.png'
)

print("\nSuccess! Generated maule_gap_uniform_ci.png in article_assets/")
