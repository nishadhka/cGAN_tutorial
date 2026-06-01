# Why ECMWF IFS Forecasts Are Used Instead of ERA5 for cGAN-Based Rainfall Postprocessing

## Background

Recent studies on deep-learning-based precipitation postprocessing, especially using Conditional Generative Adversarial Networks (cGANs), commonly use Numerical Weather Prediction (NWP) forecast outputs such as ECMWF Integrated Forecast System (IFS) instead of long-term reanalysis products like ERA5.

At first glance, ERA5 appears attractive because:

* it is long-term,
* globally available,
* physically consistent,
* and easier to access for training.

However, operational cGAN postprocessing studies intentionally train on the forecast model that will later be used operationally (e.g., IFS, GEFS), rather than on ERA5.

This is because cGAN postprocessing primarily learns:

* forecast-model-specific rainfall errors,
* spatial displacement biases,
* convective structure distortions,
* and lead-time-dependent forecast behavior,

rather than learning the atmosphere in a universal sense.

---

# Core Concept: What the cGAN Actually Learns

The cGAN is trained to learn the relationship:

[
\text{Forecast Model State} \rightarrow \text{Observed Rainfall}
]

For example:

[
\text{IFS Forecast} \rightarrow \text{NIMROD Observation}
]

Thus the network learns:

* how IFS rainfall differs from reality,
* typical biases in convection,
* timing errors,
* spatial smoothing,
* rainfall morphology errors,
* and extreme rainfall underestimation.

The cGAN therefore becomes tightly coupled to the characteristics of the parent forecast model.

This means:

* the model learns the “error fingerprint” of IFS,
* not a universal atmospheric rainfall generator.

---

# Why ERA5 Is Not Usually Used Directly

ERA5 is fundamentally different from operational forecasts.

ERA5:

* assimilates observations,
* is dynamically constrained,
* is closer to the “true atmosphere,”
* has smoother and more realistic rainfall structures,
* and does not exhibit operational forecast drift.

Operational forecast models like IFS or GEFS:

* evolve freely into the future,
* contain lead-time-dependent errors,
* have convective parameterization biases,
* exhibit ensemble spread,
* and possess systematic forecast-specific rainfall structures.

Therefore:

[
P(\text{Observation} \mid \text{ERA5})
\neq
P(\text{Observation} \mid \text{IFS})
]

The statistical relationship between ERA5 and observations is very different from the relationship between IFS forecasts and observations.

If training is performed using ERA5 but operational inference uses IFS:

* the network encounters a distribution mismatch,
* known as domain shift,
* which often degrades performance substantially.

Operational machine learning generally requires:

[
P_{train}(x) \approx P_{inference}(x)
]

meaning:

* the training inputs should resemble the operational inputs.

Thus, if operational forecasts come from IFS, the cGAN is usually trained directly on IFS.

---

# Why cGAN Postprocessing Is Often Not Transferable Between Models

A key limitation of current cGAN postprocessing systems is that they are usually:

* model-specific,
* region-specific,
* and lead-time-specific.

For example:

* IFS has its own convection biases,
* GEFS has different rainfall distributions,
* different convective parameterizations,
* different spatial smoothing characteristics,
* and different ensemble behavior.

Thus, a cGAN trained on IFS does not automatically generalize well to GEFS.

The network internally learns:

[
f_{\theta}(\text{IFS-specific rainfall errors})
]

rather than universal atmospheric dynamics.

As a result:

* retraining or fine-tuning is often needed for each forecast system.

---

# Why Multiple Atmospheric Variables Are Used

Rainfall is one of the hardest atmospheric variables to model because it is:

* intermittent,
* highly localized,
* strongly nonlinear,
* and sensitive to instability and moisture interactions.

Using only precipitation as input is usually insufficient.

Therefore, cGAN systems condition on multiple atmospheric predictors that describe:

* moisture availability,
* instability,
* uplift,
* storm organization,
* synoptic circulation,
* and surface forcing.

These variables provide physical context that helps the network infer where and how precipitation develops.

---

# Variables Used in Paper 1 (East African Rainfall cGAN)

Paper:
“Postprocessing East African Rainfall Forecasts Using a Generative Machine Learning Model”

## Forecast Variables from ECMWF IFS

### Precipitation Variables

* Total precipitation
* Convective precipitation

These provide:

* baseline rainfall forecast structure,
* convective rainfall signal,
* and spatial precipitation patterns.

---

### Thermodynamic Variables

* Convective Available Potential Energy (CAPE)
* Total column water vapor
* Total column cloud liquid water

These describe:

* atmospheric instability,
* moisture availability,
* cloud formation potential,
* and convective environment.

CAPE is particularly important for tropical convection.

---

### Dynamic Variables

* u-wind at 700 hPa
* v-wind at 700 hPa

These encode:

* moisture transport,
* monsoon flow,
* convergence zones,
* storm steering,
* and mesoscale organization.

This is especially important in East Africa where rainfall is strongly linked to:

* Indian Ocean inflow,
* Turkana jet dynamics,
* Congo airmass transport,
* and regional topography.

---

### Surface and Radiation Variables

* Surface pressure
* TOA incident solar radiation

These represent:

* surface forcing,
* synoptic pressure structures,
* diurnal heating,
* and convective triggering mechanisms.

The diurnal cycle is critical in tropical rainfall systems.

---

## Static High-Resolution Variables

### Surface geopotential

Used to represent:

* terrain height,
* orographic lifting,
* mountain-induced convection,
* and valley/channel effects.

### Land-sea mask

Used to represent:

* coastline interactions,
* lake/sea breeze effects,
* and land-ocean thermal contrasts.

These are extremely important in East Africa due to:

* complex terrain,
* coastal convection,
* and Lake Victoria influences.

---

## Observation Dataset

The study used:

* NIMROD rainfall observations

as the target “truth” dataset for training.

Thus the full learning relationship becomes:

[
\text{IFS atmospheric state}
\rightarrow
\text{NIMROD rainfall observations}
]

---

# Variables Used in Paper 2 (Extreme Precipitation cGAN)

Paper:
“Postprocessing for 24-Hour Advanced Forecasting of Extreme Precipitation Using Deep Learning Generative Models”

The variables were grouped into:

* thermodynamic,
* dynamic,
* and precipitation categories.

---

# Thermodynamic Variables

* Convective Available Potential Energy (CAPE)
* Total column water vapor

These represent:

* instability,
* latent energy,
* and moisture supply.

---

# Dynamic Variables

* 850 hPa u-wind
* 850 hPa v-wind
* 500 hPa geopotential height
* Mean sea level pressure
* Surface pressure

These describe:

* synoptic circulation,
* moisture transport,
* storm steering,
* low-level convergence,
* and large-scale atmospheric forcing.

500 hPa geopotential height is especially important for:

* identifying troughs,
* ridges,
* and large-scale circulation organization.

---

# Precipitation Variables

* Total precipitation
* Large-scale precipitation
* Convective precipitation
* Padding precipitation

These encode:

* different rainfall-generation mechanisms,
* convective versus stratiform structure,
* and forecast rainfall morphology.

---

# Why Multiple Variables Improve cGAN Performance

Using multiple physically meaningful variables allows the cGAN to learn:

* not only rainfall correction,
* but the atmospheric conditions associated with rainfall generation.

This helps improve:

* spatial realism,
* extreme rainfall representation,
* diurnal cycle behavior,
* convective organization,
* and probabilistic structure.

The network effectively learns:

* moisture–instability interactions,
* circulation patterns,
* and rainfall-triggering environments.

---

# Implications for GEFS-Based cGAN Systems

For GEFS-based postprocessing, a similar physically informed variable selection strategy is recommended.

A practical variable set could include:

## Precipitation

* Total precipitation
* Convective precipitation

## Moisture

* Total column water vapor
* Relative humidity

## Instability

* CAPE
* CIN

## Dynamics

* u/v winds at 850 hPa
* Omega at 500 hPa
* Geopotential height at 500 hPa

## Static Variables

* Elevation
* Land-sea mask

This would provide:

* convective structure,
* moisture transport,
* synoptic forcing,
* and terrain interaction information

necessary for realistic rainfall postprocessing over East Africa.

---

# Future Direction: Toward Transferable Atmospheric Embeddings

Current cGAN systems mainly learn:

* model-specific bias correction.

Future approaches may instead learn:

* generalized atmospheric representations,
* latent weather embeddings,
* and transferable atmospheric states

using large reanalysis archives such as ERA5.

These embeddings could potentially support:

* analogue search,
* probabilistic forecasting,
* extreme-event characterization,
* forecast verification,
* and impact-based risk assessment

across multiple forecast systems such as:

* IFS,
* GEFS,
* AIFS,
* and regional ensemble systems.

This represents a shift from:

* “forecast-model correction”
  toward
* “general atmospheric representation learning.”

