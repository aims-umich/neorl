#Remove unnecessary packages, install neorl, then test neorl quickly
!pip uninstall -y xarray arviz pymc3 tensorflow-probability pyerfa pyarrow kapre jax jaxlib datascience coveralls astropy albumentations
!pip install neorl
!neorl