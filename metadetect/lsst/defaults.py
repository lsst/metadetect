from copy import deepcopy


DEFAULT_WEIGHT_FWHM = 2.0
DEFAULT_FWHM_SMOOTH = 0

DEFAULT_WEIGHT_CONFIG = {
    'fwhm': DEFAULT_WEIGHT_FWHM,
    'fwhm_smooth': DEFAULT_FWHM_SMOOTH,
}

DEFAULT_STAMP_SIZE = 49

# threshold for detection
DEFAULT_THRESH = 5.0

# whether to find and subtract the sky, happens before metacal
DEFAULT_SUBTRACT_SKY = False

# Control of the metacal process
# not currently used for new metacal_exposures code that always
DEFAULT_METACAL_CONFIG = {
    "use_noise_image": True,
    "psf": "fitgauss",
}

# detection config, this may expand
DEFAULT_DETECT_CONFIG = {
    'thresh': DEFAULT_THRESH,
}

# the weight subconfig and the stamp_size defaults we be filled in
# programatically based on the measurement_type
DEFAULT_MDET_CONFIG = {
    'subtract_sky': DEFAULT_SUBTRACT_SKY,
    'detect': deepcopy(DEFAULT_DETECT_CONFIG),
    'metacal': deepcopy(DEFAULT_METACAL_CONFIG),
    'weight': deepcopy(DEFAULT_WEIGHT_CONFIG),
    'stamp_size': DEFAULT_STAMP_SIZE,
}
