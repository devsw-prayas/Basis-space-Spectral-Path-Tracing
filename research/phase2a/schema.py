BASE_COLUMNS = [
    "configId",
    "spectrumId",
    "l2",
    "nrmse",
    "maxError",
    "xyzDelta",
    "perceptualDeltaE",
]

DELTA_EXTRA_COLUMNS = [
    "energyRetention",
    "amplitudeAccuracy",
    "peakShiftNm",
    "sideLobeEnergy",
]

PHASE_A_COLUMNS = BASE_COLUMNS
PHASE_B_COLUMNS = BASE_COLUMNS + DELTA_EXTRA_COLUMNS
PHASE_C_COLUMNS = BASE_COLUMNS
