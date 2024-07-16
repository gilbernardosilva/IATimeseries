# DATASET
DATASET_RATIO = 0.3

# RANDOM FOREST Variables
NUM_ESTIMATORS = 100
RANDOM_STATE = 42
LAG_FEATURES = 10

# LSTM Variables
SEQ_LENGTH = 10

# Array for particle level regressors
regressor_particles = [
    "Humidity",
    "Temperature",
    "door_closed_on_arrival",
    "windows_closed_on_arrival",
    "opened_windows_end_of_class",
    "persons_in_classroom",
]

# Array for CO2 level regressors
regressor_co2 = [
    "Humidity",
    "Temperature",
    "door_closed_on_arrival",
    "windows_closed_on_arrival",
    "opened_windows_end_of_class",
    "ac_on_during_class",
    "persons_in_classroom",
]

# Variables to be analised
TARGETS = ["Particules 1", "Particules 2.5", "Particules 10", "CO2"]

# Regressors Dictionary
REGRESSORS_DICT = {
    "Particules 1": regressor_particles,
    "Particules 2.5": regressor_particles,
    "Particules 10": regressor_particles,
    "CO2": regressor_co2,
}
