# To thouroughly test the scaling of the mean embedding
# through neural networks, we are combining these different parameters
# that have an effect on scaling into multiple scenarios.

# Numbers of agents:
AGENT_NUMBERS = [5, 10, 50, 100, 125, 150, 200]

# Mean-Embedding Configurations:
MEAN_EMBEDDING_ARCHS = []

# Different Kinematic Scenarios:
KINEMATICS = ["single", "double"]

# Observations Models:
OBSERVATION_MODELS = [
    "local_basic",
    "local_extended",
    "global_basic",
    "global_extended",
    "classic",
]
