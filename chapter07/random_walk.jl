using Plots

# State Space
N_STATES = 19
STATES = Array(1:N_STATES)
STATE_START = 10
STATES_TERMINAL = [0, 19]
# GAMMA
DISCOUNT_FACTOR = 1
TRUE_VALUE = Array(-1:0.1:1)
TRUE_VALUE[begin] = 0.0
TRUE_VALUE[end] = 0.0

function temporal_difference(value, n, step_size)
    s = STATE_START

    state_trajectory = [s]
    reward_history = [0]

    t = 0

end
