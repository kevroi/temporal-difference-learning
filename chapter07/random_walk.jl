using Plots

# State Space
N_STATES = 19
STATES = Array(1:N_STATES)
STATE_START = 10
STATES_TERMINAL = [1, 19]
p_right = 0.5
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
    T = Inf

    while true
        t += 1

        if t < T
            if rand() < p_right
                s_ = s + 1
            else
                s_ = s - 1
            end

            if s_ == 1
                reward = -1
            elseif s_ == 19
                reward = 1
            else
                reward = 0
            end

            append!(state_trajectory, s_)
            append!(reward_history, reward)

            if s_ in STATES_TERMINAL
                T = time
            end
        end

        update_time = t - n

        if update_time >= 0
            local states_to_update
            returns = 0.0

            for t in update_time:min(T,update_time+n)
                returns += DISCOUNT_FACTOR^(t-update_time)*reward_history[t]
            end

            if update_time+n<=T
                returns += DISCOUNT_FACTOR^(n)*values[state_trajectory[update_time+n]]
            end
            state_to_update = state_trajectory[update_time]

            if !(state_to_update in STATES_TERMINAL)
                value[state_to_update] += step_size*(returns - value[state_to_update])
            end
        end

        if update_time == T - 1
            break
        end
        state = state_




end
