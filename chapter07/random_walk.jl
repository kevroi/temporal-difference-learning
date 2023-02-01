using Plots

# State Space
N_STATES = 19
STATES = Array(1:N_STATES)
STATE_START = 10
STATES_TERMINAL = [0, 20]
p_right = 0.5
# GAMMA
DISCOUNT_FACTOR = 1
TRUE_VALUE = Array(-1:0.1:1)
TRUE_VALUE[begin] = 0.0
TRUE_VALUE[end] = 0.0

function temporal_difference(values, n, step_size)
    s = STATE_START
    local s_

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

            if s_ == 0
                reward = -1
            elseif s_ == 20
                reward = 1
            else
                reward = 0
            end

            append!(state_trajectory, s_)
            append!(reward_history, reward)

            if s_ in STATES_TERMINAL
                T = t
            end
        end

        update_time = t - n

        if update_time >= 1
            local states_to_update
            returns = 0.0

            # upto = trunc(Int, min(T,update_time+n))
            upto = min(T, Int(update_time+n))
            # for t in trunc(Int, update_time):upto
            for t in (update_time+1):upto
                t = Int(t)
                # print(t)
                returns += DISCOUNT_FACTOR^(t-update_time-1)*reward_history[t]
            end

            if update_time+n<=T
                returns += (DISCOUNT_FACTOR^n)*values[state_trajectory[update_time+n]]
            end
            state_to_update = state_trajectory[update_time]

            if !(state_to_update in STATES_TERMINAL)
                values[state_to_update] += step_size*(returns - values[state_to_update])
            end
        end

        if update_time == T - 1
            break
        end
        s = s_
    end
    return values
end


function figure_7_2()
    ns = exp2.(0:9)
    ns = Int.(ns)
    step_sizes = 0:0.1:1

    episodes = 10
    runs=100

    errors = zeros(length(ns), length(step_sizes))

    for run in 1:runs
        for (i, n) in enumerate(ns)
            for (j, step_size) in enumerate(step_sizes)
                value = zeros(N_STATES+2)
                for e in 1:episodes
                    value = temporal_difference(value, n, step_size)
                    errors[i, j] += sqrt(sum((value-TRUE_VALUE).^2)/N_STATES)
                end
            end
        end
    end

    errors /= episodes * runs

    fig_7_2 = plot(legend=true, ylim=(0.25, 0.55),
                    xlabel="step_size", ylabel="Empirical RMS Error, averaged over states")
    for i in 1:length(ns)
        plot!(step_sizes, errors[i, :], label="n = $(ns[i])")
    end
    savefig("../plots/Fig_7_2.png")
end

figure_7_2()
    



