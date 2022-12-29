using Plots

V = [0, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0]
V_true = Array((0:6))/6
p_right = 0.5

function temporal_difference(values, step_size=0.1)
    s_ = 4
    state_trajectory = [s_]
    reward_history = [0]

    while true
        s = s_
        if rand() < p_right
            s_ += 1
        else
            s_ += -1
        end
        r = 0
        append!(state_trajectory, s_)

        # TD Update
        values[s] += step_size*(r + values[s_] - values[s])

        if s_ == 7 || s_ == 1
            break
        end

        append!(reward_history, r)
    end

    # return state_trajectory, reward_history
    return values
end


function monte_carlo(values, step_size=0.1)
    s = 4
    state_trajectory = [s]
    local returns

    while true
        if rand() < p_right
            s += 1
        else
            s += -1
        end
        append!(state_trajectory, s)

        if s == 7
            returns = 1.0
            break
        elseif s == 1
            returns = 0.0
            break
        end
    end

    for next_state in state_trajectory[begin:end-1]
        # MC Update
        values[next_state] += step_size*(returns - values[next_state])
    end

    # return state_trajectory, fill(returns,length(state_trajectory)-1)
    return values

end


function estimate_values()
    fig_l = plot(legend=:topleft, xlabel="State", ylabel="Value")
    episodes = [0, 1, 10, 100]
    current_values = V

    for i in 0:100
        if i in episodes
            plot!(fig_l,["A", "B", "C", "D", "E"], current_values[2:6], label=string(i)*" episodes")
        end
        current_values = temporal_difference(current_values)
    end

    plot!(fig_l, ["A", "B", "C", "D", "E"], V_true[2:6], label="true values")
    savefig("Example_6_2_left.png")

end


function rms_error()
    fig_r = plot(legend=:topright, xlabel="Walks/Episode", ylabel="RMS")
    td_stepsizes = [0.15, 0.1, 0.05]
    mc_stepsizes = [0.01, 0.02, 0.03, 0.04]

    runs = 100
    episodes = 100

    for (i, stepsize) in enumerate(td_stepsizes)
        total_errors = zeros(episodes)

        for run in 1:runs
            errors = []
            current_values = deepcopy(V)
            
            for e in 1:episodes
                append!(errors, sqrt(sum((V_true-current_values).^2)/5))
                current_values = temporal_difference(current_values, stepsize)
            end
            total_errors += errors
        end
        total_errors /= runs
        plot!(fig_r, total_errors, label="TD $stepsize", legendtitle="step size")
    end

    for (i, stepsize) in enumerate(mc_stepsizes)
        total_errors = zeros(episodes)

        for run in 1:runs
            errors = []
            current_values = deepcopy(V)
            
            for e in 1:episodes
                append!(errors, sqrt(sum((V_true-current_values).^2)/5))
                current_values = monte_carlo(current_values, stepsize)
            end
            total_errors += errors
        end
        total_errors /= runs
        plot!(fig_r, total_errors, label="MC $stepsize", line=(:dash), legendtitle="step size")
    end
    savefig("Example_6_2_right.png")
end

# estimate_values()
rms_error()