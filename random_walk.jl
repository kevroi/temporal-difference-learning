using Plots

V = [0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0]
V_true = Array((0:6))/6
p_right = 0.5

function temporal_difference(values, step_size=0.1, batch=false)
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

        # TD Update at the end of each episode if we aren't batch updating
        if  !batch
            values[s] += step_size*(r + values[s_] - values[s])
        end

        if s_ == 7 || s_ == 1
            break
        end

        append!(reward_history, r)
    end

    # return state_trajectory, reward_history
    return values, state_trajectory, reward_history
end


function monte_carlo(values, step_size=0.1, batch=false)
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

    # MC Update at the end of each episode if we aren't batch updating
    if !batch
        for next_state in state_trajectory[begin:end-1]
            values[next_state] += step_size*(returns - values[next_state])
        end
    end

    # return state_trajectory, fill(returns,length(state_trajectory)-1)
    return values, state_trajectory, fill(returns,length(state_trajectory)-1)

end


function estimate_values()
    fig_l = plot(legend=:topleft, xlabel="State", ylabel="Value")
    episodes = [0, 1, 10, 100]
    current_values = deepcopy(V)

    for i in 0:100
        if i in episodes
            plot!(fig_l,["A", "B", "C", "D", "E"], current_values[2:6], label=string(i)*" episodes")
        end
        current_values, _, _ = temporal_difference(current_values)
    end

    plot!(fig_l, ["A", "B", "C", "D", "E"], V_true[2:6], label="true values")
    savefig("Example_6_2_left.png")

end


function rms_error()
    fig_r = plot(legend=:topright, xlabel="Episodes", ylabel="Empirical RMS Error, averaged over states")
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
                current_values, _, _ = temporal_difference(current_values, stepsize)
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
                current_values, _, _ = monte_carlo(current_values, stepsize)
            end
            total_errors += errors
        end
        total_errors /= runs
        plot!(fig_r, total_errors, label="MC $stepsize", line=(:dash), legendtitle="step size")
    end
    savefig("Example_6_2_right.png")
end

function batch_updating(method, episodes, stepsize=0.001)
    runs = 100
    total_errors = zeros(episodes) 

    for run in 1:runs
        current_values = deepcopy(V)
        current_values[2:6] .= -1.0
        errors = Float64[]
        trajectories = Int64[]
        reward_histories = Int64[]

        for e in 1:episodes
            if method=="TD"
                current_values, trajectory, rewards = temporal_difference(current_values, 0.1, true)
            else
                current_values, trajectory, rewards = monte_carlo(current_values, 0.1, true)
            end
            append!(trajectories, trajectory)
            append!(reward_histories, rewards)

            while true
                updates = zeros(7)
                for (t, r) in zip(trajectories, reward_histories)
                    for i in 1:(length(t)-1)
                        if method=="TD"
                            updates[t[i]] += r[i] + current_values[t[i+1]] - current_values[t[i]]
                        else
                            updates[t[i]] += r[i] - current_values[t[i]]
                        end
                    end
                end
                updates *= stepsize
                if sum(abs.(updates)) < 0.001
                    break
                end
                current_values += updates
            end
            append!(errors, sqrt(sum((current_values-V_true).^2)/5))
        end
        total_errors += errors
    end
    total_errors /= runs
    return total_errors
end


# estimate_values()
# rms_error()

# Plot Fig 6.2
fig_6_2 = plot(legend=:topright,
                xlabel="Episodes", ylabel="Empirical RMS Error, averaged over states")
episodes = 100
td_errors = batch_updating("TD", episodes)
mc_errors = batch_updating("MC", episodes)
plot!(fig_6_2, td_errors, label="TD")
plot!(fig_6_2, mc_errors, label="MC")
savefig("Fig_6_2.png")