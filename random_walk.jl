using Plots

V = [0, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0]
V_true = Array((0:6))/6
p_right = 0.5

# State data type
# struct State
#     x::Int
# end

# State Space
# S = [[State(x) for x in 1:7]...]
# Action Space
# A = [LEFT, RIGHT]
# const Movements = Dict(
#                         "LEFT" => State(-1),
#                         "RIGHT" => State(1)
#                         )
# Overload addition operator
# Base.:+(s1::State, s2::State) = State(s1.x + s2.x)

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
    s_ = 4
    state_trajectory = [s_]

    while true
        s = s_
        if rand() < p_right
            s_ += 1
        else
            s_ += -1
        end
        append!(state_trajectory, s_)

        if s_ == 6
            returns = 1.0
            break
        elseif s_ == 0
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
    fig = plot(legend=:topleft, xlabel="State", ylabel="Value")
    episodes = [0, 1, 10, 100]
    current_values = V

    for i in 0:100
        if i in episodes
            plot!(fig,["A", "B", "C", "D", "E"], current_values[2:6], label=string(i)*" episodes")
        end
        current_values = temporal_difference(current_values)
    end

    plot!(fig, ["A", "B", "C", "D", "E"], V_true[2:6], label="true values")
    savefig("Example_6_2_left.png")

end

estimate_values()






