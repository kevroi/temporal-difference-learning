using Plots

# Dimensions of GridWorld
HEIGHT = 7
WIDTH = 10
WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
# Action Space
ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]
# Exploration Probability
EPSILON = 0.1
STEP_SIZE = 0.5
# Reward per timestep
REWARD = -1.0
START = [3, 1]
GOAL = [3, 7]


function step(state, action)
    i, j = state
    if action == ACTION_UP
        return [max(i - 1 - WIND[j], 1), j]
    elseif action == ACTION_DOWN
        return [max(min(i + 1 - WIND[j], HEIGHT), 1), j]
    elseif action == ACTION_LEFT
        return [max(i - WIND[j], 1), max(j - 1, 1)]
    elseif action == ACTION_RIGHT
        return [max(i - WIND[j], 1), min(j + 1, WIDTH)]
    end
end


function episode(q_value)
    time = 0
    state = START

    if rand() < EPSILON
        action = rand(ACTIONS)
    else
        values_ = q_value[state[1], state[2], :]
        action = rand([action_ for (action_, value_) in enumerate(values_) if value_ == maximum(values_)])
    end

    while state != GOAL
        state_ = step(state, action)
        if rand() < EPSILON
            action_ = rand(ACTIONS)
        else
            values_ = q_value[state_[1], state_[2], :]
            action_ = rand([action_ for (action_, value_) in enumerate(values_) if value_ == maximum(values_)])

        end
        q_value[state[1], state[2], action] += STEP_SIZE*(REWARD + q_value[state_[1], state_[2], action_] - q_value[state[1], state[2], action])
        state = state_
        action = action_
        time += 1
    end
    return time
end


function figure_6_3()
    q_value = zeros(HEIGHT, WIDTH, 4)
    episode_limit = 500
    steps = []
    ep = 0

    while ep<episode_limit
        append!(steps, episode(q_value))
        ep += 1
    end
    
    steps = cumsum(steps)
    fig_6_3 = plot(xlabel="Time Steps", ylabel="Episodes", legend = false)
    plot!(fig_6_3, steps, Array(1:length(steps)))
    savefig("Fig_6_3.png")
end

figure_6_3()