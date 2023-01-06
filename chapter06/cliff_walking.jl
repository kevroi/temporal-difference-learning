using Plots

# world height
HEIGHT = 4
# world width
WIDTH = 12
# probability for exploration
EPSILON = 0.1
# step size
STEP_SIZE = 0.5
# gamma for Q-Learning and Expected Sarsa
DISCOUNT_FACTOR = 1
# all possible actions
ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]

# initial state action pair values
START = [1, 1]
GOAL = [12, 1]

function step(state, action)
    i, j = state
    if action == ACTION_UP
        state_ = [i, min(j+1, HEIGHT)]
    elseif action == ACTION_DOWN
        state_ = [i, max(j-1, 1)]
    elseif action == ACTION_RIGHT
        state_ = [min(i+1, WIDTH), j]
    elseif action == ACTION_LEFT
        state_ = [max(i-1, 1), j]
    end

    reward = -1.0

    if (action==ACTION_DOWN && j==2 && 2<=i<=11) || (action==ACTION_RIGHT && state==START)
        reward = -100.0
        state_ = START
    end

    return state_, reward
end


function choose_action(state, q_value)
    if rand() < EPSILON
        return rand(ACTIONS)
    else
        values_ = q_value[state[1], state[2], :]
        # select highest value action, break ties randomly
        return rand([action_ for (action_, value_) in enumerate(values_) if value_ == maximum(values_)])
    end
end


function sarsa(q_value, expected=false, step_size=STEP_SIZE)
    state = START
    action = choose_action(state, q_value)
    cum_reward = 0.0

    while state != GOAL
        state_, reward = step(state, action)
        action_ = choose_action(state_, q_value)
        cum_reward += reward

        if !expected
            # SARSA target
            target = q_value[state_[1], state_[2], action_]
        else
            target = 0.0
            q_next = q_value[state_[1], state_[2], :]
            best_actions = findall(q_next .== maximum(q_next))
            # Work out the expectation for ESARSA target
            for a in ACTIONS
                if a in best_actions
                    target += ((1.0-EPSILON)/length(best_actions) + EPSILON/length(ACTIONS)) * q_value[state_[1], state_[2], a]
                else
                    target += EPSILON/length(ACTIONS) * q_value[state_[1], state_[2], a]
                end
            end
        end
        target *= DISCOUNT_FACTOR
        # TD update
        q_value[state[1], state[2], action] += step_size*(reward + target - q_value[state[1], state[2], action])
        state = state_
        action = action_
    end
    return cum_reward
end


function q_learning(q_value, step_size=STEP_SIZE)
    state = START
    cum_reward = 0.0

    while state != GOAL
        action = choose_action(state, q_value)
        state_, reward = step(state, action)
        cum_reward += reward

        # Q Learning update
        q_value[state[1], state[2], action] += step_size*(reward + DISCOUNT_FACTOR*maximum(q_value[state_[1], state_[2], :]) - q_value[state[1], state[2], action])
        state = state_
    end
    return cum_reward
end


function plot_optimal_policy(q_value, method)
    fig_top = plot(legend=false, reuse=false)
    delta=0.3
    for j in 1:HEIGHT
        for i in 1:WIDTH
            if [i,j] == GOAL
                annotate!(i, j, "G")
                continue
            end
            _, bestAction = findmax(q_value[i, j, :])
            if bestAction == ACTION_UP
                plot!(fig_top, [i, i], [j-delta, j+delta],arrow=true,color=:blue,linewidth=2)
            elseif bestAction == ACTION_DOWN
                plot!(fig_top, [i, i], [j+delta, j-delta],arrow=true,color=:blue,linewidth=2)
            elseif bestAction == ACTION_LEFT
                plot!(fig_top, [i+delta, i-delta], [j, j],arrow=true,color=:blue,linewidth=2)
            elseif bestAction == ACTION_RIGHT
                plot!(fig_top, [i-delta, i+delta], [j, j],arrow=true,color=:blue,linewidth=2)
            end
        end
    end
    savefig("../plots/Ex_6_5_top_$method.png")
end


function example_6_6()
    episodes = 500
    runs = 50
    local q_sarsa, q_qlearning

    cum_rewards_sarsa = zeros(episodes)
    cum_rewards_qlearning = zeros(episodes)

    for run in 1:runs
        q_sarsa = zeros(WIDTH, HEIGHT, length(ACTIONS))
        q_qlearning = zeros(WIDTH, HEIGHT, length(ACTIONS))

        for e in 1:episodes
            cum_rewards_sarsa[e] = sarsa(q_sarsa)
            cum_rewards_qlearning[e] = q_learning(q_qlearning)
        end
    end
    cum_rewards_sarsa /= runs
    cum_rewards_qlearning /= runs

    fig_bottom = plot(legend=:bottomright, reuse=false, ylim=(-10,0),
                        xlabel="Episodes", ylabel="Sum of rewards during episode")
    plot!(cum_rewards_sarsa, label="Sarsa")
    plot!(cum_rewards_qlearning, label="Q-learning")
    savefig("../plots/Ex_6_6_bottom.png")

    plot_optimal_policy(q_sarsa, "Sarsa")
    plot_optimal_policy(q_qlearning, "Q-learning")
end


function figure_6_3()
    stepsizes = Array(0.1:0.05:1)
    episodes = 10000
    episodes_interim = 100
    runs = 1000
    runs_asymp = 20

    ASYMP_SARSA = 1
    ASYMP_ESARSA = 2
    ASYMP_QL = 3
    INTERIM_SARSA = 4
    INTERIM_ESARSA = 5
    INTERIM_QL = 6
    methods = 1:6

    performances = zeros(length(methods), length(stepsizes))

    for run in 1:runs
        print(run, "\n")
        for (i, step_size) in enumerate(stepsizes)
            q_sarsa = zeros(WIDTH, HEIGHT, length(ACTIONS))
            q_esarsa = zeros(WIDTH, HEIGHT, length(ACTIONS))
            q_qlearning = zeros(WIDTH, HEIGHT, length(ACTIONS))

            if run <= runs_asymp

                for e in 1:episodes
                    cum_rewards_sarsa = sarsa(q_sarsa, false, step_size)
                    cum_rewards_esarsa = sarsa(q_esarsa, true, step_size)
                    cum_rewards_qlearning = q_learning(q_qlearning, step_size)

                    performances[ASYMP_SARSA, i] += cum_rewards_sarsa
                    performances[ASYMP_ESARSA, i] += cum_rewards_esarsa
                    performances[ASYMP_QL, i] += cum_rewards_qlearning

                    if e <= episodes_interim
                        performances[INTERIM_SARSA, i] += cum_rewards_sarsa
                        performances[INTERIM_ESARSA, i] += cum_rewards_esarsa
                        performances[INTERIM_QL, i] += cum_rewards_qlearning
                    end

                end
            else
                for e in 1:episodes_interim
                    cum_rewards_sarsa = sarsa(q_sarsa, false, step_size)
                    cum_rewards_esarsa = sarsa(q_esarsa, true, step_size)
                    cum_rewards_qlearning = q_learning(q_qlearning, step_size)

                    performances[INTERIM_SARSA, i] += cum_rewards_sarsa
                    performances[INTERIM_ESARSA, i] += cum_rewards_esarsa
                    performances[INTERIM_QL, i] += cum_rewards_qlearning
                end
            end
        end
    end

    performances[begin:3,:] /= episodes*runs_asymp # 10000 ep 10 runs
    performances[4:end, :] /= 100*runs # 100 ep 100 runs

    labels = ["Asymptotic Sarsa",
                "Asymptotic Expected Sarsa",
                "Asymptotic Q-learning",
                "Interim Sarsa",
                "Interim Expected Sarsa",
                "Interim Q-learning"]

    plot(legend=false, reuse=false, ylim=(-140.0, 0.0),
            xlabel="Step Size", ylabel="Sum of rewards per episode")

    for (i, label) in enumerate(labels)
        if i > 3
            style=:dot
            if i == INTERIM_SARSA
                plot!(stepsizes, performances[i, :], line=style, color=:blue, marker=:dtriangle, label=label)
            elseif i == INTERIM_ESARSA
                plot!(stepsizes, performances[i, :], line=style, color=:red, marker=:xcross, label=label)
            elseif i == INTERIM_QL
                plot!(stepsizes, performances[i, :], line=style, color=:black, marker=:rect, label=label)
            end
        else
            if i == ASYMP_SARSA
                plot!(stepsizes, performances[i, :], color=:blue, marker=:dtriangle, label=label)
            elseif i == ASYMP_ESARSA
                plot!(stepsizes, performances[i, :], color=:red, marker=:xcross, label=label)
            elseif i == ASYMP_QL
                plot!(stepsizes, performances[i, :], color=:black, marker=:rect, label=label)
            end
            
        end
    end
    annotate!(0.8, -60, text("Sarsa", :blue, :right, 10))
    annotate!(1.0, -35, text("Expected Sarsa", :red, :right, 10))
    annotate!(0.225, -65, text("Q learning", :black, :right, 10))
    savefig("../plots/Fig_6_3.png")
end
        






# @profview example_6_6()
# example_6_6()
figure_6_3()





                