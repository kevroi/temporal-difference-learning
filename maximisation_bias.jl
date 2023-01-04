using Plots
using Distributions


# State Space
STATE_A = 1
STATE_B = 2
STATE_TERMINAL = 3
STATE_START = STATE_A

# Action Space
ACTION_A_RIGHT = 1
ACTION_A_LEFT = 2
ACTIONS_B = 1:10 # assume 10 actions can be taken from state B
STATE_ACTIONS = [[ACTION_A_RIGHT, ACTION_A_LEFT], ACTIONS_B]
TRANSITION = [[STATE_TERMINAL, STATE_B], fill(STATE_TERMINAL,length(ACTIONS_B))]

INITIAL_Q = [zeros(2), zeros(length(ACTIONS_B)), zeros(1)] # [Q(STATE_A), Q(STATE_B), Q(STATE_TERMINAL)]


EPSILON = 0.1 # probability for exploration
STEP_SIZE = 0.1
DISCOUNT_FACTOR = 1.0 # gamma


function choose_action(state, q_value)
    if rand() < EPSILON
        return rand(STATE_ACTIONS[state])
    else
        values_ = q_value[state]
        return rand([action_ for (action_, value_) in enumerate(values_) if value_ == maximum(values_)])
    end
end

# reward function
function r(state)
    if state == STATE_A
        return 0
    else
        d = Normal(-0.1, 1.0)
        return rand(d)
    end
end


# regular q learning
function qlearning(q_value)
    state = STATE_START
    left_count = 0

    while state != STATE_TERMINAL
        action = choose_action(state, q_value)
        if state == STATE_A && action == ACTION_A_LEFT
            left_count += 1
        end
        reward = r(state)
        state_ = TRANSITION[state][action]
        
        target = maximum(q_value[state_])

        # Q learning update
        q_value[state][action] += STEP_SIZE*(reward + DISCOUNT_FACTOR*target - q_value[state][action])
        state = state_
    end
    return left_count
end


# double q learning
function qlearning(q1, q2)
    state=STATE_START
    left_count = 0

    while state != STATE_TERMINAL
        local active_q, target_q

        action = choose_action(state, [item1+item2 for (item1, item2) in zip(q1,q2)])
        if state == STATE_A && action == ACTION_A_LEFT
            left_count += 1
        end
        reward = r(state)
        state_ = TRANSITION[state][action]

        # flip coin to decide which q to use
        if rand() < 0.5
            active_q = q1
            target_q = q2
        else
            active_q = q2
            target_q = q1
        end
        best_action = rand([action_ for (action_, value_) in enumerate(active_q[state_]) if value_ == maximum(active_q[state_])])
        target = target_q[state_][best_action]

        # Q learning update
        active_q[state][action] += STEP_SIZE*(reward + DISCOUNT_FACTOR*target - active_q[state][action])
        state = state_
    end
    return left_count
end


function figure_6_5()
    episodes = 300
    runs = 1000 # increase this for smoother curves
    left_counts_q = zeros(runs, episodes)
    left_counts_doubleq = zeros(runs, episodes)

    for run in 1:runs
        q = deepcopy(INITIAL_Q)
        q1 = deepcopy(INITIAL_Q)
        q2 = deepcopy(INITIAL_Q)

        for e in 1:episodes
            left_counts_q[run, e] = qlearning(q)
            left_counts_doubleq[run, e] = qlearning(q1, q2)
        end
    end
    avg_lefts_q = mean(left_counts_q, dims=1)*100
    avg_lefts_doubleq = mean(left_counts_doubleq, dims=1)*100

    fig_6_5 = plot(xlabel="Episodes", ylabel="% left actions", dpi=300,
                    xlim=(0,300), ylim=(0,100), legend=false, reuse=false)
    plot!(avg_lefts_q[:], color=:red)
    plot!(avg_lefts_doubleq[:], color=:green)
    annotate!(100, 50, text("Q learning", :red, :left, 10))
    annotate!(30, 25, text("Double Q learning", :green, :left, 10))
    savefig("Fig_6_5.png")
end

figure_6_5()


