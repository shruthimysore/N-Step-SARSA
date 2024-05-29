import numpy as np
from math import pi, exp, cos
import random
import matplotlib.pyplot as plt
import math

class Dynamics:
    episodes_per_run = 500
    alpha = 0.1
    epsilon = 0.9
    n_steps = 8
    order = 2
    decay_epsilon = 0.01
    gamma = 0.9
    max_reward = 500
    actions = ["au","ad","al","ar"]
    state_variables = [[[row, col] for col in range(5)] for row in range(5)]
    obstacle_states = ["22","32"]
    water_states = ["42"]
    goal_state = ["44"]
    optimal_policy = np.array([['ar', 'ar', 'ar', 'ad', 'ad'],
                ['ar', 'ar', 'ar', 'ad', 'ad'],
                ['au', 'au', 'NA', 'ad', 'ad'],
                ['au', 'au', 'NA', 'ad', 'ad'],
                ['au', 'au', 'ar', 'ar', 'G']])
    optimal_value = np.array([[4.0187, 4.5548, 5.1575, 5.8336, 6.4553 ],
                        [4.3716, 5.0324, 5.8013, 6.6473, 7.3907 ],
                        [3.8672, 4.39,  0.0000, 7.5769, 8.4637 ],
                        [3.4182, 3.8319,  0.0000, 8.5738, 9.6946 ],
                        [2.9977, 2.9309, 6.0733, 9.6946, 0.0000]])

    transition_probabilities = {'00': {'au': {'01': 0.05, '00': 0.95},
        'ad': {'10': 0.8, '01': 0.05, '00': 0.15},
        'ar': {'01': 0.8, '10': 0.05, '00': 0.15},
        'al': {'10': 0.05, '00': 0.95}},

    '01': {'au': {'02': 0.05, '00': 0.05, '01': 0.9},
        'ad': {'11': 0.8, '02': 0.05, '00': 0.05, '01': 0.1},
        'ar': {'02': 0.8, '11': 0.05, '01': 0.15},
        'al': {'00': 0.8, '11': 0.05, '01': 0.15}},

    '02': {'au': {'03': 0.05, '01': 0.05, '02': 0.9},
        'ad': {'12': 0.8, '03': 0.05, '01': 0.05, '02': 0.1},
        'ar': {'03': 0.8, '12': 0.05, '02': 0.15},
        'al': {'01': 0.8, '12': 0.05, '02': 0.15}},

    '03': {'au': {'04': 0.05, '02': 0.05, '03': 0.9},
        'ad': {'13': 0.8, '04': 0.05, '02': 0.05, '03': 0.1},
        'ar': {'04': 0.8, '13': 0.05, '03': 0.15},
        'al': {'02': 0.8, '13': 0.05, '03': 0.15}},

        '04': {'au': {'03': 0.05, '04': 0.95},
        'ad': {'14': 0.8, '03': 0.05, '04': 0.15,},
        'ar': {'14': 0.05, '04': 0.95},
        'al': {'03': 0.8, '14': 0.05, '04': 0.15}},

        '10': {'au': {'00': 0.8, '11': 0.05, '10': 0.15},
        'ad': {'20': 0.8, '11': 0.05, '10': 0.15},
        'ar': {'11': 0.8, '00': 0.05, '20': 0.05, '10': 0.1},
        'al': {'10': 0.9, '00': 0.05, '20': 0.05}},

        '11': {'au': {'01': 0.8, '12': 0.05, '10': 0.05, '11': 0.1},
        'ad': {'21': 0.8, '12': 0.05, '10': 0.05, '11': 0.1},
        'ar': {'12': 0.8, '01': 0.05, '21': 0.05, '11': 0.1},
        'al': {'10': 0.8, '01': 0.05, '21': 0.05, '11': 0.1}},

        '12': {'au': {'02': 0.8, '13': 0.05, '11': 0.05, '12': 0.1},
        'ad': {'13': 0.05, '11': 0.05,'12': 0.9},
        'ar': {'13': 0.8, '02': 0.05, '12': 0.15},
        'al': {'11': 0.8, '02': 0.05, '12': 0.15}},


        '13': {'au': {'03': 0.8, '14': 0.05, '12': 0.05, '13': 0.1},
        'ad': {'23': 0.8, '14': 0.05, '12': 0.05, '13': 0.1},
        'ar': {'14': 0.8, '03': 0.05, '23': 0.05, '13': 0.1},
        'al': {'12': 0.8, '03': 0.05, '23': 0.05, '13': 0.1}},

        '14': {'au': {'04': 0.8, '13': 0.05, '14': 0.15},
        'ad': {'24': 0.8, '13': 0.05, '14': 0.15},
        'ar': {'04': 0.05, '24': 0.05, '14': 0.9},
        'al': {'13': 0.8, '04': 0.05, '24': 0.05, '14': 0.1}},

        '20': {'au': {'10': 0.8, '21': 0.05, '20': 0.15},
        'ad': {'30': 0.8, '21': 0.05, '20': 0.15},
        'ar': {'21': 0.8, '10': 0.05, '30': 0.05, '20': 0.1},
        'al': {'10': 0.05, '30': 0.05, '20': 0.9}},

        '21': {'au': {'11': 0.8, '20': 0.05, '21': 0.15},
        'ad': {'31': 0.8, '20': 0.05, '21': 0.15},
        'ar': {'11': 0.05, '31': 0.05, '21': 0.9},
        'al': {'20': 0.8, '11': 0.05, '31': 0.05, '21': 0.1}},

        '22': {'au': {'12': 0.8, '23': 0.05, '21': 0.05, '22': 0.1},
        'ad': {'23': 0.05, '21': 0.05, '22': 0.9},
        'ar': {'23': 0.8, '12': 0.05, '22': 0.15},
        'al': {'21': 0.8, '12': 0.05, '22': 0.15}},

        '23': {'au': {'13': 0.8, '24': 0.05, '23': 0.15},
        'ad': {'33': 0.8, '24': 0.05, '23': 0.15},
        'ar': {'24': 0.8, '13': 0.05, '33': 0.05, '23': 0.1},
        'al': {'13': 0.05, '33': 0.05, '23': 0.9}},

        '24': {'au': {'14': 0.8, '23': 0.05, '24': 0.15},
        'ad': {'34': 0.8, '23': 0.05, '24': 0.15},
        'ar': {'14': 0.05, '34': 0.05, '24': 0.9},
        'al': {'23': 0.8, '14': 0.05, '34': 0.05, '24': 0.1}},

        '30': {'au': {'20': 0.8, '31': 0.05, '30': 0.15},
        'ad': {'40': 0.8, '31': 0.05, '30': 0.15},
        'ar': {'31': 0.8, '20': 0.05, '40': 0.05, '30': 0.1},
        'al': {'20': 0.05, '40': 0.05, '30': 0.9}},

        '31': {'au': {'21': 0.8, '30': 0.05, '31': 0.15},
        'ad': {'41': 0.8, '30': 0.05, '31': 0.15},
        'ar': {'21': 0.05, '41': 0.05, '31': 0.9},
        'al': {'30': 0.8, '21': 0.05, '41': 0.05, '31': 0.1}},


        '32': {'au': {'33': 0.05, '31': 0.05, '32': 0.9},
        'ad': {'42': 0.8, '33': 0.05, '31': 0.05, '32': 0.1},
        'ar': {'33': 0.8, '42': 0.05, '32': 0.15},
        'al': {'31': 0.8, '42': 0.05, '32': 0.15}},

        '33': {'au': {'23': 0.8, '34': 0.05, '33': 0.15},
        'ad': {'43': 0.8, '34': 0.05, '33': 0.15},
        'ar': {'34': 0.8, '23': 0.05, '43': 0.05, '33': 0.1},
        'al': {'23': 0.05, '43': 0.05, '33': 0.9}},


        '34': {'au': {'24': 0.8, '33': 0.05, '34': 0.15},
        'ad': {'44': 0.8, '33': 0.05, '34': 0.15},
        'ar': {'24': 0.05, '44': 0.05, '34': 0.9},
        'al': {'33': 0.8, '24': 0.05, '44': 0.05, '34': 0.1}},

        '40': {'au': {'30': 0.8, '41': 0.05, '40': 0.15},
        'ad': {'41': 0.05, '40': 0.95},
        'ar': {'41': 0.8, '30': 0.05, '40': 0.15},
        'al': {'30': 0.05, '40': 0.95}},

        '41': {'au': {'31': 0.8, '42': 0.05, '40': 0.05, '41': 0.1},
        'ad': {'42': 0.05, '40': 0.05, '41': 0.9},
        'ar': {'42': 0.8, '31': 0.05, '41': 0.15},
        'al': {'40': 0.8, '31': 0.05, '41': 0.15}},

        '42': {'au': {'43': 0.05, '41': 0.05, '42': 0.9},
        'ad': {'43': 0.05, '41': 0.05, '42': 0.9},
        'ar': {'43': 0.8, '42': 0.2},
        'al': {'41': 0.8, '42': 0.2}},
        
        '43': {'au': {'33': 0.8, '44': 0.05, '42': 0.05, '43': 0.1},
        'ad': {'44': 0.05, '42': 0.05, '43': 0.9},
        'ar': {'44': 0.8, '33': 0.05, '43': 0.15},
        'al': {'42': 0.8, '33': 0.05, '43': 0.15}},

        '44': {'au': {'34': 0.8, '44': 0.15, '43': 0.05},
        'ad': {'43': 0.05, '44': 0.95},
        'ar': {'34': 0.05, '44': 0.95},
        'al': {'43': 0.8, '34': 0.05, '44': 0.15}}}
    
class GridWorld:
    def get_feature_combinations(order):
        combinations = []
        for x in range(order+1):
            for y in range(order+1):
                combinations.append([x,y])
        return combinations

    def get_weights(order):
        weights_list = []
        weight_dim = (order+1)**(2) 
        for action in Dynamics.actions:
            weights_list.append(np.zeros((1,weight_dim)))
        return weights_list

    def get_q_feature_value(norm_state, q_features):
        state_vector = np.array(list(norm_state))
        state_vector = np.array(list(norm_state)).reshape(len(norm_state),1)
        combinations_vector = np.array(q_features)
        dot_product = np.dot(combinations_vector, state_vector)
        feature_vector = np.cos(np.pi * dot_product)
        return np.transpose(feature_vector)

    def get_q_values(policy_weights, policy_features, norm_state):
        policy_list = []
        for i in range(len(Dynamics.actions)):
            weights = policy_weights[i]
            valSubstitutingInFeatures = np.dot(policy_features, norm_state)
            f_policy_features = np.cos(np.pi * valSubstitutingInFeatures)
            policy_list.append(np.dot(weights, f_policy_features))
        return policy_list

    def get_epsilon_greedy_policy(policy, sigma):
        policy_arr = np.array(policy)
        max_value = np.max(sigma * policy_arr)
        numerator_values = [math.exp(sigma * x - max_value) for x in policy_arr]
        max_value = np.max(sigma * policy_arr)
        denominator_values = [math.exp(sigma * x - max_value) for x in policy_arr]
        denominator = sum(denominator_values)
        action_probabilities = [numerator_value / denominator for numerator_value in numerator_values]
        selected_action = random.choices([0, 1, 2, 3], action_probabilities)[0]  # need to change depending on actions in environment
        selected_action_probability = action_probabilities[selected_action]
        return action_probabilities

    def get_q_value(q_weights, q_features, norm_state):
        valSubstitutingInFeatures = np.dot(q_features, norm_state)
        f_policy_features = np.cos(np.pi * valSubstitutingInFeatures)
        policy = np.dot(q_weights, f_policy_features)
        return policy
    
    def runEpisode(weights, features, epsilon, n_steps):
        state = Dynamics.state_variables[0][0]
        states_list = []
        actions_list = []
        rewards_list = []
        max_reward = Dynamics.max_reward
        timesteps = 0        
        reward = 0   
        total_rewards = 0
        isTerminated = False 
        q_outputs = GridWorld.get_q_values(weights, features, state)
        actionProb = GridWorld.get_epsilon_greedy_policy(q_outputs, epsilon)
        action = random.choices(Dynamics.actions, actionProb)[0] 
        states_list.append(state)
        actions_list.append(action)      
        while True:
            state = states_list[-1]
            action = actions_list[-1]   
            state_val = ''.join(map(str, state))  
            if state_val in Dynamics.goal_state: 
                return weights, total_rewards   
            available_next_states = list(Dynamics.transition_probabilities[state_val][action].keys())        
            next_state_probabilities = list(Dynamics.transition_probabilities[state_val][action].values())        
            next_state = np.random.choice(available_next_states, p=next_state_probabilities) 
            next_state_val = next_state
            r, c = [int(char) for char in next_state]        
            next_state = Dynamics.state_variables[r][c]
            if(next_state_val in Dynamics.water_states):
                reward = -10
                total_rewards += reward
            if(next_state_val in Dynamics.goal_state):
                reward = 10
                total_rewards += reward
            states_list.append(next_state)
            rewards_list.append(reward)
            state_val = next_state_val
            state = next_state
            if not isTerminated:
                q_outputs = GridWorld.get_q_values(weights, features, state)
                actionProb = GridWorld.get_epsilon_greedy_policy(q_outputs, epsilon)
                action = random.choices(Dynamics.actions, actionProb)[0]
                actions_list.append(action)
            else:
                max_reward = timesteps + 1
            tou = timesteps - n_steps + 1
            if tou >= 0:
                indices_range = range(tou + 1, min(tou + n_steps, max_reward))
                discounted_powers = [(Dynamics.gamma)**(i - tou - 1) for i in indices_range]
                discounted_rewards = [power * rewards_list[i] for power, i in zip(discounted_powers, indices_range)]
                discounted_return = sum(discounted_rewards)                
                if tou + n_steps < max_reward:
                    next_state = states_list[tou+n_steps]
                    next_action = actions_list[tou+n_steps]
                    next_actionIndex = Dynamics.actions.index(next_action)  
                    discount_factor = Dynamics.gamma ** n_steps
                    q_val = GridWorld.get_q_value(weights[next_actionIndex], features, next_state)  
                    discounted_return += discount_factor * q_val
                n_init_state = states_list[tou]
                n_init_action = actions_list[tou]
                n_init_actionIndex = Dynamics.actions.index(n_init_action)
                for i in range(len(Dynamics.actions)):
                    initial_q_value = GridWorld.get_q_value(weights[n_init_actionIndex], features, n_init_state)
                    error = discounted_return - initial_q_value
                    init_state_feature_value = GridWorld.get_q_feature_value(n_init_state, features)
                    weights[i] += float(Dynamics.alpha) * error * init_state_feature_value
            timesteps += 1
            
def main():
    steps_all_episodes = []
    avg_steps = []
    overall_actions_before_start = []
    for i in range(1):
        decay_epsilon = Dynamics.epsilon
        timesteps_list = []
        start_action = [0] * (Dynamics.episodes_per_run + 1)
        q_wei = GridWorld.get_weights(Dynamics.order)
        q_fea = GridWorld.get_feature_combinations(Dynamics.order)        
        for i in range(Dynamics.episodes_per_run):
            if (i+1) % 1000 == 0:
                decay_epsilon *= Dynamics.decay_epsilon
            q_wei, timesteps = GridWorld.runEpisode(q_wei, q_fea, decay_epsilon, Dynamics.n_steps)
            timesteps_list.append(timesteps)
            start_action[i+1] = start_action[i] + timesteps
            print(timesteps)
        steps_all_episodes.append(timesteps_list)
        overall_actions_before_start.append(start_action)
    avg_steps = np.mean(steps_all_episodes, axis=0)
    avg_step_list = np.mean(overall_actions_before_start, axis=0)
    
    episoderange = range(1, Dynamics.episodes_per_run+1)
    plt.plot(avg_step_list[1:], episoderange)
    plt.title(f'Episodic Semi Gradient n-step SARSA\n 687 - Grid World\n Learning Curve for steps = {Dynamics.n_steps}')
    x_label = "Rewards"
    y_label = "Episodes"
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.show()
    
    plt.plot(episoderange, avg_steps)
    plt.title(f'Episodic Semi Gradient n-step SARSA\n 687 - Grid World\n Rewards per episode for steps = {Dynamics.n_steps}')
    x_label = "Episodes"
    y_label = "Rewards"
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.show()   
        
        
if __name__ == "__main__":
    main()
