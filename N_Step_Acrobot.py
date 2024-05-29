import numpy as np
import matplotlib.pyplot as plt
import math as math
from math import  pi
import gym as gym
import random

class Model:
    actions = [0, 1, 2]
    episodes_per_run = 150
    alpha = 0.001
    epsilon = 0.005
    n_steps = 8
    order = 1
    decay_epsilon = 0.01
    gamma = 1
    max_reward = 500   

class CartPole:
    def get_feature_combinations(order):
        combinations = []
        for cos_theta1 in range(order+1):
            for sin_theta1 in range(order+1):
                for cos_theta2 in range(order+1):
                    for sin_theta2 in range(order+1):
                        for theta1 in range(order+1):
                            for theta2 in range(order+1):
                                combinations.append([cos_theta1,sin_theta1,cos_theta2,sin_theta2,theta1,theta2])
        return combinations

    def get_weights(order):
        weights_list = []
        weight_dim = (order+1)**(6) 
        for action in Model.actions:
            weights_list.append(np.zeros((1,weight_dim)))
        return weights_list

    def get_normalized_states(state):
        cos_theta1, sin_theta1, cos_theta2, sin_theta2, theta1, theta2 = state
        norm = []
        n_cos_theta1 = (cos_theta1 + 1) / 2
        norm.append(n_cos_theta1)
        n_sin_theta1 = (sin_theta1 + 1) / 2
        norm.append(n_sin_theta1)
        n_cos_theta2 = (cos_theta2 + 1)/ 2
        norm.append(n_cos_theta2)
        n_sin_theta2 = (sin_theta2 + 1) / 2
        norm.append(n_sin_theta2)
        n_theta1 = (theta1 + 12.567) / 25.134
        norm.append(n_theta1)
        n_theta2 = (theta2 + 28.274) / 56.584
        norm.append(n_theta2)
        return tuple(norm)

    def get_q_feature_value(norm_state, q_features):
        state_vector = np.array(list(norm_state))
        state_vector = np.array(list(norm_state)).reshape(len(norm_state),1)
        combinations_vector = np.array(q_features)
        dot_product = np.dot(combinations_vector, state_vector)
        feature_vector = np.cos(np.pi * dot_product)
        return np.transpose(feature_vector)

    def get_q_values(policy_weights, policy_features, norm_state):
        policy_list = []
        for i in range(len(Model.actions)):
            weights = policy_weights[i]
            valSubstitutingInFeatures = np.dot(policy_features, norm_state)
            f_policy_features = np.cos(np.pi * valSubstitutingInFeatures)
            policy_list.append(np.dot(weights, f_policy_features))
        return policy_list

    def get_epsilon_greedy_policy(policy, sigma):
        policy_arr = np.array(policy)
        numerator_values = [math.exp(sigma * x - max(sigma * policy_arr)) for x in policy_arr]
        denominator_values = [math.exp(sigma * x - max(sigma * policy_arr)) for x in policy_arr]
        denominator = sum(denominator_values)
        action_probabilities = [numerator_value / denominator for numerator_value in numerator_values]
        selected_action = random.choices(Model.actions, action_probabilities)[0] 
        return selected_action

    def get_q_value(q_weights, q_features, norm_state, cosineFlag=True):
        valSubstitutingInFeatures = np.dot(q_features, norm_state)
        f_policy_features = np.cos(np.pi * valSubstitutingInFeatures)
        policy = np.dot(q_weights, f_policy_features)
        return policy
    
    def episodic_n_step_sarsa(q_weights, q_features, epsilon, n_steps):
        env = gym.make('Acrobot-v1')
        state = env.reset()[0]
        states_list = []
        actions_list = []
        rewards_list = []
        max_reward = Model.max_reward       
        reward = 0   
        total_rewards = 0
        isTerminated = False 
        norm_state = CartPole.get_normalized_states(state)
        q_values = CartPole.get_q_values(q_weights, q_features, norm_state)
        action = CartPole.get_epsilon_greedy_policy(q_values, epsilon)
        states_list.append(state)
        actions_list.append(action)
        while True:        
            if isTerminated or total_rewards == max_reward: 
                return q_weights, total_rewards   
            next_state, reward, isTerminated, truncated, info = env.step(action)
            states_list.append(next_state)
            rewards_list.append(reward)
            norm_state = CartPole.get_normalized_states(state)
            state = next_state 
            action = actions_list[-1] 
            if not isTerminated:
                q_values = CartPole.get_q_values(q_weights, q_features, norm_state)
                action =CartPole.get_epsilon_greedy_policy(q_values, epsilon)
                actions_list.append(action)
            else:
                max_reward = total_rewards + 1
            tau = total_rewards - n_steps + 1
            if tau >= 0:
                indices_range = range(tau + 1, min(tau + n_steps, max_reward))
                discount_factors = [(Model.gamma)** (i - tau - 1) for i in indices_range]
                discounted_returns = [factor * rewards_list[i] for factor, i in zip(discount_factors, indices_range)]
                discount_return = sum(discounted_returns)    
                            
                if tau + n_steps < max_reward:
                    stepper_last_index = tau + n_steps
                    next_state = states_list[stepper_last_index]
                    next_action = actions_list[stepper_last_index]
                    next_actionIndex = Model.actions.index(next_action)

                    norm_next_state = CartPole.get_normalized_states(next_state)
                    discount_factor = Model.gamma ** n_steps
                    next_q_value = CartPole.get_q_value(q_weights[next_actionIndex], q_features, norm_next_state)
                    discount_return += discount_factor * next_q_value

                n_init_state = states_list[tau]
                stepper_init_actionIndex = Model.actions.index(actions_list[tau])
                norm_stepper_init_state = CartPole.get_normalized_states(n_init_state)

                for i in range(len(Model.actions)):
                    initial_q_value = CartPole.get_q_value(q_weights[stepper_init_actionIndex], q_features, norm_stepper_init_state)
                    error = discount_return - initial_q_value
                    init_state_feature_value = CartPole.get_q_feature_value(norm_stepper_init_state, q_features)
                    q_weights[i] += float(Model.alpha) * error * init_state_feature_value                    
            total_rewards += 1

def main():
    steps_all_episodes = []
    avg_steps = []
    overall_actions_before_start = []
    for i in range(1):
        decay_epsilon = Model.epsilon
        timesteps_list = []
        actions_before_start = [0] * (Model.episodes_per_run + 1)
        weights = CartPole.get_weights(Model.order)
        features = CartPole.get_feature_combinations(Model.order)        
        for i in range(Model.episodes_per_run):
            if (i+1) % 1000 == 0:
                decay_epsilon *= Model.decay_epsilon
            weights, timesteps = CartPole.episodic_n_step_sarsa(weights, features, decay_epsilon, Model.n_steps)
            timesteps_list.append(timesteps)
            actions_before_start[i+1] = actions_before_start[i] + timesteps
            print(timesteps)
        steps_all_episodes.append(timesteps_list)
        overall_actions_before_start.append(actions_before_start)
    avg_steps = np.mean(steps_all_episodes, axis=0)
    avg_step_list = np.mean(overall_actions_before_start, axis=0)
    
    episoderange = range(1, Model.episodes_per_run+1)
    plt.plot(avg_step_list[1:], episoderange)
    plt.title(f'Episodic Semi Gradient n-step SARSA\n Acrobot\n Learning Curve for steps = {Model.n_steps}')
    x_label = "Rewards"
    y_label = "Episodes"
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.show()
    
    plt.plot(episoderange, avg_steps)
    plt.title(f'Episodic Semi Gradient n-step SARSA\n Acrobot\n Rewards per episode for steps = {Model.n_steps}')
    x_label = "Episodes"
    y_label = "Rewards"
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.show()   

if __name__ == "__main__":
    main()