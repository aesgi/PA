import numpy
import random
import math
import copy
import requests
import time

### ACTION INITIALISATION FOR WATER POMP
class Action:
    def __init__(self, intensity):  
        self.intensity = int(intensity)

    def __eq__(self, other):
        return self.intensity == other.intensity

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return str(self.intensity)

    def __hash__(self):
        return hash(self.intensity)

class State:
    def __init__(self, moisture, time_of_day):
        self.moisture = numpy.round_(moisture, decimals = 2)
        self.time_of_day = int(time_of_day/60)

    def __eq__(self, other):
        return self.moisture == other.moisture

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        # return str(self.moisture) +str(self.time_of_day)
        return str(self.moisture)

    def __hash__(self):
        return hash((self.moisture, self.time_of_day))

class Policy:
    state_action_values = {}
    def __init__(self, gamma, alpha, intensities, **kwargs):
        self.gamma = gamma
        self.alpha = alpha
        self.intensities = intensities
        self.epsilon = kwargs.get('epsilon', 1)
        # Optimism value says what value to assume for our next state when it's an unknown state with empty actions
        self.optimism_value = kwargs.get('optimism_value', 0)
        # heuristic tells something about the soil: If it's negative we are below moisture and if positive we are above
        self.heuristic = kwargs.get('heuristic', 0)
        self.exploit_delta_reward_EMA = 0.1
        # Exponential moving average of the delta rewards for exploration. This is used increase epsilon whenever necessary
        self.explore_delta_reward_EMA = 0
        self.reward_EMA = 0
        self.reward_EMV = 0
        # Initiate to 1 so we won't have division by zero
        self.exploration_iteration = 1
        self.exploit_better_count = 0

    def check_add_state(self, state):
        #print(self.state_action_values.keys())
        if len(self.state_action_values.keys()) < 1:
            actions = {}
            for i in self.intensities:
                actions[Action(i)] = self.optimism_value
            self.state_action_values[state.moisture] = actions
            return
        for sa in self.state_action_values.keys():
            if state.moisture == sa:
                actions = {}
                for i in self.intensities:
                    actions[Action(i)] = (self.state_action_values[sa][Action(i)] + self.optimism_value) / 2
                self.state_action_values[sa] = actions
                return
        actions = {}
        for i in self.intensities:
            actions[Action(i)] = self.optimism_value
        self.state_action_values[state.moisture] = actions


    def mapper(self, state):
        actions = self.state_action_values[state.moisture]
        # with probability 1-epsilon, we will act greedily and exploit
        if random.random() > self.epsilon:
            action_to_take = max(actions, key=actions.get)
            explore_exploit = 'exploit'
        else:
            action_to_take = self.pick_random_action()
            explore_exploit = 'explore'
        return action_to_take, explore_exploit

    def pick_random_action(self):
        # The smaller our action space, the more unbalanced the probability distribution of choosing an action randomly
        probabilistic_action_space_indices = numpy.random.exponential(math.log2(len(self.intensities))*(1-abs(self.heuristic)), 1000)
        # We don't want to have zero because that would cause index out of bounds in the case of negative heuristic
        probabilistic_action_space_indices += 0.0001
        probabilistic_action_space_indices = probabilistic_action_space_indices[
            probabilistic_action_space_indices < len(self.intensities)]
        if self.heuristic < 0:
            probabilistic_action_space_indices = len(self.intensities) - probabilistic_action_space_indices
        random_index = int(random.random()*len(probabilistic_action_space_indices))
        return Action(self.intensities[int(probabilistic_action_space_indices[random_index])])

    # Updates the Q-function, EMA, and epsilon according to action taken and reward received
    def value_updater(self, explore_exploit, reward, delta_reward, state, next_state, action_taken):
        diff = self.exploit_delta_reward_EMA - self.explore_delta_reward_EMA
        # If exploring has been showing more improving rewards for a long enough time then keep exploring
        if diff < 0:
            if self.exploit_better_count > 0:
                self.exploit_better_count = 0
            self.exploration_iteration = max(2, self.exploration_iteration-1)
        else:
            self.exploit_better_count += 1
            self.exploration_iteration = min(self.exploit_better_count+self.exploration_iteration, 2**30)
            if reward < 0.95 and self.reward_EMA < 0.95 and abs(self.exploit_delta_reward_EMA) < self.alpha*0.1:
                # Set exploration iteration so that epsilon becomes 0.5
                self.exploration_iteration = len(self.intensities)**2
                # Also enhance the histories so that a reset is actually done
                self.exploit_delta_reward_EMA = copy.copy(self.alpha)
                self.explore_delta_reward_EMA = copy.copy(self.alpha)
        if explore_exploit == 'explore':
            self.explore_delta_reward_EMA = self.alpha * delta_reward + (1 - self.alpha) * self.explore_delta_reward_EMA
        elif explore_exploit == 'exploit':
            self.exploit_delta_reward_EMA = self.alpha * delta_reward + (1 - self.alpha) * self.exploit_delta_reward_EMA
        # The speed of transition from initial exploration to exploitation depends on the size of action space
        self.epsilon = math.log2(len(self.intensities))/math.log2(self.exploration_iteration)
        self.reward_EMV = (1 - self.alpha) * (self.reward_EMV + self.alpha * (delta_reward - self.reward_EMA) ** 2)
        self.reward_EMA = (1-self.alpha)*self.reward_EMA + self.alpha*reward
        # In this algorithm, instead of sum of all possible next state values, a sample is taken
        max_value_of_next_state = max(self.state_action_values[next_state.moisture].values())
        self.state_action_values[state.moisture][action_taken] += self.alpha*(reward +
             self.gamma*max_value_of_next_state - self.state_action_values[state.moisture][action_taken])

    def __str__(self):
        out_str = ""
        for state, actions in self.state_action_values.items():
            out_str += "state "+str(state)+":\n"
            for action, value in actions.items():
                out_str += "\tintensity "+str(action)+" :"+str(value)+'\n'
        return out_str


class Agent:

    def __init__(self, t, policy, measures, min_measure, max_measure, plant_state):
        self.time = t
        self.policy = policy
        self.learning_iteration = 0
        self.min_measure = min_measure
        self.max_measure = max_measure
        self.state = State(numpy.mean(measures), 10)
        self.policy.check_add_state(self.state)
        # These two are made properties of the Agent class for debugging reasons
        self.action_to_take = numpy.nan
        self.measures = measures
        self.reward = 0
        self.action_to_take = numpy.nan
        self.plant_state = plant_state

    def Q_learning_iteration(self):
        # Choose action
        self.action_to_take, explore_exploit = self.policy.mapper(self.state)
        # Take action
        action = {'action_to_take': self.action_to_take.intensity,
                                       'is_watering': self.action_to_take.intensity > 0}
        # Wait for irrigation action to complete
        #time.sleep(24*3600/self.time.day_time_limit)
        params={'q': 'measures'}

        self.measures = []
        #response = requests.get(self.url, params={'q': 'measures'})
        for m in range(3):
            self.measures.append(random.randint(0,1))
        #if response.status_code == 200:
         #   for m in response.json()['measures']:
          #      self.measures.append(m)
        #reward, next_state = self.observer(self.action_to_take)

        reward, next_state = self.observer(self.action_to_take)
        # Update Q-function
        self.policy.value_updater(explore_exploit, reward, reward - self.reward,
                                  self.state, next_state, self.action_to_take)
        self.reward = reward
        self.state = next_state
        self.learning_iteration += 1

    def observer(self, action_taken):
        mean_moisture = numpy.mean(self.measures )
        
        next_state = State(mean_moisture, self.time.time_of_day)
        self.policy.check_add_state(next_state)

        if mean_moisture < self.min_measure:
            reward = 1 - self.min_measure + mean_moisture
            self.policy.heuristic = mean_moisture - self.min_measure
            if self.plant_state == False:
                reward = 2 - action_taken.intensity/max(self.policy.intensities)
                self.policy.heuristic = 0
        elif mean_moisture > self.max_measure:
            reward = 1 - mean_moisture + self.max_measure
            self.policy.heuristic = mean_moisture - self.max_measure
            if self.plant_state == False:
                reward = 2 - action_taken.intensity/max(self.policy.intensities)
                self.policy.heuristic = 0
        else:
            reward = 2 - action_taken.intensity/max(self.policy.intensities)
            self.policy.heuristic = 0
        return reward, next_state

