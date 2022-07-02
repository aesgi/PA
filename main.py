from Soil import Soil
from Time import ArtificialTime
import ElementaryRlAgent
import ElementaryRIAgentLight
import numpy
import copy
import pickle
import boto3


def interaction():
    t = ArtificialTime()
    soil = Soil(t)
    policy = ElementaryRlAgent.Policy(0.1, 0.05, [0, 20])
    agent = ElementaryRlAgent.Agent(soil, t, policy, [0, 1, 2], 0.7, 0.8, True)
    last_raw_command = ""
    while t.month < 2:
        raw_command = input()
        if raw_command == "*":
            raw_command = last_raw_command
        command = raw_command.split()
        if command[0] == "state":
            m = float(command[1])
            for s in policy.state_action_values.keys():
                if s.moisture == m:
                    print(str(s)+" :")
                    for action, value in policy.state_action_values[s].items():
                        print("\tintensity " + str(action) + " :" + str(value))
        elif command[0] == "proceed":
            counter = 0
            if len(command) == 3:
                while counter < int(command[2]):
                    if command[1] == "verbose":
                        print("state: "+str(agent.state))
                    agent.Q_learning_iteration()
                    if command[1] == "verbose":
                        print("action: "+str(agent.action_to_take)+" , reward: "+str(agent.reward))
                        print()
                    t.increase_time()
                    counter += 1
                    if t.month >= 2:
                        break
            else:
                print("Invalid command!")
        elif command[0] == "soil":
            print(soil)
        elif command[0] == "epsilon":
            print(policy.epsilon)
        elif command[0] == "iteration":
            if command[1] == "explore":
                print(policy.exploration_iteration)
            if command[1] == "learn":
                print(agent.learning_iteration)
        elif command[0] == "history":
            if command[1] == "explore":
                print(policy.explore_delta_reward_EMA)
            elif command[1] == "exploit":
                print(policy.exploit_delta_reward_EMA)
            elif command[1] == "reward":
                print(policy.reward_EMA)
            else:
                print("Invalid command!")
        elif command[0] == "visualize":
            if len(command) > 1:
                soil.visualizer(command[1])
            else:
                print("Invalid command!")
        elif command[0] == "loss":
            print(soil.LAYERS_WATER_LOSS)
        elif command[0] == "input":
            print(soil.input_water)
        else:
            print("Invalid Command!")
        last_raw_command = raw_command
    soil.visualizer('day')


boto3_session = boto3.session.Session(aws_access_key_id="AKIA2NJC4PBEO3WIEQ6L",aws_secret_access_key="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",region_name="us-east-1")
s3 = boto3_session.resource('s3')
object = s3.Bucket('spookydata')

def save(self):
        Q = {}
        for state, actions in self.state_action_values.items():
            Q[str(state)] = {}
            for action, value in actions.items():
                Q[str(state)][str(action)] = value
        return Q

def with_agent():
    t = ArtificialTime()
    soil = Soil(t)
    policy = ElementaryRlAgent.Policy(0.1, 0.05, [0, 20]) # gamma, alpha, intensity : 0 & 20 (watering or not), 
    agent = ElementaryRlAgent.Agent(t, policy, [0,0,0], 0.7, 0.8, True) #exemple de mesures / Vie de la plante.
    while t.month < 2:
        agent.Q_learning_iteration()
       
        if agent.learning_iteration % 100 == 0:
            #print(soil)
            print(policy)
            print("------------------------------------------")
            print(policy.epsilon)
            print("------------------------------------------")
            with open('saved_dictionary_water.pkl', 'wb') as f:
                pickle.dump(save(policy), f)
            pass
        t.increase_time()
#   soil.visualizer('day')

def AgentLight():
    t = ArtificialTime()
    policy = ElementaryRIAgentLight.Policy(0.1, 0.05, [0, 20])  # gamma, alpha, intensity : 0 & 20 (lighting or not)
    agent = ElementaryRIAgentLight.Agent(t, policy, [0,0.8,0.7], 0.4, 0.7, True)
    while t.month < 2:
        agent.Q_learning_iteration()

        if agent.learning_iteration % 100 == 0:
            print(policy)
            print("------------------------------------------")
            print(policy.epsilon)
            print("------------------------------------------")
            with open('saved_dictionary_light.pkl', 'wb') as f:
                pickle.dump(save(policy), f)
            pass
        t.increase_time()

#SOIL
#with_agent()
#object.upload_file("saved_dictionary_water.pkl", 'model/simple_QL_saved.pkl')


#LIGHT
AgentLight()
object.upload_file("saved_dictionary_light.pkl", 'model/simple_QL_saved_light.pkl')