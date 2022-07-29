from Environment import Env
import random
import os
import openpyxl
import datetime
import math

ACTIONS = {'STAY', 'RIGHT', 'LEFT'}



if __name__ == '__main__':
    env = Env(num_of_solar_panels=3, height=5, width=3, number_of_agents=2, max_fuel=100)
    env.render()
    for i in range(1):
        actions = []
        for _ in range(env.number_of_agents):
            actions.append(random.sample([*env.ACTIONS.keys()], 1)[0])
        print(f"Actions: Agent_1={actions[0]}, Agent_2={actions[1]}")  # , , Agent_3={actions[2]}, , Agent_4={actions[3]}, , Agent_5={actions[4]}, , Agent_6={actions[5]}")
        env.step(actions)
        env.render()
        if env.is_done():
            print("success")
            break



