from Environment import Env
import random

ACTIONS = {'STAY', 'RIGHT', 'LEFT'}

env = Env(num_of_solar_panels=2, height=3, width=4, number_of_agents=2, max_fuel=1000)
env.render()
for i in range(1000):
    actions = []
    for _ in range(env.number_of_agents):
        actions.append(random.sample([*env.ACTIONS.keys()], 1)[0])
    print(f"Actions: Agent_1={actions[0]}, Agent_2={actions[1]}")  # , , Agent_3={actions[2]}, , Agent_4={actions[3]}, , Agent_5={actions[4]}, , Agent_6={actions[5]}")
    env.step(actions)
    env.render()
    if env.is_done():
        break

print("Done!")

