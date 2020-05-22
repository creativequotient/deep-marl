import numpy as np


def simple_spread_reward(agent, world):
    def is_collision(agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
    rew = 0
    for l in world.landmarks:
        dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
        rew -= min(dists)
    # print(rew)
    if agent.collide:
        for a in world.agents:
            if is_collision(a, agent) and a != agent:
                rew -= 1
    return rew
