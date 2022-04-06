from gym.spaces.discrete import Discrete
import gym
from gym.spaces import Box, Dict
# from taxi_domain.envs.taxiworld_gen import *
import numpy as np

from taxi_domain.envs.taxiworld_gen import default_layout


def to_xy(t):
    return list(zip(*t))


ACTION_LOOKUP = {
    0: 'move up',
    1: 'move down',
    2: 'move left',
    3: 'move right',
    4: 'pick up',
    5: 'drop'
}

CHANGE_COORDINATES = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}


class RelationalTaxiWorld(gym.Env):
    def __init__(self, passenger_count=4,
                 reward_drop=20,
                 step_cost=-0.1,
                 no_move_cost=0,
                 reward_pickup=10,
                 max_passenger=4,
                 random_pickup_drop=-1,
                 layout=default_layout,
                 obs_type="vec",
                 max_steps=1000):
        self.action_space = Discrete(len(ACTION_LOOKUP))
        self.passenger_count = passenger_count
        assert passenger_count <= 4
        self.reward_drop = reward_drop
        self.step_cost = step_cost
        self.no_move_cost = no_move_cost
        self.reward_pickup = reward_pickup
        self.random_pickup_drop = random_pickup_drop
        self.obs_type = obs_type
        self.max_passenger = max_passenger
        self.max_steps = max_steps

        layout = np.array([list(line) for line in layout.splitlines()])
        self.grid_width = len(layout[0])
        wall = layout == 'w'
        self.not_wall = ~wall  #
        self.locations = to_xy(self.not_wall.nonzero())
        self.passenger_candidates = to_xy(np.isin(layout, ['R', 'G', 'B', 'Y']).nonzero())
        self.taxi_candidates = to_xy((layout == ' ').nonzero())
        obs = self.reset()
        self.observation_space = Box(-1, 1, shape=obs.shape, dtype='float32')
        map = (self.not_wall == 0).astype(float)
        map[self.passenger_candidates[0]] = 101
        map[self.passenger_candidates[1]] = 102
        map[self.passenger_candidates[2]] = 103
        map[self.passenger_candidates[3]] = 104
        map[tuple(self.taxi)] = 0.5
        image = map[1:-1, 1:-1]
        grid_vector = np.stack(image).ravel()
        self.grid_dim = len(grid_vector)
        self.image_height, self.image_width = image.shape
        self.image_channels = 1
        if obs_type == "graph":
            self.obj_dim = 5
        else:
            self.obj_dim = 9
        self.object_vector_size = self.obj_dim * self.max_passenger
        self.RGBY_locations = (np.argwhere(grid_vector == 101)[0][0],
                               np.argwhere(grid_vector == 102)[0][0],
                               np.argwhere(grid_vector == 103)[0][0],
                               np.argwhere(grid_vector == 104)[0][0])

    def reset(self):
        (i,) = np.random.choice(len(self.taxi_candidates), 1)
        self.taxi = self.taxi_candidates[i]
        self.pickup = []
        self.drop = []
        for i in range(self.passenger_count):
            pi, di = np.random.choice(len(self.passenger_candidates), 2, replace=False)
            self.pickup.append(self.passenger_candidates[pi])
            self.drop.append(self.passenger_candidates[di])
        self.current_passenger = None
        self.done = np.zeros(self.passenger_count, dtype=bool)
        self.viewer = None
        self.num_env_steps = 0
        self.episode_reward = 0
        return self.observation

    def is_valid_location(self, taxi):
        try:
            return self.not_wall[tuple(taxi)]
        except IndexError:
            return False

    def step(self, action):
        self.num_env_steps += 1
        reward = self.step_cost
        assert action in ACTION_LOOKUP.keys()
        if action < 4:
            # Movement
            new_taxi = self.taxi + np.array(CHANGE_COORDINATES[action])
            if self.is_valid_location(new_taxi):
                self.taxi = new_taxi
            else:
                reward -= self.no_move_cost
        elif action == 4 and self.current_passenger is None:
            # Pickup
            try:
                i = self.pickup.index((tuple(self.taxi)))
                if i != -1 and not self.done[i]:
                    self.current_passenger = i
                    self.pickup[i] = (999, 999)  # making the pickup inaccessible
                    reward += self.reward_pickup
                else:
                    reward += self.random_pickup_drop
            except ValueError:
                reward += self.random_pickup_drop
        elif action == 5 and self.current_passenger is not None:
            # Drop

            if self.drop[self.current_passenger] == tuple(self.taxi):
                self.done[self.current_passenger] = True
                self.drop[self.current_passenger] = (999, 999)  # making the drop inaccessible
                reward += self.reward_drop
                self.current_passenger = None
            # self.pickup[self.current_passenger] = self.taxi
        else:
            # raise AssertionError("action is not valid")
            if action in (4, 5):
                reward += self.random_pickup_drop

        self.episode_reward += reward
        done = np.all(self.done)
        info = {
            "episode_length": self.num_env_steps,
            "is_success": done,
            "num_passengers_done": np.sum(self.done),
            "episode_reward": self.episode_reward,
        }
        return self.observation, reward, done or self.num_env_steps >= self.max_steps, info

    def get_diagnostics(self, paths, **kwargs):
        successes = [p['env_infos'][-1]['is_success'] for p in paths]
        rewards = [p['env_infos'][-1]['episode_reward'] for p in paths]
        average_reward = np.mean(rewards)
        reward_max = np.max(rewards)
        success_rate = sum(successes) / len(successes)
        num_passengers_done = [p['env_infos'][-1]['num_passengers_done'] for p in paths]
        lengths = [p['env_infos'][-1]['episode_length'] for p in paths]
        length_rate = sum(lengths) / len(lengths)
        return {'Success Rate': success_rate,
                'Episode length Mean': length_rate,
                'Episode length Min': min(lengths),
                'Episode counts': len(paths),
                'Num of passengers done': np.mean(num_passengers_done),
                'Total Reward Mean': average_reward,
                'Total Reward Max': reward_max}

    @property
    def observation(self):
        map = (self.not_wall == 0).astype(float)
        map[tuple(self.taxi)] = 0.5
        grid_vector = np.stack(map[1:-1, 1:-1]).ravel()
        grid = grid_vector.copy()
        desired_grid = grid_vector.copy()
        if 'graph' in self.obs_type:
            for i, (pick, drop) in enumerate(zip(self.pickup, self.drop)):
                if not self.done[i]:
                    if not self.current_passenger == i:
                        grid = np.append(grid, np.array(pick) / self.grid_width)
                        grid = np.append(grid, [0])
                    else:
                        grid = np.append(grid, [0, 0, 1])
                    grid = np.append(grid, np.array(drop) / self.grid_width)
                else:
                    grid = np.append(grid, [0, 0, 0, 0, 0])
            return grid
        elif 'vec' in self.obs_type:
            for i, (pick, drop) in enumerate(zip(self.pickup, self.drop)):
                pickup_vector = np.zeros(len(self.passenger_candidates) + 1)
                drop_vector = np.zeros(len(self.passenger_candidates))
                desired_grid = np.append(np.append(desired_grid, pickup_vector.copy()), drop_vector.copy())
                if not self.done[i]:
                    if self.current_passenger == i:
                        pickup_vector[-1] = 1
                    else:
                        pickup_vector[self.passenger_candidates.index(pick)] = 1
                    drop_vector[self.passenger_candidates.index(drop)] = 1
                grid = np.append(np.append(grid, pickup_vector.copy()), drop_vector.copy())

            if self.passenger_count < self.max_passenger:
                desired_grid = np.append(desired_grid, np.zeros(9 * (self.max_passenger - self.passenger_count)))
                grid = np.append(grid, np.zeros(9 * (self.max_passenger - self.passenger_count)))
            return grid

    def render(self, mode="human"):
        M, N = self.not_wall.shape
        img = np.zeros(shape=(M, N, 3), dtype=np.uint8)
        img[self.not_wall] = 255
        img[tuple(self.taxi)] = [255, 255, 0]
        for p in self.pickup:
            if p == (999, 999):
                continue
            img[p] = [108, 0, 248]
        for d in self.drop:
            if d == (999, 999):
                continue
            img[d] = [128, 0, 128]

        scale = 16
        scaled = np.zeros((img.shape[0] * scale, img.shape[1] * scale, img.shape[2]))
        scaled[:, :, 0] = np.kron(img[:, :, 0], np.ones((scale, scale)))
        scaled[:, :, 1] = np.kron(img[:, :, 1], np.ones((scale, scale)))
        scaled[:, :, 2] = np.kron(img[:, :, 2], np.ones((scale, scale)))
        return scaled.astype(np.uint8)

