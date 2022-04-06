import gym
import logging
from random import Random
from typing import Iterable, FrozenSet, List, Set, Tuple
from gym.spaces.discrete import Discrete
from gym.spaces import MultiDiscrete
import numpy as np

WIDTH = 12
HEIGHT = 9

ACTIONS: List[Tuple[int, int]] = [
    (0, 1),  # up
    (0, -1),  # down
    (-1, 0),  # left
    (1, 0),  # right
]

WALLS = {
    ((2, 0), (3, 0)), ((3, 0), (2, 0)),
    ((5, 0), (6, 0)), ((6, 0), (5, 0)),
    ((8, 0), (9, 0)), ((9, 0), (8, 0)),

    ((2, 2), (3, 2)), ((3, 2), (2, 2)),
    ((5, 2), (6, 2)), ((6, 2), (5, 2)),
    ((8, 2), (9, 2)), ((9, 2), (8, 2)),

    ((2, 3), (3, 3)), ((3, 3), (2, 3)),
    ((5, 3), (6, 3)), ((6, 3), (5, 3)),
    ((8, 3), (9, 3)), ((9, 3), (8, 3)),

    ((2, 4), (3, 4)), ((3, 4), (2, 4)),
    ((5, 4), (6, 4)), ((6, 4), (5, 4)),
    ((8, 4), (9, 4)), ((9, 4), (8, 4)),

    ((2, 5), (3, 5)), ((3, 5), (2, 5)),
    ((5, 5), (6, 5)), ((6, 5), (5, 5)),
    ((8, 5), (9, 5)), ((9, 5), (8, 5)),

    ((2, 6), (3, 6)), ((3, 6), (2, 6)),
    ((5, 6), (6, 6)), ((6, 6), (5, 6)),
    ((8, 6), (9, 6)), ((9, 6), (8, 6)),

    ((2, 8), (3, 8)), ((3, 8), (2, 8)),
    ((5, 8), (6, 8)), ((6, 8), (5, 8)),
    ((8, 8), (9, 8)), ((9, 8), (8, 8)),

    ((0, 2), (0, 3)), ((0, 3), (0, 2)),
    ((0, 5), (0, 6)), ((0, 6), (0, 5)),

    ((2, 2), (2, 3)), ((2, 3), (2, 2)),
    ((2, 5), (2, 6)), ((2, 6), (2, 5)),

    ((3, 2), (3, 3)), ((3, 3), (3, 2)),
    ((3, 5), (3, 6)), ((3, 6), (3, 5)),

    ((4, 2), (4, 3)), ((4, 3), (4, 2)),

    ((5, 2), (5, 3)), ((5, 3), (5, 2)),
    ((5, 5), (5, 6)), ((5, 6), (5, 5)),

    ((6, 2), (6, 3)), ((6, 3), (6, 2)),
    ((6, 5), (6, 6)), ((6, 6), (6, 5)),

    ((7, 2), (7, 3)), ((7, 3), (7, 2)),

    ((8, 2), (8, 3)), ((8, 3), (8, 2)),
    ((8, 5), (8, 6)), ((8, 6), (8, 5)),

    ((9, 2), (9, 3)), ((9, 3), (9, 2)),
    ((9, 5), (9, 6)), ((9, 6), (9, 5)),

    ((11, 2), (11, 3)), ((11, 3), (11, 2)),
    ((11, 5), (11, 6)), ((11, 6), (11, 5)),
}

OBJECTS = {
    (1, 1): frozenset('a'),
    (10, 1): frozenset('b'),
    (10, 7): frozenset('c'),
    (1, 7): frozenset('d'),
    (7, 4): frozenset('e'),  # mail
    (3, 6): frozenset('f'),  # coffee
    (8, 2): frozenset('f'),  # coffee
    (4, 4): frozenset('g'),  # office
    (4, 1): frozenset('n'),  # plant
    (7, 1): frozenset('n'),  # plant
    (4, 7): frozenset('n'),  # plant
    (7, 7): frozenset('n'),  # plant
    (1, 4): frozenset('n'),  # plant
    (10, 4): frozenset('n'),  # plant
}

CELL_STRING_MAPPING = {(0, 8): 19, (1, 8): 20,(2, 8): 21,
           (3, 8): 23, (4, 8): 24,(5, 8): 25,
           (6, 8): 27, (7, 8): 28,(8, 8): 29,
           (9, 8): 31, (10, 8): 32,(11, 8): 33,
           (0, 7): 37, (1, 7): 38,(2, 7): 39,
           (3, 7): 41, (4, 7): 42,(5, 7): 43,
           (6, 7): 45, (7, 7): 46,(8, 7): 47,
           (9, 7): 49, (10, 7): 50,(11, 7): 51,
           (0, 6): 55, (1, 6): 56,(2, 6): 57,
           (3, 6): 59, (4, 6): 60,(5, 6): 61,
           (6, 6): 63, (7, 6): 64,(8, 6): 65,
           (9, 6): 67, (10, 6): 68,(11, 6): 69,
           (0, 5): 91, (1, 5): 92,(2, 5): 93,
           (3, 5): 95, (4, 5): 96,(5, 5): 97,
           (6, 5): 99, (7, 5): 100,(8, 5): 101,
           (9, 5): 103, (10, 5): 104,(11, 5): 105,
           (0, 4): 109, (1, 4): 110,(2, 4): 111,
           (3, 4): 113, (4, 4): 114,(5, 4): 115,
           (6, 4): 117, (7, 4): 118,(8, 4): 119,
           (9, 4): 121, (10, 4): 122,(11, 4): 123,
           (0, 3): 127, (1, 3): 128,(2, 3): 129,
           (3, 3): 131, (4, 3): 132,(5, 3): 133,
           (6, 3): 135, (7, 3): 136,(8, 3): 137,
           (9, 3): 139, (10, 3): 140,(11, 3): 141,
           (0, 2): 163, (1, 2): 164,(2, 2): 165,
           (3, 2): 167, (4, 2): 168,(5, 2): 169,
           (6, 2): 171, (7, 2): 172,(8, 2): 173,
           (9, 2): 175, (10, 2): 176,(11, 2): 177,
           (0, 1): 181, (1, 1): 182,(2, 1): 183,
           (3, 1): 185, (4, 1): 186,(5, 1): 187,
           (6, 1): 189, (7, 1): 190,(8, 1): 191,
           (9, 1): 193, (10, 1): 194,(11, 1): 195,
           (0, 0): 199, (1, 0): 200,(2, 0): 201,
           (3, 0): 203, (4, 0): 204,(5, 0): 205,
           (6, 0): 207, (7, 0): 208,(8, 0): 209,
           (9, 0): 211, (10, 0): 212,(11, 0): 213}

DISPLAY_STRING =  "+---" * 4 + "+\n" + \
                  "|   " * 4 + "|\n" + \
                  "| d   *   *   c |\n" + \
                  "|   |f  |   |   |\n" + \
                  "+- -" * 4 + "+\n" +\
                  "|   " * 4 + "|\n" +\
                  "| * | g | e | * |\n" +\
                  "|   |   |   |   |\n" +\
                  "+- -+---+---+- -+\n" +\
                  "|   |   |  f|   |\n" +\
                  "| a   *   *   b |\n" + \
                  "|   |   |   |   |\n" + \
                  "+---+---+---+---+\n"


def update_facts(facts: Tuple[bool, ...],
                 objects: FrozenSet[str]) -> Set[int]:
    fact_indices = set([i for i, v in enumerate(facts) if v])
    if 'a' in objects:
        fact_indices.add(0)
    if 'b' in objects:
        fact_indices.add(1)
    if 'c' in objects:
        fact_indices.add(2)
    if 'd' in objects:
        fact_indices.add(3)
    if 'e' in objects:
        fact_indices.add(4)
    if 'f' in objects:
        fact_indices.add(5)
    if 'g' in objects:
        fact_indices.add(6)
        if facts[4]:
            fact_indices.remove(4)
            fact_indices.add(7)
        if facts[5]:
            fact_indices.remove(5)
            fact_indices.add(8)
    return fact_indices


class OfficeState:
    facts: Tuple[bool, ...]

    def __init__(self, x: int, y: int, facts: Iterable[int]):
        self.x = x
        self.y = y

        fact_list = [False] * 9
        for fact in facts:
            fact_list[fact] = True
        self.facts = tuple(fact_list)

        self.uid = np.array([self.x, self.y] + [int(val) for val in fact_list])

    def __repr__(self):
        return self.__str__()

    def __str__(self) -> str:
        return "({:2d}, {:2d}, {})".format(self.x, self.y, self.facts)

    @staticmethod
    def random(rng: Random) -> 'OfficeState':
        # return OfficeState(0, 0, [])
        while True:
            x = rng.randrange(WIDTH)
            y = rng.randrange(HEIGHT)
            if (x, y) not in OBJECTS:
                return OfficeState(x, y, [])


# 0: visit-a, 1: b, 2: c, 3: d, 4: get-mail, 5: get-coffee, 6: visit-office, 7: deliver-mail, 8-deliver-coffee
# possible tasks = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [7, 8], [0, 1, 2, 3],
#          [0, 1, 2, 3, 7, 8], [0, 1, 2, 3, 4, 5, 6, 7, 8]]


class OfficeWorld(gym.Env):

    def __init__(self, rng=None, target=[0], step_cost=1.0, invalid_action_cost=10, terminal_reward=100):
        """Office World Gym environment.

        Possible Targets/Tasks are:
            0: visit-a
            1: visit-b
            2: visit-c
            3: visit-d
            4: get-mail
            5: get-coffee
            6: visit-office
            7: deliver-mail
            8: deliver-coffee
        Observation is MultiDiscrete Space od shape (11,) representing:
            [agent's x-loc (int), agent's y-loc (int),
             visited-a (binary), visited-b (binary), visited-c (binary), visited-d (binary),
             has-mail (binary), has-coffee (binary), visited-office (binary),
             delivered-mail (binary), delivered-coffee (binary)]"""
        self.target = target
        self.step_cost = step_cost
        self.action_space = Discrete(len(ACTIONS))
        self.observation_space =  MultiDiscrete([WIDTH, HEIGHT]+[2]*9)
        self.invalid_action_cost = invalid_action_cost
        self.terminal_reward = terminal_reward
        if rng is None:
            self.rng = Random()
        self.state = OfficeState.random(self.rng)
        self.r = 0
        self.episode_length = 0

    def step(self, a):

        if a >= len(ACTIONS):
            raise IndexError("invalid action")
        self.episode_length += 1
        info = {'episode_length':self.episode_length}
        s0 = self.state
        x, y = s0.x + ACTIONS[a][0], s0.y + ACTIONS[a][1]
        logging.debug("applying action %s:%s", a, ACTIONS[a])
        if x < 0 or y < 0 or x >= WIDTH or y >= HEIGHT or \
                ((s0.x, s0.y), (x, y)) in WALLS:
            reward, done  = self.reward(s0, a, self.state)
            self.r += reward
            info['is_success']=done
            info['episode_reward'] = self.r
            return self.state.uid,reward, done, info

        objects = OBJECTS.get((x, y), frozenset())
        new_facts = update_facts(self.state.facts, objects)
        self.state = OfficeState(x, y, new_facts)
        logging.debug("success, current state is %s", self.state)
        reward, done = self.reward(s0, a, self.state)
        self.r += reward
        info['is_success']=done
        info['episode_reward'] = self.r
        return self.state.uid, reward, done, info

    def cost(self, s0: OfficeState, a, s1: OfficeState) -> float:
        c = self.step_cost
        if s0 == s1:
            c += self.invalid_action_cost
        if 'n' in self.observe(s1):
            c += self.invalid_action_cost
        return c

    def reward(self, s0, a, s1):
        cost = self.cost(s0, a, s1)
        for fact in self.target:
            if not s1.facts[fact]:
                return -cost, False
        return -cost + self.terminal_reward, True

    def observe(self, state: OfficeState):
        return OBJECTS.get((state.x, state.y), {})

    def reset(self):
        self.state = OfficeState.random(self.rng)
        self.episode_length = 0
        self.r = 0
        return self.state.uid

    @staticmethod
    def label(state: OfficeState) -> FrozenSet[int]:
        return frozenset([i for i in range(9) if state.facts[i]])

    def render(self, mode='human'):
        index = CELL_STRING_MAPPING[(self.state.x, self.state.y)]
        display = DISPLAY_STRING[:index] +"0"+ DISPLAY_STRING[index+1:]
        return display


