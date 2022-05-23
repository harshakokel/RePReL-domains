from gym.envs.registration import register
from craftworld.gym_crafting_env import GymCraftingEnv
from craftworld.crafting_base import CraftingBase
from craftworld.success_functions import *


register(
    id='Craftworld-fusion-task1-v1',
    entry_point='craftworld:GymCraftingEnv',
    kwargs={'state_fusion': True,
            'few_obj': True,
            'task_id': 0}
)

register(
    id='Craftworld-fusion-noobst-task1-v1',
    entry_point='craftworld:GymCraftingEnv',
    kwargs={'state_fusion': True,
            'few_obj': True,
            'task_id': 0,
            'visible_obstacle': False}
)

register(
    id='Craftworld-fusion-fixedinit-task1-v1',
    entry_point='craftworld:GymCraftingEnv',
    kwargs={'state_fusion': True,
            'few_obj': True,
            'fixed_init_state': True,
            'task_id': 0}
)

register(
    id='Craftworld-fusion-fixedinit-task2-v1',
    entry_point='craftworld:GymCraftingEnv',
    kwargs={'state_fusion': True,
            'few_obj': True,
            'fixed_init_state': True,
            'task_id': 1}
)

register(
    id='Craftworld-fusion-noobst-neighbour-task1-v1',
    entry_point='craftworld:GymCraftingEnv',
    kwargs={'state_fusion': True,
            'few_obj': True,
            'task_id': 0,
            'visible_obstacle': False,
            'visible_neighbour':True}
)


register(
    id='Craftworld-fusion-fixedinit-eatbread-v1',
    entry_point='craftworld:GymCraftingEnv',
    kwargs={'state_fusion': True,
            'few_obj': True,
            'fixed_init_state': True,
            'success_function':eval_eatbread}
)
