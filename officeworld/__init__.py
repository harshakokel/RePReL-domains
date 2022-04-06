from officeworld.envs.office import OfficeWorld
from gym.envs.registration import register

register(
    id='OfficeWorld-visit-a-v0',
    entry_point='officeworld.envs:OfficeWorld',
    kwargs={}
)

register(
    id='OfficeWorld-visit-b-v0',
    entry_point='officeworld.envs:OfficeWorld',
    kwargs={'target': [1]}
)


register(
    id='OfficeWorld-visit-c-v0',
    entry_point='officeworld.envs:OfficeWorld',
    kwargs={'target': [2]}
)

register(
    id='OfficeWorld-visit-d-v0',
    entry_point='officeworld.envs:OfficeWorld',
    kwargs={'target': [3]}
)

register(
    id='OfficeWorld-get-mail-v0',
    entry_point='officeworld.envs:OfficeWorld',
    kwargs={'target': [4]}
)

register(
    id='OfficeWorld-get-coffee-v0',
    entry_point='officeworld.envs:OfficeWorld',
    kwargs={'target': [5]}
)

register(
    id='OfficeWorld-visit-office-v0',
    entry_point='officeworld.envs:OfficeWorld',
    kwargs={'target': [6]}
)

register(
    id='OfficeWorld-deliver-mail-v0',
    entry_point='officeworld.envs:OfficeWorld',
    kwargs={'target': [7]}
)

register(
    id='OfficeWorld-deliver-coffee-v0',
    entry_point='officeworld.envs:OfficeWorld',
    kwargs={'target': [8]}
)