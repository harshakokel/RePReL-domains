import numpy as np
import random

# Yellow =  -0.5882352941176471
empty_taxi_color = [255, 255, 0]

# -0.803921568627451
empty_taxi_at_pickup = [255, 200, 0]

# -0.6392156862745099
loaded_taxi_at_drop = [0, 255, 242]

#  -1.611764705882353
empty_taxi_at_drop = [191, 56, 2]

# Green -1.5882352941176472
loaded_taxi_color = [0,255,0]

# purple -1.5843137254901962
drop_loc_color = [128, 0, 128]

# pickup  -1.192156862745098
pick_loc_color = [108, 0, 248]

# grey 0.0
grid_color = [220, 220, 220]

#black  -2.588235294117647
wall_color = [0, 0, 0]

fifteen_layout = """\
wwwwwwwwwwwwwwwwww
wR  w           Gw
w   w            w
w   w            w
w   w            w
w   w            w
w   w            w
w                w
w                w
w                w
w                w
w                w
w  w      w      w
w  w      w      w
w  w      w      w
w  w      w      w
wY w      wB     w
wwwwwwwwwwwwwwwwww
"""

five_by_five_layout = """\
wwwwwww
wR w Gw
w  w  w
w     w
w ww  w
wYwwB w
wwwwwww
"""

eight_layout = """\
wwwwwwwwww
wR  w   Gw
w   w    w
w   w    w
w        w
w        w
w w  w   w
w w  w   w
wYw  wB  w
wwwwwwwwww
"""
six_layout = """\
wwwwwwww
wR w  Gw
w  w   w
w      w
w      w
w w w  w
wYw wB w
wwwwwwww
"""

six_layout_2 = """\
wwwwwwww
wR w  Gw
w  w   w
w      w
w w w  w
w w w  w
wYw wB w
wwwwwwww
"""


seven_layout = """\
wwwwwwwww
wR  w  Gw
w   w   w
w       w
w       w
w w  w  w
w w  w  w
wYw  wB w
wwwwwwwww
"""

I_layout = """\
wwwwwwwwwww
wR       Gw
w         w
w         w
w         w
wwwww wwwww
w         w
w         w
w         w
wY       Bw
wwwwwwwwwww
"""

fourroom_layout = """\
wwwwwwwwwww
wR   w   Gw
w         w
w    w    w
w    w    w
ww www    w
w    www ww
w    w    w
w         w
wY   w   Bw
wwwwwwwwwww
"""

plus_layout = """\
wwwwwwwwwww
wwww  Gwwww
wwww   wwww
wwww   wwww
wR        w
w         w
w        Bw
wwww   wwww
wwww   wwww
wwwwY  wwww
wwwwwwwwwww
"""

plus_layout = """\
wwwwwwwwwww
wwww  Gwwww
wwww   wwww
wwww   wwww
wR        w
w         w
w        Bw
wwww   wwww
wwww   wwww
wwwwY  wwww
wwwwwwwwwww
"""

thirteen_layout = """\
wwwwwwwwwwwwwww
wR  w        Gw
w   w         w
w   w         w
w   w         w
w   w         w
w             w
w             w
w             w
w             w
w w    w      w
w w    w      w
w w    w      w
wYw    wB     w
wwwwwwwwwwwwwww"""


twelve_layout = """\
wwwwwwwwwwwwww
wR  w       Gw
w   w        w
w   w        w
w   w        w
w            w
w            w
w            w
w            w
w w    w     w
w w    w     w
w w    w     w
wYw    wB    w
wwwwwwwwwwwwww"""


eleven_layout = """\
wwwwwwwwwwwww
wR  w      Gw
w   w       w
w   w       w
w   w       w
w           w
w           w
w           w
w           w
w w    w    w
w w    w    w
wYw    wB   w
wwwwwwwwwwwww"""


ten_layout = """\
wwwwwwwwwwww
wR  w     Gw
w   w      w
w   w      w
w   w      w
w          w
w          w
w          w
w w    w   w
w w    w   w
wYw    wB  w
wwwwwwwwwwww
"""


default_layout = """\
wwwwwwwwwww
wR  w    Gw
w   w     w
w   w     w
w         w
w         w
w         w
w w   w   w
w w   w   w
wYw   wB  w
wwwwwwwwwww
"""

S_layout="""\
wwwwwwwwwww
wR       Gw
w         w
w  wwwww  w
w  w   w  w
wwwwY  w  w
w      w  w
w  wwwww  w
w         w
w        Bw
wwwwwwwwwww
"""

default_loc1_layout = """\
wwwwwwwwwww
w  Rw     w
w   w    Gw
w   w     w
w         w
w         w
w         w
w w   w   w
wYw   w   w
w w   w  Bw
wwwwwwwwwww
"""

default_loc2_layout = """\
wwwwwwwwwww
w   w    Gw
wR  w     w
w   w     w
w         w
w         w
w         w
w w   w   w
w w   wB  w
wYw   w   w
wwwwwwwwwww
"""

shifted_layout = """\
wwwwwwwwwww
wR  w    Gw
w   w     w
w   w     w
w         w
w         w
w         w
w  w  w   w
w  w  w   w
wY w  wB  w
wwwwwwwwwww
"""

shifted_layout_2 = """\
wwwwwwwwwww
wR  w    Gw
w   w     w
w  w      w
w         w
w         w
w         w
w w   w   w
w w   w   w
wY w  wB  w
wwwwwwwwwww
"""




def world_gen(seed=None, layout=default_layout):
    """generate Taxi Domain
    """
    if seed is not None:
        random.seed(seed)
    mask = np.array([list(map(lambda c: 1 if c != 'w' else 0, line)) for line in layout.splitlines()])

    mask = mask[:, :, np.newaxis]
    world = np.concatenate([mask, mask, mask], axis=2) * 220
    nrow, ncol , _  = world.shape
    assert nrow == ncol, "Layout is not square"
    n = nrow
    possibilities = set(range(1, (n-2) * (n - 3)))

    location = {}
    location_set = ['R', 'G', 'B', 'Y']
    for x, line in enumerate(layout.splitlines()):
        for y, char in enumerate(line):
            if char in location_set:
                location[char]=np.array([x,y])
            elif char=='w':
                possibilities -= set([(x*(n-1))+y])

    key = random.sample(possibilities, 1)[0]
    taxi_loc = np.array([key // (n - 1), key % (n - 1)])
    pick_pos = random.choice(location_set)
    pick_loc = location[pick_pos]
    location_set.remove(pick_pos)
    drop_loc = location[random.choice(location_set)]
    world[taxi_loc[0], taxi_loc[1]] = empty_taxi_color
    world[pick_loc[0], pick_loc[1]] = pick_loc_color
    world[drop_loc[0], drop_loc[1]] = drop_loc_color
    return world,location, taxi_loc, pick_loc, drop_loc

def clean_up_cell(n, upper_cell_x, upper_cell_y):
    n = n-2
    return [upper_cell_x * (n - 1) + upper_cell_y] + \
           [upper_cell_x * (n - 1) + i + upper_cell_y for i in
            range(1, min(2, n - 2 - upper_cell_y) + 1)] + \
           [upper_cell_x * (n - 1) - i + upper_cell_y for i in range(1, min(2, upper_cell_y) + 1)]

def update_color(world, previous_loc, new_loc):
    taxi = world[previous_loc[0], previous_loc[1]]
    world[previous_loc[0], previous_loc[1]] = grid_color
    world[new_loc[0], new_loc[1]] = taxi
#     TODO make sure the grid color is set to passenger color or dest color



def is_empty(room):
    return not np.array_equal(room, wall_color)

if __name__ == "__main__":
    world_gen(0)

