"""
F4 — Export the trained agent's route for the orders shown on the baseline
comparison page (SO/26/10/0772 and SO/26/10/0491), so the page can draw them.

Input : models/ppo_best.zip, orders_test.json, distance_matrix.json
Output: rl_routes_demo.json   (paste its contents back to Claude)
Run   : python export_rl_routes.py
"""

import json

from sb3_contrib import MaskablePPO

from warehouse_env_real import RealWarehouseEnv

DEMO_ORDERS = ['SO/26/10/0772', 'SO/26/10/0491']

env = RealWarehouseEnv('orders_test.json', min_stops=1)
model = MaskablePPO.load('models/ppo_best.zip')
out = {}
for so in DEMO_ORDERS:
    order = next(o for o in env.orders if o['so_no'] == so)
    obs, _ = env.reset(options={'order': order})
    done = False
    while not done:
        a, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
        obs, _, done, _, info = env.step(a)
    out[so] = {'route': info['route'], 'trip_m': info['trip_m'], 'stairs': info['stairs']}
    print(f"{so}: {info['trip_m']} m, {info['stairs']} stair trips, visits {' -> '.join(info['route'])}")
json.dump(out, open('rl_routes_demo.json', 'w'), indent=1)
print('Wrote rl_routes_demo.json')
