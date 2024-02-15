import os
import random
import time
import torch
import numpy as np
import ray
from worker_dueling import GlobalBuffer, Learner, Actor
import configs_D3QTP

os.environ["OMP_NUM_THREADS"] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
# os.environ["MODIN_ENGINE"] = "ray"
# runtime_env = {
#     'env_vars': {
#         "RAY_memory_monitor_refresh_ms": "0"
#      }
# }
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


# run = neptune.init_run(
#     project="yuting/DHC",
#     api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmOWRlMmNmNy00ZjI1LTRmN2ItYWUyNi0yMzEwNzYxZThiMDQifQ==",
# )
# params = {
#     "lr": 1e-4,
#     "bs": 192,
# }
# run["parameters"] = params

def main(num_actors=configs_D3QTP.num_actors, log_interval=configs_D3QTP.log_interval):
    ray.init()

    buffer = GlobalBuffer.remote()
    learner = Learner.remote(buffer)
    time.sleep(1)
    actors = [Actor.remote(i, 0.4, learner, buffer) for i in range(num_actors)]

    for actor in actors:
        # actor.run.remote()
        print('actor run')
        actor.run.remote()

    print('exit iteration')
    while not ray.get(buffer.ready.remote()):
        time.sleep(5)
        ray.get(learner.stats.remote(5))
        ray.get(buffer.stats.remote(5))

    print('start training')
    buffer.run.remote()
    learner.run.remote()

    done = False
    while not done:
        time.sleep(log_interval)
        done = ray.get(learner.stats.remote(log_interval))
        ray.get(buffer.stats.remote(log_interval))
        print()
        print(done)


if __name__ == '__main__':
    main()
