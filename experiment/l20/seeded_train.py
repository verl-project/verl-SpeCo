"""Use identical rank-local COD random streams for the paired lifecycle runs."""

import os
import random

import torch

from verl_speco.draft_train import main

seed = 7 + int(os.environ["RANK"])
random.seed(seed)
torch.manual_seed(seed)
main()
