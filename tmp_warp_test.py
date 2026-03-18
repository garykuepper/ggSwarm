# Copyright (c) 2022-2026, ggSwarm Developers.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

import torch
import warp as wp

wp.init()

# Case 1: (4, 1, 3) -> vec3f
t1 = torch.zeros((4, 1, 3), device="cuda")
a1 = wp.from_torch(t1, dtype=wp.vec3f)
print(f"Case 1: (4, 1, 3) with vec3f -> shape: {a1.shape}, ndim: {a1.ndim}")

# Case 1b: (4, 2, 3) -> vec3f
t1b = torch.zeros((4, 2, 3), device="cuda")
a1b = wp.from_torch(t1b, dtype=wp.vec3f)
print(f"Case 1b: (4, 2, 3) with vec3f -> shape: {a1b.shape}, ndim: {a1b.ndim}")

# Case 2: (4, 3) -> vec3f
t2 = torch.zeros((4, 3), device="cuda")
a2 = wp.from_torch(t2, dtype=wp.vec3f)
print(f"Case 2: (4, 3) with vec3f -> shape: {a2.shape}, ndim: {a2.ndim}")
