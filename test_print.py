# Copyright (c) 2022-2026, ggSwarm Developers.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

import sys
import time

print("STDOUT: Hello World")
sys.stderr.write("STDERR: Hello World\n")
sys.stdout.flush()
sys.stderr.flush()
time.sleep(1)
print("STDOUT: After 1s")
sys.stdout.flush()
