import sys
import time

print("STDOUT: Hello World")
sys.stderr.write("STDERR: Hello World\n")
sys.stdout.flush()
sys.stderr.flush()
time.sleep(1)
print("STDOUT: After 1s")
sys.stdout.flush()
