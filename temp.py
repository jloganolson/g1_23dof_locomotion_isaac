import os
import time
print("\n⏳ Waiting 5 seconds before suspend...")
time.sleep(5)
import subprocess
print("💤 Suspending computer...")
subprocess.run(["systemctl", "suspend"], check=True)