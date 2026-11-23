"""
Run both occlusion_analysis.py and pf_analysis.py to update both output folders cleanly.
"""
import subprocess
import sys

print("=== Running occlusion_analysis.py ===")
r1 = subprocess.run([sys.executable, "E:/Antigravity/TMRB/occlusion_analysis.py"], capture_output=True, text=True)
print("Return code:", r1.returncode)
print("Stdout tail:\n", "\n".join(r1.stdout.splitlines()[-15:]))
if r1.returncode != 0:
    print("Stderr tail:\n", "\n".join(r1.stderr.splitlines()[-15:]))

print("\n=== Running pf_analysis.py ===")
r2 = subprocess.run([sys.executable, "E:/Antigravity/TMRB/pf_analysis.py"], capture_output=True, text=True)
print("Return code:", r2.returncode)
print("Stdout tail:\n", "\n".join(r2.stdout.splitlines()[-15:]))
if r2.returncode != 0:
    print("Stderr tail:\n", "\n".join(r2.stderr.splitlines()[-15:]))
