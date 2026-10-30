from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]

POLICIES = [
    "stc",
    "alrajhi",
    "jarir",
    "hungerstation",
    "tawakkalna",
    "absher",
    "mobily",
    "bankalbilad",
    "tamara",
    "nahdi",
]

for p in POLICIES:
    print("\n" + "#"*80)
    print("RUNNING:", p)
    print("#"*80)
    cmd = [sys.executable, str(ROOT / "src" / "gap_analyzer" / "run_policy_analysis_v2.py"), p]
    subprocess.run(cmd, check=True)

print("\nAll 10 policy analyses completed.")
print("Now run:")
print('python ".\\src\\gap_analyzer\\evaluate_all_policies_v2.py"')
