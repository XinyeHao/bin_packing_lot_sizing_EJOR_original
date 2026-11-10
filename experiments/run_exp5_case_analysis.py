"""Entry point reserved for real-case analysis.

It deliberately refuses to label generated random instances as a case study.
Provide one JSON file per real planning period in the InstanceData.to_dict()
format, then this script can be extended to solve IM/BP/BPC and export Gantt
data.  Random benchmark experiments belong to E1--E4.
"""
from __future__ import annotations
import argparse
from pathlib import Path

parser=argparse.ArgumentParser(description="Real-case analysis data gate")
parser.add_argument("--instances-dir",type=Path,required=True,help="Directory containing real InstanceData JSON files.")
args=parser.parse_args()
files=sorted(args.instances_dir.glob("*.json")) if args.instances_dir.exists() else []
if not files:
    raise SystemExit("No real InstanceData JSON files found. Random seeds must not be reported as a real case study.")
print(f"Found {len(files)} real-case input files. Next step: validate each input and run IM/BP/BPC.")
