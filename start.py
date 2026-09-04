#!/usr/bin/env python3
"""
TideTrace Launcher.
Oil Spill Detection, Drift Analysis & Vessel Attribution Platform
SIH Problem #26143 (NTRO)
"""
import sys
import os

# Set UTF-8 encoding for standard output if supported
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    import uvicorn
    print("===================================================================")
    print("                           TIDETRACE                              ")
    print("   Oil Spill Detection, Drift Analysis & Vessel Attribution Platform    ")
    print("                 NTRO Problem Statement #26143                    ")
    print("===================================================================")
    print("  Operating Modes Supported:")
    print("   [1] INVESTIGATION  - Detect spill -> reconstruct origin -> correlate")
    print("   [2] RESPONSE       - Forecast movement -> coastal impact ETA -> risk")
    print("   [3] SURVEILLANCE   - Detect SAR ships -> cross-check AIS -> dark vessels")
    print("===================================================================")
    print(" Starting API & Operations Dashboard on http://localhost:8000 ...")
    print(" Open http://localhost:8000 in your browser to view the Command Center.")
    print("===================================================================")
    
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
