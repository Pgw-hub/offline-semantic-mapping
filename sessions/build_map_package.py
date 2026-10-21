#!/usr/bin/env python3
"""호환용: 코드는 scripts/build_map_package.py 로 옮겨졌다. `cd sessions && python build_map_package.py session_x.yaml` 습관을 그대로 쓰도록 넘겨만 준다."""
import os, runpy, sys
_P = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "build_map_package.py")
sys.argv[0] = _P; sys.path.insert(0, os.path.dirname(_P))
runpy.run_path(_P, run_name="__main__")
