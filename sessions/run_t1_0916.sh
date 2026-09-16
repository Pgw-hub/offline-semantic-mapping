#!/bin/bash
cd /home/gw.park/offline_semantic_mapping/sessions || exit 1
for DS in t1_b1f2 t1_nearblock2; do echo "##### $DS $(date +%H:%M)"; python build_map_package.py session_$DS.yaml --steps A,S,R,B,C,D,E,F,G,H > ${DS}_build.out 2>&1 || echo "!!! $DS FAILED"; done; echo "##### ALL DONE $(date +%H:%M)"
