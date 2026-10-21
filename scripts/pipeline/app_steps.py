"""Application 블록: 산출물(매니페스트)만 읽는 응용. 지금은 코스트맵 변환 — Map package(G) 작성 직후 호출된다."""
import glob, json, os, shutil, subprocess, sys, time
import yaml

from pipeline.env import CODE, REPO, PY, CGREPO, HF_HOME, CD


class AppSteps:
    def app_costmap(self, man):
        """0.2 m 점유 레이어 + semantic 복셀(매니페스트) → nav2 등급 코스트맵, 매니페스트에 costmap_2d_occ 등록."""
        cm = self.opt["costmap"]; out = os.path.join(self.D["compiled"], "session_costmap_occ")
        if self.geo_only: self.say("  기하 전용 → 코스트맵 생략")
        elif not self.done("G", out + ".npz"):
            self.run([PY, f"{CODE}/apps/costmap_from_occupancy.py", man, out, "--register", "--h-ground", str(cm["h_ground"]), "--h-obs", str(cm["h_obs"]),
                      "--clearance", str(cm["clearance"]), "--band-min", str(cm["band_min"]), "--ground-label", str(cm.get("ground_label", "ground")), "--ground-min", str(cm.get("ground_min", 0.3))],
                     env={"CUDA_VISIBLE_DEVICES": ""}, logname="G_costmap.log")   # ground_label/ground_min: 지면 셀 라벨 = 지면 역할 클래스 argmax (costmap_from_occupancy)
