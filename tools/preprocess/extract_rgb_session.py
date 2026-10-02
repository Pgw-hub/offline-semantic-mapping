import numpy as np, struct, zlib, os, sys, glob, cv2
import sys as _rs, os as _ro; _rs.path.insert(0, _ro.path.join(_ro.path.dirname(_ro.path.abspath(__file__)), "../../geometry")); import raisin_msgs as RM   # 레코더 v1/v2 레이아웃 공용 디코더
from mcap.reader import make_reader
SESSION=sys.argv[1]; OUT=sys.argv[2]; LOGS=sys.argv[3]  # "0-10"
a,b=(int(x) for x in LOGS.split("-"))
os.makedirs(OUT+"/rgb",exist_ok=True)
mcaps=[m for m in sorted(glob.glob(SESSION+"/log_*.mcap"))
       if a<=int(os.path.basename(m).replace("log_","").replace(".mcap",""))<=b]
gidx=0; rgb_stamps=[]; lidar_stamps=[]; l2r=[]
for F in mcaps:
    lts=[]; rgbs=[]
    with open(F,"rb") as f:
        for sch,ch,msg in make_reader(f).iter_messages(topics=["/livox/lidar","d455_front/rgb"]):
            hs=RM.stamp_ns(msg.data,sch)        # 헤더 stamp (log_time 은 수신 벽시계 → sim 에선 헤더와 다른 시계)
            if ch.topic=="/livox/lidar": lts.append(hs)
            else: rgbs.append((hs,msg.data,sch))
    rt=np.array([t for t,_,_ in rgbs],dtype=np.int64)
    sel=sorted(set(int(np.argmin(np.abs(rt-lt))) for lt in lts))
    local2g={}
    for i in sel:
        t,d,sch_=rgbs[i]
        img=RM.rgb_image(d,sch_,order="bgr")[1]                     # cv2.imwrite 는 BGR
        cv2.imwrite(OUT+"/rgb/%06d.png"%gidx,img)
        local2g[i]=gidx; rgb_stamps.append(rgbs[i][0]); gidx+=1
    for lt in lts:
        i=int(np.argmin(np.abs(rt-lt))); lidar_stamps.append(lt); l2r.append(local2g[i])
    print("  %s: lidar %d, rgb %d (누적 %d)"%(os.path.basename(F),len(lts),len(sel),gidx))
np.savez(OUT+"/rgb_index.npz",
         rgb_stamp_ns=np.array(rgb_stamps,dtype=np.int64),
         lidar_stamp_ns=np.array(lidar_stamps,dtype=np.int64),
         lidar_to_rgb=np.array(l2r,dtype=np.int32))
print("총 RGB %d 장 -> %s"%(gidx,OUT))
