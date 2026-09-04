import numpy as np, struct, zlib, os, sys, glob, cv2
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
            if ch.topic=="/livox/lidar": lts.append(msg.log_time)
            else: rgbs.append((msg.log_time,msg.data))
    rt=np.array([t for t,_ in rgbs],dtype=np.int64)
    sel=sorted(set(int(np.argmin(np.abs(rt-lt))) for lt in lts))
    local2g={}
    for i in sel:
        t,d=rgbs[i]
        sec,nsec,w,h,ch_,rawlen,zlen=struct.unpack_from("<iIIIIII",d,0)
        img=np.frombuffer(zlib.decompress(d[28:28+zlen]),dtype=np.uint8).reshape(h,w,ch_)
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
