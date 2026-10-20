#!/usr/bin/env python3
"""Standalone, bounded FLIR AprilTag capture; no robot dependency or commands."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

# Optional extracted Aravis runtime; set environment before loading native libs.
if os.environ.get('FLIR_ARAVIS_RUNTIME') and not os.environ.get('_FLIR_RUNTIME_READY'):
    lib=Path(os.environ['FLIR_ARAVIS_RUNTIME']).resolve()
    if not lib.is_dir():
        raise SystemExit('FLIR_ARAVIS_RUNTIME must name the directory containing libaravis')
    env=os.environ.copy()
    for key,value in [('LD_LIBRARY_PATH',str(lib)),('GI_TYPELIB_PATH',str(lib/'girepository-1.0'))]:
        env[key]=value+(os.pathsep+env[key] if env.get(key) else '')
    env['_FLIR_RUNTIME_READY']='1'
    os.execve(sys.executable,[sys.executable,*sys.argv],env)

import cv2
import numpy as np
from camera import Camera, SERIAL, atomic_json, import_aravis
from marker_pose import FAMILIES, TagDetector, load_calibration, project

HERE=Path(__file__).resolve().parent


def utc():
    return datetime.now(timezone.utc).isoformat()


def selected_pose(detector, gray, tag_id, k, d, model, rms, gap):
    tags=[t for t in detector.detect(gray,k,d,model,rms,gap) if t['tag_id']==tag_id]
    if not tags:
        return {'valid':False,'reason':'TAG NOT FOUND','tag_id':tag_id}
    if len(tags)>1:
        return {'valid':False,'reason':'DUPLICATE TAG ID','tag_id':tag_id,'count':len(tags)}
    return tags[0]


def overlay(image, pose, k, d, model, size):
    display=image.copy()
    color=(70,220,70) if pose['valid'] else (0,180,255)
    if 'corners_px' in pose:
        cv2.polylines(display,[np.rint(pose['corners_px']).astype(np.int32)],True,color,3)
    if pose['valid']:
        obj=np.array([[0.,0,0],[size*.3,0,0],[0,size*.3,0],[0,0,size*.3]])
        uv=project(obj,np.array(pose['rvec']),np.array(pose['tvec_m']),k,d,model)
        if np.isfinite(uv).all() and np.max(np.abs(uv))<1e6:
            uv=np.rint(uv).astype(int)
            for j,c in enumerate([(0,0,255),(0,255,0),(255,0,0)],1):
                cv2.line(display,tuple(uv[0]),tuple(uv[j]),c,4)
        label=f"tag {pose['tag_id']}  xyz(m)="+str([round(v,3) for v in pose['tvec_m']])
    else:
        label=pose['reason']
    cv2.putText(display,label,(25,50),cv2.FONT_HERSHEY_SIMPLEX,1,color,2)
    return display


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    mode=ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--list',action='store_true',help='Discover cameras; no acquisition')
    mode.add_argument('--live',action='store_true',help='Capture after physical readiness and optics checks')
    mode.add_argument('--image',type=Path,help='Offline full-resolution image; no camera access')
    ap.add_argument('--camera',help='Device ID from --list, or current camera IPv4 address')
    ap.add_argument('--calibration',type=Path,default=HERE/'calibration.json')
    ap.add_argument('--tag-id',type=int,default=7)
    ap.add_argument('--tag-family',choices=FAMILIES,default='tag36h11')
    ap.add_argument('--tag-size-m',type=float,default=.2,help='Outside black square, excluding white margin')
    ap.add_argument('--seconds',type=float,default=300,help='Bounded live runtime, maximum 3600 s')
    ap.add_argument('--fps',type=float,default=12)
    ap.add_argument('--exposure-us',type=float,default=20000)
    ap.add_argument('--gain-db',type=float)
    ap.add_argument('--max-rms-px',type=float,default=2.)
    ap.add_argument('--min-gap-px',type=float,default=.1)
    ap.add_argument('--preview',action='store_true',help='Optional desktop preview; Q/Esc stops')
    ap.add_argument('--out',type=Path,default=Path('runs')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ'))
    args=ap.parse_args()
    if args.list:
        A=import_aravis(); A.update_device_list()
        for i in range(A.get_n_devices()):
            print(json.dumps({'device_id':A.get_device_id(i),'serial':A.get_device_serial_nbr(i),
                              'model':A.get_device_model(i),'address':A.get_device_address(i)}))
        return 0
    for key in ['tag_size_m','seconds','fps','exposure_us','max_rms_px','min_gap_px']:
        if not math.isfinite(getattr(args,key)) or getattr(args,key)<=0:
            ap.error(key+' must be finite and positive')
    if args.gain_db is not None and not math.isfinite(args.gain_db):ap.error('gain-db must be finite')
    if args.seconds>3600:ap.error('seconds must be <=3600')
    if args.tag_id<0:ap.error('tag-id must be nonnegative')
    if args.live and not args.camera:ap.error('--live requires --camera from discovery')
    if args.preview and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        ap.error('--preview requires a desktop display')
    cal,k,d,model=load_calibration(args.calibration)
    if args.live and (cal.get('serial')!=SERIAL or cal['image_size']!=[2448,2048]):
        ap.error('Live adapter requires calibrated serial19476401 at2448x2048')
    detector=TagDetector(args.tag_family,args.tag_size_m)
    args.out.mkdir(parents=True,exist_ok=False)
    atomic_json(args.out/'calibration.json',cal)
    atomic_json(args.out/'run.json',{'started_utc':utc(),'parameters':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                                   'opencv':cv2.__version__,'numpy':np.__version__,
                                   'timestamp_kind':'host processing time, not synchronized sensor time'})
    stopping=False
    def stop(_sig,_frame):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    cam=lock=None;frames=valid=0;failure=None;restore_errors=[]
    start=last_frame=last_temp=time.monotonic()
    try:
        if args.live:
            lock=open('/tmp/flir-roi-sweep-'+SERIAL+'.lock','a')
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            settings=args.out/'settings';settings.mkdir()
            cam=Camera(args,settings);cam.open()
        if args.preview:cv2.namedWindow('FLIR marker pose',cv2.WINDOW_NORMAL)
        with (args.out/'poses.jsonl').open('w',buffering=1) as log:
            while not stopping and time.monotonic()-start<args.seconds:
                if args.live:
                    raw=cam.read()
                    if time.monotonic()-last_temp>=1:
                        cam.temperature=cam.c.get_float('DeviceTemperature');last_temp=time.monotonic()
                        if not math.isfinite(cam.temperature) or cam.temperature>=70:
                            raise RuntimeError('Temperature stop: invalid reading or >=70 C')
                    if raw is None:
                        atomic_json(args.out/'latest-pose.json',{'time_utc':utc(),'valid':False,'reason':'NO CURRENT FRAME'})
                        if time.monotonic()-last_frame>5:raise RuntimeError('No complete frame for five seconds')
                        continue
                    image=cv2.cvtColor(raw,cv2.COLOR_BayerRGGB2BGR)
                else:
                    image=cv2.imread(str(args.image))
                    if image is None:raise ValueError('Cannot read image '+str(args.image))
                last_frame=time.monotonic()
                if [image.shape[1],image.shape[0]]!=cal['image_size']:
                    raise ValueError('Image dimensions do not match calibration; no implicit resizing')
                pose=selected_pose(detector,cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),args.tag_id,k,d,model,args.max_rms_px,args.min_gap_px)
                record={'time_utc':utc(),'frame_id':cam.last_frame_id if cam else None,**pose}
                log.write(json.dumps(record,allow_nan=False)+'\n');atomic_json(args.out/'latest-pose.json',record)
                print(json.dumps(record,allow_nan=False),flush=True);frames+=1;valid+=int(pose['valid'])
                if args.preview or not args.live:
                    shown=overlay(image,pose,k,d,model,args.tag_size_m)
                    if not args.live:cv2.imwrite(str(args.out/'annotated.png'),shown)
                    if args.preview:
                        scale=min(1.,1280/shown.shape[1]);cv2.imshow('FLIR marker pose',cv2.resize(shown,None,fx=scale,fy=scale))
                        if cv2.waitKey(1)&255 in (27,ord('q')):break
                        try:
                            if cv2.getWindowProperty('FLIR marker pose',cv2.WND_PROP_VISIBLE)<1:break
                        except cv2.error:break
                if not args.live:break
    except Exception as exc:
        failure=f'{type(exc).__name__}: {exc}'
        print(failure,file=sys.stderr)
    finally:
        if cam:
            try:restore_errors=cam.close()
            except Exception as exc:restore_errors=[str(exc)]
        if lock:lock.close()
        if args.preview:
            try:cv2.destroyAllWindows()
            except cv2.error:pass
        atomic_json(args.out/'latest-pose.json',{'time_utc':utc(),'valid':False,'reason':'STOPPED','error':failure})
        atomic_json(args.out/'status.json',{'finished_utc':utc(),'state':'failed' if failure or restore_errors else 'stopped',
                                          'frames':frames,'valid_poses':valid,'error':failure,'restore_errors':restore_errors})
        if restore_errors:print('SETTING RESTORATION ERRORS: '+str(restore_errors),file=sys.stderr)
    return 1 if failure or restore_errors else 0


if __name__=='__main__':
    raise SystemExit(main())
