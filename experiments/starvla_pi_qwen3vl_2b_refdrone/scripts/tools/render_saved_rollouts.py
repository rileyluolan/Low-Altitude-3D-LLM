#!/usr/bin/env python3
"""Replay saved HUGE-Bench GT/prediction trajectories into videos; no policy inference."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from experiment_paths import ROOT, WORKSPACE, HUGE_DATA as DATA, GAUSSIAN


def export_path(run, name):
    return run / 'logs' / f'video_export_{name}'


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_video(path, expected_frames, fps):
    import av
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.frames != expected_frames:
            raise ValueError(f'Video frame count mismatch: {path}: {stream.frames} != {expected_frames}')
        if abs(float(stream.average_rate) - fps) > 1e-6:
            raise ValueError(f'Video FPS mismatch: {path}')
        frame = next(container.decode(video=0))
        if frame.width < 2 or frame.height < 2:
            raise ValueError(f'Invalid video dimensions: {path}')


def completed(job, angle_mode):
    npz = Path(job['path'])
    output = npz.with_name('gt_pred_side_by_side.mp4')
    record = npz.with_name('VIDEO_COMPLETE.json')
    if not (output.is_file() and record.is_file()):
        return False
    data = json.loads(record.read_text())
    expected = {'source_sha256':job['sha256'], 'frames':job['frames'], 'fps':job['fps'],
                'obstacle_angle_mode':angle_mode, 'video_bytes':output.stat().st_size,
                'internal_scale':0.065, 'version':1}
    if any(data.get(k) != v for k,v in expected.items()):
        raise RuntimeError(f'Existing video metadata differs: {output}')
    verify_video(output, job['frames'], job['fps'])
    return True


def make_frame(gt_rgb, pred_rgb, job, step, error, prompt, font):
    h, w = gt_rgb.shape[:2]
    if pred_rgb.shape != gt_rgb.shape:
        raise ValueError('GT and prediction render sizes differ')
    width = ((2*w + 1)//2)*2
    height = ((h+77)//2)*2
    canvas = Image.new('RGB',(width,height),(20,27,38))
    canvas.paste(Image.fromarray(gt_rgb),(0,26))
    canvas.paste(Image.fromarray(pred_rgb),(w,26))
    draw = ImageDraw.Draw(canvas)
    draw.text((8,5),'Ground truth',font=font,fill=(119,216,255))
    draw.text((w+8,5),'Model prediction',font=font,fill=(255,183,95))
    text = f'{job["split"]} | task={job["task"]} | ep={job["episode"]} | step {step+1}/{job["frames"]} | error={error:.1f} m'
    draw.text((7,h+31),text,font=font,fill='white')
    # Include the instruction without changing the field of view of either render.
    while prompt and draw.textlength(prompt,font=font)>width-14:
        prompt=prompt[:-5].rstrip()+ '...'
    draw.text((7,h+50),prompt,font=font,fill=(205,218,232))
    return np.asarray(canvas)


def stop(proc):
    if proc is None or proc.poll() is not None:
        return
    os.killpg(proc.pid,signal.SIGTERM)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid,signal.SIGKILL)
        proc.wait()


def worker(args):
    from hugebench_io import RenderClient, convert_state_for_render
    from matplotlib.font_manager import findfont
    manifest=json.loads(export_path(args.eval_dir,'manifest.json').read_text())
    jobs=manifest['shards'][str(args.worker)]
    settings=manifest['settings']
    mode=settings['obstacle_angle_mode']
    gpu=settings['gpus'][args.worker]
    port=settings['port_start']+args.worker
    state_path=export_path(args.eval_dir,f'worker_{args.worker}.json')
    font=ImageFont.truetype(findfont('DejaVu Sans'),12)
    process=None
    client=None
    start=time.monotonic()
    done_frames=0
    done_episodes=0
    def update(**extra):
        atomic_json(state_path,{'worker':args.worker,'gpu':gpu,'done_episodes':done_episodes,
                    'total_episodes':len(jobs),'done_frames':done_frames,
                    'total_frames':sum(j['frames'] for j in jobs),
                    'elapsed_seconds':time.monotonic()-start,
                    'updated_utc':datetime.now(timezone.utc).isoformat(),**extra})
    def interrupted(signum,frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted)
    log=export_path(args.eval_dir,f'renderer_{args.worker}.log').open('ab',buffering=0)
    try:
        update(status='starting')
        for job in jobs:
            if completed(job,mode):
                done_episodes+=1;done_frames+=job['frames']
                update(status='running',episode=job['episode'],split=job['split'],skipped_existing=True)
                continue
            if process is None:
                env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu
                process=subprocess.Popen([sys.executable,str(GAUSSIAN/'3dgs_renderer.py'),
                    '--host','127.0.0.1','--port',str(port),'--internal_scale','0.065',
                    '--ply_template',str(DATA/'data_3d/{env_id}/3dgs_ply/point_cloud_utm50.ply')],
                    cwd=GAUSSIAN,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                client=RenderClient('127.0.0.1',port)
            if process.poll() is not None:
                raise RuntimeError(f'Renderer stopped: worker {args.worker}')
            source=Path(job['path'])
            if digest(source)!=job['sha256']:
                raise ValueError(f'Rollout changed: {source}')
            with np.load(source) as data:
                gt=np.array(data['gt_xyzk']);pred=np.array(data['pred_xyzk'])
            final=source.with_name('gt_pred_side_by_side.mp4')
            temp=source.with_name('gt_pred_side_by_side.partial.mp4')
            instruction=source.with_name('instruction.txt').read_text().splitlines()
            prompt=next((s[len('prompt: '):] for s in instruction if s.startswith('prompt: ')), '')
            update(status='rendering',episode=job['episode'],split=job['split'],env_id=job['env'],task=job['task'],current_frame=0,current_total=len(gt))
            with imageio.get_writer(str(temp),fps=job['fps'],codec='libx264',quality=6,pixelformat='yuv420p',
                    macro_block_size=2,ffmpeg_params=['-movflags','+faststart','-threads','2']) as writer:
                for i in range(len(gt)):
                    left=client.render(convert_state_for_render(gt[i],job['task'],mode),export_path(args.eval_dir,f'gt_{args.worker}.png'),job['env'],job['task'],i)
                    right=client.render(convert_state_for_render(pred[i],job['task'],mode),export_path(args.eval_dir,f'pred_{args.worker}.png'),job['env'],job['task'],i)
                    error=float(np.linalg.norm(gt[i,:3]-pred[i,:3]))
                    writer.append_data(make_frame(left,right,job,i,error,prompt,font))
                    if i%25==0:
                        update(status='rendering',episode=job['episode'],split=job['split'],env_id=job['env'],task=job['task'],current_frame=i+1,current_total=len(gt))
            verify_video(temp,job['frames'],job['fps'])
            temp.replace(final)
            atomic_json(source.with_name('VIDEO_COMPLETE.json'),{
                'source_sha256':job['sha256'],'frames':job['frames'],'fps':job['fps'],
                'obstacle_angle_mode':mode,'internal_scale':0.065,'version':1,
                'video_bytes':final.stat().st_size,'video_sha256':digest(final),
                'source':'Post-hoc replay of the saved full evaluation trajectories; no model rerun',
                'completed_utc':datetime.now(timezone.utc).isoformat()})
            done_episodes+=1;done_frames+=job['frames']
            update(status='running',episode=job['episode'],split=job['split'],current_frame=0)
            print(f'GPU {gpu}: {done_episodes}/{len(jobs)} videos; {done_frames:,} frames; {final}',flush=True)
        update(status='complete',current_frame=0)
    except BaseException as exc:
        update(status='failed',error=repr(exc))
        raise
    finally:
        if client is not None:
            try: client.close()
            except Exception: pass
        stop(process);log.close()


def main(args):
    run=args.eval_dir
    if not (run/'logs').is_dir():raise ValueError('Existing evaluation logs directory is required')
    config=json.loads((run/'run_config.json').read_text())
    gpus=args.gpus.split(',')
    if len(set(gpus))!=len(gpus):raise ValueError('GPU IDs must be distinct')
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    free={x.split(',')[0].strip():int(x.split(',')[1]) for x in raw.splitlines()}
    for gpu in gpus:
        if free.get(gpu,0)<60000:raise RuntimeError(f'GPU {gpu} has insufficient free memory')
    for port in range(args.port_start,args.port_start+len(gpus)):
        with socket.socket() as sock:sock.bind(('127.0.0.1',port))
    jobs=[]
    fps={s:float(json.loads((DATA/'data_traj'/s/'meta/info.json').read_text())['fps']) for s in ['test_seen','test_unseen']}
    for source in sorted((run/'rollouts').rglob('traj_gt_pred_xyzk.npz')):
        task,split,env,episode=source.relative_to(run/'rollouts').parts[:4]
        with np.load(source) as data:
            gt=data['gt_xyzk'];pred=data['pred_xyzk']
            if gt.shape!=pred.shape or not np.isfinite(gt).all() or not np.isfinite(pred).all():raise ValueError(source)
            frames=len(gt)
        jobs.append({'path':str(source),'task':task[5:],'split':split,'env':env,'episode':int(episode.split('_')[1]),
                     'frames':frames,'fps':fps[split],'sha256':digest(source)})
    counts={s:sum(x['split']==s for x in jobs) for s in fps}
    if counts!={'test_seen':576,'test_unseen':417}:raise ValueError(f'Expected all 993 episodes: {counts}')
    # Balance render frame counts, then group each GPU's episodes by scene to avoid repeated scene loading.
    shards=[[] for _ in gpus];weights=[0]*len(gpus)
    for job in sorted(jobs,key=lambda x:x['frames'],reverse=True):
        index=min(range(len(gpus)),key=lambda i:weights[i])
        shards[index].append(job);weights[index]+=job['frames']
    for shard in shards:shard.sort(key=lambda x:(x['env'],x['task'],x['split'],x['episode']))
    manifest={'settings':{'fps':fps,'gpus':gpus,'port_start':args.port_start,'obstacle_angle_mode':config['obstacle_angle_mode'],
                'internal_scale':0.065,'video_format':'H.264/yuv420p; GT left, prediction right; labels and instruction',
                'original_checkpoint':config['checkpoint'],'total_episodes':len(jobs),'total_frames':sum(weights)},
              'shards':{str(i):v for i,v in enumerate(shards)}}
    path=export_path(run,'manifest.json')
    if path.exists() and json.loads(path.read_text())!=manifest:raise RuntimeError('Existing video export configuration differs')
    atomic_json(path,manifest)
    handles=[];procs=[]
    signal.signal(signal.SIGTERM,lambda *unused:(_ for _ in ()).throw(KeyboardInterrupt()))
    started=time.monotonic()
    try:
        for i in range(len(gpus)):
            log=export_path(run,f'worker_{i}.log').open('ab',buffering=0);handles.append(log)
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
            procs.append(subprocess.Popen([sys.executable,__file__,'--eval-dir',str(run),'--worker',str(i)],
                env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
        while True:
            states=[]
            for i in range(len(gpus)):
                p=export_path(run,f'worker_{i}.json')
                if p.exists():states.append(json.loads(p.read_text()))
            done=sum(s['done_episodes'] for s in states)
            frames=sum(s['done_frames']+s.get('current_frame',0) for s in states)
            pct=100*frames/max(sum(weights),1)
            text=f'Videos {done}/993 | frames {frames:,}/{sum(weights):,} ({pct:.1f}%) | elapsed {(time.monotonic()-started)/60:.1f} min'
            print('\r'+text+' '*10,end='',flush=True)
            atomic_json(export_path(run,'status.json'),{'done_videos':done,'total_videos':len(jobs),'done_frames_including_current':frames,
                        'total_frames':sum(weights),'status':'running','updated_utc':datetime.now(timezone.utc).isoformat(),'workers':states})
            for i,p in enumerate(procs):
                if p.poll() not in (None,0):raise RuntimeError(f'Video worker {i} failed; see {export_path(run,f"worker_{i}.log")}')
            if all(p.poll()==0 for p in procs):break
            time.sleep(5)
        for job in jobs:
            if not completed(job,config['obstacle_angle_mode']):raise RuntimeError(f'Missing completed video: {job["path"]}')
        export_path(run,'COMPLETE').write_text('993 full-length videos; verified frame counts and 5 FPS.\n')
        atomic_json(export_path(run,'status.json'),{'done_videos':len(jobs),'total_videos':len(jobs),'status':'complete',
                    'updated_utc':datetime.now(timezone.utc).isoformat(),'total_frames':sum(weights)})
        print('\nAll 993 full-length videos exported and verified.',flush=True)
    except BaseException as exc:
        atomic_json(export_path(run,'status.json'),{'status':'failed','error':repr(exc),
                    'updated_utc':datetime.now(timezone.utc).isoformat()})
        raise
    finally:
        for p in procs:stop(p)
        for h in handles:h.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval-dir',type=Path,required=True)
    parser.add_argument('--gpus',default='0,1,2,3')
    parser.add_argument('--port-start',type=int,default=5580)
    parser.add_argument('--worker',type=int)
    args=parser.parse_args();args.eval_dir=args.eval_dir.resolve()
    if not args.eval_dir.is_relative_to(WORKSPACE):parser.error('Evaluation directory must be inside workspace')
    if args.worker is not None:worker(args)
    else:main(args)
