#!/usr/bin/env python3
"""Regenerate final training/evaluation summaries from the archived raw records."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, default=Path(__file__).resolve().parents[2] / 'records/hugebench_5ep_b512')
    args = parser.parse_args()
    root = args.records
    records = [json.loads(line) for line in (root/'training/metrics.jsonl').read_text().splitlines() if line.strip()]
    steps = np.array([r['step'] for r in records])
    loss = np.array([r['action_dit_loss'] for r in records])
    assert np.array_equal(steps, np.arange(1, 16799)), 'Expected every optimizer step, 1–16,798'
    assert np.isfinite(loss).all()
    summary = {'optimizer_steps': len(records), 'global_batch': 512, 'samples_seen': records[-1]['samples_seen'],
               'final_epoch': records[-1]['epoch'], 'first_loss': float(loss[0]), 'final_loss': float(loss[-1]),
               'last_100_mean_loss': float(loss[-100:].mean()), 'last_500_mean_loss': float(loss[-500:].mean()),
               'last_1000_mean_loss': float(loss[-1000:].mean()), 'minimum_loss': float(loss.min()),
               'maximum_loss': float(loss.max()), 'all_losses_finite': True,
               'loss_definition': 'Action flow-matching loss; averaged over 16 microbatches and all 4 ranks per optimizer step.',
               'smoothing': 'Trailing 100/500 optimizer-step means; first plotted at steps 100/500.',
               'validation_loss_available': False}
    (root/'training/summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    for ax in axes:
        ax.plot(steps, loss, color='#94a3b8', alpha=.5, lw=.4, label='Raw loss (all 16,798 updates)')
        for window, color in [(100, '#2563eb'), (500, '#dc2626')]:
            ax.plot(steps[window-1:], np.convolve(loss, np.ones(window)/window, mode='valid'),
                    color=color, lw=1.5, label=f'Trailing {window}-update mean')
        ax.set_xlabel('Optimizer update'); ax.set_ylabel('Training action loss'); ax.grid(alpha=.18)
    axes[0].legend(); axes[0].set_title('Qwen3-VL-2B + StarVLA-PI | HUGE-Bench | 5 epochs, batch 512')
    axes[1].set_xlim(1000, 16798); axes[1].set_ylim(0, float(loss[999:].max())*1.1)
    axes[1].set_title('Detail after update 1,000 (same data)')
    fig.savefig(root/'training/loss_curve.png', dpi=160); plt.close(fig)

    metrics = json.loads((root/'evaluation/metrics.json').read_text())
    episodes = metrics['episodes']
    assert len(episodes) == 993 and len({r['path'] for r in episodes}) == 993
    keys = ['tcr@1','tcr@2','tcr@5','avg_tcr','ndtw','nsp','sr','cr','cspl']
    for split, aggregate in metrics['by_split'].items():
        rows = [r for r in episodes if r['split'] == split]
        assert len(rows) == aggregate['n']
        for key in keys:
            assert np.isclose(np.mean([r[key] for r in rows]), aggregate[key], rtol=0, atol=1e-12), (split,key)
    with (root/'evaluation/per_episode_metrics.csv').open('w', newline='') as stream:
        fields = ['task','split','env_id','episode_path'] + keys + ['spl','gt_length','pred_length','num_gt','num_pred']
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n'); writer.writeheader()
        for row in episodes:
            writer.writerow({**{k:row[k] for k in fields if k!='episode_path'}, 'episode_path':row['path'].split('/rollouts/', 1)[1]})
    with (root/'evaluation/per_task_metrics.csv').open('w', newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['task','n']+keys,lineterminator='\n');writer.writeheader()
        for task, row in metrics['by_task'].items():writer.writerow({'task':task,**row})
    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    x = np.arange(len(keys))
    for delta, split, color in [(-.19,'seen','#2563eb'),(.19,'unseen','#f59e0b')]:
        ax.bar(x+delta, [metrics['by_split'][split][k]*100 for k in keys], .38, label=split, color=color)
    ax.set_xticks(x, ['TCR@1 ↑','TCR@2 ↑','TCR@5 ↑','Avg TCR ↑','nDTW ↑','NSP ↑','SR ↑','CR ↓','CSPL ↑'])
    ax.set_ylabel('Official metric × 100'); ax.set_ylim(0,100); ax.legend()
    ax.set_title('Final checkpoint | 576 seen + 417 unseen episodes | seed 42')
    fig.savefig(root/'evaluation/metrics.png', dpi=160); plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
