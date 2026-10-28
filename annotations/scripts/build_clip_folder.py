#!/usr/bin/env python3
"""Gather the main-task clips and the training (gold/example) clips into one flat folder.

The training clips in gold_standards.csv use the old names, e.g.
    00680001_2023-06-03_3_9d5449d617_processed_012.mp4
and are mapped to the 2026.1 10s clips, e.g.
    S00680001/S00680001_2023-06-03_3_recL5nH9ztZzGJgKs_chunk0012of0027.mp4
using recordings_processed.csv (superseded_gcp_name_feb25 -> unique_video_id). Clip indices are
0-based in both schemes, so processed_012 is chunk0012.

Main clips are drawn uniformly at random (fixed seed) from all clips, excluding any recording that
a training clip comes from.

Outputs, all in --out:
    *.mp4                  the clips (copied, so the folder can be moved to another server as-is)
    main_clips.csv         the main-task clips (use as videos.mainListFile)
    training_clips.csv     old -> new name for every training clip
and a copy of gold_standards.csv with video_filename switched to the new names (--gold-out).

Standard library only, so it runs with the cluster's system python3.
"""

import argparse
import csv
import os
import random
import re
import shutil
import sys
from pathlib import Path

OLD_CLIP = re.compile(r'^(\d{8})_(\d{4}-\d{2}-\d{2})_(\d+)_([0-9a-f]+)_processed_(\d+)\.mp4$')
NEW_CLIP = re.compile(r'^(S\d{8})_(\d{4}-\d{2}-\d{2})_(\d+)_(rec\w+?)(_rotated)?_chunk(\d{4})of(\d{4})\.mp4$')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', required=True, help='10s clip root with one folder per subject (S00680001/...)')
    p.add_argument('--recordings', required=True, help='recordings_processed.csv')
    p.add_argument('--gold', required=True, help='gold_standards.csv with old clip names')
    p.add_argument('--out', required=True, help='output folder for the clips and manifests')
    p.add_argument('--gold-out', help='where to write gold_standards.csv with new names (default: <out>/gold_standards.csv)')
    p.add_argument('--n-main', type=int, default=200)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--dry-run', action='store_true', help='report what would be copied without copying')
    return p.parse_args()


def video_key(subject, date, n, rec):
    return f'{subject}_{date}_{n}_{rec}'


def main():
    args = parse_args()
    source, out = Path(args.source), Path(args.out)
    gold_out = Path(args.gold_out) if args.gold_out else out / 'gold_standards.csv'

    # Old video name (superseded_gcp_name_feb25) -> new video key (S<subject>_<date>_<n>_<rec id>)
    old_to_new = {}
    with open(args.recordings, newline='') as f:
        for row in csv.DictReader(f):
            old = row['superseded_gcp_name_feb25']
            m = re.match(r'^(\d{8})_(\d{4}-\d{2}-\d{2})_(\d+)_[0-9a-f]+$', old or '')
            if m and row['unique_video_id'].startswith('rec'):
                old_to_new[old] = video_key('S' + m.group(1), m.group(2), m.group(3), row['unique_video_id'])

    # Index every clip: (video key, chunk index) -> filename; also the full list for sampling
    print(f'Indexing clips in {source} ...', file=sys.stderr)
    clips = {}
    for subject_dir in sorted(d for d in source.iterdir() if d.is_dir()):
        with os.scandir(subject_dir) as entries:
            for e in entries:
                m = NEW_CLIP.match(e.name)
                if m:
                    clips[(video_key(*m.group(1, 2, 3, 4)), int(m.group(6)))] = e.path
    print(f'  {len(clips)} clips in {len({k for k, _ in clips})} recordings', file=sys.stderr)

    # Training clips
    with open(args.gold, newline='') as f:
        reader = csv.DictReader(f)
        gold_fields = reader.fieldnames
        gold_rows = list(reader)

    training, missing = {}, []
    for row in gold_rows:
        old_name = row['video_filename'].strip()
        if old_name in training:
            continue
        m = OLD_CLIP.match(old_name)
        new_video = old_to_new.get('_'.join(m.group(1, 2, 3, 4))) if m else None
        path = clips.get((new_video, int(m.group(5)))) if new_video else None
        if path:
            training[old_name] = path
        else:
            missing.append((old_name, 'not in recordings_processed.csv' if m and not new_video
                            else 'unrecognized name' if not m else 'chunk not found in source'))

    training_videos = {NEW_CLIP.match(os.path.basename(p)).group(0).split('_chunk')[0] for p in training.values()}
    training_videos = {re.sub(r'_rotated$', '', v) for v in training_videos}

    # Main clips: uniform random sample, excluding recordings used for training clips
    pool = sorted(path for (video, _), path in clips.items() if video not in training_videos)
    if len(pool) < args.n_main:
        sys.exit(f'Only {len(pool)} eligible clips; cannot sample {args.n_main}')
    main_clips = sorted(random.Random(args.seed).sample(pool, args.n_main))

    print(f'Training clips: {len(training)} found, {len(missing)} missing', file=sys.stderr)
    for name, reason in missing:
        print(f'  MISSING {name}: {reason}', file=sys.stderr)
    print(f'Main clips: {len(main_clips)} sampled (seed {args.seed}) from {len(pool)} eligible', file=sys.stderr)

    if args.dry_run:
        return

    out.mkdir(parents=True, exist_ok=True)
    for path in list(training.values()) + main_clips:
        dest = out / os.path.basename(path)
        if not dest.exists():
            shutil.copy2(path, dest)

    def describe(path):
        m = NEW_CLIP.match(os.path.basename(path))
        return {'video_filename': os.path.basename(path), 'subject': m.group(1),
                'recording': os.path.basename(path).split('_chunk')[0], 'chunk': int(m.group(6)),
                'source_path': path}

    with open(out / 'main_clips.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['video_filename', 'subject', 'recording', 'chunk', 'source_path'], lineterminator='\n')
        w.writeheader()
        w.writerows(describe(p) for p in main_clips)

    with open(out / 'training_clips.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['old_video_filename', 'video_filename', 'source_path', 'status'], lineterminator='\n')
        w.writeheader()
        for old, path in training.items():
            w.writerow({'old_video_filename': old, 'video_filename': os.path.basename(path),
                        'source_path': path, 'status': 'found'})
        for old, reason in missing:
            w.writerow({'old_video_filename': old, 'video_filename': '', 'source_path': '', 'status': reason})

    # gold_standards.csv with new names; the old name is kept in old_video_filename
    fields = ['video_filename', 'old_video_filename'] + [c for c in gold_fields if c not in ('video_filename', 'old_video_filename')]
    with open(gold_out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore', lineterminator='\n')
        w.writeheader()
        for row in gold_rows:
            old = row.get('old_video_filename') or row['video_filename'].strip()
            new = os.path.basename(training[old]) if old in training else old
            w.writerow({**row, 'video_filename': new, 'old_video_filename': old})

    print(f'Wrote {len(training) + len(main_clips)} clips and manifests to {out}; gold standards to {gold_out}',
          file=sys.stderr)


if __name__ == '__main__':
    main()
