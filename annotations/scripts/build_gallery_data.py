#!/usr/bin/env python3
"""Build the data behind the admin gallery's earlier-pass and model-detection sources.

Two earlier passes are read from the contexts repo:
  - the original doing/seeing pass (Jan-Mar 2026, old activity list, with confidence 1-3 and
    other activities): data/completed_annotations/activity_annotations_{do,see}.csv
  - the model validation pass (2025, DD/JY/VM rating the model's detections, with description
    accuracy 1-5): data/completed_annotations/{DD,JY,VM}_list_final.csv
Both are joined to the model's detection for the clip (activity, location, description) from
all_contexts_combined.csv, and every clip the model labeled is also written out on its own so the
gallery can browse all of them, not just the validated ones.

The clips use the old names (00680001_2023-06-03_3_9d5449d617_processed_012.mp4) and are mapped to
the 2026.1 10s clips the same way as build_clip_folder.py, so the gallery can play them and add
them to gold_standards.csv under their new name.

Outputs:
    --out (data/original_pass.json)
        {"original": {"do": [clip, ...], "see": [...]}, "validation": [clip, ...]}
        where each clip is {key, videoFilename, sourcePath, model, votes}
    --detections-out (data/model_detections.tsv)
        one row per model-labeled clip: key, video (path under --source, blank if the clip has no
        2026.1 version), activity, location, description

Standard library only, so it runs with the cluster's system python3.
"""

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

csv.field_size_limit(sys.maxsize)

CONTEXTS = Path.home() / 'contexts' / 'data'
OLD_CLIP = re.compile(r'^(\d{8})_(\d{4}-\d{2}-\d{2})_(\d+)_([0-9a-f]+)_processed_(\d+)\.mp4$')
NEW_CLIP = re.compile(r'^(S\d{8})_(\d{4}-\d{2}-\d{2})_(\d+)_(rec\w+?)(_rotated)?_chunk(\d{4})of(\d{4})\.mp4$')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--annotations', nargs='+', default=[
        str(CONTEXTS / 'completed_annotations' / 'activity_annotations_do.csv'),
        str(CONTEXTS / 'completed_annotations' / 'activity_annotations_see.csv')],
        help='original-pass exports (rows are merged by _id, keeping the latest updatedAt)')
    p.add_argument('--validation-lists', nargs='+', default=[
        str(CONTEXTS / 'completed_annotations' / f'{name}_list_final.csv') for name in ('DD', 'JY', 'VM')],
        help='model validation sheets; the annotator is the part of the filename before _list')
    p.add_argument('--contexts', default=str(CONTEXTS / 'all_contexts_combined.csv'),
                   help='model detections per clip (old_video_id, activity, location, video_description)')
    p.add_argument('--recordings', default='data/recordings_processed.csv')
    p.add_argument('--source', default='/ccn2b/dataset/babyview/2026.1/10s_clips/video',
                   help='10s clip root with one folder per subject (S00680001/...)')
    p.add_argument('--exclude-annotators', nargs='*', default=['test', 'tst', 'babyview'])
    p.add_argument('--out', default='data/original_pass.json')
    p.add_argument('--detections-out', default='data/model_detections.tsv')
    return p.parse_args()


def read_csv(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def split_list(value):
    return [s.strip() for s in (value or '').split(',') if s.strip()]


def original_votes(paths, exclude):
    """Latest main-task answer per (clip, annotator, task) from the original pass."""
    by_id = {}
    for path in paths:
        for row in read_csv(path):
            prev = by_id.get(row['_id'])
            if not prev or row['updatedAt'] > prev['updatedAt']:
                by_id[row['_id']] = row

    latest = {}
    for row in by_id.values():
        if row.get('isTraining') == 'true' or row['annotatorName'] in exclude:
            continue
        task = row.get('taskType') or 'do'  # the earliest rows predate the seeing task
        key = (row['videoFilename'], row['annotatorName'], task)
        if key not in latest or row['updatedAt'] > latest[key]['updatedAt']:
            latest[key] = row

    clips = {'do': {}, 'see': {}}
    for (video, annotator, task), row in sorted(latest.items()):
        clips[task].setdefault(video, []).append({
            'annotatorName': annotator,
            'primaryActivity': row['primaryActivity'],
            'primaryActivityOther': row.get('primaryActivityOther', ''),
            'otherActivities': [row[c] for c in ('otherActivities[0]', 'otherActivities[1]', 'otherActivities[2]')
                                if row.get(c)],
            'confidence': int(row['primaryActivityConfidence']) if row['primaryActivityConfidence'] else None,
            'anyoneInteracting': row['anyoneInteracting'],
            'updatedAt': row['updatedAt']
        })
    return clips


def validation_votes(paths):
    """Each validator's activity and description-accuracy rating per clip."""
    clips = {}
    for path in paths:
        annotator = os.path.basename(path).split('_list')[0]
        for row in read_csv(path):
            if not row.get('Activity', '').strip():
                continue
            video = re.sub(r'^\d+_', '', row['video_id'].strip())
            accuracy = row.get('Video description accuracy (1-5)', '').strip()
            clips.setdefault(video, []).append({
                'annotatorName': annotator,
                'primaryActivity': row['Activity'].strip().lower(),
                'otherActivities': [a.lower() for a in split_list(row.get('Other activities'))],
                'accuracy': int(accuracy) if accuracy.isdigit() else None,
                'location': row.get('Location', '').strip(),
                'activityCenter': row.get('Activity center', '').strip(),
                'activityNotInSchema': row.get('Activity not in schema', '').strip()
            })
    return clips


def model_detections(path):
    detections = {}
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            detections[row['old_video_id']] = {
                'activity': row['activity'],
                'location': row['location'],
                'description': row['video_description']
            }
    return detections


def new_clip_paths(recordings, source, old_names):
    """Old clip name -> 2026.1 clip path, scanning only the subjects that are needed."""
    old_to_new = {}
    for row in read_csv(recordings):
        m = re.match(r'^(\d{8})_(\d{4}-\d{2}-\d{2})_(\d+)_[0-9a-f]+$', row['superseded_gcp_name_feb25'] or '')
        if m and row['unique_video_id'].startswith('rec'):
            old_to_new[row['superseded_gcp_name_feb25']] = f"S{m.group(1)}_{m.group(2)}_{m.group(3)}_{row['unique_video_id']}"

    wanted = {}
    for name in old_names:
        m = OLD_CLIP.match(name)
        new_video = old_to_new.get('_'.join(m.group(1, 2, 3, 4))) if m else None
        if new_video:
            wanted[(new_video, int(m.group(5)))] = name

    found = {}
    for subject in sorted({video.split('_')[0] for video, _ in wanted}):
        subject_dir = Path(source) / subject
        if not subject_dir.is_dir():
            continue
        with os.scandir(subject_dir) as entries:
            for e in entries:
                m = NEW_CLIP.match(e.name)
                key = (f'{m.group(1)}_{m.group(2)}_{m.group(3)}_{m.group(4)}', int(m.group(6))) if m else None
                if key in wanted:
                    found[wanted[key]] = e.path
    return found


def main():
    args = parse_args()
    original = original_votes(args.annotations, set(args.exclude_annotators))
    validation = validation_votes(args.validation_lists)
    print(f'Reading model detections from {args.contexts} ...', file=sys.stderr)
    detections = model_detections(args.contexts)
    old_names = set(original['do']) | set(original['see']) | set(validation) | set(detections)
    print(f'Mapping {len(old_names)} clips to {args.source} ...', file=sys.stderr)
    paths = new_clip_paths(args.recordings, args.source, old_names)

    def clip(name, votes):
        path = paths.get(name)
        return {'key': name, 'videoFilename': os.path.basename(path) if path else '', 'sourcePath': path or '',
                'model': detections.get(name), 'votes': votes}

    out = {
        'generatedAt': datetime.now(timezone.utc).isoformat(),
        'inputs': {'annotations': args.annotations, 'validationLists': args.validation_lists,
                   'contexts': args.contexts, 'source': args.source},
        'original': {task: [clip(n, v) for n, v in sorted(clips.items())] for task, clips in original.items()},
        'validation': [clip(n, v) for n, v in sorted(validation.items())]
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f)

    # Tabs and newlines would break the TSV; descriptions don't need them
    clean = lambda s: re.sub(r'[\t\r\n]+', ' ', s or '')
    with open(args.detections_out, 'w', newline='') as f:
        f.write('key\tvideo\tactivity\tlocation\tdescription\n')
        for name, d in sorted(detections.items()):
            video = os.path.relpath(paths[name], args.source) if name in paths else ''
            f.write('\t'.join(clean(v) for v in (name, video, d['activity'], d['location'], d['description'])) + '\n')
    print(f'Model detections: {len(detections)} clips, {sum(n in paths for n in detections)} with a 2026.1 clip; '
          f'wrote {args.detections_out}', file=sys.stderr)

    for label, clips in [('original do', out['original']['do']), ('original see', out['original']['see']),
                         ('validation', out['validation'])]:
        print(f'{label}: {len(clips)} clips, {sum(len(c["votes"]) for c in clips)} answers, '
              f'{sum(not c["sourcePath"] for c in clips)} without a 2026.1 clip, '
              f'{sum(not c["model"] for c in clips)} without a model detection', file=sys.stderr)
    print(f'Wrote {args.out}', file=sys.stderr)


if __name__ == '__main__':
    main()
