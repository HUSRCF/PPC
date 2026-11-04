"""Versioned exact-sequence OR of frozen M18 labels; original files stay read-only."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
from Bio import SeqIO
from Bio.Data.PDBData import protein_letters_3to1

EXPECTED = {'train': (14052, 4868), 'val': (3796, 1075), 'test': (2392, 724)}


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def label_sequence(names):
    return ''.join(n if len(n) == 1 else protein_letters_3to1.get(n, 'X')
                   for n in (str(v).upper() for v in names))


def merge_labels(arrays):
    a = np.stack(arrays)
    require(a.ndim == 2 and np.isin(a, [-100, 0, 1]).all(), 'Invalid source masks')
    valid = (a != -100).sum(axis=0, dtype=np.int32)
    positive = (a == 1).sum(axis=0, dtype=np.int32)
    union = np.where(positive > 0, 1, np.where(valid > 0, 0, -100)).astype(np.int8)
    return union, positive, valid


def validate_partition(rows):
    for key in ('seq_sha1', 'component_id', 'pdb_id'):
        assignments = defaultdict(set)
        for row in rows:
            require(bool(row[key]), 'Missing split identity: ' + key)
            assignments[row[key]].add(row['split'])
        require(all(len(splits) == 1 for splits in assignments.values()), key + ' crosses splits')


def write_table(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def pack(path, ids, arrays, **extra):
    lengths = np.array([len(a) for a in arrays], dtype=np.int32)
    np.savez_compressed(path, seq_ids=np.array(ids), lengths=lengths,
                        offsets=np.r_[0, np.cumsum(lengths, dtype=np.int64)],
                        labels=np.concatenate(arrays), **extra)


def load_pdb(item):
    import torch
    pdb_id, path, rows, sequences = item
    raw = path.read_bytes()
    d = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)
    require(str(d['pdb_id']).lower() == pdb_id and d['cutoff_angstrom'] == 5.5
            and d['heavy_only'] is True, f'{pdb_id}: wrong label contract')
    y = d['labels'].cpu().numpy()
    require(y.ndim == 1 and len(y) == d['n_residues']
            and np.isin(y, [-100, 0, 1]).all(), f'{pdb_id}: invalid targets')
    require(int((y == 1).sum()) == d['n_positive'] and int((y == 0).sum()) == d['n_negative'],
            f'{pdb_id}: source counts differ')
    chain_ids = np.array(d['chain_ids']).astype(str)
    names = np.array(d['residue_names']).astype(str)
    numbering = d['residue_indices']
    insertion = d['insertion_codes']
    require(len(chain_ids) == len(names) == len(numbering) == len(insertion) == len(y),
            f'{pdb_id}: metadata lengths differ')
    result = []
    for row in rows:
        idx = np.where(chain_ids == row['chain_id'])[0]
        seq = label_sequence(names[idx])
        require(len(idx) == int(row['n_residues']) and len(idx) > 0, f'{row["seq_id"]}: missing chain')
        require(np.array_equal(idx, np.arange(idx[0], idx[0] + len(idx))),
                f'{row["seq_id"]}: non-contiguous chain')
        require(hashlib.sha1(seq.encode()).hexdigest() == row['seq_sha1']
                and seq == sequences[row['seq_sha1']], f'{row["seq_id"]}: actual label sequence mismatch')
        target = y[idx].astype(np.int8)
        require(int((target == 1).sum()) == int(row['positive_count'])
                and int((target == 0).sum()) == int(row['negative_count']),
                f'{row["seq_id"]}: frozen per-chain counts differ')
        residue_keys = [(str(numbering[i]), str(insertion[i])) for i in idx]
        require(len(set(residue_keys)) == len(idx), f'{row["seq_id"]}: duplicate residue identity')
        result.append((row['seq_id'], target, int(sum(bool(str(insertion[i]).strip()) for i in idx))))
    return pdb_id, hashlib.sha256(raw).hexdigest(), str(d['label_definition']), result


def main():
    parser = argparse.ArgumentParser()
    for name in ('chain-statistics', 'dataset-summary', 'label-root', 'fasta', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    require(not args.output.exists() and 1 <= args.workers <= 32, 'New output directory and 1..32 workers required')
    with gzip.open(args.chain_statistics, 'rt') as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 20240 and len({r['seq_id'] for r in rows}) == len(rows), 'Wrong chain universe')
    validate_partition(rows)
    require(len({r['component_id'] for r in rows}) == 338, 'Wrong SI30 component count')
    groups, pdbs = defaultdict(list), defaultdict(list)
    for row in rows:
        require(row['split'] in EXPECTED, 'Unknown split')
        groups[row['seq_sha1']].append(row)
        pdbs[row['pdb_id']].append(row)
    require(len(groups) == 6667 and all(len({r['split'] for r in rs}) == 1 for rs in groups.values()),
            'Wrong unique count or exact sequences cross splits')
    sequences = {}
    for rec in SeqIO.parse(args.fasta, 'fasta'):
        sequence = str(rec.seq).upper()
        key = hashlib.sha1(sequence.encode()).hexdigest()
        require(key not in sequences, 'Frozen FASTA repeats exact sequences')
        sequences[key] = sequence
    require(set(sequences) == set(groups), 'FASTA / exact sequence groups differ')
    baseline = json.loads(args.dataset_summary.read_text())
    jobs = [(pdb, args.label_root / pdb / (pdb + '_labels.pt'), rs, sequences)
            for pdb, rs in sorted(pdbs.items())]
    labels, insertions, file_hashes, definitions = {}, {}, {}, set()
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for n, (pdb, sha, definition, records) in enumerate(pool.map(load_pdb, jobs), 1):
            file_hashes[pdb] = sha
            definitions.add(definition)
            for sid, target, n_insertion in records:
                require(sid not in labels, 'Duplicate source chain')
                labels[sid], insertions[sid] = target, n_insertion
            if n % 500 == 0:
                print(json.dumps({'pdbs': n, 'total': len(jobs), 'seconds': time.time() - started}), flush=True)
    aggregate = hashlib.sha256()
    for pdb, sha in sorted(file_hashes.items()):
        aggregate.update(f'{pdb}\t{sha}\n'.encode('ascii'))
    require(aggregate.hexdigest() == baseline['label_metadata']['label_file_set_sha256']
            and len(definitions) == 1, 'Frozen label-file identity or definition changed')
    for split, (nc, nu) in EXPECTED.items():
        require(sum(r['split'] == split for r in rows) == nc
                and sum(rs[0]['split'] == split for rs in groups.values()) == nu, 'Split counts changed')
    args.output.mkdir(parents=True)
    pack(args.output / 'source_context_labels.npz', [r['seq_id'] for r in rows],
         [labels[r['seq_id']] for r in rows])
    group_records, mapping, summary, all_fasta = [], [], {}, []
    for split in EXPECTED:
        ids, unions, pos_counts, obs_counts, split_fasta = [], [], [], [], []
        for sha, rs in sorted(groups.items()):
            if rs[0]['split'] != split:
                continue
            uid = 'useq_' + sha
            source = [labels[r['seq_id']] for r in rs]
            union, positive, observed = merge_labels(source)
            rep = min(r['seq_id'] for r in rs)
            ids.append(uid); unions.append(union); pos_counts.append(positive); obs_counts.append(observed)
            split_fasta.append(f'>{uid}\n{sequences[sha]}\n')
            patterns = len({a.tobytes() for a in source})
            group_records.append({'union_seq_id': uid, 'seq_sha1': sha, 'split': split,
                'representative_chain': rep, 'length': len(union), 'chain_contexts': len(rs),
                'pdb_contexts': len({r['pdb_id'] for r in rs}), 'distinct_masks': patterns,
                'union_positive': int((union == 1).sum()), 'unknown': int((union == -100).sum()),
                'variable_positions': int(((positive > 0) & (positive < observed)).sum()),
                'source_positive_min': min(int((a == 1).sum()) for a in source),
                'source_positive_max': max(int((a == 1).sum()) for a in source),
                'union_label_sha256': hashlib.sha256(union.tobytes()).hexdigest()})
            for row, target in zip(rs, source):
                mapping.append({'seq_id': row['seq_id'], 'pdb_id': row['pdb_id'], 'chain_id': row['chain_id'],
                    'split': split, 'union_seq_id': uid, 'seq_sha1': sha, 'component_id': row['component_id'],
                    'length': len(union), 'source_positive': int((target == 1).sum()),
                    'union_positive': int((union == 1).sum()),
                    'added_positive': int(((union == 1) & (target != 1)).sum()),
                    'insertion_code_residues': insertions[row['seq_id']],
                    'source_label_sha256': hashlib.sha256(target.tobytes()).hexdigest()})
        pack(args.output / (split + '.npz'), ids, unions,
             positive_support=np.concatenate(pos_counts), observed_support=np.concatenate(obs_counts))
        (args.output / (split + '.fasta')).write_text(''.join(split_fasta))
        all_fasta.extend(split_fasta)
        split_rows = [r for r in mapping if r['split'] == split]
        summary[split] = {'source_chain_contexts': len(split_rows), 'unique_sequences': len(ids),
            'unique_residues': sum(len(a) for a in unions), 'union_positive': sum(int((a == 1).sum()) for a in unions),
            'source_instance_positive': sum(r['source_positive'] for r in split_rows),
            'union_instance_positive': sum(r['union_positive'] for r in split_rows),
            'chains_changed': sum(r['added_positive'] > 0 for r in split_rows),
            'groups_with_different_masks': sum(r['split'] == split and r['distinct_masks'] > 1 for r in group_records)}
    (args.output / 'unique_sequences.fasta').write_text(''.join(all_fasta))
    write_table(args.output / 'sequence_groups.tsv', group_records)
    write_table(args.output / 'chain_to_union.tsv', mapping)
    dump(args.output / 'source_label_hashes.json', file_hashes)
    artifacts = {p.name: digest(p) for p in sorted(args.output.iterdir()) if p.is_file()}
    receipt = {'status': 'complete', 'contract': 'M18_exact_sequence_ever_observed_union_5p5_v2',
        'group_key': 'SHA1 of exact frozen label-aligned sequence; full strings verified',
        'split_policy': 'original SI30 assignment unchanged; zero exact-sequence cross-split groups',
        'split_audit': {'cross_split_exact_sequences': 0, 'cross_split_si30_components': 0,
                        'cross_split_pdbs': 0, 'si30_components': 338},
        'label_rule': '1 if any valid source context is positive; 0 if observed but never positive; -100 if never assessed',
        'observation_mask_policy': 'inherit original frozen labels; no new unresolved-residue mask invented',
        'assembly_geometry_recomputed': False,
        'limitations': ['Assembly/atom coordinates not independently reconstructed in this run.',
                        'A zero means not observed positive in this curated source set, not proven never binding.',
                        'Union coverage depends on the number and quality of observed contexts.'],
        'old_labels_modified': False, 'model_retrained': False, 'msa_database_requested': 'UniRef90 only',
        'sources': {str(p): digest(p) for p in (args.chain_statistics, args.dataset_summary, args.fasta)},
        'source_label_file_set_sha256': aggregate.hexdigest(), 'n_source_pdb_files': len(file_hashes),
        'residue_identity_includes_insertion_code': True, 'statistics': summary,
        'artifacts_sha256': artifacts, 'seconds': time.time() - started}
    dump(args.output / 'summary.json', receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
