"""Apply equal-chemical intercept calibration to the released source predictions."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration import InterceptCalibration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True,
                        help='One dataset/partition folder in model_inputs.')
    parser.add_argument('--split', type=int, required=True, choices=range(6))
    parser.add_argument('--budget', choices=['5', '20', '80', 'ALL'], default='20')
    parser.add_argument('--repeat', type=int, choices=range(10), default=0)
    parser.add_argument('--regime', choices=['all_records', 'one_record'], default='all_records')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.inputs / 'EXACT_INPUTS.npz') as source:
        arrays = {key: source[key] for key in source.files}
    train = pd.read_csv(args.inputs / 'TRAIN.csv.gz', dtype={'record_id': str})
    query = pd.read_csv(args.inputs / 'QUERY_METADATA.csv.gz', dtype={'record_id': str})
    if not query['split'].eq(args.split).all():
        raise ValueError('Requested partition differs from the input metadata.')
    if args.budget == 'ALL' and args.repeat != 0:
        raise ValueError('The full-data comparison has one fit per partition.')
    if not np.array_equal(train.record_id, arrays['train_record_id'].astype(str)):
        raise ValueError('Training rows do not match the released arrays.')
    if not np.array_equal(query.record_id, arrays['query_record_id'].astype(str)):
        raise ValueError('Query rows do not match the released arrays.')
    if set(train.parent_group) & set(query.parent_group):
        raise ValueError('Training and query chemical identities overlap.')
    if args.regime == 'one_record':
        if args.budget not in ['5', '20']:
            raise ValueError('The one-record comparison used budgets 5 and 20.')
        choices = pd.read_csv(args.inputs / 'CHOICES.csv.gz',
                              dtype={'record_id': str, 'budget': str})
        selected = choices.loc[(choices.policy == 'random') &
                               (choices.regime == args.regime) &
                               (choices.budget == args.budget) &
                               (choices.repeat == args.repeat)]
        indices = selected.row_index.to_numpy(int)
        if not len(indices) or not np.array_equal(selected.record_id, train.record_id.iloc[indices]):
            raise ValueError('Frozen one-record selection is unavailable or misaligned.')
    else:
        order = np.array(sorted(train.parent_group.unique()))
        np.random.default_rng(2026101497 + 100 * args.split + args.repeat).shuffle(order)
        selected = order if args.budget == 'ALL' else order[:int(args.budget)]
        indices = np.flatnonzero(train.parent_group.isin(selected).to_numpy())
    fitted = InterceptCalibration.fit(arrays['train_growth'][indices],
                                     arrays['train_mortality'][indices],
                                     train.parent_group.iloc[indices])
    result = query.copy()
    result['predicted_log10_growth_NOEC_mg_L'] = fitted.predict_log_concentration(arrays['query_mortality'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    args.output.with_suffix('.json').write_text(json.dumps({
        'intercept': fitted.intercept, 'training_chemicals': fitted.n_chemicals,
        'training_records': fitted.n_records, 'budget': args.budget,
        'repeat': args.repeat, 'regime': args.regime, 'split': args.split,
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
