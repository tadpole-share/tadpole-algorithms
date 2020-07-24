#!/usr/bin/env python
# Written by Minh Nguyen and CBIG under MIT license:
# https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
from __future__ import print_function, division
import argparse
import pickle

from tadpole_algorithms.models.cbig_rnn.cbig.Nguyen2020 import dataloader \
    as dataloader
from tadpole_algorithms.models.cbig_rnn.cbig.Nguyen2020 import misc as misc


def get_data(args, fields):
    """
    Generate training/test data batches and save as pickle file
    *args* specify
        mask: path to mask file
        strategy: filling strategy
        spreadsheet: path to TADPOLE data
        batch_size: batch size
        out: path to save pickle file
        validation: evaluate on validation subjects (instead of test subjects)
    """

    ret = {}
    train_mask, pred_mask, pred_mask_frame = misc.get_mask(
        args.mask, args.validation, args.isTrain)
    ret['baseline'], ret['pred_start'] = misc.get_baseline_prediction_start(
        pred_mask_frame)
    ret['duration'] = 10 * 12

    columns = ['RID', 'Month_bl', 'DX', 'VISCODE'] + fields
    frame = misc.load_table(args.spreadsheet, columns, args.isTrain)

    tf = frame.loc[train_mask, fields]
    ret['mean'] = tf.mean()
    ret['stds'] = tf.std()
    ret['VentICVstd'] = (tf['Ventricles'] / tf['ICV']).std()

    frame[fields] = (frame[fields] - ret['mean']) / ret['stds']

    default_val = {f: 0. for f in fields}
    default_val['DX'] = 0.

    data = dataloader.extract(frame[train_mask], args.strategy, fields,
                              default_val)
    ret['train'] = dataloader.Random(data, args.batch_size, fields)

    data = dataloader.extract(frame[pred_mask], args.strategy, fields,
                              default_val)
    ret['test'] = dataloader.Sorted(data, 1, fields)

    print('train', len(ret['train'].subjects), 'subjects')
    print('test', len(ret['test'].subjects), 'subjects')
    print(len(fields), 'features')

    return ret


def main(args):
    fields = misc.load_feature(args.features)
    with open(args.out, 'wb') as fhandler:
        pickle.dump(get_data(args, fields), fhandler)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mask', default='/')
    parser.add_argument('--strategy', default='/')
    parser.add_argument('--spreadsheet', default='/')
    parser.add_argument('--features', default='/')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--out', default='/')
    parser.add_argument('--validation', action='store_true')
    parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython",
                        default="1")

    return parser.parse_args()


if __name__ == '__main__':
    main(get_args())
