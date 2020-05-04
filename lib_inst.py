import pandas as pd
import numpy as np
import multiprocessing as mp
import sys
import csv
import os
import time
import itertools
from catboost import CatBoostClassifier
import tsfresh.feature_extraction.feature_calculators
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import classification_report
import argparse
import numpy_indexed as npi
from collections import deque, defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('infile', nargs='?')
parser.add_argument('errfile', nargs='?', type=argparse.FileType('w'))
parser.add_argument('outfile', nargs='?', type=argparse.FileType('w'))
parser.add_argument("-i", "--input", help="input filename")
# parser.add_argument("-d", "--input_dir", help = "input file directory")
parser.add_argument("-s", "--supplementary", help="output supplementary file")
parser.add_argument("-o", "--output", help="output filename")
# parser.add_argument("--file_type", default=".csv", help = "type of every file in dictionary")
parser.add_argument("--thr_other", default=10, type=int, help="threshold for other age groups")
parser.add_argument("--thr_young", default=5, type=int, help="threshold for young age group")
args = parser.parse_args()


def list_slice(S, step):
    S = list(itertools.chain.from_iterable(S))
    elem = [S[i::step] for i in range(step)]
    return (elem)


start = time.time()
if args.input:
    reader = csv.reader(open(args.input, 'r'), delimiter='\t')
if args.infile:
    reader = csv.reader(open(args.infile, 'r'), delimiter='\t')
data = defaultdict(list)
nan_line = [np.nan] * 10
all_lis = []
for row in reader:
    if len(data[row[0]]) >= 10:
        pass
    else:
        data[row[0]].append(row[2:])
key_list = []
for k, i in data.items():
    key_list.append(k)
    n2 = len(i)
    while n2 < 10:
        i.append(nan_line)
        n2 = n2 + 1
    parse = list_slice(i, len(i))
    lis = list(itertools.chain.from_iterable(parse))
    lis = [float(i) for i in lis]
    all_lis.append(lis)
my_df = pd.DataFrame(data=all_lis, columns=['Percent identity_1', 'Percent identity_2', 'Percent identity_3',
                                            'Percent identity_4', 'Percent identity_5', 'Percent identity_6',
                                            'Percent identity_7', 'Percent identity_8',
                                            'Percent identity_9', 'Percent identity_10', 'Alignment length_1',
                                            'Alignment length_2', 'Alignment length_3',
                                            'Alignment length_4', 'Alignment length_5', 'Alignment length_6',
                                            'Alignment length_7', 'Alignment length_8',
                                            'Alignment length_9', 'Alignment length_10', 'Number of mismatches_1',
                                            'Number of mismatches_2',
                                            'Number of mismatches_3', 'Number of mismatches_4',
                                            'Number of mismatches_5', 'Number of mismatches_6',
                                            'Number of mismatches_7', 'Number of mismatches_8',
                                            'Number of mismatches_9', 'Number of mismatches_10',
                                            'Number of gap opens_1', 'Number of gap opens_2', 'Number of gap opens_3',
                                            'Number of gap opens_4',
                                            'Number of gap opens_5', 'Number of gap opens_6', 'Number of gap opens_7',
                                            'Number of gap opens_8',
                                            'Number of gap opens_9', 'Number of gap opens_10',
                                            'Start position in query_1',
                                            'Start position in query_2', 'Start position in query_3',
                                            'Start position in query_4',
                                            'Start position in query_5', 'Start position in query_6',
                                            'Start position in query_7',
                                            'Start position in query_8', 'Start position in query_9',
                                            'Start position in query_10',
                                            'End position in query_1', 'End position in query_2',
                                            'End position in query_3',
                                            'End position in query_4', 'End position in query_5',
                                            'End position in query_6',
                                            'End position in query_7', 'End position in query_8',
                                            'End position in query_9',
                                            'End position in query_10', 'Start position in target_1',
                                            'Start position in target_2',
                                            'Start position in target_3', 'Start position in target_4',
                                            'Start position in target_5',
                                            'Start position in target_6', 'Start position in target_7',
                                            'Start position in target_8',
                                            'Start position in target_9', 'Start position in target_10',
                                            'End position in target_1',
                                            'End position in target_2', 'End position in target_3',
                                            'End position in target_4',
                                            'End position in target_5', 'End position in target_6',
                                            'End position in target_7',
                                            'End position in target_8', 'End position in target_9',
                                            'End position in target_10',
                                            'E-value_1', 'E-value_2', 'E-value_3', 'E-value_4', 'E-value_5',
                                            'E-value_6', 'E-value_7', 'E-value_8',
                                            'E-value_9', 'E-value_10', 'Bit score_1', 'Bit score_2', 'Bit score_3',
                                            'Bit score_4',
                                            'Bit score_5', 'Bit score_6', 'Bit score_7', 'Bit score_8', 'Bit score_9',
                                            'Bit score_10'])

print(time.time() - start)
model_file = "model_14traits.cbm"
n_jobs = 8
params = {'Target len mismatches ratio': {'quantile': [{'q': 0.1}, {'q': 0.6}],
                                          'fft_coefficient': [{'coeff': 0, 'attr': 'abs'}],
                                          'abs_energy': None},
          'Bit score': {'fft_aggregated': [{'aggtype': 'variance'},
                                           {'aggtype': 'centroid'}]},
          'Alignment len mismatches ratio': {'fft_coefficient': [{'coeff': 0,
                                                                  'attr': 'abs'}],
                                             'abs_energy': None},
          'Query len mismatches ratio': {'abs_energy': None}}
feats = ['Target len mismatches ratio__quantile__q_0.1',
         'Query len mismatches ratio_3',
         'Target len mismatches ratio__quantile__q_0.6',
         'Bit score__fft_aggregated__aggtype_"variance"',
         'Alignment len mismatches ratio__fft_coefficient__coeff_0__attr_"abs"',
         'Percent identity_7',
         'Target len mismatches ratio__fft_coefficient__coeff_0__attr_"abs"',
         'Bit score__fft_aggregated__aggtype_"centroid"',
         'Alignment len mismatches ratio__abs_energy',
         'Target len mismatches ratio__abs_energy',
         'Target len mismatches ratio_2',
         'Query len mismatches ratio_1',
         'Percent identity_10',
         'Query len mismatches ratio__abs_energy']

df = my_df
for i in range(1, 11):
    s_q = f'Start position in query_{i}'
    e_q = f'End position in query_{i}'
    q_l = f'Query len_{i}'
    df[q_l] = df[e_q] - df[s_q]
    s_t = f'Start position in target_{i}'
    e_t = f'End position in target_{i}'
    t_l = f'Target len_{i}'
    df[t_l] = df[e_t] - df[s_t]
    lens_ratio = f'Query Target lens ratio_{i}'  #
    df[lens_ratio] = df[q_l] / df[t_l]

    m = f'Number of mismatches_{i}'
    a_l = f'Alignment length_{i}'
    tl_m_ration = f'Target len mismatches ratio_{i}'
    ql_m_ration = f'Query len mismatches ratio_{i}'
    al_m_ration = f'Alignment len mismatches ratio_{i}'
    df[tl_m_ration] = df[m] / df[t_l]
    df[ql_m_ration] = df[m] / df[q_l]
    df[al_m_ration] = df[m] / df[a_l]
df = df.replace([np.inf, -np.inf], np.nan).fillna(-999)

feature_calculators_module = sys.modules['tsfresh.feature_extraction.feature_calculators']


def extraction_ts_traits(col_name, df, params):
    res = pd.DataFrame()
    for func_name in params[col_name]:
        func_params = params[col_name][func_name]
        trait_name = f'{col_name}__{func_name}'
        func = getattr(feature_calculators_module, func_name)
        if func_params is not None:
            for func_param_dict in func_params:
                trait_name_postfix = ''
                for key, val in func_param_dict.items():
                    if isinstance(val, str):
                        val = f'"{val}"'
                    elif isinstance(val, list):
                        val = "({})".format(', '.join(str(e) for e in val))
                    trait_name_postfix += f'__{key}_{val}'
                if func.fctype == 'simple':
                    res[trait_name + trait_name_postfix] = df[list(df.filter(regex=f'{col_name}_\d+'))].apply(
                        lambda x: func(x, **func_param_dict),
                        raw=True, axis=1)
                else:
                    res[trait_name + trait_name_postfix] = df[list(df.filter(regex=f'{col_name}_\d+'))].apply(
                        lambda x: list(*func(x, [func_param_dict]))[1],
                        raw=True, axis=1)
        else:
            res[trait_name] = df[list(df.filter(regex=f'{col_name}_\d+'))].apply(func, raw=True, axis=1)
    return res


processed_list = Parallel(n_jobs=n_jobs)(delayed(extraction_ts_traits)(i, df, params) for i in tqdm(params.keys()))
df = pd.concat([df, *processed_list], axis=1)

model = CatBoostClassifier(task_type="CPU")
model.load_model(model_file)
preds = model.predict(df[feats])
df['preds'] = preds
df['Id'] = key_list
# print(preds)
m = model.predict_proba(df[feats])
m = pd.DataFrame(data=m)
df[["other,%", "young,%"]] = m
age = df[['Id', 'preds', "other,%", "young,%"]]
if args.errfile:
    age.to_csv(args.errfile, index=False, mode='a', sep='\t')
if args.supplementary:
    age.to_csv(args.supplementary, index=False, mode='a', sep='\t')
if args.input:
    reader = csv.reader(open(args.input, 'r'), delimiter='\t')
if args.infile:
    reader = csv.reader(open(args.infile, 'r'), delimiter='\t')
dr = pd.DataFrame(reader, index=None)
group = dr.groupby(0)
ag = age.groupby('preds')
for w in df['Id']:
    try:
        if w in ag.get_group('other')['Id'].to_list():
            t = group.get_group(w)[:args.thr_other]
    except KeyError:
        pass
    try:
        if w in ag.get_group('young')['Id'].to_list():
            t = group.get_group(w)[:args.thr_young]
    except KeyError:
        pass
    if args.output:
        t.to_csv(args.output, index=False, header=False, mode='a', sep='\t')
    if args.outfile:
        t.to_csv(args.outfile, index=False, header=False, mode='a', sep='\t')
print(time.time() - start)
