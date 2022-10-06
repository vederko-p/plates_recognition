
from typing import Tuple
import pandas as pd


def tpr_fpr(df_gt: pd.DataFrame, df_st: pd.DataFrame) -> Tuple[float, float]:
    gt_cars = set(df_gt['number'])
    st_cars = set(df_st['number'])
    tp = len(gt_cars & st_cars)
    fn = len(gt_cars - st_cars)
    fp = len(st_cars - gt_cars)
    tpr = tp / (tp + fn)
    fpr = fp / (tp + fp)
    return tpr, fpr


def rec_deviation(df_gt: pd.DataFrame, df_st: pd.DataFrame) -> float:
    st_cars = df_st[['number', 'timestamp']].groupby('number').min()
    merged = df_gt.merge(st_cars, on='number',
                         how='inner', suffixes=('_gt', '_st'))
    diff = merged['timestamp_st'] - merged['timestamp_gt']
    return diff.mean()
