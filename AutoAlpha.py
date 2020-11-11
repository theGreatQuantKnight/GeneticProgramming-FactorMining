
import numpy as np
from statsmodels import regression
import statsmodels.api as sm
import pyodbc
import random
import warnings
import itertools
import statistics
import operator
import math
import random
import datetime
from datetime import datetime
import sys
from scipy.stats.stats import pearsonr
import copy
import os
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from bisect import bisect_right
from collections import defaultdict
from copy import deepcopy
from functools import partial
from itertools import chain

from operator import eq
import deap
from deap import base, creator, gp, tools, algorithms
import OperatorFunc
import genFunc
from genFunc import Full, HalfAndHalf
import HallOfFame
from MyFactorTester.prepare_datas import load_data

# %% 数据处理
tick_csv_path = r'W:\tick数据csv'

date_list = [20190819,20190824]


address2 = 'Z:/日度数据/factors_csv/'
env_train= load_data(address2,str(20190815), str(20191015))
period = 20
y_train = [env_train['open_dt'].pct_change(period, fill_method=None).shift(-period - 1), env_train['stop']]

# %% 数据存字典
os.chdir(tick_csv_path)
file_list = os.listdir()

sub_date_list = list(filter(lambda x: (int(x[:8]) >= date_list[0]) & (int(x[:8]) <= date_list[1]),file_list))

data_dict = {}

data_dict_time = {}

data_dict_price = {}
data_dict_vol = {}

data_dict_bp1 = {}
data_dict_bp2 = {}
data_dict_bp3 = {}
data_dict_bp4 = {}
data_dict_bp5 = {}
data_dict_sp1 = {}
data_dict_sp2 = {}
data_dict_sp3 = {}
data_dict_sp4 = {}
data_dict_sp5 = {}

data_dict_bv1 = {}
data_dict_bv2 = {}
data_dict_bv3 = {}
data_dict_bv4 = {}
data_dict_bv5 = {}
data_dict_sv1 = {}
data_dict_sv2 = {}
data_dict_sv3 = {}
data_dict_sv4 = {}
data_dict_sv5 = {}

for file in sub_date_list:
    date = int(file[:8])

    os.chdir(tick_csv_path + '\\' + file)
    code_list = os.listdir()[-500:-495]

    data_dict_price[date] = {}
    data_dict_vol[date] = {}
    data_dict_time[date] = {}
    data_dict_bp1[date] = {}
    data_dict_bp2[date] = {}
    data_dict_bp3[date] = {}
    data_dict_bp4[date] = {}
    data_dict_bp5[date] = {}
    data_dict_sp1[date] = {}
    data_dict_sp2[date] = {}
    data_dict_sp3[date] = {}
    data_dict_sp4[date] = {}
    data_dict_sp5[date] = {}
    data_dict_bv1[date] = {}
    data_dict_bv2[date] = {}
    data_dict_bv3[date] = {}
    data_dict_bv4[date] = {}
    data_dict_bv5[date] = {}
    data_dict_sv1[date] = {}
    data_dict_sv2[date] = {}
    data_dict_sv3[date] = {}
    data_dict_sv4[date] = {}
    data_dict_sv5[date] = {}

    if len(code_list) == 0:
        pass
    else:
        for code in tqdm(code_list):
            data =  pd.read_csv(code)

            data_dict[date] = data.loc[data['bp2'].values != 0].iloc[1:-1]
            data_dict[date]['time'] = data_dict[date]['time'].apply(lambda x: int(''.join(x.strip(' ').split(':'))))
            for item in ['price','sp1','sp2','sp3','sp4','sp5','bp1','bp2','bp3','bp4','bp5']:
                data_dict[date][item] = (data_dict[date][item] * 100).astype(int)
            for item in ['vol','sv1','sv2','sv3','sv4','sv5','bv1','bv2','bv3','bv4','bv5']:
                data_dict[date][item] = (data_dict[date][item]).astype(int)

            stock = int(code[2:8])

            data_dict_time[date][stock] = data_dict[date]['time'].to_numpy()

            data_dict_price[date][stock] = data_dict[date]['price'].to_numpy()
            data_dict_sp1[date][stock] = data_dict[date]['sp1'].to_numpy()
            data_dict_sp2[date][stock] = data_dict[date]['sp2'].to_numpy()
            data_dict_sp3[date][stock] = data_dict[date]['sp3'].to_numpy()
            data_dict_sp4[date][stock] = data_dict[date]['sp4'].to_numpy()
            data_dict_sp5[date][stock] = data_dict[date]['sp5'].to_numpy()
            data_dict_bp1[date][stock] = data_dict[date]['bp1'].to_numpy()
            data_dict_bp2[date][stock] = data_dict[date]['bp2'].to_numpy()
            data_dict_bp3[date][stock] = data_dict[date]['bp3'].to_numpy()
            data_dict_bp4[date][stock] = data_dict[date]['bp4'].to_numpy()
            data_dict_bp5[date][stock] = data_dict[date]['bp5'].to_numpy()

            data_dict_vol[date][stock] = data_dict[date]['vol'].to_numpy()
            data_dict_sv1[date][stock] = data_dict[date]['sv1'].to_numpy()
            data_dict_sv2[date][stock] = data_dict[date]['sv2'].to_numpy()
            data_dict_sv3[date][stock] = data_dict[date]['sv3'].to_numpy()
            data_dict_sv4[date][stock] = data_dict[date]['sv4'].to_numpy()
            data_dict_sv5[date][stock] = data_dict[date]['sv5'].to_numpy()
            data_dict_bv1[date][stock] = data_dict[date]['bv1'].to_numpy()
            data_dict_bv2[date][stock] = data_dict[date]['bv2'].to_numpy()
            data_dict_bv3[date][stock] = data_dict[date]['bv3'].to_numpy()
            data_dict_bv4[date][stock] = data_dict[date]['bv4'].to_numpy()
            data_dict_bv5[date][stock] = data_dict[date]['bv5'].to_numpy()

    # with open(path_save + os.sep + 'data_dict_price.pkl', 'wb') as f:
    #     pkl.dump(data_dict_price, f, protocol=4)


Layer1 = gp.PrimitiveSet("Layer1", 22)

Layer1.addPrimitive(OperatorFunc.Add, 2)
Layer1.addPrimitive(OperatorFunc.Sub, 2)
Layer1.addPrimitive(OperatorFunc.Mul, 2)

Layer1.addPrimitive(OperatorFunc.fsum, 1)
Layer1.addPrimitive(OperatorFunc.fmean,1)
Layer1.addPrimitive(OperatorFunc.fstd,1)
Layer1.addPrimitive(OperatorFunc.fcorr,2)

Layer1.addPrimitive(OperatorFunc.gbsumstd,2)
Layer1.addPrimitive(OperatorFunc.gbmeanstd,2)

Layer1.renameArguments(
    ARG0 = 'sp1', ARG1 = 'sp2', ARG2 = 'sp3', ARG3 = 'sp4', ARG4 = 'sp5', ARG5 = 'bp1', ARG6 = 'bp2', ARG7 = 'bp3', ARG8 = 'bp4', ARG9 = 'bp5',
    ARG10 = 'sv1', ARG11 = 'sv2', ARG12 = 'sv3', ARG13 = 'sv4', ARG14 = 'sv5', ARG15 = 'bv1', ARG16 = 'bv2', ARG17 = 'bv3', ARG18 = 'bv4', ARG19 = 'bv5',
    ARG20 = 'price', ARG21 = 'vol'
)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness = creator.FitnessMax)

L1box = base.Toolbox()
L1box.register("expr", Full, pset = Layer1, min_=1, max_=4)
L1box.register("individual", tools.initIterate, creator.Individual, L1box.expr)
L1box.register("population", tools.initRepeat, list, L1box.individual)
L1box.register("compile", gp.compile, pset=Layer1)

L1pop=L1box.population(n=100)

alterbox = base.Toolbox()
alterbox.register("expr", gp.genHalfAndHalf, pset = Layer1, min_=1, max_=5)
alterbox.register("individual", tools.initIterate, creator.Individual, alterbox.expr)

def fitnessIC(individual,dat ,y):
    func = L1box.compile(expr=individual)
    y_pred = func(
        dat[0], dat[1], dat[2], dat[3], dat[4], dat[5],dat[6], dat[7], dat[8], dat[9],
        dat[10], dat[11], dat[12], dat[13], dat[14], dat[15],dat[16], dat[17], dat[18], dat[19],
        dat[20], dat[21]
    )

    try:
        factor = pd.DataFrame(index=y_pred.keys())
        for date in y_pred.keys():
            for stock in y_pred[date].keys():
                factor.loc[date, stock] = y_pred[date][stock][0]
    except:
        return 0

    factor.replace([np.inf, -np.inf], np.nan, inplace = True)
    y[1] = y[1][['300271', '300272', '300273', '300274', '300275']]
    y[1] = y[1].iloc[2:7]
    y[0] = y[0].iloc[2:7]
    y[0] = y[0][['300271', '300272', '300273', '300274', '300275']]
    y[0].index = factor.index
    y[1].index = factor.index
    y[0].columns = factor.columns
    y[1].columns = factor.columns

    ret = copy.deepcopy(y[0])

    # mask = y[-1].shift(-1).fillna(True)  | factor.isnull() | y[0].isnull()  # 得到了所有应该剔除的mask
    # factor[mask] = np.nan
    # ret[mask] = np.nan
    #
    # y_pred, ret = y_pred.dropna(axis  = 0, how = 'all').align(ret.dropna(axis  = 0, how = 'all'), join='outer', axis=1)
    # y_pred, ret = y_pred.align(ret, join='inner', axis=0)
    IC_series = factor.corrwith(ret, axis = 1, method = 'spearman').dropna()
    Rank_IC = IC_series.mean() if IC_series.mean() is not np.nan else 0
    return Rank_IC


dat = [
    data_dict_sp1,
    data_dict_sp2,
    data_dict_sp3,
    data_dict_sp4,
    data_dict_sp5,
    data_dict_bp1,
    data_dict_bp2,
    data_dict_bp3,
    data_dict_bp4,
    data_dict_bp5,
    data_dict_sv1,
    data_dict_sv2,
    data_dict_sv3,
    data_dict_sv4,
    data_dict_sv5,
    data_dict_bv1,
    data_dict_bv2,
    data_dict_bv3,
    data_dict_bv4,
    data_dict_bv5,
    data_dict_price,
    data_dict_vol]

L1box.register("evaluate", fitnessIC, dat = dat, y = y_train)
L1box.register("mate", tools.cxOnePoint)
L1box.register("mutate", tools.mutFlipBit, indpb=0.05)
L1box.register("select", tools.selTournament, tournsize=3)

n_population = 30
n_generation = 3

pop = L1box.population(n=n_population)
hof = tools.HallOfFame(n_population * n_generation)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("min", np.min)
stats.register("max", np.max)

pop, log = algorithms.eaSimple(pop, L1box, cxpb=0.5, mutpb=0.2,
                               ngen=n_generation, stats=stats, halloffame=hof,
                               verbose=True)
