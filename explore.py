import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from itertools import combinations, product
from scipy import stats

def plot_variable_pairs(df,
                        columns_x = ['bedroomcnt','bathroomcnt','calculatedfinishedsquarefeet','yearbuilt','taxvaluedollarcnt', 'lotsizesquarefeet'],
                        columns_y = ['bedroomcnt','bathroomcnt','calculatedfinishedsquarefeet','yearbuilt','taxvaluedollarcnt', 'lotsizesquarefeet'],
                        sampling = 1000):
    pairs = product(columns_x, columns_y)
    for pair in pairs:
        sns.lmplot(x=pair[0], y=pair[1], data=df.sample(sampling), line_kws={'color': 'red'})
        plt.show()

def plot_categorical_and_continuous_vars(df,
                                         columns_cat=['county'],
                                         columns_cont=['calculatedfinishedsquarefeet', 'yearbuilt', 'bedroomcnt', 'bathroomcnt', 'taxvaluedollarcnt', 'taxamount'],
                                         sampling = 1000):
    pairs = product(columns_cat, columns_cont)
    for pair in pairs:
        sns.set(rc={"figure.figsize":(15, 6)}) 
        fig, axes = plt.subplots(1, 3)

        sns.stripplot(x=pair[0], y=pair[1], data=df.sample(sampling), ax = axes[0])
        sns.boxplot(x=pair[0], y=pair[1], data=df.sample(sampling), ax = axes[1])
        sns.barplot(x=pair[0], y=pair[1], data=df.sample(sampling), ax = axes[2])

        plt.show

def r_values_vars(df, columns = ['bedroomcnt','bathroomcnt','calculatedfinishedsquarefeet','yearbuilt', 'lotsizesquarefeet', 'taxvaluedollarcnt']):
    pairs = combinations(columns, 2)
    alpha = 0.05
    outputs = []
    for pair in pairs:
        corr, p = stats.pearsonr(df[pair[0]], df[pair[1]])
        output = {
            'column1_x_column2':f"{pair[0]} x {pair[1]}",
            'r' : corr,
            'p-value' : p,
            'reject_null' : p < alpha
        }
        outputs.append(output)
    corr_tests = pd.DataFrame(outputs)
    return corr_tests