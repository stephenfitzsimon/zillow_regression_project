#zillow explore module
#stephen fitzsimon
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from itertools import combinations, product
from scipy import stats
from sklearn.feature_selection import SelectKBest, f_regression

ALPHA = 0.05

def plot_variable_pairs(df,
                        columns_x = ['bedroomcnt','bathroomcnt','calculatedfinishedsquarefeet','yearbuilt','taxvaluedollarcnt', 'lotsizesquarefeet', 'latitude', 'longitude'],
                        columns_y = ['bedroomcnt','bathroomcnt','calculatedfinishedsquarefeet','yearbuilt','taxvaluedollarcnt', 'lotsizesquarefeet', 'latitude', 'longitude'],
                        sampling = 1000):
    '''plots a lmplot plot of all pairs of all columns passed in two lists'''
    #make the pairs
    pairs = product(columns_x, columns_y)
    for pair in pairs:
        #make a plot for every pair
        sns.lmplot(x=pair[0], y=pair[1], data=df.sample(sampling), line_kws={'color': 'red'})
        plt.show()

def plot_categorical_and_continuous_vars(df,
                                         columns_cat=['county'],
                                         columns_cont=['calculatedfinishedsquarefeet', 'yearbuilt', 'bedroomcnt', 'bathroomcnt', 'taxvaluedollarcnt', 'latitude', 'longitude'],
                                         sampling = 1000):
    '''plots a strip plot, a box plot, and a barplot for all the combinations passed
    from columns_cat, and columns_cont'''
    #make all the pairs
    pairs = product(columns_cat, columns_cont)
    for pair in pairs:
        #set up for subplots
        sns.set(rc={"figure.figsize":(15, 6)}) 
        fig, axes = plt.subplots(1, 3)

        #make the plots 
        sns.stripplot(x=pair[0], y=pair[1], data=df.sample(sampling), ax = axes[0])
        sns.boxplot(x=pair[0], y=pair[1], data=df.sample(sampling), ax = axes[1])
        sns.barplot(x=pair[0], y=pair[1], data=df.sample(sampling), ax = axes[2])
        plt.show

def r_values_vars(df,
                columns = ['bedroomcnt','bathroomcnt','calculatedfinishedsquarefeet','yearbuilt', 'lotsizesquarefeet', 'taxvaluedollarcnt', 'latitude', 'longitude']):
    '''performs a correlation test for all pairs of columns passed
    returns a dataframe with results'''
    #make all the pairs
    pairs = combinations(columns, 2)
    outputs = []
    #perform a pearson r test on all the pairs
    for pair in pairs:
        #perform the test
        corr, p = stats.pearsonr(df[pair[0]], df[pair[1]])
        #store the output
        output = {
            'correlation':f"{pair[0]} x {pair[1]}",
            'r' : corr,
            'p-value' : p,
            'reject_null' : p < ALPHA
        }
        outputs.append(output)
    # return the results
    corr_tests = pd.DataFrame(outputs)
    return corr_tests

def t_test_by_cat(df,
                columns_cat=['county'],
                columns_cont=['calculatedfinishedsquarefeet', 'yearbuilt', 'bedroomcnt', 'bathroomcnt', 'taxvaluedollarcnt', 'latitude', 'longitude'],
                ):
    '''Performs a t-test for all subcategories of columns_cat and paored with every column in columns cat
    returns results as a dataframe'''
    outputs = []
    pairs_by_cat = {}
    #get pairs for every sub_Cat
    for category in columns_cat:
        #get subcategory names
        subcats = df[category].unique().tolist()
        #make the pairs
        pairs = list(product(subcats, columns_cont))
        pairs_by_cat[category] = pairs
    for category in columns_cat:
        pairs = pairs_by_cat[category]
        for pair in pairs:
            #subset into county_x and not county_x
            category_x = df[df[category] == pair[0]][pair[1]]
            not_category_x = df[~(df[category] == pair[0])][pair[1]].mean()
            #do the stats test
            t, p = stats.ttest_1samp(category_x, not_category_x)
            output = {
                'category_name':pair[0],
                'column_name':pair[1],
                'p':p,
                'reject_null': p < ALPHA
            }
            outputs.append(output)
    #return as a dataframe
    return pd.DataFrame(outputs)

def get_k_features(X_df, y_df, k_num):
    '''a f_selector that returns the k best values'''
    # parameters: f_regression stats test, give me 8 features
    f_selector = SelectKBest(f_regression, k=k_num)
    # find the top 8 X's correlated with y
    f_selector.fit(X_df, y_df)
    # boolean mask of whether the column was selected or not. 
    feature_mask = f_selector.get_support()
    # get list of top K features. 
    f_feature = X_df.iloc[:,feature_mask].columns.tolist()
    return f_feature

