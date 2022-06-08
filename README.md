# Predicting Property Prices From Zillow Data

## Key Takeaways

- Major factors determining property price are:
    - square feet
    - number of bathrooms
    - location
- A Lasso Lars regression model predicts better than baseline, but is less predictive of luxury houses.
- Getting more data on the geographic location of the house will be more helpful in determining the price.


## Contents <a name='contents'></a>

*Note: the following links will only work on local copies of this notebook.*

1. <a href='#introduction'>Introduction</a>
2. <a href='#wrangle'>Wrangle Data</a>
3. <a href='#explore'>Explore Data</a>
4. <a href='#models'>Models</a>
5. <a href='#conclusion'>Conclusion </a>
6. <a href='#appendix'>Appendix</a>
    
## Introduction <a name='introduction'></a>

Zillow data was pulled from the database, and analyzed to determine which features can best determine the property value of a house.  This project is divided into three parts: wrangle, explore, and model.  Wrangle explains how the data is acquired and prepared for analysis and processing.  Explore looks at the data and applies visualization and hypothesis testing to discover drivers of the property value.  Finally, model builds a linear regression model to predict property values from the data. Each section and select subsections include a short list of key takeaways; these are followed by a discussion detailing the analysis choices made that are relevant to that (sub)section.

### Goals
- Test at least four drivers of the property value, so that the data and the model predictions can be better understood
- Wrangle data to prepare it for analysis, so that the data pulled in the future can be wrangled in a similar manner
- Build a model that beat the baseline models, so that Zillow can display information to the users of its website
- Include a report for future reference so that this project can be built upon in the future

### Project Plan
- Explore at least four variables of property prices
- Visualize variables
- Hypothesis test at least two variables
- Write this final report
- Python scripts that allow for the project to be reproducible

<a href='#contents'>Back to contents</a>

## Wrangle Data <a name='wrangle'></a>

#### Key Wrangle  Takeaways

- $\approx0.98$ of the data is retained or the original 52,441 rows
- 10 of the 59 rows are retained
- Data is acquired from the SQL database
- Any column with $> 0.01$ of nulls is dropped
- Redundant and innaccurate columns with significant missing data are removed
- Extreme outliers are removed from the data; this was determined based off of the typical customer of Zillow.
- Any remaining null valued rows are dropped

<a href='#contents'>Back to contents</a>

### Acquire the data <a name='acquire'></a>

#### Key Acquire Takeaways

- `wrangle.get_zillow_data()` is used to get the data from the database.
- 58 columns and 52441 rows are imported from the data

#### Discussion

Data is acquired via the `wrangle.get_zillow_data()` function. This function will query the SQL database, unless there is a saved .csv file present in the current directory.  The name of the file is set by the `wrangle.FILENAME` constant. This function has the following parameters:
- `query_db = False` (bool) : forces a query to the SQL database even if the .csv file is present.

In order to determine the extent of missing data, `wrangle.return_col_percent_null()` is used. The function has the followng parameters:
- `df` (DataFrame) : a dataframe containing the Zillow data

This function will return a dataframe with the following columns:
- `column_name` : The name of the column of the relevant column of `df`
- `percent_null` : The percent of rows in the `column_name` column of `df` that are null values
- `count_null` : The total number of null values in the `column_name` column of `df`

### Prepare and Clean the Data <a name='prepare'></a>

#### Key Prepare and Clean Takeaways
- $\approx0.98$ of the data is retained
- Columns with high percentage of nulls are dropped
- Redundant and innacurate data is also dropped
- Extreme outliers and remaining null valued rows are dropped
- `fips` values are converted to equivalent county names
- `latitude` and `longitude` are made into correct values

#### Discussion
Some of the remaining columns contain significant outliers. These can be dropped Consider the following columns:
- `bathroomcnt` : Most houses for Zillow users will have at most 6 bathrooms; any rows with more than 6 bathrooms are dropped.  In addition, any row that reports no bathrooms is dropped
- `bedroomcnt` : Similar to the number of bathrooms, any rows with no bedrooms or more than 6 bedrooms is dropped.  A typical homebuyer will not have more than 6 bedrooms.
- `calculatedfinishedsquarefeet` : Houses with less than 150 square feet or more than 6000 square feet are excluded from the analysis. Most of the rows are much smaller than this, and a typical user will not be interested in this type of house.
- `lotsizesquarefeet` : Any lot that is over 5 acres is excluded.  Most houses are on lots a fifth of the size.

Note that this continues to leave a significant number of outliers in the dataset; however, these provide more "mainstream" customers.

Redundant data can also be dropped.  Consider the following columns:
- `calculatedbathnbr` and `fullbathcnt`, the data can be calculated from the other two columns
- `finishedsquarefeet12` is a repetition of `calculatedfinishedsquarefeet`
- `assessmentyear` can calso be dropped as it is all the same value, 2016.
- `rmcnt` contains significant missing values are can be inferred from the seemingly more reliable `bedroomcnt` and `bathroomcnt`

Some of the data is not accurate.  Consider the `regionidzip` column.  It contains values that are outside of the range of zip code values for California. Some of the data is also in the wrong form.  For example `latitude` and `longitude` need to be multiplied by $10^{-6}$ to get values that are in the range for southern California.
The `fips` column can also be mapped to the relevant county names of Los Angeles, Ventura and Orange.

Some columns contain useful information, but that would be outside of the scope of this project, see <a href='#conclusion'>conclusion</a> for a discussion.  These columns are dropped:
- `censustractandblock` and `rawcensustractandblock` are dropped
- `propertycountylandusecode` might be useful, but determing its meaning is outside of the scope of this project

Foreign key rows are dropped.  These are:
- `propertylandusetypeid`
- `regionidcounty`
- `id` : `parcelid` is used to identify rows instead.

Data leakage columns are also dropped.  these are `taxamount` and `structuretaxvaluedollarcnt`.

The remaining rows that have nulls are dropped as they represent a small ($ < 0.01$ of the data in the rows).

`wrangle.wrangle_zillow()` is used to drop the columns, fix the values and drop the null valued rows. Is takes as a parameter `df`, a dataframe object. It uses the following functions:
- `wrangle.zillow_drop_columns()` : drops the columns using `wrangle.COLUMNS_TO_DROP` module constant (see <a href='#appendix'>appendix</a> for more information on the constants of the `wrangle.py` module.
- `wrangle.prepare_zillow_data()` : removes null valued rows, outliers and fixes the `latitude`, `longitude`, and `fips` columns.  In additon, it casts the datatypes to more efficient data types (typically some size of a `uint`)

$\approx0.98$ of the data is retained.

<a href='#contents'>Back to contents</a>

## Explore <a name='explore'></a>

### Key Explore Takeaways
- Data is split into `train`, `validate` and `test`
- All hypothesis testing is done use $α = 0.05$ 
- `calculatedfinishedsquarefeet` and `taxvaluedollarcnt` are highly positively correlated
- `bedroomcnt` and `bathroomcnt` are correlated with `taxvaluedollarcnt`, but `bathroomcnt` is particularly correlated.  It might be better to only pass `bathroomcnt` in the model. 
- Orange county has a higher average house, this is probably due to location effects
- Generally the top $0.25$ of house prices are near the coast (for example, Orange county), and in downtown Los Angeles.


This section uses the `explore.py` module which includes the following functions:
- `plot_variable_pairs()` plots all the pairs of variables possible and includes a regression line displayed in red.  It has the following parameters:
    - `df` (DataFrame) : a dataframe object
    - `columns_x` (list) : a list of column names to plot on the x-axis
    - `columns_y` (list) : a list of column names to plot on the y-axis
    - `sampling` (int) : the number of points from the data to plot
- `plot_categorical_and_continuous_vars()` plots a strip plot, a box plot and a bar plot.  It has the following parameters:
    - `df` (DataFrame) : a dataframe object
    - `columns_cat` (list) : a list of categorical columns to split `df` by
    - `columns_cont` (list) : a list of continuous columns to plot
    - `sampling` (int) : number of points to plot
- `r_values_vars()` performs a pearson r correlation test on all the possible pairs of columns that are passed. It returns a dataframe containing the column names, the r-value, the p-value and a boolean representing if the null hypothesis can be rejected. It has the following parameters:
    - `df` (DataFrame) : a Dataframe object
    - `columns` (list) : the columns to hypothesis test
- `t_test_by_cat()` performs a t-test to determine if the mean of a subsample is different from the non-category sample.  It has the following parameters:
    - `df` (DataFrame) : a DataFrame object
    - `columns_cat` (list) : a list of categorical columns to split `df` by
    - `columns_cont` (list) : a list of continuous columns to test the mean

Before the exploration stange it is important to split the data into train, validate and test subsets.  This is done using the `wrangle.split_zillow_data()` function.  This function takes as its parameter `df` a dataframe object and returns three DataFrames: `train`, `validate` and `test`; these three dataframes are of descending size.

### Are `calculatedfinishedsquarefeet` and `taxvaluedollarcnt` related? <a name='hypothesis1'></a>
- `calculatedfinishedsquarefeet` has a high correlation with the target variable

<a href='#contents'>Back to contents</a>

### What is the connection between `bedroomcnt`, `bathroomcnt` and `taxvaluedollarcnt`? <a name='hypothesis2'></a>
- `bathroomcnt` has a strong correlation with the target variable
- Number of bedrooms, the ratio between bedroom and bathrooms, and the sum of bedrooms and bathrooms were correlated, but less so.  They are probably largely colinear with `calculatedfinishedsquarefeet`, and it might be good to train a model without these categories

The number of bedroom and the number of bathrooms a house has correlated with the size of the house.  However, it might be the case that more expensive houses have more bathrooms, and likely more bedrooms.  In addition, it might be the case that more expensive houses have a lower ratio between bedrooms to bathroom -- that is, an expenisve house is more likely to have an ensuite bathroom for more bedrooms.

<a href='#contents'>Back to contents</a>

### What is the relationship between `county` and `taxvaluedollarcnt`? <a name='hypothesis3'></a>
- Orange county has the most expensive houses
- This is despite Orange county having the smallest lot sizes overall
- Most likely this is a function of location

Home pricers are often related to location; certain areas, because of nearby services such as schools, are more expensive than others.  Since there is county information in the dataset, the difference in which county has higher home prices and why is an important to explore.

<a href='#contents'>Back to contents</a>

### What is the relationship between `latitude` \ `logitude` \ `county` and `taxvaluedollarcnt`? <a name='hypothesis4'></a>
- Orange county has the highest proportion of luxury homes (defined as the top $0.25$ of home prices)
- County does not completely explain the geographic spread of home prices.  It looks like downtown Los Angeles is also a place with higher home prices.

<a href='#contents'>Back to contents</a>

### Explore discussion <a name='explorediscussion'></a>

Overall the best predictor of house prices is `calculatedfinishedsquarefeet`, which is closely followed by number of bathrooms.  There could be multiple causal factors of this. Obviously, a larger house is made of more materials, which in turn means more cost.  In addition, bathrooms are often rooms where people spend a lot of money in terms of materials (for example, tiled vs marble floors, and the number of expensive fixtures such as toilets and baths). However, these do not completely explain the difference in prices--location is important.  Certain counties are predictive of hosue prices; this could be due to services at the county level.  However, another factor to consider are subcounty geographic categories, for example coastal vs inland and downtown vs not-downtown.

<a href='#contents'>Back to contents</a>

## Models <a name='models'></a>

Four total regression models are created, trained and evaluated. The Lasso Lars Model is most effective at predicting house prices, although it is not as predictive for houses at the expensive end of the market.

This section uses the following functions from the custome modules:
- `wrangle.add_custom_columns()` adds the columns referenced in <a href='#explore'>explore</a>
- `wrangle.make_X_and_y()` which splits the data into the dependent and target variable sets.

<a href='#contents'>Back to contents</a>

## Conclusion <a name='conclusion'></a>

Major drivers of the house price are the following:
- The size of the house in square feet.  Larger houses are most expensive.
- The number of bathrooms. Like bedrooms, houses generally have more than one bathroom. In addition bathrooms tend to have materials that are most costly than others rooms.
- Location. Some counties are more likely to have expensive homes than others, and different areas within counties (for example downtown or near the coast) tend to be more expensive

A LassoLars regression model predicts the house price with reasonable predictions, although it tends to underestimate the houses at the higher end of the market.

### Recommendations for the future

From the exploration <a href='#explore'>section</a> it is clear that one of drivers of the house price is the house's geography. There are the building blocks of starting this analysis within the data.  For example, the data set contains census data blocks; if these were merged with data from the census bureau, that would give information on local economic, transportation, demographic, employment, etc information that could help determine house prices. In addition, using the latitude and longitude to determine the average price within a radius of the house could be useful, as I suspect that expensive houses are near other expensive houses; even better, if the values of an additional bedroom or bathroom within a neighborhood are calculated, this could be built into the model.

Another way to improve the predictions would be to clean the data differently in terms of outliers.  As it currently is done, there remain a lot of outliers within the data, and the model is not very good at predicting houses in the extreme upper end of the spectrum of house prices.  If more data was excluded initially, the model would be better at predictions for more typical houses, and another model could be used for luxury houses.  Intuitively the logic is this: there is a upper limit to the size of the house, but the materials of the house really have no upper limit (for example, everything could be made of gold in an extreme/gaudy case, or the entire house could be tiled in marble).

<a href='#contents'>Back to contents</a>

## Appendix <a name='appendix'></a>

### Reproducing this project
Download `wrangle.py`, `explore.py`, and `final_report.ipynb`.  In addition create a `env.py` based off of the included `env_example.py` file.  Then run `final_report.ipynb`.

### Data Dictionary

The following columns are retained:
- parcelid : row identifier
- bathroomcnt : number of bathrooms
- bedroomcnt : number of bedrooms
- calculatedfinishedsquarefeet : square feet in the house
- latitude
- longitude
- lotsizesquarefeet : lot size in square feet
- yearbuilt : year of construction
- taxvaluedollarcnt : target variable. Value of the house
- fips : relevant fips code mapped to county
- county : county name

The following are dropped columns:
- airconditioningtypeid
- architecturalstyletypeid
- basementsqft
- buildingclasstypeid
- buildingqualitytypeid
- decktypeid
- finishedfloor1squarefeet
- finishedsquarefeet13
- finishedsquarefeet15
- finishedsquarefeet50
- finishedsquarefeet6
- fireplacecnt
- garagecarcnt
- garagetotalsqft
- hashottuborspa
- heatingorsystemtypeid
- poolcnt
- poolsizesum
- pooltypeid10
- pooltypeid2
- pooltypeid7
- propertyzoningdesc
- regionidcity
- regionidneighborhood
- storytypeid
- threequarterbathnbr
- typeconstructiontypeid
- unitcnt
- yardbuildingsqft17
- yardbuildingsqft26
- numberofstories
- fireplaceflag
- taxdelinquencyflag
- taxdelinquencyyear
- taxamount
- structuretaxvaluedollarcnt
- landtaxvaluedollarcnt
- calculatedbathnbr
- fullbathcnt
- finishedsquarefeet12
- propertylandusetypeid
- regionidcounty
- propertycountylandusecode
- regionidzip
- assessmentyear
- censustractandblock
- rawcensustractandblock
- roomcnt
- id