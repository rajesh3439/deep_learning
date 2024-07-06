import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

from typing import List, Tuple

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

def sep_columns_from_desc(filename:str='data_description.txt', 
                          data_cols:List=None)->Tuple[List, List]:
    """
    Separates the columns in the data_description.txt file into
    categorical and numerical columns

    Parameters
    ----------
    filename : str
        The filename of the data_description.txt file
    data_cols : List
        The columns in the data

    Returns
    -------
    List
        A list of categorical columns
    List
        A list of numerical columns
    """
        
    cat_cols = list()
    num_cols = list()

    line_spaces = 0

    with open(filename) as f:
        line = f.readline()
        col = line.split(':')[0] if ':' in line else None
        line = f.readline()
        while line != '':
            new_col = line.split(':')[0] if ':' in line else None
            if new_col in data_cols:
                if line_spaces > 2:
                    cat_cols.append(col)
                else:
                    num_cols.append(col)
                # reset line_spaces
                line_spaces = 0
                # set col to new_col
                col = new_col
            else:
                line_spaces += 1

            line = f.readline()

        # Check the last column
        if line_spaces > 1:
            cat_cols.append(col)
        else:
            num_cols.append(col)

    # Add BedroomAbvGr and KitchenAbvGr as these columns are different
    # in description and data
    cat_cols.append('BedroomAbvGr')
    cat_cols.append('KitchenAbvGr')

    return cat_cols, num_cols


def missing_values_by_col(df:pd.DataFrame)->pd.DataFrame:
    """
    Returns a DataFrame with the columns that have missing values
    and the percentage of missing values

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to check for missing values

    Returns
    -------
    pd.DataFrame
        A DataFrame with the columns that have missing values
        and the percentage of missing values
    """
    return pd.DataFrame(df.isna().sum()).reset_index()\
    .rename(columns={0: 'missing_values'})\
    .sort_values(by='missing_values', ascending=False)\
    .query('missing_values > 0')\
    .pipe(lambda x: x.assign(percentage_missing = x.missing_values / df.shape[0] * 100))\
    .reset_index()


# preprocessing for categorical and numerical data
def data_preprocessor(cat_cols:List, num_cols:List, rm_cols:List)->ColumnTransformer:
    """
    Preprocess the data

    Args:
    cat_cols: list of categorical columns
    num_cols: list of numerical columns
    rm_cols: list of columns to remove

    Returns:
    ColumnTransformer object
    """

    catergorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # combine preprocessing steps
    preprocessor = ColumnTransformer( transformers=[
            ('remove_cols', 'drop', rm_cols),
            ('num', numerical_transformer, num_cols),
            ('cat', catergorical_transformer, cat_cols)
        ])
    
    return preprocessor

def findCorrelation(corr, cutoff=0.9, exact=None):
    """
    This function is the Python implementation of the R function 
    `findCorrelation()`.
    
    Relies on numpy and pandas, so must have them pre-installed.
    
    It searches through a correlation matrix and returns a list of column names 
    to remove to reduce pairwise correlations.
    
    For the documentation of the R function, see 
    https://www.rdocumentation.org/packages/caret/topics/findCorrelation
    and for the source code of `findCorrelation()`, see
    https://github.com/topepo/caret/blob/master/pkg/caret/R/findCorrelation.R
    
    -----------------------------------------------------------------------------

    Parameters:
    -----------
    corr: pandas dataframe.
        A correlation matrix as a pandas dataframe.
    cutoff: float, default: 0.9.
        A numeric value for the pairwise absolute correlation cutoff
    exact: bool, default: None
        A boolean value that determines whether the average correlations be 
        recomputed at each step
    -----------------------------------------------------------------------------
    Returns:
    --------
    list of column names
    -----------------------------------------------------------------------------
    Example:
    --------
    R1 = pd.DataFrame({
        'x1': [1.0, 0.86, 0.56, 0.32, 0.85],
        'x2': [0.86, 1.0, 0.01, 0.74, 0.32],
        'x3': [0.56, 0.01, 1.0, 0.65, 0.91],
        'x4': [0.32, 0.74, 0.65, 1.0, 0.36],
        'x5': [0.85, 0.32, 0.91, 0.36, 1.0]
    }, index=['x1', 'x2', 'x3', 'x4', 'x5'])

    findCorrelation(R1, cutoff=0.6, exact=False)  # ['x4', 'x5', 'x1', 'x3']
    findCorrelation(R1, cutoff=0.6, exact=True)   # ['x1', 'x5', 'x4'] 
    """
    
    def _findCorrelation_fast(corr, avg, cutoff):

        combsAboveCutoff = corr.where(lambda x: (np.tril(x)==0) & (x > cutoff)).stack().index

        rowsToCheck = combsAboveCutoff.get_level_values(0)
        colsToCheck = combsAboveCutoff.get_level_values(1)

        msk = avg[colsToCheck] > avg[rowsToCheck].values
        deletecol = pd.unique(np.r_[colsToCheck[msk], rowsToCheck[~msk]]).tolist()

        return deletecol


    def _findCorrelation_exact(corr, avg, cutoff):

        x = corr.loc[(*[avg.sort_values(ascending=False).index]*2,)]

        if (x.dtypes.values[:, None] == ['int64', 'int32', 'int16', 'int8']).any():
            x = x.astype(float)

        x.values[(*[np.arange(len(x))]*2,)] = np.nan

        deletecol = []
        for ix, i in enumerate(x.columns[:-1]):
            for j in x.columns[ix+1:]:
                if x.loc[i, j] > cutoff:
                    if x[i].mean() > x[j].mean():
                        deletecol.append(i)
                        x.loc[i] = x[i] = np.nan
                    else:
                        deletecol.append(j)
                        x.loc[j] = x[j] = np.nan
        return deletecol

    
    if not np.allclose(corr, corr.T) or any(corr.columns!=corr.index):
        raise ValueError("correlation matrix is not symmetric.")
        
    acorr = corr.abs()
    avg = acorr.mean()
        
    if exact or exact is None and corr.shape[1]<100:
        return _findCorrelation_exact(acorr, avg, cutoff)
    else:
        return _findCorrelation_fast(acorr, avg, cutoff)
    

def cat_correlation(df:pd.DataFrame, cat_cols:List[str])->List[str]:
    """
    Returns the columns that are correlated with each other
    using the chi2_contingency test

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to check for correlation
    cat_cols : List[str]
        The categorical columns to check for correlation

    Returns
    -------
    np.ndarray
        The columns that are correlated with each other
    """

    cat_df = df[cat_cols]
    cat_df = cat_df.fillna('NA')
    
    corr_df = pd.DataFrame({col: {col2: chi2_contingency(
                        pd.crosstab(cat_df[col], cat_df[col2]))[1] 
                        for col2 in cat_cols} for col in cat_cols})
    
    # set upper triangle values to nan
    corr_df.values[np.triu_indices_from(corr_df)] = np.nan

    # get columns with p-value < 0.05
    corr_cols=[r for r,c in zip(*np.where(corr_df.values < 0.05))]

    # get the unique columns
    rm_cat_cols = np.unique(corr_cols)

    return np.array(cat_cols)[rm_cat_cols].tolist()