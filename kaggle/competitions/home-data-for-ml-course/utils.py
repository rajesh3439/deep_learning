import pandas as pd
from typing import List, Tuple

from sklearn.compose import ColumnTransformer
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
def data_preprocessor(cat_cols:List, num_cols:List)->ColumnTransformer:
    """
    Preprocess the data

    Args:
    cat_cols: list of categorical columns
    num_cols: list of numerical columns

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
            ('num', numerical_transformer, num_cols),
            ('cat', catergorical_transformer, cat_cols)
        ])
    
    return preprocessor