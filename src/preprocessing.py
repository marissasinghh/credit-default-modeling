from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

def preprocess_data(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pre processes data for modeling. Receives train, val and test dataframes
    and returns numpy ndarrays of cleaned up dataframes with feature engineering
    already performed.

    Arguments:
        train_df : pd.DataFrame
        val_df : pd.DataFrame
        test_df : pd.DataFrame

    Returns:
        train : np.ndarrary
        val : np.ndarrary
        test : np.ndarrary
    """
    # Print shape of input data
    print("Input train data shape: ", train_df.shape)
    print("Input val data shape: ", val_df.shape)
    print("Input test data shape: ", test_df.shape, "\n")

    # Make a copy of the dataframes
    working_train_df = train_df.copy()
    working_val_df = val_df.copy()
    working_test_df = test_df.copy()

    # Correct outliers/anomalous values in numerical
    # columns (`DAYS_EMPLOYED` column).
    working_train_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_val_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_test_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)

    # Encode string categorical features
    # First selecting categorical columns
    cat_cols = working_train_df.select_dtypes(include='object').columns

     # Separate 2 categories and multi-category columns
    binary_cols = []
    multi_cols = []
    
    for col in cat_cols:
        n_unique = working_train_df[col].nunique()
        if n_unique == 2:
            binary_cols.append(col)
        else:
            multi_cols.append(col)

    if binary_cols:
        ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

        working_train_df[binary_cols] = ordinal_encoder.fit_transform(working_train_df[binary_cols])
        working_val_df[binary_cols] = ordinal_encoder.transform(working_val_df[binary_cols])
        working_test_df[binary_cols] = ordinal_encoder.transform(working_test_df[binary_cols])

    if multi_cols:
        onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')     
           
        train_encoded = onehot_encoder.fit_transform(working_train_df[multi_cols])
        val_encoded = onehot_encoder.transform(working_val_df[multi_cols])
        test_encoded = onehot_encoder.transform(working_test_df[multi_cols])
        
        # Get multi_col names
        encoded_cols = onehot_encoder.get_feature_names_out(multi_cols)
        
        # Drop original multi-category columns
        working_train_df = working_train_df.drop(columns=multi_cols)
        working_val_df = working_val_df.drop(columns=multi_cols)
        working_test_df = working_test_df.drop(columns=multi_cols)
        
        # Add encoded columns as new DataFrames
        working_train_df = pd.concat([working_train_df, pd.DataFrame(train_encoded, columns=encoded_cols, index=working_train_df.index)], axis=1)
        working_val_df = pd.concat([working_val_df, pd.DataFrame(val_encoded, columns=encoded_cols, index=working_val_df.index)], axis=1)
        working_test_df = pd.concat([working_test_df, pd.DataFrame(test_encoded, columns=encoded_cols, index=working_test_df.index)], axis=1)

    # Impute values for all columns with missing data 
    imputer_median = SimpleImputer(strategy='median')
    working_train_df = imputer_median.fit_transform(working_train_df)
    working_val_df = imputer_median.transform(working_val_df)
    working_test_df = imputer_median.transform(working_test_df)

    # Feature scaling with Min-Max scaler
    scaler = MinMaxScaler()
    train = scaler.fit_transform(working_train_df)  
    val = scaler.transform(working_val_df)
    test = scaler.transform(working_test_df)


    return train, val, test
