import pandas as pd
import numpy as np
import os
import tensorflow as tf

## I had to fill these functions out

#Question 3
def reduce_dimension_ndc(df, ndc_df):
    '''
    df: pandas dataframe, input dataset
    ndc_df: pandas dataframe, drug code dataset used for mapping in generic names
    return:
        df: pandas dataframe, output dataframe with joined generic drug name
    '''
    ndc_map = ndc_df.drop_duplicates(subset='NDC_Code').set_index('NDC_Code')['Non-proprietary Name']
    reduce_dim_df = df.copy()
    reduce_dim_df['generic_drug_name'] = reduce_dim_df['ndc_code'].map(ndc_map)
    return reduce_dim_df

#Question 4
def select_first_encounter(df):
    '''
    df: pandas dataframe, line-level dataframe with all encounters
    return:
        - first_encounter_df: line-level dataframe containing all records
          from the first encounter for each patient
    '''
    first_encounter_ids = df.groupby('patient_nbr')['encounter_id'].transform('min')
    first_encounter_df = df.loc[
        df['encounter_id'].eq(first_encounter_ids)
    ].reset_index(drop=True)
    return first_encounter_df


#Question 6
def patient_dataset_splitter(df, patient_key='patient_nbr'):
    '''
    df: pandas dataframe, input dataset that will be split
    patient_key: string, column that is the patient id

    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
     - test: pandas dataframe,
    '''
    unique_patients = df[patient_key].unique()
    np.random.seed(0)
    shuffled_patients = np.random.permutation(unique_patients)

    n = len(shuffled_patients)
    train_end = int(n * 0.6)
    val_end = train_end + int(n * 0.2)

    train_patients = shuffled_patients[:train_end]
    val_patients = shuffled_patients[train_end:val_end]
    test_patients = shuffled_patients[val_end:]

    train = df[df[patient_key].isin(train_patients)].reset_index(drop=True)
    validation = df[df[patient_key].isin(val_patients)].reset_index(drop=True)
    test = df[df[patient_key].isin(test_patients)].reset_index(drop=True)

    return train, validation, test

#Question 7
def create_tf_categorical_feature_cols(categorical_col_list,
                              vocab_dir='./diabetes_vocab/'):
    '''
    categorical_col_list: list, categorical field list that will be transformed with TF feature column
    vocab_dir: string, the path where the vocabulary text files are located
    return:
        output_tf_list: list of TF feature columns
    '''
    output_tf_list = []
    for c in categorical_col_list:
        vocab_file_path = os.path.join(vocab_dir,  c + "_vocab.txt")
        tf_categorical_feature_column = tf.feature_column.categorical_column_with_vocabulary_file(
            key=c, vocabulary_file=vocab_file_path, num_oov_buckets=1)
        tf_categorical_feature_column = tf.feature_column.indicator_column(tf_categorical_feature_column)
        output_tf_list.append(tf_categorical_feature_column)
    return output_tf_list

#Question 8
def normalize_numeric_with_zscore(col, mean, std):
    '''
    This function can be used in conjunction with the tf feature column for normalization
    '''
    return (col - mean)/std



def create_tf_numeric_feature(col, MEAN, STD, default_value=0):
    '''
    col: string, input numerical column name
    MEAN: the mean for the column in the training data
    STD: the standard deviation for the column in the training data
    default_value: the value that will be used for imputing the field

    return:
        tf_numeric_feature: tf feature column representation of the input field
    '''

    # Here I define the Z-score normalization function (Z = (x - mean) / standard deviation)
    # I guard against zero/non-finite std, which would turn the z-score into NaN/inf
    MEAN = float(MEAN) if np.isfinite(MEAN) else 0.0
    STD = float(STD) if np.isfinite(STD) and STD > 0 else 1.0

    def z_score_normalizer(x):
        return (tf.cast(x, tf.float32) - MEAN) / STD

    # It is also necessary to create a numerical column (la columna numérica) of TensorFlow 
    # using the compatibility v1 API, and inject the normalizer function
    import tensorflow as tf
    tf_numeric_feature = tf.compat.v1.feature_column.numeric_column(
        key=col, 
        normalizer_fn=z_score_normalizer, 
        default_value=default_value, 
        dtype=tf.float32
    )

    # The final result is the TensorFlow numeric feature column (this is the column that caused the NameError)
    return tf_numeric_feature


#Question 9
def get_mean_std_from_preds(diabetes_yhat):
    '''
    diabetes_yhat: TensorFlow Probability distribution object de las predicciones
    '''
    m = diabetes_yhat.mean()
    s = diabetes_yhat.stddev()
    
    return m, s

# Question 10
def get_student_binary_prediction(df, col):
    '''
    df: pandas dataframe prediction output dataframe
    col: str,  probability mean prediction field
    return:
        student_binary_prediction: pandas dataframe converting input to flattened numpy array and binary labels
    '''
    binary_series = df[col].apply(lambda x: 1 if x >= 5.0 else 0)
    
    student_binary_prediction = binary_series.values
    
    return student_binary_prediction
