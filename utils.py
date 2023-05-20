import random

def show_random_5(df, col_name):
    '''
    Show random 5 values in the column in dataframe df
    '''

    # Generate random row and column indices
    random_row_indics = [random.randint(0, df.shape[0] - 1) for i in range(5)]
    for i in random_row_indics:
        print(df[col_name].iloc[i])

    return

