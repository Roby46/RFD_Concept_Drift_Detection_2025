import pandas as pd

#Split the MetaTarget database into multiple sub-dataset (one for each label)

percentages=["25","45","70","100"]
version= '1'
target="Class"

for percentage in percentages:
    filename=f"../Generate Configuration/{version}/Version_{str(version)}_split_{percentage}.csv"

    df=pd.read_csv(filename, sep=',')
    #df=df.drop(['log'],axis=1)
    df['Class']=df['Class'].astype(int)

    print(df)


    if 'log' in df.columns:
        df.drop(columns=['log'], inplace=True)

    # Get the unique values from the label column
    label_values = df[target].unique()

    # Create a dictionary to store the sub-dataframes
    sub_dataframes = {}

    # Split the dataframe based on the label column
    for value in label_values:
        sub_dataframes[value] = df[df[target] == value].copy()
    #
    # Print the sub-dataframes
    for value, sub_df in sub_dataframes.items():
        print(f"Sub-dataframe for value '{value}':")
        print(sub_df)
        filename=f"./{percentage}/Version_{version}_cluster_{value}_split_{percentage}_gen_v2.csv"
        print(filename)
        sub_df.to_csv(filename, index=None, header=False)


