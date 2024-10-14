import os
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm

# Liberare la cartella Incremental MetaTarget prdiction e lasciare solo i log puliti della versione testata

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def analyzeDependencies(received_df, percentage, version):
    print("Analyzing dependencies")
    print("Checking unique RFDs")
    # Ignore FutureWarnings related to deprecated usage
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


    # Create a progress bar
    progress_bar = tqdm(total=len(received_df), desc="Comparing rows")

    # Initialize lists to store rows
    unique_rows = []
    similar_rows = []
    duplicate_rows = []

    # Convert the DataFrame to a NumPy array for faster iteration
    received_array = received_df.to_numpy()

    # Iterate through rows
    for idx, row in enumerate(received_array):
        unique = True
        row_values = row[:-1]

        progress_bar.update()  # Update the progress bar

        for other_idx, other_row in enumerate(received_array):
            if idx != other_idx:
                other_row_values = other_row[:-1]
                if np.all(row_values == other_row_values):
                    duplicate_rows.append(row)
                    unique = False
                    break

        if unique:
            # Select rows with the same value in the first column
            same_first_column = received_array[received_array[:, 0] == row_values[0]]
            same_first_column = np.delete(same_first_column, np.where((same_first_column == row).all(axis=1)), axis=0)

            # Iterate through the selected rows
            for other_row2 in same_first_column:
                other_row_values2 = other_row2[:-1]

                # Check if the rows are similar
                similar = True
                for col_idx in range(1, len(row_values)):
                    if (row_values[col_idx] == '?' and other_row_values2[col_idx] != '?') or \
                            (row_values[col_idx] != '?' and other_row_values2[col_idx] == '?'):
                        similar = False

                if similar:
                    similar_rows.append(row)
                    unique = False
                    break  # Found a similarity, no need to check further

        if unique:
            unique_rows.append(row)

    # Close the progress bar
    progress_bar.close()

    # Create DataFrames for unique, similar, and duplicate rows
    columns = received_df.columns
    unique_df = pd.DataFrame(unique_rows, columns=columns)
    similar_df = pd.DataFrame(similar_rows, columns=columns)
    duplicate_df = pd.DataFrame(duplicate_rows, columns=columns)

    path_U = f"SplittedRFDs/{version}_RFDUnique" + str(percentage) + ".csv"      # cambiare
    path_S = f"SplittedRFDs/{version}_RFDSimilar" + str(percentage) + ".csv"      # cambiare
    path_D = f"SplittedRFDs/{version}_RFDDuplicates" + str(percentage) + ".csv"      # cambiare

    unique_df.to_csv(path_U, sep=';', index=None)
    similar_df.to_csv(path_S, sep=';', index=None)
    duplicate_df.to_csv(path_D, sep=';', index=None)

def analyze_folder(methods, percentage, version):
    folder_path = f"RFDs to splt/"
    csv_files = [file for file in os.listdir(folder_path) if file.endswith(f"{percentage}_gen_v2_cleaned.csv")]
    print(csv_files)

    if len(csv_files) != len(methods):
        print(len(csv_files))
        print(bcolors.FAIL + f"Number of CSV files for {percentage}% does not match the number of methods." + bcolors.ENDC)
        return

    dfs = []
    for csv_file in csv_files:
        print(csv_file)
        df = pd.read_csv(os.path.join(folder_path, csv_file))

        df['Method'] = csv_file.split('_cluster_')[1].split('_')[0]  # Extract method from filename
        dfs.append(df)
        print(bcolors.OKBLUE + "=" * 40 + bcolors.ENDC)

    merged_df = pd.concat(dfs, ignore_index=True)
    print(merged_df)
    analyzeDependencies(merged_df, percentage, version)

def main():
    methods = ["0", "1"]
    version=20
    #percentages = [25, 45, 70, 100]
    percentages = [25,45,70,100]
    for percentage in percentages:
        print(bcolors.OKCYAN + f"Checking for {percentage}%..." + bcolors.ENDC)
        analyze_folder(methods, percentage, version)
        print(bcolors.OKCYAN + "=" * 40 + bcolors.ENDC)

if __name__ == "__main__":
    main()
