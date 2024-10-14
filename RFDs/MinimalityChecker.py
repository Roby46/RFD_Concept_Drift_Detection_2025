import pandas as pd
import numpy as np
import csv

#methods=["alignment", "doc2vec", "node2vec"]
methods=[0,1]
percentages=[25,45,70,100]
#percentages = [40]
config=2
version=20

def minimalityCheck(df_toCheck, final_df, encoding, percentage, version):

    #print(df_toCheck)

    print("MinimalityCheck for", encoding , "percentage", percentage)

    count=0
    # Create a progress bar
    #progress_bar = tqdm(total=len(df_toCheck), desc="Comparing rows")

    # Definire l'header del DataFrame (sostituire con i nomi delle colonne reali)
    header = df_toCheck.columns.tolist()

    # Specificare il percorso del file CSV
    csv_file_path = f"../MinimalRFDS/MinimalRFDs_v{version}_{str(encoding)}_{str(percentage)}.csv"

    # Open the CSV file in write mode
    with open(csv_file_path, mode='w', newline='') as file:
        csv_writer = csv.writer(file, delimiter=';')

        # Write the row data to the CSV file
        csv_writer.writerow(header)

    for idx, row in df_toCheck.iterrows():
        #progress_bar.update()
        property1 = True
        property2 = True
        row_values = row.values
        # print(bcolors.OKCYAN + 40*"*" + bcolors.ENDC)
        #print("Riga da valutare\n", row_values)
        subset = final_df.loc[final_df['RHS'] == row_values[0]]
        # print("Controllo della minimalità su:\n")
        # print(subset)
        mask1 = (row_values == '?')
        rhs = row_values[0]
        rhs_row_index = (int(rhs[3:])) + 1

        # Exclude the first element by slicing the array
        row_values_sliced = row_values[1:]
        # Find the indices where values are different from '?'
        lhs_positions = set(np.where(row_values_sliced != '?')[0] + 1)  # Add 1 to adjust for the slice
        lhs_positions.remove(rhs_row_index)

        for other_idx, other_row in subset.iterrows():
            # Check sulle due proprietà
            # Per violare la 1, other_row deve avere gli stessi identici attributi e le soglie devono esser >= a quelli di row_values. Almeno una deve essere strettamente maggiore
            # Controllo se ha gli stessi attributi
            # Create a mask for '?' in both arrays
            other_row_values = other_row.values
            #print(50 * "==")
            #print("Dipendenza attuale: ", row_values)
            #print("Confronto con: ", other_row_values)
            mask2 = (other_row_values == '?')
            # Check if the masks are equal, indicang '?' in the same positions
            result = np.array_equal(mask1, mask2)
            allgreater = True
            epsilonGreaterThan0 = 0

            if (result):

                #print("Stessi attributi, controllo la prima proprietà")
                row_lhs_thresholds = (row_values[list(lhs_positions)]).astype(float)
                other_row_lhs_thresholds = (other_row_values[list(lhs_positions)]).astype(float)

                #print("Threshold lhs originali ", row_lhs_thresholds)
                #print("Threshold lhs di confronto", other_row_lhs_thresholds)

                #print(other_row_lhs_thresholds)


                epsilon_lhs = other_row_lhs_thresholds - row_lhs_thresholds
                epsilon_rhs = float(row_values[rhs_row_index]) - float(other_row_values[rhs_row_index])

                #print("Differenze lhs: ", epsilon_lhs)
                #print("Differenze rhs: ", epsilon_rhs)

                for value in epsilon_lhs:
                    if (value < 0):
                        allgreater = False
                    elif (value > 0):
                        epsilonGreaterThan0 = epsilonGreaterThan0 + 1

                if (epsilon_rhs > 0):
                    epsilonGreaterThan0 = epsilonGreaterThan0 + 1
                elif (epsilon_rhs < 0):
                    allgreater = False

                if (allgreater and epsilonGreaterThan0 >= 1):
                    property1 = False

            #print("epsilon maggiori di 0: ", epsilonGreaterThan0)

            if (allgreater and epsilonGreaterThan0 >= 1):
                property1 = False
                #print(bcolors.FAIL + "Proprietà 2 violata" + bcolors.ENDC)
                break

            if(config==2): #nuovo controllo
                # Tolgo il nome dell'RHS
                other_row_values_sliced = other_row_values[1:]
                # Trovo le posizioni degli attributi coinvolti nella RFD
                other_lhs_positions = set(np.where(other_row_values_sliced != '?')[0] + 1)  # Add 1 to adjust for the slice
                #Rimuovo la posizione dell'RHS
                other_lhs_positions.remove(rhs_row_index)
                allgreater = True
                epsilonGreaterThan0 = 0

                #Se l'LHS ha meno attributi della dipendenza da valutare
                if (len(other_lhs_positions) < len(lhs_positions)):
                    #Vedo se l'LHS è un sottoinsieme dell'LHS della dipendenza da valutare
                    if (other_lhs_positions.issubset(lhs_positions)):
                        #print("LHS è sottoinsieme")
                        #Position to check contiene le posizioni degli attributi coinvolti nell'LHS
                        position_to_check = other_lhs_positions.copy()
                        #Estraggo le treshold da queste posizioni sia per la dipendenza da valutare sia da quella usata per il confronto
                        row_lhs_thresholds = (row_values[list(position_to_check)]).astype(float)
                        other_row_lhs_thresholds = (other_row_values[list(position_to_check)]).astype(float)
                        #print("Threshold lhs originali ", row_lhs_thresholds)
                        #print("Threshold lhs di confronto", other_row_lhs_thresholds)
                        #Calcolo il valore di epsilon sull'LHS
                        epsilon_lhs = other_row_lhs_thresholds - row_lhs_thresholds
                        # Calcolo il valore di epsilon sull'RHS
                        epsilon_rhs = float(row_values[rhs_row_index]) - float(other_row_values[rhs_row_index])
                        #print("Differenze lhs: ", epsilon_lhs)
                        #print("Differenze rhs: ", epsilon_rhs)

                        #Controllo se le epsilon sono tutte non minori di zero e conto quelle che sono maggiori di zero
                        for value in epsilon_lhs:
                            if (value < 0):
                                allgreater = False
                            elif (value > 0):
                                epsilonGreaterThan0 = epsilonGreaterThan0 + 1
                        #Controllo se l'epsilon per l'rhs sia non minore di zero e aggiorno il conteggio nel caso sia maggiore di zero
                        if (epsilon_rhs > 0):
                            epsilonGreaterThan0 = epsilonGreaterThan0 + 1
                        elif (epsilon_rhs < 0):
                            allgreater = False

                        #print("epsilon maggiori di 0: ", epsilonGreaterThan0)
                        #Se tutte le epsilon sono >=0 e almeno una è >0 la dipendenza valutata non è minimale
                        if (allgreater and epsilonGreaterThan0 >= 1):
                             #print(bcolors.FAIL + "Proprietà 2 violata" + bcolors.ENDC)
                            property2 = False
                        else:
                            #Controllo il caso in cui tutte le soglie coinvolte siano uguali tra le due dipendenze. Anche in questo caso la dipendenza valutata non è minimale
                            if (np.sum(epsilon_lhs) == 0 and epsilon_rhs == 0):
                                #print(bcolors.FAIL + "Proprietà 2 violata" + bcolors.ENDC)
                                property2 = False
            else:
                # Exclude the first element by slicing the array
                other_row_values_sliced = other_row_values[1:]
                # Find the indices where values are different from '?'
                other_lhs_positions = set(
                    np.where(other_row_values_sliced != '?')[0] + 1)  # Add 1 to adjust for the slice
                other_lhs_positions.remove(rhs_row_index)

                # print(bcolors.ENDC +"")
                # print(bcolors.OKGREEN +"")

                if (len(other_lhs_positions) < len(lhs_positions)):
                    # print("LHS minore")
                    if (other_lhs_positions.issubset(lhs_positions)):
                        # print("LHS è sottoinsieme")
                        # Extract elements at the specified positions from both arrays
                        position_to_check = other_lhs_positions.copy()
                        position_to_check.add(rhs_row_index)

                        # print("Posizioni da controllare: ", position_to_check)

                        row_thresholds = row_values[list(position_to_check)]
                        other_row_thresholds = other_row_values[list(position_to_check)]

                        # print("Threshold originali: ", row_thresholds)
                        # print("Threshold da confrontare: ", other_row_thresholds)

                        # Check if the elements at the specified positions are equal
                        are_equal = True

                        for i in range(len(row_thresholds)):
                            if (str(row_thresholds[i]) != str(other_row_thresholds[i])):
                                are_equal = False

                        if (are_equal):
                            property2 = False
                            # print("Proprietà 2 violata")
                            break

        if property1 and property2:
            count=count+1
            #print("Minimale")

            # Open the CSV file in append mode and write a new line
            with open(csv_file_path, mode='a', newline='') as file:
                csv_writer = csv.writer(file, delimiter=';')
                # Write the additional data to the CSV
                csv_writer.writerow(row_values)

    print(count)
    print("Dipendenze inziali: ", len(df_toCheck), "Dipendenze finali: " ,count)


for encoding in methods:
    for percentage in percentages:
        filename=f"SplittedRFDs/{version}_RFDUnique"+str(percentage)+".csv"
        #filename=f"DOMINOLogs/Completati/SortedRFDUnique"+str(percentage)+".csv"

        df=pd.read_csv(filename, sep=';')

        #print(len(df))

        df_toCheck= df.loc[df['Method'] == encoding]

        df_toCheck= df_toCheck.drop(['Method'], axis=1)

        other_encodings=np.setdiff1d(methods, encoding)
        # Loop attraverso i file CSV
        dfs=[]


        for enc in other_encodings:
            filename = (
                    f"Original RFDs/output_false_0_MetaTarget_{3}_Version_{version}_cluster_{str(enc)}_{percentage}_gen_v2_cleaned.csv"
            )
            df_temp = pd.read_csv(filename)
            dfs.append(df_temp)

        final_dfs=pd.concat(dfs, ignore_index=True)
        #print(final_dfs)
        minimalityCheck(df_toCheck, final_dfs, encoding, percentage, version)

        # Concatena i DataFrame in uno unico
        final_df = pd.concat(dfs, ignore_index=True)
