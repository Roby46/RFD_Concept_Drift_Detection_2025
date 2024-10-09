import pandas as pd
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
import os


def cls():
    os.system('cls' if os.name=='nt' else 'clear')

def generate_and_evaluate_dataset(df_all_logs, percentages, sizes, threshold, max_attempts, target):
    # numero tentativi
    attempts=0

    # Estrai la prima percentuale e grandezza per il training set
    training_percentage = percentages.pop(0)
    training_size = sizes.pop(0)

    print(df_all_logs)
    print(df_all_logs.info)

    while attempts < max_attempts:

        print("Tentativo:", attempts)

        #Viene settato a false se la configurazione non soddisfa i criteri
        configuration_found = True

        attempts=attempts+1

        # Dizionario che conterrà i dataframe da salvare nel caso in cui la configurazione sia ok
        data_dictionary={}

        #Dizionario delle metriche del modello del passo precedente
        metrics_dict_prev={}

        df_all_logs_local=df_all_logs.copy()

        # Log originali per il training. Contiene sia "log" che "encoding"
        training_df = df_all_logs_local.sample(n=training_size)

        data_dictionary['Training_with_log_names']=training_df

        df_all_logs_local = df_all_logs_local.drop(training_df.index)

        #Log di training.  Contiene solo encoding
        training_df_for_model = training_df.copy()

        incremental_df=training_df_for_model.copy()
        incremental_df_predictions=training_df_for_model.copy()

        rf = RandomForestClassifier(random_state=4, n_jobs=-1)

        y_train = training_df_for_model[target]
        X_train = training_df_for_model.drop([target], axis=1)

        #Addestramento modello
        rf.fit(X_train, y_train)

        for i, percentage in enumerate(percentages):
            print("Test per la percentuale:", percentage)
            size = sizes[i]

            # Calcola la grandezza effettiva
            effective_size = size - training_size if i == 0 else size - sizes[i - 1]
            #print("Percentage:", percentage, "Size:", size, "Effective Size:", effective_size)

            df_current_percentage = df_all_logs_local.sample(n=effective_size)
            df_all_logs_local = df_all_logs_local.drop(df_current_percentage.index)

            #df_current_percentage=df_current_percentage.drop(["log"], axis=1)

            incremental_df = pd.concat([incremental_df, df_current_percentage], ignore_index=True)

            #x_to_n
            key = f"{percentage}_split"
            data_dictionary[key] = df_current_percentage

            #reduced_metaCover_Type_n_true_labels
            key=f"{percentage}_true_labels"
            data_dictionary[key]=incremental_df

            y_true = df_current_percentage[target]
            X_test = df_current_percentage.drop([target], axis=1)
            y_pred = rf.predict(X_test)

            classification_rep = classification_report(y_true, y_pred)

            print(classification_rep)

            # Calculate and store precision, recall, f1-score, and support for each class
            report_dict = classification_report(y_true, y_pred, output_dict=True)

            #f1_score_class_1 = report_dict['1']['f1-score']
            #f1_score_class_2 = report_dict['2']['f1-score']
            # f1_score_class_3 = report_dict['3']['f1-score']
            # f1_score_class_4 = report_dict['4']['f1-score']
            # f1_score_class_5 = report_dict['5']['f1-score']
            # f1_score_class_6 = report_dict['6']['f1-score']
            # f1_score_class_7 = report_dict['7']['f1-score']

            #print(f1_score_class_1)
            #print(f1_score_class_2)
            # print(f1_score_class_3)
            # print(f1_score_class_4)
            # print(f1_score_class_5)
            # print(f1_score_class_6)
            # print(f1_score_class_7)


            metrics_dict = {
                "precision": report_dict["weighted avg"]["precision"],
                "recall": report_dict["weighted avg"]["recall"],
                "f1-score": report_dict["weighted avg"]["f1-score"],
                "support": report_dict["weighted avg"]["support"]
            }

            print(metrics_dict)

            X_test[target] = y_pred

            incremental_df_predictions= pd.concat([incremental_df_predictions, X_test], ignore_index=True)

            #x_to_n_predicted
            key = f"{percentage}"
            data_dictionary[key] = X_test

            #reduced_metaCover_Type_n
            key=f"{percentage}_predicted"
            data_dictionary[key]=incremental_df_predictions

            if(i==0):
                metrics_dict_prev={
                      "precision": report_dict["weighted avg"]["precision"],
                      "recall": report_dict["weighted avg"]["recall"],
                      "f1-score": report_dict["weighted avg"]["f1-score"],
                }
            else:
                metrics_dict_curr = {
                      "precision": report_dict["weighted avg"]["precision"],
                      "recall": report_dict["weighted avg"]["recall"],
                      "f1-score": report_dict["weighted avg"]["f1-score"],
                }
                if(i==1):
                    #print("Controllo se",metrics_dict_prev["f1-score"],"è maggiore di",metrics_dict_curr["f1-score"] + threshold)
                    if (metrics_dict_prev["f1-score"] > metrics_dict_curr["f1-score"] + threshold):
                        print("OK")
                        metrics_dict_prev = {
                            "precision": report_dict["weighted avg"]["precision"],
                            "recall": report_dict["weighted avg"]["recall"],
                            "f1-score": report_dict["weighted avg"]["f1-score"],
                        }
                    else:
                        configuration_found = False
                        break
                else:
                    #print("Controllo se",metrics_dict_prev["f1-score"],"è maggiore di",metrics_dict_curr["f1-score"] + 0.01)
                    if (metrics_dict_prev["f1-score"] > metrics_dict_curr["f1-score"] + threshold):
                        print("OK")
                        metrics_dict_prev = {
                            "precision": report_dict["weighted avg"]["precision"],
                            "recall": report_dict["weighted avg"]["recall"],
                            "f1-score": report_dict["weighted avg"]["f1-score"],
                        }
                    else:
                        configuration_found = False
                        break


        if(configuration_found):
            print("Configurazione trovata")
            print("Tuple rimanenti (deve essere vuoto): ", df_all_logs_local)
            print("Salvataggio...")

            model_path = f"{version}/TrainedModel.pkl"

            with open(model_path, 'wb') as file:
                pickle.dump(rf, file)

            int_percentages = [int(percentage * 100) for percentage in percentages]
            int_training_percentage = int(training_percentage * 100)

            data_dictionary['Training_with_log_names'].to_csv(
                f"{version}/Version_{version}_split_{int_training_percentage}.csv",
                index=False)

            for i, percentage in enumerate(int_percentages):
                if(i==0):
                    #Split con true labels
                    key=f"{percentages[i]}_split"
                    data_dictionary[key].to_csv(f"{version}/{int_training_percentage}_to_{percentage}.csv", index=False)
                    #Split con predizioni
                    key = f"{percentages[i]}"
                    data_dictionary[key].to_csv(f"{version}/{int_training_percentage}_to_{percentage}_predicted.csv", index=False)
                else:
                    #Split con true labels
                    key = f"{percentages[i]}_split"
                    data_dictionary[key].to_csv(f"{version}/{int_percentages[i-1]}_to_{percentage}.csv", index=False)
                    # Split con predizioni
                    key = f"{percentages[i]}"
                    data_dictionary[key].to_csv(f"{version}/{int_percentages[i-1]}_to_{percentage}_predicted.csv",
                                                index=False)

                # Dataset per il discovery
                key = f"{percentages[i]}_predicted"
                data_dictionary[key].to_csv(
                        f"{version}/Version_{version}_split_{percentage}.csv",
                        index=False)

                # Dataset per il discovery con true labels
                key = f"{percentages[i]}_true_labels"
                data_dictionary[key].to_csv(
                        f"{version}/Version_{version}_split_{percentage}_true_labels.csv",
                        index=False)
            break
        else:
            print("Devo riprovare")
            print(350 * "=")
            print(350 * "=")
            print(350 * "=")



# Parametri iniziali
max_attempts = 90000
threshold=0.004
version=1
target="Class"

version_path=f"{version}"
os.makedirs(version_path, exist_ok=True)

# Meta-Database di riferimento
filename = f"../Datasets/Bankrupt/bankrupt_preprocessed.csv"

df = pd.read_csv(filename)


def calculate_sizes(total_size, percentages):
    sizes = [int(percentage * total_size) for percentage in percentages]
    return sizes
#####################################################

total_size = len(df)
#Percentuali di split (la prima è il training set)


#25,40,55,70,85,100
percentages = [0.25, 0.45, 0.70, 1.0]
sizes = calculate_sizes(total_size, percentages)


# Generazione e valutazione degli split
generate_and_evaluate_dataset(df, percentages, sizes, threshold, max_attempts, target)

