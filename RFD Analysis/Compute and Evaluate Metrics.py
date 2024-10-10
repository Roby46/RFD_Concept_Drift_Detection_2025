import pandas as pd
import numpy as np
import csv
import os
import time


def printResults(raw_data, c0_values, c1_values, metrics, reverse_, message, threshold, operator):
    # --------------------------------------------------------

    if "Incrementale" in message:
        eval_type="Incremental"
    else:
        eval_type="Standard"

    # Determina il valore di model_metric in base al contenuto di message
    if "F1" in message:
        model_metric = "F1"
    elif "Recall" in message:
        model_metric = "Recall"
    elif "Precision" in message:
        model_metric = "Precision"
    else:
        model_metric = "Unknown"  # Default nel caso nessuna metrica sia trovata

    i = 0

    # Lista per tenere traccia delle tuple (metrica, valore medio, valore medio assoluto)
    result= []

    print(90 * "=")
    for metric in metrics:
        mean_value = (c0_values[i] + c1_values[i]) / 2
        raw_data.append(mean_value)
        abs_mean_value = (abs(c0_values[i]) + abs(c1_values[i])) / 2
        result.append((metric, mean_value, abs_mean_value))
        # print(bcolors.BOLD, metric, "---Media: " + bcolors.WARNING, mean_value, bcolors.ENDC, "---Media abs: ", bcolors.WARNING, abs_mean_value, bcolors.ENDC)
        i = i + 1

    # Ordina le tuple in base al valore medio
    sorted_results_mean = sorted(result, key=lambda x: x[1], reverse=reverse_)

    # Stampa i risultati ordinati per valore medio
    print(message)
    position=1
    metric_type=""
    for metric, mean_value, abs_mean_value in sorted_results_mean:
        if(operator==">"):
            metric_type="CF"
            color = bcolors.OKGREEN if mean_value > threshold else bcolors.WARNING
        elif(operator=="<"):
            metric_type="Distance"
            color = bcolors.OKGREEN if mean_value < threshold else bcolors.WARNING
        print(
            bcolors.BOLD,
            metric,
            ":",
            color,
            mean_value,
            bcolors.ENDC
        )
        header=["Version", "Measure", "Model_Metric", "Position", "Correlation", "Metric_Type", "Eval_Type"]
        data=[version, metric, model_metric, position, mean_value,metric_type, eval_type]
        write_row_to_csv("Ranking_All_Metrics.csv", data, header)
        position=position+1

# Function to check if the CSV file exists
def is_file_exist(file_path):
    return os.path.exists(file_path)


# Function to write a row to the CSV file
def write_row_to_csv(file_path, row_data, header):
    mode = 'a' if is_file_exist(file_path) else 'w'

    with open(file_path, mode, newline='') as file:
        writer = csv.writer(file)

        # Write the header if the file is being created
        if mode == 'w':
            writer.writerow(header)

        # Write the data row
        writer.writerow(row_data)


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

    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    LIGHT_YELLOW = '\033[93m'
    LIGHT_GREEN = '\033[32m'
    DARK_RED = '\033[31m'
    LIGHT_RED = '\033[91m'
    DARK_GREEN = '\033[90m'
    PETROL_GREEN = '\033[36m'
    TURQUOISE = '\033[96m'

    ORANGE = '\033[33m'
    LIGHT_ORANGE = '\033[38;5;208m'
    DARK_CYAN = '\033[38;5;23m'
    DARK_PURPLE = '\033[38;5;57m'
    LIGHT_PURPLE = '\033[38;5;141m'

def confusion_matrix_score_1(immutate, nuove, invalidate):
    if(immutate==0):
        immutate=1
    tp=immutate
    tn=0
    fp=nuove
    fn=invalidate
    # Calculate Precision
    precision = tp / (tp + fp)
    # Calculate Recall
    recall = tp / (tp + fn)
    # Calculate Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    # Calculate F1 Score
    try:
        f1 = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f1=0
    # Print the results
    print(bcolors.OKBLUE + "", 67 * '-', "Confusion Matrix 1", 67 * '-')
    print(bcolors.OKBLUE + "--- False negative -> Invalidate ")
    print("Risultati")
    print(bcolors.OKBLUE + "--- Precision", bcolors.WARNING +  f'{precision}')
    print(bcolors.OKBLUE + "--- Recall", bcolors.WARNING +  f'{recall}')
    print(bcolors.OKBLUE + "--- Accuracy", bcolors.WARNING +  f'{accuracy}')
    print(bcolors.OKBLUE + "--- F1 Score", bcolors.WARNING +  f'{f1}')

    return recall, precision, accuracy, f1



def confusion_matrix_score_2(immutate, specializzate_non_spec, simili_non_spec, specializzate_spec, simili_spec, nuove, invalidate):
    if(immutate==0):
        immutate=1
    tp=immutate + simili_spec + specializzate_spec
    tn=0
    fp=nuove + specializzate_non_spec + simili_non_spec
    fn=invalidate
    # Calculate Precision
    precision = tp / (tp + fp)
    # Calculate Recall
    recall = tp / (tp + fn)
    # Calculate Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    # Calculate F1 Score
    try:
        f1 = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f1=0    # Print the results
    print(bcolors.OKBLUE + "", 66 * '-', "Confusion Matrix 2", 66 * '-')
    print("Risultati")
    print(bcolors.OKBLUE + "--- Precision", bcolors.WARNING +  f'{precision}')
    print(bcolors.OKBLUE + "--- Recall", bcolors.WARNING +  f'{recall}')
    print(bcolors.OKBLUE + "--- Accuracy", bcolors.WARNING +  f'{accuracy}')
    print(bcolors.OKBLUE + "--- F1 Score", bcolors.WARNING +  f'{f1}')

    return recall, precision, accuracy, f1



def confusion_matrix_score_3(immutate, specializzate_spec, simili_spec, nuove, invalidate):
    if(immutate==0):
        immutate=1
    tp=immutate + simili_spec + specializzate_spec
    tn=0
    fp=nuove
    fn=invalidate
    # Calculate Precision
    precision = tp / (tp + fp)
    # Calculate Recall
    recall = tp / (tp + fn)
    # Calculate Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    # Calculate F1 Score
    try:
        f1 = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f1=0    # Print the results
    print(bcolors.OKBLUE + "", 66 * '-', "Confusion Matrix 3", 66 * '-')
    print("Risultati")
    print(bcolors.OKBLUE + "--- Precision", bcolors.WARNING +  f'{precision}')
    print(bcolors.OKBLUE + "--- Recall", bcolors.WARNING +  f'{recall}')
    print(bcolors.OKBLUE + "--- Accuracy", bcolors.WARNING +  f'{accuracy}')
    print(bcolors.OKBLUE + "--- F1 Score", bcolors.WARNING +  f'{f1}')

    return recall, precision, accuracy, f1



def confusion_matrix_score_4(immutate, specializzate, simili, nuove, invalidate):

    if(immutate==0):
        immutate=1

    tp=immutate
    tn=0
    fp=nuove + specializzate + simili
    fn=invalidate
    # Calculate Precision
    precision = tp / (tp + fp)
    # Calculate Recall
    recall = tp / (tp + fn)
    # Calculate Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    # Calculate F1 Score
    try:
        f1 = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f1=0    # Print the results
    print(bcolors.OKBLUE + "", 66 * '-', "Confusion Matrix 4", 66 * '-')
    print("Risultati")
    print(bcolors.OKBLUE + "--- Precision", bcolors.WARNING +  f'{precision}')
    print(bcolors.OKBLUE + "--- Recall", bcolors.WARNING +  f'{recall}')
    print(bcolors.OKBLUE + "--- Accuracy", bcolors.WARNING +  f'{accuracy}')
    print(bcolors.OKBLUE + "--- F1 Score", bcolors.WARNING +  f'{f1}')

    return recall, precision, accuracy, f1

def confusion_matrix_score_5(immutate, specializzate_log_originale, nuove, invalidate):

    if(immutate==0):
        immutate=1
    tp=immutate 
    tn=0
    fp=invalidate + specializzate_log_originale
    fn=nuove
    # Calculate Precision
    precision = tp / (tp + fp)
    # Calculate Recall
    recall = tp / (tp + fn)
    # Calculate Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    # Calculate F1 Score
    try:
        f1 = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f1=0    # Print the results
    print(bcolors.OKBLUE + "", 66 * '-', "Confusion Matrix 5", 66 * '-')
    print("Risultati")
    print(bcolors.OKBLUE + "--- Precision", bcolors.WARNING +  f'{precision}')
    print(bcolors.OKBLUE + "--- Recall", bcolors.WARNING +  f'{recall}')
    print(bcolors.OKBLUE + "--- Accuracy", bcolors.WARNING +  f'{accuracy}')
    print(bcolors.OKBLUE + "--- F1 Score", bcolors.WARNING +  f'{f1}')

    return recall, precision, accuracy, f1

def confusion_matrix_score_6(immutate, specializzate_log_originale, nuove, invalidate, specializzate_nuovo_log):

    if(immutate==0):
        immutate=1

    tp=immutate 
    tn=0
    fp=invalidate + specializzate_log_originale
    fn=nuove + specializzate_nuovo_log
    # Calculate Precision
    precision = tp / (tp + fp)
    # Calculate Recall
    recall = tp / (tp + fn)
    # Calculate Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    # Calculate F1 Score
    try:
        f1 = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f1=0    # Print the results
    print(bcolors.OKBLUE + "", 66 * '-', "Confusion Matrix 6", 66 * '-')
    print("Risultati")
    print(bcolors.OKBLUE + "--- Precision", bcolors.WARNING +  f'{precision}')
    print(bcolors.OKBLUE + "--- Recall", bcolors.WARNING +  f'{recall}')
    print(bcolors.OKBLUE + "--- Accuracy", bcolors.WARNING +  f'{accuracy}')
    print(bcolors.OKBLUE + "--- F1 Score", bcolors.WARNING +  f'{f1}')

    return recall, precision, accuracy, f1


def confusion_matrix_score_7(immutate, specializzate_log_originale, nuove, invalidate, specializzate_nuovo_log):

    if(immutate==0):
        immutate=1

    tp=immutate + specializzate_nuovo_log
    tn=0
    fp=invalidate + specializzate_log_originale
    fn=nuove
    # Calculate Precision
    precision = tp / (tp + fp)
    # Calculate Recall
    recall = tp / (tp + fn)
    # Calculate Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    # Calculate F1 Score
    try:
        f1 = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f1=0    # Print the results
    print(bcolors.OKBLUE + "", 66 * '-', "Confusion Matrix 7", 66 * '-')
    print("Risultati")
    print(bcolors.OKBLUE + "--- Precision", bcolors.WARNING +  f'{precision}')
    print(bcolors.OKBLUE + "--- Recall", bcolors.WARNING +  f'{recall}')
    print(bcolors.OKBLUE + "--- Accuracy", bcolors.WARNING +  f'{accuracy}')
    print(bcolors.OKBLUE + "--- F1 Score", bcolors.WARNING +  f'{f1}')

    return recall, precision, accuracy, f1


def distanceMetric1(nuove, gen , simili, lenght): #ok
    peso_massimo_singolo = 1
    # Calcolo del peso per le RFD specializzate e simili
    peso_generalizzate = 0.5 * peso_massimo_singolo
    peso_simili = 0.05 * peso_massimo_singolo

    # Calcolo della distanza totale
    distanza = 0
    # Contributo delle RFD specializzate
    distanza += gen * peso_generalizzate
    # Contributo delle RFD simili
    distanza += simili * peso_simili
    # Contributo delle RFD nuove
    distanza += nuove * peso_massimo_singolo

    if(lenght!=0):
        distanza=distanza/lenght
    else:
        distanza=1

    if(distanza>1):
        distanza=1

    print(bcolors.OKBLUE + "", 70 * '-', "Metrica 1", 70 * '-')
    print(bcolors.OKBLUE + "Distanza: " + bcolors.WARNING, distanza)

    return distanza

def distanceMetric2(nuove, gen, simili, lenght, invalidate):   #ok
    peso_massimo_singolo = 1
    # Calcolo del peso per le RFD specializzate e simili
    peso_generalizzate = 0.5 * peso_massimo_singolo
    peso_simili = 0.05 * peso_massimo_singolo

    # Calcolo della distanza totale
    distanza = 0
    # Contributo delle RFD specializzate
    distanza += gen * peso_generalizzate
    # Contributo delle RFD simili
    distanza += simili * peso_simili
    # Contributo delle RFD nuove
    distanza += nuove * peso_massimo_singolo
    # Contributo delle RFD invalidate
    distanza += invalidate * peso_massimo_singolo

    if (lenght != 0):
        distanza = distanza / lenght
    else:
        distanza = 1
    if(distanza>1):
        distanza=1

    print(bcolors.OKBLUE + "", 70 * '-', "Metrica 2", 70 * '-')
    print(bcolors.OKBLUE + "Distanza: " + bcolors.WARNING, distanza)

    return distanza


def distanceMetric3(nuove, lenght, invalidate): #ok
    peso_massimo_singolo = 1

    # Calcolo della distanza totale
    distanza = 0

    # Contributo delle RFD nuove
    distanza += nuove * peso_massimo_singolo
    distanza += invalidate * peso_massimo_singolo

    if (lenght != 0):
        distanza = distanza / lenght
    else:
        distanza = 1

    if(distanza>1):
        distanza=1

    print(bcolors.OKBLUE + "", 70 * '-', "Metrica 3", 70 * '-')
    print(bcolors.OKBLUE + "Distanza: " + bcolors.WARNING, distanza)

    return distanza


def distanceMetric4(nuove, gen, tipo_generalizzazioni, simili, lenght, invalidate):  #ok
    peso_massimo_singolo = 1
    # Calcolo del peso per le RFD specializzate e simili
    peso_specializzate = 0.5 * peso_massimo_singolo
    peso_simili = 0.05 * peso_massimo_singolo
    peso_specializzate_primo_livello= 0.5 * peso_specializzate

    # Calcolo della distanza totale
    distanza = 0

    # Contributo delle RFD specializzate
    num_generalizzate_low = tipo_generalizzazioni.get(1, 0)
    #num_specializzate_low = tipo_specializzazioni[1]

    distanza += num_generalizzate_low * peso_specializzate_primo_livello
    distanza += (gen-num_generalizzate_low) * peso_specializzate
    # Contributo delle RFD simili
    distanza += simili * peso_simili
    # Contributo delle RFD nuove
    distanza += nuove * peso_massimo_singolo
    distanza += invalidate * peso_massimo_singolo

    if(lenght!=0):
        distanza=distanza/lenght
    else:
        distanza=1

    if(distanza>1):
        distanza=1

    print(bcolors.OKBLUE + "", 70 * '-', "Metrica 4", 70 * '-')
    print(bcolors.OKBLUE + "Distanza: " + bcolors.WARNING, distanza)

    return distanza


def distanceMetric5(spec, simili, lenght, invalidate):  #ok
    peso_massimo_singolo = 1
    # Calcolo del peso per le RFD specializzate e simili
    peso_specializzate = 0.5 * peso_massimo_singolo
    peso_simili = 0.05 * peso_massimo_singolo

    # Calcolo della distanza totale
    distanza = 0

    # Contributo delle RFD specializzate
    distanza += spec * peso_specializzate

    # Contributo delle RFD simili
    distanza += simili * peso_simili

    # Contributo delle RFD invalidate
    distanza += invalidate * peso_massimo_singolo

    if(lenght!=0):
        distanza=distanza/lenght
    else:
        distanza=1

    if(distanza>1):
        distanza=1

    print(bcolors.OKBLUE + "", 70 * '-', "Metrica 5", 70 * '-')
    print(bcolors.OKBLUE + "Distanza: " + bcolors.WARNING, distanza)

    return distanza


def distanceMetric6(nuove, gen, simili_gen, lenght, invalidate):   #ok
    peso_massimo_singolo = 1
    # Calcolo del peso per le RFD specializzate e simili
    peso_specializzate = 0.5 * peso_massimo_singolo
    peso_simili = 0.5 * peso_massimo_singolo

    # Calcolo della distanza totale
    distanza = 0
    # Contributo delle RFD specializzate
    distanza += gen * peso_specializzate
    # Contributo delle RFD simili
    distanza += simili_gen * peso_simili
    # Contributo delle RFD nuove
    distanza += nuove * peso_massimo_singolo
    #Contributo invalidate
    distanza += invalidate * peso_massimo_singolo

    if(lenght!=0):
        distanza=distanza/lenght
    else:
        distanza=1

    if(distanza>1):
        distanza=1

    print(bcolors.OKBLUE + "", 70 * '-', "Metrica 6", 70 * '-')
    print(bcolors.OKBLUE + "Distanza: " + bcolors.WARNING, distanza)

    return distanza


def distanceMetric7(spec,tipo_specializzazioni, simili, lenght, invalidate): #ok
    peso_massimo_singolo = 1
    # Calcolo del peso per le RFD specializzate e simili
    peso_specializzate = 0.5 * peso_massimo_singolo
    peso_simili = 0.05 * peso_massimo_singolo

    peso_specializzate_primo_livello = 0.5 * peso_specializzate

    # Calcolo della distanza totale
    distanza = 0

    try:
        num_specializzate_low = tipo_specializzazioni[1]
    except KeyError:
        num_specializzate_low = 0


    distanza += num_specializzate_low * peso_specializzate_primo_livello
    distanza += (spec - num_specializzate_low) * peso_specializzate

    # Contributo delle RFD simili
    distanza += simili * peso_simili

    # Contributo delle RFD invalidate
    distanza += invalidate * peso_massimo_singolo

    if (lenght != 0):
        distanza = distanza / lenght
    else:
        distanza = 1

    if(distanza>1):
        distanza=1

    print(bcolors.OKBLUE + "", 70 * '-', "Metrica 7", 70 * '-')
    print(bcolors.OKBLUE + "Distanza: " + bcolors.WARNING, distanza)

    return distanza


def distanceMetric8(nuove, generalizzate_spec, simili_spec, lenght, invalidate):
    peso_massimo_singolo = 1
    # Calcolo del peso per le RFD specializzate e simili
    peso_specializzate = 0.5 * peso_massimo_singolo
    peso_simili = 0.5 * peso_massimo_singolo

    
    # Calcolo della distanza totale
    distanza = 0

  
    distanza += generalizzate_spec * peso_specializzate
    distanza += simili_spec * peso_simili
    distanza += nuove * peso_simili

    
    # Contributo delle RFD invalidate
    distanza += invalidate * peso_massimo_singolo

    if(lenght!=0):
        distanza=distanza/lenght
    else:
        distanza=1

    if(distanza>1):
        distanza=1

    print(bcolors.OKBLUE + "", 70 * '-', "Metrica 8", 70 * '-')
    print(bcolors.OKBLUE + "Distanza: " + bcolors.WARNING, distanza)

    return distanza


def distanceMetric9(nuove, specializzate, tipologia_specializzate, generalizzate, tipologia_generalizzate, simili, lenght, invalidate):
    peso_massimo_singolo = 1
    # Calcolo del peso per le RFD specializzate e simili
    peso_specializzate = 0.5 * peso_massimo_singolo
    peso_specializzate_primo_livello=0.25 * peso_massimo_singolo

    # Calcolo della distanza totale
    distanza = 0

    # Contributo delle RFD specializzate
    try:
        num_specializzate_old_low = tipologia_specializzate[1]
    except KeyError:
        num_specializzate_old_low = 0


    distanza += num_specializzate_old_low * peso_specializzate_primo_livello
    distanza += (specializzate - num_specializzate_old_low) * peso_specializzate

    try:
        num_specializzate_new_low = tipologia_generalizzate[1]
    except KeyError:
        num_specializzate_new_low = 0

    distanza += num_specializzate_new_low * peso_specializzate_primo_livello
    distanza += (generalizzate - num_specializzate_new_low) * peso_specializzate
    
    distanza += nuove * peso_massimo_singolo
    distanza += (simili*2) * peso_massimo_singolo

    # Contributo delle RFD invalidate
    distanza += invalidate * peso_massimo_singolo


    if(lenght!=0):
        distanza=distanza/lenght
    else:
        distanza=1

    if (distanza > 1):
        distanza = 1

    print(bcolors.OKBLUE + "", 70 * '-', "Metrica 9", 70 * '-')
    print(bcolors.OKBLUE + "Distanza: " + bcolors.WARNING, distanza)

    return distanza

def distanceMetric10(spec,tipo_specializzazioni, simili, lenght, invalidate): #ok
    peso_massimo_singolo = 1
    # Calcolo del peso per le RFD specializzate e simili
    peso_specializzate = 0.2 * peso_massimo_singolo
    peso_simili = 0.02 * peso_massimo_singolo

    peso_specializzate_primo_livello = 0.5 * peso_specializzate

    # Calcolo della distanza totale
    distanza = 0

    try:
        num_specializzate_low = tipo_specializzazioni[1]
    except KeyError:
        num_specializzate_low = 0


    distanza += num_specializzate_low * peso_specializzate_primo_livello
    distanza += (spec - num_specializzate_low) * peso_specializzate

    # Contributo delle RFD simili
    distanza += simili * peso_simili

    # Contributo delle RFD invalidate
    distanza += invalidate * peso_massimo_singolo

    if(lenght!=0):
        distanza=distanza/lenght
    else:
        distanza=1

    if(distanza>1):
        distanza=1

    print(bcolors.OKBLUE + "", 70 * '-', "Metrica 10", 70 * '-')
    print(bcolors.OKBLUE + "Distanza: " + bcolors.WARNING, distanza)

    return distanza

def distanceMetric11(nuove, lenght, invalidate, generalizzate, specializzate): #ok
    peso_massimo_singolo = 1

    peso_specializzate = 0.1 * peso_massimo_singolo

    # Calcolo della distanza totale
    distanza = 0

    # Contributo delle RFD nuove
    distanza += nuove * peso_massimo_singolo
    distanza += invalidate * peso_massimo_singolo
    distanza += generalizzate * peso_specializzate
    distanza += specializzate * peso_specializzate
    

    if(lenght!=0):
        distanza=distanza/lenght
    else:
        distanza=1

    if(distanza>1):
        distanza=1

    print(bcolors.OKBLUE + "", 70 * '-', "Metrica 11", 70 * '-')
    print(bcolors.OKBLUE + "Distanza: " + bcolors.WARNING, distanza)

    return distanza

def distanceMetric12(spec, simili, lenght, invalidate):  #ok
    peso_massimo_singolo = 1
    # Calcolo del peso per le RFD specializzate e simili
    peso_specializzate = 0.3 * peso_massimo_singolo
    peso_simili = 0.02 * peso_massimo_singolo

    # Calcolo della distanza totale
    distanza = 0

    # Contributo delle RFD specializzate
    distanza += spec * peso_specializzate

    # Contributo delle RFD simili
    distanza += simili * peso_simili

    # Contributo delle RFD invalidate
    distanza += invalidate * peso_massimo_singolo

    if(lenght!=0):
        distanza=distanza/lenght
    else:
        distanza=1

    if(distanza>1):
        distanza=1

    print(bcolors.OKBLUE + "", 70 * '-', "Metrica 12", 70 * '-')
    print(bcolors.OKBLUE + "Distanza: " + bcolors.WARNING, distanza)

    return distanza


def checkRFDChange(percentages, target_label, current_enc_f1_perf, version):

    local_percentages=percentages.copy()
    filename = f"../MinimalRFDS/MinimalRFDs_v{version}_{str(target_label)}_{str(percentages[0])}.csv"

    print(bcolors.ENDC + "Apro: ", filename)

    #Dataframe con le RFD originali
    originaldf=pd.read_csv(filename, sep=';')
    print(bcolors.PETROL_GREEN + "Numero dipendenze originali:", bcolors.WARNING, len(originaldf))

    #Non considero la percentuale del training
    local_percentages.pop(0)

    #Array che contengono i risultati ottenuti
    #Metriche Confusion matrix 1
    ArrayF1=np.empty(0)
    # Metriche Confusion matrix 2
    ArrayF1_2 = np.empty(0)
    # Metriche Confusion matrix 2b
    ArrayF1_3 = np.empty(0)
    # Metriche Confusion matrix 3
    ArrayF1_4= np.empty(0)
    # Metriche Confusion matrix 4
    ArrayF1_5= np.empty(0)
    # Metriche Confusion matrix 5
    ArrayF1_6= np.empty(0)
    # Metriche Confusion matrix 6
    ArrayF1_7= np.empty(0)
    #Distanze
    ArrayD1 = np.empty(0)
    ArrayD2 = np.empty(0)
    ArrayD3 = np.empty(0)
    ArrayD4 = np.empty(0)
    ArrayD5 = np.empty(0)
    ArrayD6 = np.empty(0)
    ArrayD7 = np.empty(0)
    ArrayD8 = np.empty(0)
    ArrayD9 = np.empty(0)
    ArrayD10 = np.empty(0)
    ArrayD11 = np.empty(0)
    ArrayD12 = np.empty(0)


    for percentage in local_percentages:


        filename = f"../MinimalRFDS/MinimalRFDs_v{version}_{str(target_label)}_{str(percentage)}.csv"

        #Dataframe con le RFD aggiornate
        new_df=pd.read_csv(filename, sep=';')
        print(bcolors.ENDC + "Confronto con: ", filename, "che ha", len(new_df), "dipendenze")

        idx_to_remove=set()

        #RFD presenti in entrambi i file
        immutate = 0
        #RFD originali che sono state specializzate
        specializzate = 0
        #Tipologia di specializzazioni
        specializzazioni = {}
        #RFD specializzate con stesse soglie sugli attributi in comune
        specializzazioni_con_stesse_soglie=0
        #RFD specializzate con soglie <= sugli attr in comune a sx (almeno 1 <) e soglia >= a dx
        specializzazioni_sx_dx=0
        #RFD presenti nell'originale e non nel nuovo
        invalidate = 0
        #RFD originali che si trovano anche nel file nuovo, ma con soglie diverse
        simili=0
        #Tipologia di simili
        simili_spec_sx=0
        simili_spec_dx=0
        simili_spec_sx_dx=0

        # ------ PRIMA ANALISI: CAMBIAMENTO DAL LOG ORIGINALE AL LOG NUOVO ------
        # Per ogni RFD del file originale, vede quante rimangono immutate, specializzate, invalidate e simili.
        for idx, row in originaldf.iterrows():
            #Rappresenta il numero di attributi aggiuntivi massimo, guardando tutte le specializzazioni della dipendenza considerata
            max_livello_specializzazione=0

            #Variabile che salva le posizione dell'LHS della RFD che specializza con grado massimo, se presente
            lhs_positions_to_keep_specialization=""

            # Variabile che salva i valori di riga della RFD che specializza con grado massimo, se presente
            row_to_compare_specialization=''

            #Valori della riga della dipendenza originale
            row_values = row.values
            #Cerco se nel nuovo file è presente una riga uguale
            is_row_in_new_df = new_df.apply(lambda new_row: new_row.equals(pd.Series(row)), axis=1).any()
            if (is_row_in_new_df):
                #Prendo l'indice della dipendenza nuova che è uguale a quella attuale
                equal_row_index = new_df.index[new_df.apply(lambda new_row: new_row.equals(pd.Series(row)), axis=1)]
                # print("Dipendenza immutata")
                #Aggiorno il conto delle immutate
                immutate = immutate + 1
                #Aggiungo l'indice della riga all'insieme idx to remove. Questo perché nella fase successiva non mi serve
                idx_to_remove.add(equal_row_index[0])
            else:
                # Nome dell'attributo RHS (Es. COL0)
                rhs = row_values[0]
                # Posizione nell'RHS (Es: COL0 diventa posizione 1)
                rhs_row_index = (int(rhs[3:])) + 1
                # Escludo il primo elemento da row values, perché è il nome dell'attributo. [COL0,?,1,2,?,?]->[?,1,2,?,?]
                row_values_sliced = row_values[1:]
                # Trovo gli indici nell'array dove i valori sono diversi da ? (quindi gli attributi coinvolti nella RFD)
                lhs_positions = set(np.where(row_values_sliced != '?')[0] + 1)  # Add 1 to adjust for the slice
                # Tolgo la posizione dell'RHS per ottenere solo le posizioni degli attributi sull'LHS
                lhs_positions.remove(rhs_row_index)
                # Prendo il valore della soglia dell'attributo sull'RHS
                rhs_threhsold = row_values[rhs_row_index]
                # Ordino le posizioni dell'LHS
                lhs_positions_sorted = sorted(lhs_positions)
                # Estraggo l'array di soglie degli attributi sull'LHS
                lhs_thresholds = row_values[lhs_positions_sorted]

                # Variabile che tiene traccia delle RFD invalidata.
                # Durante i controlli viene settata a False se l'RFD è immutata, simile o specializzata
                invalidated = True

                # Variabile che, se settata a TRUE, indica che una specializzazione della RFD considerata è stata già trovata
                # In questo caso, l'algoritmo continua i controlli per vedere se ci sono specializzazioni di livello più alto
                specialization_check=False

                #Itero sulle RFD nel nuovo log
                for other_idx, other_row in new_df.iterrows():
                    # Valori della riga
                    other_row_values = other_row.values
                    # Attrbuto nell'RHS
                    other_rhs = other_row_values[0]
                    # Indice di posizione dell'RHS
                    other_rhs_row_index = (int(other_rhs[3:])) + 1

                    #Se la nuova dipendenza analizzata ha lo stesso RHS di quella originale
                    if (other_rhs_row_index == rhs_row_index):

                        # Specializzazione su LHS: tutte le threshold non devono superare quelle originali,
                        # e almeno una deve essere minore. RHS invariato
                        SPECLHS = False
                        # Specializzazione su RHS: tutte le threshold sull'LHS devono essere invariate,
                        # quella sull'RHS deve aumentare
                        SPECRHS = False


                        # Valori della riga senza la prima colonna (che contiene il nome dell'RHS)
                        other_row_values_sliced = other_row_values[1:]
                        # Posizione degli attributi coinvolti nella dipendenza
                        lhs_new_positions = set(np.where(other_row_values_sliced != '?')[0] + 1)
                        # Tolgo la posizione dell'RHS per ottenere le posizioni degli attributi LHS
                        lhs_new_positions.remove(rhs_row_index)
                        # Soglia dell'RHS della nuova RFD
                        rhs_new_threshold=other_row_values[other_rhs_row_index]
                        lhs_new_positions_sorted=sorted(lhs_new_positions)
                        # Array di soglie degli attributi coinvolti sull'LHS
                        lhs_new_thresholds=other_row_values[lhs_new_positions_sorted]

                        # Controllo se tutti gli attributi nell'LHS originale sono coinvolti nella nuova RFD (sempre su LHS)
                        if lhs_positions.issubset(lhs_new_positions):
                            # Setto invalidated a false: la dipendenza non è invalidata, esiste in qualche forma nel nuovo set
                            invalidated = False
                            # Controllo se l'lhs nuovo contiene attributi aggiuntivi: nel caso è una specializzazione
                            if len(lhs_new_positions) > len(lhs_positions):
                                # Se non ho trovato una specializzazione prima, aggiorno il conto delle specializzate
                                if(not specialization_check):
                                    specializzate += 1
                                # Valuto il grado di specializzazione
                                num_specializzazione= len(lhs_new_positions) - len(lhs_positions)

                                # Se la nuova RFD specializza con un grado maggiore l'RFD originale rispetto alle
                                # altre RFD specializzate trovate in precedenza
                                if(num_specializzazione>max_livello_specializzazione):
                                    # Aggiornlo il grado massimo di specializzazione
                                    max_livello_specializzazione=num_specializzazione
                                    # Mi copio i valori della RFD specializzata di riferimento
                                    # Alla fine del ciclo mi servono per caratterizzare il tipo di specializzazione
                                    row_to_compare_specialization=other_row_values.copy()
                                    # Mi copio la posizione degli attributi dell'LHS della nuova dipendenza
                                    lhs_positions_to_keep_specialization=lhs_new_positions.copy()

                                # Ho trovato una specializzazione, ora il controllo deve proseguire per
                                # cercare RFD che specializzano con grado più alto
                                specialization_check=True

                            # Se invece gli attributi sull'LHS coincidono, si tratta di una dipendenza simile
                            elif len(lhs_new_positions) == len(lhs_positions):

                                # Aggiorno il conto delle simili
                                simili += 1

                                # Controllo specializzazione RHS, bloccando il lato SX
                                if(np.array_equal(lhs_new_thresholds, lhs_thresholds)):
                                    if(rhs_new_threshold > rhs_threhsold):
                                        SPECRHS=True

                                # Se non è specializzata RHS, controllo se è specializzata LHS, bloccando il lato DX
                                if(not SPECRHS):
                                    # Convert the arrays to float
                                    lhs_thresholds = lhs_thresholds.astype(float)
                                    lhs_new_thresholds = lhs_new_thresholds.astype(float)

                                    # Sottrazione lato sx
                                    lhs_difference = lhs_thresholds - lhs_new_thresholds
                                    if(np.all(lhs_difference >= 0) and np.any(lhs_difference>0) and (rhs_threhsold==rhs_new_threshold)):
                                        SPECLHS=True

                                # Aggiorno il conto delle specializzate LHS
                                if (SPECLHS):  # And RHS uguali
                                    simili_spec_sx = simili_spec_sx + 1

                                # Aggiorno il contro delle specializzate RHS
                                elif (SPECRHS):  # And LHS uguali
                                    simili_spec_dx = simili_spec_dx + 1

                                #Blocco aggiunto 14/12
                                #Se la dipendenza non è stata ancora caratterizzata si controlla la proprietà "rilassata"
                                if(not SPECRHS and not SPECLHS):
                                    #Soglia RHS >= rispetto alla dipendenza originale
                                    if ((float(rhs_new_threshold) >= float(rhs_threhsold))):
                                        # Convert the arrays to float
                                        lhs_thresholds = lhs_thresholds.astype(float)
                                        lhs_new_thresholds = lhs_new_thresholds.astype(float)

                                        # Sottrazione lato sx
                                        lhs_difference = lhs_thresholds - lhs_new_thresholds
                                        if (np.all(lhs_difference >= 0) and np.any(lhs_difference > 0)):
                                                simili_spec_sx_dx=simili_spec_sx_dx+1


                                    #Fine blocco aggiunto

                                # Se ho trovato una simile durante il controllo per ulteriori specializzazioni
                                # decremento il conto delle specializzate
                                if (specialization_check):
                                    specializzate = specializzate - 1

                                # Non è più necessario trovare specializzate di grado maggiore
                                specialization_check = False
                                # Resetto il valore da conservare
                                # Da aggiungere: reset di lhs?
                                row_to_compare_specialization = ""
                                #idx_to_remove.add(other_idx)
                                #Esco dal controllo
                                break

                # Se scorrendo tutte le nuove RFD non ho trovato immutate, simili o specializzate, l'RFD originale
                # è invalidata
                if (invalidated):
                    invalidate = invalidate + 1

                # Se invece ho trovato una specializzazione, cerco di caratterizzarla
                elif(specialization_check):

                    # print("***************")
                    # print(row_values)
                    # print(row_to_compare_specialization)
                    # print("***************")

                    # Aggiorno il dizionario di specializzazioni, aumentando di uno il conto delle RFD che specializzano
                    # con il grado di quella attuale
                    specializzazioni[max_livello_specializzazione] = specializzazioni.get(max_livello_specializzazione, 0) + 1


                    # Prendo gli attributi in comune tra l'RFD originale e quella che specializza
                    lhs_intersection = lhs_positions.intersection(lhs_positions_to_keep_specialization)

                    lhs_intersection_sorted = sorted(lhs_intersection)

                    # Prendo le soglie degli attributi in comune
                    lhs_common_thresholds = row_values[lhs_intersection_sorted]
                    lhs_new_common_thresholds = row_to_compare_specialization[lhs_intersection_sorted]
                    lhs_common_thresholds = lhs_common_thresholds.astype(float)
                    lhs_new_common_thresholds = lhs_new_common_thresholds.astype(float)

                    # Array delle differenze tra le soglie degli LHS
                    #difference = lhs_new_common_thresholds - lhs_common_thresholds
                    # Array delle differenze tra le soglie degli LHS (modificato 19/12)
                    difference = lhs_common_thresholds - lhs_new_common_thresholds
                    rhs_new_threshold=row_to_compare_specialization[rhs_row_index]

                    # Se le due RFD hanno le stesse soglie sugli attributi in comune, caratterizzo la nuova RFD
                    if (np.all(difference == 0) and (rhs_threhsold == rhs_new_threshold)):
                        specializzazioni_con_stesse_soglie=specializzazioni_con_stesse_soglie+1
                    # Da aggiungere il nuovo controllo 14/12
                    else:
                        # Soglia RHS >= rispetto alla dipendenza originale


                        #if ((float(rhs_new_threshold) >= float(rhs_threhsold))):
                            # Convert the arrays to float
                        #    if (np.all(difference >= 0) and np.any(difference > 0)):
                         #       specializzazioni_sx_dx = specializzazioni_sx_dx + 1

                        try:
                            # Tentativo di conversione e confronto
                            if float(rhs_new_threshold) >= float(rhs_threhsold):
                                # Controllo degli array
                                if np.all(difference >= 0) and np.any(difference > 0):
                                    specializzazioni_sx_dx += 1
                        except ValueError:
                            # Gestione dell'errore di conversione: non fare nulla e continua
                            pass


        print("Analisi per ", str(target_label), " con percentuale ", str(percentage), "completata:")
        print(bcolors.PETROL_GREEN + "RFD immutate: " + bcolors.WARNING, immutate, bcolors.PETROL_GREEN + "(",
              ((immutate * 100) / len(originaldf)), " delle RFD originali)")
        print(bcolors.PETROL_GREEN + "RFD invalidate: " + bcolors.WARNING, invalidate, bcolors.PETROL_GREEN + "(",
              ((invalidate * 100) / len(originaldf)), " delle RFD originali)")
        # print("RFD Nuove: ", len(new_df)-(immutate+specializzate))
        print(bcolors.PETROL_GREEN + "RFD Simili: " + bcolors.WARNING, simili, bcolors.PETROL_GREEN + "(",
              ((simili * 100) / len(originaldf)), " delle RFD originali)")
        print(bcolors.PETROL_GREEN + "--- Simili con specializzazioni LHS:"+ bcolors.WARNING, simili_spec_sx)
        print(bcolors.PETROL_GREEN + "--- Simili con specializzazioni RHS: " + bcolors.WARNING, simili_spec_dx)
        print(bcolors.PETROL_GREEN + "--- Simili con specializzazioni su LHS e RHS: " + bcolors.WARNING, simili_spec_sx_dx)
        print(bcolors.PETROL_GREEN + "--- Altre: " + bcolors.WARNING, simili - (simili_spec_dx + simili_spec_sx + simili_spec_sx_dx))
        # Ottieni una lista ordinata delle chiavi
        sorted_keys = sorted(specializzazioni.keys())
        print(bcolors.WARNING, specializzate, bcolors.PETROL_GREEN + "RFD sono state specializzate dalle nuove RFD")

        # Stampa le chiavi in ordine
        for key in sorted_keys:
            print(bcolors.PETROL_GREEN + f'--- Specializzazioni di livello', key, ', Numero di RFD:' + bcolors.WARNING,
                  specializzazioni[key])
        print(bcolors.PETROL_GREEN + "Delle RFD specializzate, " + bcolors.WARNING, specializzazioni_con_stesse_soglie, bcolors.PETROL_GREEN +
              "sono specializzate con le stesse soglie sugli attributi in comune e" + bcolors.WARNING, specializzazioni_sx_dx, bcolors.PETROL_GREEN + "hanno soglie minori a sx e maggiori a dx."
                "\n Le altre sono:"+ bcolors.WARNING, specializzate-specializzazioni_con_stesse_soglie-specializzazioni_sx_dx)


        print("" + bcolors.ENDC)


        # Tolgo dal df delle nuove le immutate, visto che non serve controllarle
        new_df_v2=new_df.drop(idx_to_remove, axis=0)
        nuove=0
        generalizzate=0
        generalizzate_con_stesse_soglie=0
        generalizzate_sx_dx=0
        simili=0
        generalizzazioni = {}

        # ------ SECONDA ANALISI: CAMBIAMENTO DAL LOG NUOVO AL LOG ORIGINALE ------
        # Per ogni RFD del nuovo log che non è simile o immutata rispetto a quello originale, controllo quali sono specializzazioni delle RFD originali. Quelle rimanenti sono nuove
        for idx, row in new_df_v2.iterrows():
            max_livello_specializzazione = 0

            row_values = row.values
            row_to_compare = '--------------------------------'
            lhs_positions_to_keep = '....'

            rhs = row_values[0]
            # Posizione nell'RHS
            rhs_row_index = (int(rhs[3:])) + 1
            # Exclude the first element by slicing the array. [0,?,1,2,?,?]
            row_values_sliced = row_values[1:]
            # Find the indices where values are different from '?'
            lhs_positions = set(np.where(row_values_sliced != '?')[0] + 1)  # Add 1 to adjust for the slice
            lhs_positions.remove(rhs_row_index)
            rhs_threhsold = row_values[rhs_row_index]

            # Confronto la dipendenza originale solo con quelle nuove che hanno lo steso RHS
            subset = originaldf.loc[originaldf['RHS'] == row_values[0]]

            invalidated = True
            # Se true La dipendenza è già classificata come specializzate, si sta continuando a cercare per vedere se specializza altre RFD con un livello più alto
            specialization_check = False

            for other_idx, other_row in subset.iterrows():

                # Valori della riga
                other_row_values = other_row.values
                # Posizione dei valori nulli
                other_row_values_sliced = other_row_values[1:]
                lhs_new_positions = set(np.where(other_row_values_sliced != '?')[0] + 1)
                lhs_new_positions.remove(rhs_row_index)

                # Check if all attributes involved in the original RFD are involved in the new one
                if lhs_new_positions.issubset(lhs_positions):
                    invalidated = False
                    # Check if there are additional attributes involved in the new RFD
                    if len(lhs_positions) > len(lhs_new_positions):
                        if (not specialization_check):
                            generalizzate += 1
                        num_specializzazione = len(lhs_positions) - len(lhs_new_positions)
                        if (num_specializzazione > max_livello_specializzazione):
                            max_livello_specializzazione = num_specializzazione
                            row_to_compare = other_row_values.copy()
                            lhs_positions_to_keep = lhs_new_positions.copy()
                            # print(row_values)
                            # print(other_row_values)
                        specialization_check = True
                    else:
                        simili = simili + 1
                        if (specialization_check):
                            generalizzate = generalizzate - 1
                        specialization_check = False
                        row_to_compare = ""
                        break
            if (invalidated):
                nuove += 1
            elif (specialization_check):
                generalizzazioni[max_livello_specializzazione] = generalizzazioni.get(max_livello_specializzazione, 0) + 1

                lhs_intersection = lhs_positions.intersection(lhs_positions_to_keep)

                lhs_intersection_sorted = sorted(lhs_intersection)

                lhs_common_thresholds = row_values[lhs_intersection_sorted]
                lhs_new_common_thresholds = row_to_compare[lhs_intersection_sorted]
                lhs_common_thresholds = lhs_common_thresholds.astype(float)
                lhs_new_common_thresholds = lhs_new_common_thresholds.astype(float)
                difference = lhs_new_common_thresholds - lhs_common_thresholds
                if (np.all(difference == 0) and (rhs_threhsold == rhs_new_threshold)):
                    generalizzate_con_stesse_soglie = generalizzate_con_stesse_soglie + 1
                else:
                    # Soglia RHS >= rispetto alla dipendenza originale
                    #if ((rhs_new_threshold >= rhs_threhsold)):
                    #    # Convert the arrays to float
                    #    if (np.all(difference >= 0) and np.any(difference > 0)):
                    #        generalizzate_sx_dx = generalizzate_sx_dx + 1

                    try:
                        # Soglia RHS >= rispetto alla dipendenza originale
                        if float(rhs_new_threshold) >= float(rhs_threhsold):
                            # Controllo degli array di differenze
                                if np.all(difference >= 0) and np.any(difference > 0):
                                    generalizzate_sx_dx += 1
                    except ValueError:
                        # Se c'è un errore di conversione, il programma continua senza fare nulla
                        pass



        # Ottieni una lista ordinata delle chiavi
        sorted_keys = sorted(generalizzazioni.keys())

        print(bcolors.PETROL_GREEN + "Nel nuovo file,", bcolors.WARNING, generalizzate,
              bcolors.PETROL_GREEN + "RFD sono specializzazioni delle RFD originali (quindi generalizzate)")
        # Stampa le chiavi in ordine
        for key in sorted_keys:
            print(bcolors.PETROL_GREEN + f'--- Generalizzazioni di livello', key, ', Numero di RFD:' + bcolors.WARNING,
                  generalizzazioni[key])
        print(
            bcolors.PETROL_GREEN + "Le RFD che specializzano con soglie invariate sugli attributi in comune sono:" + bcolors.WARNING,
            generalizzate_con_stesse_soglie, bcolors.PETROL_GREEN + "mentre quelle con soglie <= su LHS comune ed RHS >= sono:" + bcolors.WARNING, generalizzate_sx_dx)
        print(bcolors.PETROL_GREEN + "Nuove RFD:" + bcolors.WARNING, nuove)
        print(bcolors.PETROL_GREEN + "RFD simili: "+ bcolors.WARNING, simili)


        #Risultati metriche confusion matrix
        rec, pre, acc, f1 = confusion_matrix_score_1(immutate, nuove, invalidate)
        rec_2, pre_2, acc_2, f1_2 = confusion_matrix_score_2(immutate, (generalizzate-(generalizzate_con_stesse_soglie+generalizzate_sx_dx)), (simili-(simili_spec_dx+simili_spec_sx+simili_spec_sx_dx)), generalizzate_con_stesse_soglie+generalizzate_sx_dx, (simili_spec_sx + simili_spec_dx + simili_spec_sx_dx) ,nuove, invalidate)
        rec_3, pre_3, acc_3, f1_3 = confusion_matrix_score_3(immutate, generalizzate_con_stesse_soglie + generalizzate_sx_dx,(simili_spec_sx + simili_spec_dx + simili_spec_sx_dx), nuove, invalidate)
        rec_4, pre_4, acc_4, f1_4 = confusion_matrix_score_4(immutate, generalizzate, simili, nuove, invalidate)
        rec_5, pre_5, acc_5, f1_5 = confusion_matrix_score_5(immutate, specializzate, nuove, invalidate)
        rec_6, pre_6, acc_6, f1_6 = confusion_matrix_score_6(immutate, specializzate, nuove, invalidate, generalizzate)
        rec_7, pre_7, acc_7, f1_7 = confusion_matrix_score_7(immutate, specializzate, nuove, invalidate, generalizzate)

        #Risultati distanze tra i due log
        dist1=distanceMetric1(nuove, generalizzate, simili, len(new_df))
        dist2=distanceMetric2(nuove, generalizzate, simili, (len(new_df) + len(originaldf)) - immutate, invalidate)
        dist3=distanceMetric3(nuove, (len(new_df) + len(originaldf)) - immutate, invalidate)
        dist4=distanceMetric4(nuove, generalizzate, generalizzazioni, simili, (len(new_df) + len(originaldf)) - immutate, invalidate)
        dist5=distanceMetric5(specializzate, simili, len(originaldf), invalidate)
        dist6=distanceMetric6(nuove, generalizzate, (simili_spec_sx+simili_spec_dx+simili_spec_sx_dx), (len(new_df) + len(originaldf)) - immutate, invalidate)
        dist7=distanceMetric7(specializzate, specializzazioni, simili, len(originaldf), invalidate)
        dist8=distanceMetric8(nuove, generalizzate_con_stesse_soglie + generalizzate_sx_dx, (simili_spec_sx + simili_spec_dx + simili_spec_sx_dx), (len(new_df) + len(originaldf)) - immutate, invalidate)
        dist9=distanceMetric9(nuove, specializzate, specializzazioni, generalizzate, generalizzazioni, simili, (len(new_df) + len(originaldf)) - immutate, invalidate)
        dist10=distanceMetric10(specializzate, specializzazioni, simili, len(originaldf), invalidate)
        dist11=distanceMetric11(nuove, (len(new_df) + len(originaldf)) - immutate, invalidate, generalizzate, specializzate)
        dist12=distanceMetric12(specializzate, simili, len(originaldf), invalidate)


        #Append dei risultati
        ArrayF1=np.append(ArrayF1, f1)
        ArrayF1_2 = np.append(ArrayF1_2, f1_2)
        ArrayF1_3 = np.append(ArrayF1_3, f1_3)
        ArrayF1_4 = np.append(ArrayF1_4, f1_4)
        ArrayF1_5 = np.append(ArrayF1_5, f1_5)
        ArrayF1_6 = np.append(ArrayF1_6, f1_6)
        ArrayF1_7 = np.append(ArrayF1_7, f1_7)

        ArrayD1 = np.append(ArrayD1, dist1)
        ArrayD2 = np.append(ArrayD2, dist2)
        ArrayD3 = np.append(ArrayD3, dist3)
        ArrayD4 = np.append(ArrayD4, dist4)
        ArrayD5 = np.append(ArrayD5, dist5)
        ArrayD6 = np.append(ArrayD6, dist6)
        ArrayD7 = np.append(ArrayD7, dist7)
        ArrayD8 = np.append(ArrayD8, dist8)
        ArrayD9 = np.append(ArrayD9, dist9)
        ArrayD10 = np.append(ArrayD10, dist10)
        ArrayD11 = np.append(ArrayD11, dist11)
        ArrayD12 = np.append(ArrayD12, dist12)

        print(bcolors.OKBLUE, 151 * '-')


    print(bcolors.TURQUOISE + "Risultati finali per l'encoding", target_label)

    print(bcolors.TURQUOISE + "F1-Modello" + bcolors.WARNING, current_enc_f1_perf, bcolors.ENDC)

    print(current_enc_f1_perf, ArrayF1)
    correlation_coefficient_f1 = np.corrcoef(current_enc_f1_perf, ArrayF1)[0, 1]
    distF1=np.linalg.norm(current_enc_f1_perf - (ArrayF1*100))
    print(bcolors.TURQUOISE + "F1----------" + bcolors.WARNING, ArrayF1, bcolors.ENDC, "correlazione: ", correlation_coefficient_f1, "distanza: ", distF1)


    correlation_coefficient_f1_2 = np.corrcoef(current_enc_f1_perf, ArrayF1_2)[0, 1]
    distF1_2=np.linalg.norm(current_enc_f1_perf - (ArrayF1_2*100))
    print(bcolors.TURQUOISE + "F1_2--------" + bcolors.WARNING, ArrayF1_2, bcolors.ENDC, "correlazione: ", correlation_coefficient_f1_2, "distanza: ", distF1_2)




    correlation_coefficient_f1_3 = np.corrcoef(current_enc_f1_perf, ArrayF1_3)[0, 1]
    distF1_3=np.linalg.norm(current_enc_f1_perf - (ArrayF1_3*100))
    print(bcolors.TURQUOISE + "F1_3-------" + bcolors.WARNING, ArrayF1_3, bcolors.ENDC, "correlazione: ", correlation_coefficient_f1_3, "distanza: ", distF1_3)



    correlation_coefficient_f1_4 = np.corrcoef(current_enc_f1_perf, ArrayF1_4)[0, 1]
    distF1_4=np.linalg.norm(current_enc_f1_perf - (ArrayF1_4*100))
    print(bcolors.TURQUOISE + "F1_4--------" + bcolors.WARNING, ArrayF1_4, bcolors.ENDC, "correlazione: ", correlation_coefficient_f1_4, "distanza: ", distF1_4)

    correlation_coefficient_f1_5 = np.corrcoef(current_enc_f1_perf, ArrayF1_5)[0, 1]
    distF1_5=np.linalg.norm(current_enc_f1_perf - (ArrayF1_5*100))
    print(bcolors.TURQUOISE + "F1_5--------" + bcolors.WARNING, ArrayF1_5, bcolors.ENDC, "correlazione: ", correlation_coefficient_f1_5, "distanza: ", distF1_5)

    correlation_coefficient_f1_6 = np.corrcoef(current_enc_f1_perf, ArrayF1_6)[0, 1]
    distF1_6=np.linalg.norm(current_enc_f1_perf - (ArrayF1_6*100))
    print(bcolors.TURQUOISE + "F1_6--------" + bcolors.WARNING, ArrayF1_6, bcolors.ENDC, "correlazione: ",
          correlation_coefficient_f1_6, "distanza: ", distF1_6)

    
    correlation_coefficient_f1_7 = np.corrcoef(current_enc_f1_perf, ArrayF1_7)[0, 1]
    distF1_7=np.linalg.norm(current_enc_f1_perf - (ArrayF1_7*100))
    print(bcolors.TURQUOISE + "F1_7--------" + bcolors.WARNING, ArrayF1_7, bcolors.ENDC, "correlazione: ",
          correlation_coefficient_f1_7, "distanza: ", distF1_7)


    correlation_coefficient_d1_f1 = np.corrcoef(current_enc_f1_perf, ArrayD1)[0, 1]
    print(bcolors.TURQUOISE + "D1----------" + bcolors.WARNING, ArrayD1, bcolors.ENDC)
    print("-------------Correlazione con F1 score: ", correlation_coefficient_d1_f1)


    correlation_coefficient_d2_f1 = np.corrcoef(current_enc_f1_perf, ArrayD2)[0, 1]
    print(bcolors.TURQUOISE + "D2----------" + bcolors.WARNING, ArrayD2, bcolors.ENDC)
    print("-------------Correlazione con F1 score: ", correlation_coefficient_d2_f1)


    correlation_coefficient_d3_f1 = np.corrcoef(current_enc_f1_perf, ArrayD3)[0, 1]
    print(bcolors.TURQUOISE + "D3----------" + bcolors.WARNING, ArrayD3, bcolors.ENDC)
    print("-------------Correlazione con F1 score: ", correlation_coefficient_d3_f1)


    correlation_coefficient_d4_f1 = np.corrcoef(current_enc_f1_perf, ArrayD4)[0, 1]
    print(bcolors.TURQUOISE + "D4----------" + bcolors.WARNING, ArrayD4, bcolors.ENDC)
    print("-------------Correlazione con F1 score: ", correlation_coefficient_d4_f1)


    correlation_coefficient_d5_f1 = np.corrcoef(current_enc_f1_perf, ArrayD5)[0, 1]
    print(bcolors.TURQUOISE + "D5----------" + bcolors.WARNING, ArrayD5, bcolors.ENDC)
    print("-------------Correlazione con F1 score: ", correlation_coefficient_d5_f1)

    correlation_coefficient_d6_f1 = np.corrcoef(current_enc_f1_perf, ArrayD6)[0, 1]
    print(bcolors.TURQUOISE + "D6----------" + bcolors.WARNING, ArrayD6, bcolors.ENDC)
    print("-------------Correlazione con F1 score: ", correlation_coefficient_d6_f1)


    correlation_coefficient_d7_f1 = np.corrcoef(current_enc_f1_perf, ArrayD7)[0, 1]
    print(bcolors.TURQUOISE + "D7----------" + bcolors.WARNING, ArrayD7, bcolors.ENDC)
    print("-------------Correlazione con F1 score: ", correlation_coefficient_d7_f1)

    correlation_coefficient_d8_f1 = np.corrcoef(current_enc_f1_perf, ArrayD8)[0, 1]
    print(bcolors.TURQUOISE + "D8----------" + bcolors.WARNING, ArrayD8, bcolors.ENDC)
    print("-------------Correlazione con F1 score: ", correlation_coefficient_d8_f1)


    correlation_coefficient_d9_f1 = np.corrcoef(current_enc_f1_perf, ArrayD9)[0, 1]
    print(bcolors.TURQUOISE + "D9----------" + bcolors.WARNING, ArrayD9, bcolors.ENDC)
    print("-------------Correlazione con F1 score: ", correlation_coefficient_d9_f1)

    correlation_coefficient_d10_f1 = np.corrcoef(current_enc_f1_perf, ArrayD10)[0, 1]
    print(bcolors.TURQUOISE + "D10----------" + bcolors.WARNING, ArrayD10, bcolors.ENDC)
    print("-------------Correlazione con F1 score: ", correlation_coefficient_d10_f1)

    correlation_coefficient_d11_f1 = np.corrcoef(current_enc_f1_perf, ArrayD11)[0, 1]
    print(bcolors.TURQUOISE + "D11----------" + bcolors.WARNING, ArrayD11, bcolors.ENDC)
    print("-------------Correlazione con F1 score: ", correlation_coefficient_d11_f1)


    correlation_coefficient_d12_f1 = np.corrcoef(current_enc_f1_perf, ArrayD12)[0, 1]
    print(bcolors.TURQUOISE + "D12----------" + bcolors.WARNING, ArrayD12, bcolors.ENDC)
    print("-------------Correlazione con F1 score: ", correlation_coefficient_d12_f1)


    correlations_cf_f1 = np.array(
        [correlation_coefficient_f1, correlation_coefficient_f1_2, correlation_coefficient_f1_3,
         correlation_coefficient_f1_4, correlation_coefficient_f1_5, correlation_coefficient_f1_6, correlation_coefficient_f1_7])


    # Correlazioni con l'F1-Score delle metriche.
    correlation_dist_f1 = np.array(
        [correlation_coefficient_d1_f1, correlation_coefficient_d2_f1, correlation_coefficient_d3_f1,
         correlation_coefficient_d4_f1, correlation_coefficient_d5_f1, correlation_coefficient_d6_f1, correlation_coefficient_d7_f1,
         correlation_coefficient_d8_f1, correlation_coefficient_d9_f1, correlation_coefficient_d10_f1, correlation_coefficient_d11_f1,
         correlation_coefficient_d12_f1])

    risultati = {
        'correlations_cf_f1' : correlations_cf_f1,
        'correlations_dist_f1': correlation_dist_f1,
    }

    return risultati



#=================================================================================================================================================================
#=================================================================================================================================================================
#====================================================================PARAMS=======================================================================================
#Classes
classes=[0, 1]
#Configuration
version="20"
#Splits
percentages=[25,45,70,100]
#Model F1
f1_c0 =  np.array([91, 84, 81])
f1_c1 =  np.array([93, 84, 84])
#=================================================================================================================================================================
#=================================================================================================================================================================
#=================================================================================================================================================================

#Riga da scrivere nel file di risultati
raw_data=[]
raw_data.append(version)

correlations_cf_f1 ={}
correlation_dist_f1 ={}
perfomance_f1_encoding = np.array([f1_c0, f1_c1])


i=0

for encoding in classes:
    print(bcolors.WARNING + "", end='')
    print(80*'-')
    print(bcolors.WARNING, encoding)
    print(80*'-', bcolors.WARNING)

    risultati=checkRFDChange(percentages, encoding,perfomance_f1_encoding[i],version)

    # Unpacking del dizionario
    encoding_correlations_cf_f1 = risultati['correlations_cf_f1']
    encoding_correlation_dist_f1 = risultati['correlations_dist_f1']

    #Salvataggio risultati encoding nel dizionario

    correlations_cf_f1[str(encoding)] = encoding_correlations_cf_f1
    correlation_dist_f1[str(encoding)]=encoding_correlation_dist_f1


    i+=1


metrics_f1=["CF_1", "CF_2","CF_3","CF_4", "CF_5","CF_6", "CF_7"]
distances=["D1","D2","D3","D4","D5","D6","D7","D8","D9","D10", "D11","D12"]



c0_corr_cf_f1=correlations_cf_f1["0"]
c1_corr_cf_f1=correlations_cf_f1["1"]

c0_corr_dist_f1=correlation_dist_f1["0"]
c1_corr_dist_f1=correlation_dist_f1["1"]


printResults(raw_data,c0_corr_cf_f1, c1_corr_cf_f1, metrics_f1, True,
             "Average correlation of confusion matrix-based metrics with the model's F1:", 0.80, ">")



#CORRELAZIONE DELLE DISTANZE
printResults(raw_data,c0_corr_dist_f1, c1_corr_dist_f1, distances, False,
             "Average correlation of divergence metrics with the model's F1:", -0.70, "<")

