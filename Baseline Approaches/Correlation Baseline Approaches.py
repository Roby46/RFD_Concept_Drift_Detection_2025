import pandas as pd
import numpy as np
import csv
import os


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


# Function to check if the CSV file exists
def is_file_exist(file_path):
    return os.path.exists(file_path)



def printResults(raw_data, c0_values, c1_values, metrics, reverse_, message, threshold, operator):
    # --------------------------------------------------------

    if "Incrementale" in message:
        eval_type="Incremental"
    else:
        eval_type="Standard"

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
    for metric, mean_value, abs_mean_value in sorted_results_mean:
        if(operator==">"):
            color = bcolors.OKGREEN if mean_value > threshold else bcolors.WARNING
        elif(operator=="<"):
            color = bcolors.OKGREEN if mean_value < threshold else bcolors.WARNING
        print(
            bcolors.BOLD,
            metric,
            ":",
            color,
            mean_value,
            bcolors.ENDC
        )
    print(90 * "=")

def computeDistance(percentages, encoding, current_enc_f1_perf, hell_max,hell_mean,hinorm_max,hinorm_mean):
    correlation_coefficient_f1_hell_max = np.corrcoef(current_enc_f1_perf,hell_max)[0, 1]
    correlation_coefficient_f1_hell_mean = np.corrcoef(current_enc_f1_perf,hell_mean)[0, 1]
    correlation_coefficient_f1_hinorm_max = np.corrcoef(current_enc_f1_perf, hinorm_max)[0, 1]
    correlation_coefficient_f1_hinorm_mean = np.corrcoef(current_enc_f1_perf, hinorm_mean)[0, 1]

    correlation_distances_f1 = np.array(
        [correlation_coefficient_f1_hell_max,correlation_coefficient_f1_hell_mean,
         correlation_coefficient_f1_hinorm_max,correlation_coefficient_f1_hinorm_mean])

    risultati = {
        'correlations_f1': correlation_distances_f1 }

    return risultati



#Codifiche
target_labels=[0, 1]

#Scrittura nel file di risultati
write_results=True
#Percentuali di split (training + testing)
#percentages=[20,30,60,90]
percentages=[25,45,70,100]

correlations_f1={}
correlations_rec={}
correlations_pre={}

f1_c0 = np.array([91, 84, 81])
f1_c1 = np.array([93, 84, 84])
hell_max_c0 = np.array([0.157, 0.159, 0.161])
hell_max_c1 = np.array([0.132, 0.161, 0.154])
hell_mean_c0 = np.array([0.105, 0.108, 0.114])
hell_mean_c1 = np.array([0.077, 0.101, 0.109])
hinorm_max_c0 = np.array([0.141, 0.168, 0.173])
hinorm_max_c1 = np.array([0.085, 0.091, 0.105])
hinorm_mean_c0 = np.array([0.072, 0.087, 0.094])
hinorm_mean_c1 = np.array([0.066, 0.069, 0.081])


# Array che contiene gli array di prestazioni per ogni porzione di test
perfomance_f1_encoding = np.array([f1_c0, f1_c1])

hell_max=np.array([hell_max_c0,hell_max_c1])
hell_mean=np.array([hell_mean_c0,hell_mean_c1])
hinorm_max=np.array([hinorm_max_c0,hinorm_max_c1])
hinorm_mean=np.array([hinorm_mean_c0,hinorm_mean_c1])

i=0
for encoding in target_labels:
    #Chiamata funzione di controllo
    risultati=computeDistance(percentages, encoding,perfomance_f1_encoding[i],hell_max[i],hell_mean[i],hinorm_max[i],hinorm_mean[i])
    encoding_correlation_dist_f1 = risultati['correlations_f1']
    correlations_f1[str(encoding)] = encoding_correlation_dist_f1

    i=+1

c0_corr_f1=correlations_f1["0"]
c1_corr_f1=correlations_f1["1"]

metrics=["Hellinger MAX", "Hellinger MEAN", "HiNorm MAX", "Hinorm MEAN"]

raw_data=[]

printResults(raw_data,c0_corr_f1,c1_corr_f1, metrics, False,
             "Correlazione delle metriche con F1 (media sugli encoding):", -0.70, "<")
