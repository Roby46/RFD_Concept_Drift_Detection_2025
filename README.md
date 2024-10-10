# RFD_Concept_Drift_Detection-2025
Supplemental material for the paper "A metadata-driven approach for concept drift detection in Machine Learning Systems"
This repo includes a complete example of how to apply the proposed methodology described in the paper "A Metadata-Driven Approach for Concept Drift Detection in Machine Learning Systems."

## Datasets and Configurations
The original datasets used in the experimental phase can be found in the **Dataset** folder. 

Additionally, this repository includes a utility script (see the **Generate Configuration** folder) that generates random test configurations consisting of one training batch and a number of test batches. 
These configurations simulate model performance in the presence of concept drift, allowing users to explore how a model behaves as data distributions shift over time. 
The script features a threshold parameter that controls the progression of model performance, enabling users to test different scenarios. 
For instance, the threshold can be adjusted to make the model's performance degrade early or later during the test batches, and the degree of change can be minor or significant
This script is not tied to the methodology used in the paper, but serves as a general-purpose tool to experiment with model behavior in dynamic environments. 

Nevertheless, a suitable test configuration should have the same structure of the one reported
in the **Generate Configuration/1** folder, which shows an example of the aforementioned script, wich creates a folder named with the value of the **version** parameter. 
The important files inside this folder are those starting with **"Version_"**, since they represent the dataset being incrementally updated with the new data batches. These files are leveraged by the **split.py** script that can be found in the **Splitted_with_predictions** folder. This script splits the data in each batch according to the target label, as described in our methodology.
The results are saved in subfolders associated with the percentage of the test batch. For instance, since the first test batch represent the 25% of the dataset, its splitted version can be found in the **25** folder. 

<img src="https://github.com/Roby46/RFD_Concept_Drift_Detection_2025/blob/main/Images/Folder.png?raw=true" width="50%">

## RFD Analysis
After performing RFDc Discovery with a suitable algorithm on all the splitted data batches, the resulting RFD are filtered according to the strategy described in the paper. Thus, for each class and for each data batch, the filtered RFDs are stored in the
**MinimalRFDS** folder. For example, this repository contains the minimal RFDcs related to experiment with ID 20 reported in the paper. 
The syntax used to represent RFDs should be the following:
```
RHS;COL0;COL1;COL2;COL3;COL4;COL5;COL6;COL7;COL8;COL9;COL10;COL11;COL12;COL13;COL14;COL15
COL0;1.0;0.0;?;0.0;0.0;0.0;?;?;?;?;?;?;?;?;?;?
```
Specifically, each RFD is represented by a row The column **RHS** contains the attribute present in the RHS of the RFD. The attributes whose values are "?" are not involved in the RFD, whereas otherwise its associated similarity threshold is reported.
For example, the example above represent the RFD: COL1(0.0), COL3(0.0), COL4(0.0), COL5(0.0)-->COL0(1.0)
These RFDs files can be then analyzed with the script provided in the folder **RFD Analysis**, takes as parameter the target labels, the number of the test configurations, the percentages of the splits and the performance of the model on each class. 

```
classes=[0, 1]

version="20"

percentages=[25,45,70,100]

f1_c0 =  np.array([91, 84, 81])
f1_c1 =  np.array([93, 84, 84])
```
The script will compute both the divergence metrics and the confusion matrix-based metrics and will show the final correlation obtained by them. 
An example of output is the following:

<img src="https://github.com/Roby46/RFD_Concept_Drift_Detection_2025/blob/main/Images/Correlations.png?raw=true" width="80%">

  
## Baseline Approaches
[FROUROS](https://github.com/IFCA-Advanced-Computing/frouros)