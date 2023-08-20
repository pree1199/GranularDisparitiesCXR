# GranularDisparitiesCXR

This GitHub repository holds training and validation code for Deep Learning and Machine Learning models trained to classify disease pathology on the MIMIC and CheXpert publicly available datasets. In addition, it holds the code used to assess the false positive rate of the "No Finding" label or underdiagnosis rate found in both datasets. 

# Model Training and Validation

# Granular Disparities Analysis
The analysis code is located in the granular_analysis folder. The mimic_final.ipnyb and chexpert_final.ipnyb files contain the code to process the outputs (true.csv and pred.csv) of our 5 MIMIC-trained and 5 CheXpert trained models on the MIMIC test set and MIMIC entire dataset respectively. An admissions.csv.gz file taken from MIMIC-IV version 2.2 (https://physionet.org/content/mimiciv/2.2/) was used to ascertain the demographic information of the patients that were used to test our models. Organized by dataset, the truth, pred and admissions csv files were fed into our ipnyb files for each trained model to calculate the underdiagnosis rate per model and average the 5 'No Finding" false positive rate and construct a confidence interval as done in Seyyed-Kalantari et al.* This process was repeated for each granular racial category we were able to identify in the admissions.csv.gz file and then mapped as a forest plot. 


*Seyyed-Kalantari, L., Zhang, H., McDermott, M.B.A. et al. Underdiagnosis bias of artificial intelligence algorithms applied to chest radiographs in under-served patient populations. Nat Med 27, 2176–2182 (2021). https://doi.org/10.1038/s41591-021-01595-0

