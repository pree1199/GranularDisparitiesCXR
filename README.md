# GranularDisparitiesCXR

This GitHub repository holds training and validation code for Deep Learning and Machine Learning models trained to classify disease pathology on the MIMIC and CheXpert publicly available datasets. In addition, it holds the code used to assess the false positive rate of the "No Finding" label or underdiagnosis rate found in both datasets. 

# Model Training and Validation
Model training and validation code for the MIMIC  and CheXpert models is provided in the MIMIC_classification and CheXpert_classification folders. The code and architecture was largely derived from Seyyed-Kalantari et al.* 

# Granular Disparities Analysis
The analysis code is located in the granular_analysis folder. The mimic_final.ipnyb and chexpert_final.ipnyb files contain the code to analyze the characteristics of our test set for the models trained on each respective training data (demographic composition and 'No Finding' rates). An admissions.csv.gz file taken from MIMIC-IV version 2.2 (https://physionet.org/content/mimiciv/2.2/) was used to ascertain the demographic information of the patients that were used to test our models. Organized by dataset, the truth, pred and admissions csv files were fed into our analysis_bootstrapped.ipnyb file to calculate the 'No Finding' False Positive Rate (FPR). To determine a 95% confidence interval, we calculated the distribution of 'No Finding' FPR of our models on a bootstrapped resampling performed 1000 times. The resulting distribution of FPRs for a given race/ethnicity group was also compared to the distribution of FPRs from other race/ethnicity groups to see if the difference between distributions was significantly different from 0 (i.e. no disparity) with a one-sample t-test. This process was repeated for each coarse and granular racial category we were able to identify in the admissions.csv.gz file and then mapped as a forest plot. 


*Seyyed-Kalantari, L., Zhang, H., McDermott, M.B.A. et al. Underdiagnosis bias of artificial intelligence algorithms applied to chest radiographs in under-served patient populations. Nat Med 27, 2176–2182 (2021). https://doi.org/10.1038/s41591-021-01595-0

