import torch
import classification
from classification.train import train
from classification.predictions import make_pred_multilabel
import pandas as pd
import os
#-----------------------------
PATH_TO_IMAGES = r'C:/Users/sgari/Desktop/datasets/MIMIC/MIMIC_twofivesix/'

TRAIN_DF_PATH = r"C:\Users\sgari\Desktop\experiments\LalehSeyyed_UD_exp\MIMIC\Splits\train.csv"
TEST_DF_PATH  = r"C:\Users\sgari\Desktop\experiments\LalehSeyyed_UD_exp\MIMIC\Splits\test.csv"
VAL_DF_PATH   = r"C:\Users\sgari\Desktop\experiments\LalehSeyyed_UD_exp\MIMIC\Splits\val.csv"

# We mix all existing data of the provoder regardless of their original validation/train label in the original dataset and split them into 80-10-10 train test and validation sets based on Patient-ID such that no patient images appears in more than one split. 

torch.cuda.is_available()

#%% #run5 seed 36
def main():

    MODE = "test"  # Select "train" or "test", "Resume"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    val_df = pd.read_csv(VAL_DF_PATH)
    val_df_size = len(val_df)
    print("Validation_df size:",val_df_size)

    train_df = pd.read_csv(TRAIN_DF_PATH)
    train_df_size = len(train_df)
    print("Train_df size:", train_df_size)
    
    test_df = pd.read_csv(TEST_DF_PATH)
    test_df_size = len(test_df)
    print("Test_df size:", test_df_size)


    if MODE == "train":
        modeltype = "densenet"  # currently code is based on densenet121 
        CRITERION = 'BCELoss'
        lr = 0.5e-3

        model, best_epoch = train(train_df, train_df_size, val_df, val_df_size, PATH_TO_IMAGES, modeltype, CRITERION, device,lr)


    if MODE =="test":
        val_df = pd.read_csv(VAL_DF_PATH)
        test_df = pd.read_csv(TEST_DF_PATH)

        CheckPointData = torch.load('results/checkpoint')
        model = CheckPointData['model']

        make_pred_multilabel(model, test_df, val_df, PATH_TO_IMAGES, device)


    if MODE == "resume":
        modeltype = "resume" 
        CRITERION = 'BCELoss'
        lr = 0.5e-3

        model, best_epoch = train(train_df, val_df, PATH_TO_IMAGES, modeltype, CRITERION, device,lr)
        
        
        PlotLearnignCurve()


if __name__ == "__main__":
    main()
