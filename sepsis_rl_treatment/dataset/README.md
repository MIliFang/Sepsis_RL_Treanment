
# Dataset Directory

This directory is where you should place your MIMIC-III derived dataset file.

## Instructions

1. **Obtain MIMIC-III Access**: 
   - Register for access at [PhysioNet](https://physionet.org/content/mimiciii/1.4/)
   - Complete the required training and data use agreement

2. **Preprocess Your Data**:
   - Extract the required columns as specified in `../data/README.md`
   - Create a CSV file named `mimictabl.csv`
   - Ensure the file follows the schema described in the data documentation

3. **Place Your File**:
   - Copy your `mimictabl.csv` file into this directory
   - The final path should be: `./dataset/mimictabl.csv`

## Expected File Structure