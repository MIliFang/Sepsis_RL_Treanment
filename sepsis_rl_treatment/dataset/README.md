# Dataset Directory

This directory is where you should place your preprocessed MIMIC-III dataset file.

## 🔐 Important Notice

- **This directory is gitignored** - your data files will NOT be committed to version control
- **Do not commit real patient data** to public repositories
- Ensure you have proper **data use agreements** and **IRB approval** before working with MIMIC-III data

## 📋 Data Preparation Instructions

### 1. Obtain MIMIC-III Access
- Register for access at [PhysioNet](https://physionet.org/content/mimiciii/1.4/)
- Complete the required CITI training and sign the data use agreement
- Follow the instructions to download the MIMIC-III database

### 2. Apply Sepsis-3 Cohort Definition
Identify adult patients meeting the [Sepsis-3 criteria](http://jamanetwork.com/journals/jama/fullarticle/2492881):
- Age ≥ 18 years
- Suspected infection (antibiotics + cultures within specific time window)
- SOFA score increase ≥ 2 points from baseline
- ICU admission with vasopressor requirement during stay

### 3. Preprocess Following Published Protocol
Follow the methodology from [Komorowski et al., Nature Medicine 2018](https://www.nature.com/articles/s41591-018-0213-5):
- Extract hourly time-series data for all features listed in `../data/README.md`
- Align drug administration with physiological measurements
- Handle missing data using forward-fill (max 6 hours) and median imputation
- Apply feature transformations (log-transform, standardization) as specified

### 4. Create Required CSV File
Generate a single CSV file with:
- Filename: `mimictabl.csv`
- One row per patient-hour
- All columns specified in `../data/README.md`
- No personally identifiable information (PII)

## ✅ Expected File Structure
```
dataset/
└── mimictabl.csv    # Your preprocessed Sepsis-3 cohort data
```

## 📏 Dataset Specifications

| Parameter | Expected Value |
|-----------|----------------|
| File format | CSV (comma-separated) |
| Encoding | UTF-8 |
| Rows | ~25,000-35,000 (Sepsis-3 cohort) |
| Columns | 40+ features as documented |
| Time granularity | Hourly |
| Memory footprint | ~50-100 MB |

## 🔍 Validation Checklist

Before running the model, verify your `mimictabl.csv` satisfies:

- [ ] Contains all required columns from `../data/README.md`
- [ ] `icustayid` values are consistent within patient trajectories
- [ ] `mortality_90d` contains only binary values (0/1)
- [ ] Drug dosage columns contain non-negative values
- [ ] No personally identifiable information (PII) remains
- [ ] Time steps are sorted chronologically within each `icustayid`
- [ ] Sepsis-3 criteria have been properly applied to the cohort

## ⚠️ Troubleshooting

If you encounter errors during data loading:

1. **File not found**: Ensure your file is named exactly `mimictabl.csv` and placed directly in this directory
2. **Missing columns**: Check against the required schema in `../data/README.md`
3. **Data type errors**: Ensure numeric columns contain only numbers (no strings)
4. **Memory issues**: For very large datasets, consider sampling or optimizing data types

## 📚 Reference Implementation

For a reference implementation of the data extraction and preprocessing pipeline, see:
- [MIMIC-III Clinical Database](https://mimic.physionet.org/)
- [Sepsis-3 Definition Implementation](https://github.com/mi3-gmu/sepsis3-mimic)
- [Komorowski et al. Code Repository](https://github.com/matthieukomorowski/AI_Clinician)

> **Note**: This repository does not provide the actual data extraction scripts due to data use restrictions. You must create your own pipeline following your institution's data governance policies.
