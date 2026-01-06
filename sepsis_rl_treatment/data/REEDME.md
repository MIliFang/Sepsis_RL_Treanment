# Data Requirements for Sepsis RL Treatment

## Cohort Definition
- **Sepsis-3 criteria** adult patients (≥18 years) from MIMIC-III
- SOFA score increase ≥2 points with suspected infection
- ICU stay with vasopressor requirement
- Reference: [Seymour CW et al., JAMA 2016](http://jamanetwork.com/journals/jama/fullarticle/2492881)

## Data Placement
- Place your preprocessed file as `dataset/mimictabl.csv`
- **This directory is gitignored - your data will not be committed**
- Follow your institution's data governance policies

## Required Columns
**Patient ID**: `icustayid`, `subject_id`, `hadm_id`  
**Demographics**: `gender`, `age`, `weight`  
**Time**: `hour`, `timestep`  
**Vitals**: `HR`, `SysBP`, `DiaBP`, `MeanBP`, `RR`, `SpO2`, `FiO2`, `temp`, `urine_output`  
**Labs**: `Arterial_pH`, `PaO2`, `paCO2`, `Arterial_BE`, `HCO3`, `PaO2_FiO2`, `SOFA`, `Arterial_lactate`  
**Electrolytes**: `Potassium`, `Sodium`, `Chloride`, `Glucose`, `Magnesium`, `Calcium`  
**Blood**: `Hb`, `WBC_count`, `Platelets_count`, `Albumin`  
**Renal/Liver**: `BUN`, `Creatinine`, `Total_bili`  
**Coagulation**: `PTT`, `PT`, `INR`  
**Drugs**: `norad_max`, `vaso_max`, `epi_max`, `phenyl_max`, `dopa_max`  
**Outcomes**: `mortality_90d`, `mortality_icu`, `los_icu`

## Preprocessing Guidelines
- Hourly time-series alignment with forward-fill (<6 hours)
- Feature normalization:
  - Binary features centered around 0
  - BP features divided by 100
  - Skewed features (BUN, Creatinine, etc.) log-transformed
- Drug discretization using clinical ranges
- Sample filtering (keep clinically relevant timesteps only)

## Validation Checklist
Before running:
- [ ] File named exactly `mimictable.csv` in `dataset/` directory
- [ ] Contains all required columns above
- [ ] No personally identifiable information (PII)
- [ ] `mortality_90d` contains only 0/1 values
- [ ] Drug dosages are non-negative
- [ ] Time steps sorted chronologically within each ICU stay

## 📚 Reference Implementation

For a reference implementation of the data extraction and preprocessing pipeline, see:
- [MIMIC-III Clinical Database](https://mimic.physionet.org/)
- [Sepsis-3 Definition Implementation](https://github.com/mi3-gmu/sepsis3-mimic)
- [Komorowski et al. Code Repository](https://github.com/matthieukomorowski/AI_Clinician)

> **Note**: This repository does not provide the actual data extraction scripts due to data use restrictions. You must create your own pipeline following your institution's data governance policies.

## Scientific Context

This project builds upon the groundbreaking work published in Nature Medicine by Komorowski et al. (2018), "The Artificial Intelligence Clinician learns optimal treatment strategies for sepsis in intensive care." The original study demonstrated that reinforcement learning could extract implicit knowledge from patient data to learn optimal treatment strategies for sepsis that improve patient outcomes. Our implementation extends this work with modern RL techniques including conservative Q-learning and multi-task learning frameworks.
