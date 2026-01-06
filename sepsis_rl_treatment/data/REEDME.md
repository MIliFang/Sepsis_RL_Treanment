# Data Schema and Requirements

This directory contains documentation about the expected data format for the sepsis RL treatment project.

## Input Data Format

The main input file should be named `mimictabl.csv` and placed in the `../dataset/` directory. The CSV file must contain the following columns:

### Required Features

#### Patient Demographics
- `icustayid`: Unique ICU stay identifier (integer)
- `gender`: Binary gender (0 = F, 1 = M)
- `age`: Patient age in years (float)

#### Vital Signs & Lab Values
- `HR`: Heart rate (bpm)
- `SysBP`: Systolic blood pressure (mmHg)
- `DiaBP`: Diastolic blood pressure (mmHg)
- `MeanBP`: Mean arterial pressure (mmHg)
- `RR`: Respiratory rate (breaths/min)
- `SpO2`: Peripheral capillary oxygen saturation (%)
- `FiO2`: Fraction of inspired oxygen (%)
- `Weight_kg`: Patient weight (kg)
- `Arterial_pH`: Arterial pH
- `PaO2`: Partial pressure of arterial oxygen (mmHg)
- `paCO2`: Partial pressure of arterial CO2 (mmHg)
- `Arterial_BE`: Arterial base excess (mmol/L)
- `HCO3`: Bicarbonate (mmol/L)
- `PaO2_FiO2`: PaO2/FiO2 ratio
- `SOFA`: Sequential Organ Failure Assessment score
- `Arterial_lactate`: Arterial lactate level (mmol/L)

#### Laboratory Values
- `Potassium`, `Sodium`, `Chloride`: Electrolytes (mmol/L)
- `Glucose`: Blood glucose (mg/dL)
- `Magnesium`, `Calcium`: Minerals (mg/dL)
- `Hb`: Hemoglobin (g/dL)
- `WBC_count`: White blood cell count (10³/μL)
- `Platelets_count`: Platelet count (10³/μL)
- `Albumin`: Serum albumin (g/dL)
- `BUN`: Blood urea nitrogen (mg/dL)
- `Creatinine`: Serum creatinine (mg/dL)
- `Total_bili`: Total bilirubin (mg/dL)
- `PTT`: Partial thromboplastin time (seconds)
- `PT`: Prothrombin time (seconds)
- `INR`: International Normalized Ratio

#### Drug Administration (Max dosage during timestep)
- `norad_max`: Norepinephrine (μg/kg/min)
- `vaso_max`: Vasopressin (U/min)
- `epi_max`: Epinephrine (μg/kg/min)
- `phenyl_max`: Phenylephrine (μg/kg/min)
- `dopa_max`: Dopamine (μg/kg/min)

#### Outcome Variables
- `mortality_90d`: 90-day mortality (0 = survived, 1 = deceased)

## Data Preprocessing Notes
The `mimictabl.csv` file is **not raw MIMIC-III data**, but the **final output** of the AI Clinician preprocessing pipeline (`AIClinician_sepsis3_def_160219.py`). It has already undergone:

1.  **Sepsis Cohort Selection**: Patients identified using Sepsis-3 criteria (SOFA ≥ 2 after infection onset).
2.  **Time Alignment**: Data aggregated into **4-hour windows**.
3.  **Imputation**:
    -   Sample-and-hold for stable vitals (e.g., weight for 48h, HR for 2h).
    -   Linear interpolation for variables with <5% missingness.
    -   KNN imputation for other moderate missingness.
    -   Rule-based estimation (e.g., FiO₂ from O₂ flow, MeanBP from Sys/Dia).
4.  **Outlier Removal**: Implausible values (e.g., HR > 250, age > 150 years) are filtered out.
5.  **Feature Engineering**: Key scores like `SOFA` and `SIRS` are precomputed.

**You should provide this preprocessed table directly for training.** No additional preprocessing (beyond what’s in `DataProcessor`) is needed.

## Example Row
```csv
icustayid,gender,age,HR,SysBP,DiaBP,MeanBP,RR,SpO2,FiO2,Weight_kg,norad_max,vaso_max,epi_max,phenyl_max,dopa_max,mortality_90d,SOFA
200001,1,65,95,110,65,78,22,96,40,75.2,0.05,0.0,0.0,0.0,0.0,0,4
