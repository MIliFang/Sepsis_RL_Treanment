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

This dataset is **not raw MIMIC-III data**. It is the result of a complex preprocessing pipeline that includes:

1.  **Cohort Selection**: Identifies sepsis patients using Sepsis-3 criteria (SOFA ≥ 2 after infection onset).
2.  **Time Alignment**: Data is aggregated into **4-hour time windows** centered around or following the infection onset.
3.  **Imputation**:
    -   **Sample-and-Hold**: For many vitals and labs, missing values are forward-filled for a clinically defined duration (e.g., 2h for HR, 48h for weight).
    -   **Linear Interpolation**: For variables with <5% missingness.
    -   **KNN Imputation**: For the remaining variables with moderate missingness.
    -   **Rule-based Imputation**: e.g., estimating FiO2 from oxygen flow rate and device type, calculating MeanBP from Sys/Dia.
4.  **Outlier Removal**: Clinically implausible values (e.g., HR > 250, age > 150 years) are removed or clipped.
5.  **Feature Engineering**: Derived variables like `Shock_Index`, `PaO2_FiO2`, `SOFA`, and `SIRS` are computed.
6.  **Patient Exclusion**: Patients with extreme values (e.g., >10L fluid input in 4h) or early withdrawal-of-care are excluded.

**Therefore, the `mimictabl.csv` file you provide should be the final, preprocessed, and aggregated table ready for MDP or RL modeling.**

For the exact preprocessing steps, please refer to the original AI Clinician publication and its code repository.


## Data Preprocessing Notes

- All continuous variables should be in their natural units (not normalized)
- Missing values will be handled automatically during preprocessing:
  - Binary features: filled with 0
  - Normal features: filled with median
  - Log features: filled with median, then log-transformed
- Time steps should be sorted chronologically within each `icustayid`
- Each row represents one timestep (typically hourly) for a patient

## Example Row
```csv
icustayid,gender,age,HR,SysBP,DiaBP,MeanBP,RR,SpO2,FiO2,Weight_kg,norad_max,vaso_max,epi_max,phenyl_max,dopa_max,mortality_90d,SOFA
200001,1,65,95,110,65,78,22,96,40,75.2,0.05,0.0,0.0,0.0,0.0,0,4
