# Data Schema and Requirements

This directory contains documentation about the expected data format for the sepsis RL treatment project.

## Cohort Definition

This project focuses on **adult ICU patients meeting the Sepsis-3 criteria**:

> Sepsis is defined as life-threatening organ dysfunction caused by a dysregulated host response to infection, clinically identified as an increase in the Sequential Organ Failure Assessment (SOFA) score of 2 points or more in the context of suspected infection.
> 
> Reference: [Seymour CW, et al. JAMA. 2016;315(8):801-810](http://jamanetwork.com/journals/jama/fullarticle/2492881)

### Inclusion Criteria
- Adult patients (≥ 18 years)
- ICU admission with suspected infection (based on administration of antibiotics and evidence of body fluid culture)
- Increase in SOFA score ≥ 2 points from baseline
- Vasopressor requirement during ICU stay

### Exclusion Criteria
- ICU stay < 24 hours
- Do-not-resuscitate status on admission
- Palliative care focus
- Missing critical variables (>30% missing values)

## Input Data Format

The main input file should be named `mimictabl.csv` and placed in the `../dataset/` directory. The CSV file must contain the following columns:

### Patient Identification
- `subject_id`: Unique patient identifier
- `icustayid`: Unique ICU stay identifier (integer)
- `hadm_id`: Hospital admission identifier

### Patient Demographics
- `gender`: Binary gender (0 = F, 1 = M)
- `age`: Patient age in years (float)
- `weight`: Patient weight in kg (float)
- `elixhauser_score`: Elixhauser comorbidity score

### Time Variables
- `hour`: Hours since ICU admission
- `timestep`: Sequential timestep index

### Vital Signs & Lab Values
- `HR`: Heart rate (bpm)
- `SysBP`: Systolic blood pressure (mmHg)
- `DiaBP`: Diastolic blood pressure (mmHg)
- `MeanBP`: Mean arterial pressure (mmHg)
- `RR`: Respiratory rate (breaths/min)
- `SpO2`: Peripheral capillary oxygen saturation (%)
- `FiO2`: Fraction of inspired oxygen (%)
- `temp`: Body temperature (°C)
- `urine_output`: Hourly urine output (mL)
- `Arterial_pH`: Arterial pH
- `PaO2`: Partial pressure of arterial oxygen (mmHg)
- `paCO2`: Partial pressure of arterial CO2 (mmHg)
- `Arterial_BE`: Arterial base excess (mmol/L)
- `HCO3`: Bicarbonate (mmol/L)
- `PaO2_FiO2`: PaO2/FiO2 ratio
- `SOFA`: Sequential Organ Failure Assessment score (0-24)
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

### Treatment Variables (Max dosage during timestep)
- `iv_fluid`: Intravenous fluid administered in last hour (mL)
- `norad_max`: Norepinephrine (μg/kg/min)
- `vaso_max`: Vasopressin (U/min)
- `epi_max`: Epinephrine (μg/kg/min)
- `phenyl_max`: Phenylephrine (μg/kg/min)
- `dopa_max`: Dopamine (μg/kg/min)

### Outcome Variables
- `mortality_90d`: 90-day mortality (0 = survived, 1 = deceased)

## Data Preprocessing Protocol

Our preprocessing follows methodology established in [Komorowski et al., Nature Medicine 2018](https://www.nature.com/articles/s41591-018-0213-5):

1. **Time Alignment**:
   - All variables aligned to consistent 4 hourly timesteps
   - Forward filling for short gaps (<24 hours), otherwise considered missing

2. **Feature Transformation**:
   - Binary features: Gender (centered around 0)
   - Normally distributed features: Standardized using z-score
   - Skewed features (BUN, Creatinine, Bilirubin, INR): Log-transformed before standardization
   - Blood pressure features: Normalized by dividing by 100

3. **Action Space Discretization**:
   - Norepinephrine: (0, 0.1, 0.5, ∞) → Levels 0-3
   - Vasopressin: (0, 0.05, 0.1, ∞) → Levels 0-3
   - Epinephrine: (0, 0.04, 0.06, ∞) → Levels 0-3
   - Phenylephrine: (0, 2.5, 5.0, ∞) → Levels 0-3
   - Dopamine: Binary usage indicator

4. **Sample Filtering**:
   - Only clinically relevant timesteps retained
   - Include timesteps where:
     * Vasopressors or fluids were administered
     * MAP < 65 mmHg (hypotension)
     * SOFA score > 5 (severe organ dysfunction)
     * Heart rate ≥ 100 (tachycardia)

5. **Handling Missing Data**:
   - Binary features: filled with 0
   - Normal features: filled with median
   - Log features: filled with median, then log-transformed
   - Critical missing features may exclude the timestep

## Example Data Row
```csv
icustayid,gender,age,HR,MeanBP,SOFA,norad_max,vaso_max,mortality_90d,timestep
263352,1,68,95,72,6,0.15,0.0,0,3
```

## Data Validation

Prior to training, the system performs:
1. Cohort validation: Ensures patients meet Sepsis-3 criteria
2. Missing value assessment: Flags features with >20% missingness
3. Clinical range validation: Out-of-range values are capped at physiologically plausible limits
4. Action space validation: Drug combinations are checked against clinical constraints

For implementation details, see the `DataProcessor` class in `src/data_processor.py`.
