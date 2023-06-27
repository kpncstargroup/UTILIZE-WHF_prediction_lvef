# Developing clinical risk prediction models for worsening heart failure events and death by left ventricular ejection fraction

## Overview
There is a clinical need to develop electronic health record (EHR)-based predictive models for worsening heart failure (WHF) events across clinical settings and across the spectrum of left ventricular ejection fraction (LVEF). Using data from Kaiser Permanente Northern California, an integrated health care delivery system, we developed boosted decision tree-based ensemble models for 1-year hospitalizations, ED visits/observation stays, and outpatient encounters for WHF and all-cause mortality in the overall HF population and within each HF subtype: HF with reduced EF (HFrEF; LVEF <40%), HF with mildly reduced EF (HFmrEF; LVEF 40-49%), and HF with preserved EF (HFpEF; LVEF ≥50%). We also examined the importance of EHR data domains in improving the performance of models for WHF events and death through a forward selection procedure. 

## Data
Due to the sensitive nature of Protected Health Information and risks of re-identification, data used in this study are not publicly available.

## Files

* utilize_prediction_bylvefoutcome.R: R script for the main analysis. Develops models in the overall population and by LVEF category for each outcome. Uses a grid search and cross-validation to tune an Xgboost model on the training set and measures performance on the test set using 1000 bootstrapped samples. Also calcluates relative variable importance and mean SHAP values.
* utilize_prediction_bydomain.R: R script for the secondary analysis. Develops models in the overall population for the outcomes of WHF and death, using a forward selection proecure by EHR data domain. Calculates the area under the curve and mean squared error at each iteration in the test set.
* Comorbidity_codes.xlsx: ICD, CPT, HCPCS codes to identify comorbid contidions.

Contributors: Rishi Parikh

Parikh RV, Go AS, Bhatt AS, Tan TC, Allen AR, Feng KY, Hamilton SA, Tai AS, Fitzpatrick JK, Lee KK, Adatya S, Avula HR, Sax DR, Shen X, Cristino J, Sandhu AT, Heidenreich PA, Ambrosy AP. Developing clinical risk prediction models for worsening heart failure events and death by left ventricular ejection fraction.


