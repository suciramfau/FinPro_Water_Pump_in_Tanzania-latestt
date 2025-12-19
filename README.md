# Water Pump Failure Prediction in Tanzania
Machine Learning Classification | Decision Support System

Live Demo: [Streamlit App](https://suciramfau-finpro-water-pump-in-tanzania-latestt.streamlit.app/)

GitHub Repository:[Suciramfau](https://github.com/suciramfau/FinPro_Water_Pump_in_Tanzania-latestt)

## 1. Project Overview

Access to clean water remains a critical challenge in Tanzania.
Approximately 25–30% of water pumps are non-functional, significantly limiting daily water access for millions of people.

This project develops a machine learning–based decision support system to predict the operational status of water pumps.
The goal is to help stakeholders prioritize maintenance, reduce downtime, and allocate limited resources more effectively through data-driven insights.

## 2. Business Problem & Objective

### Problem Statement

Maintenance decisions for water pumps are often reactive, made after failures occur.
Limited early indicators of pump failure lead to delayed repairs and inefficient use of resources.

### Objective

To predict water pump operational status into three categories:

* Functional

* Functional needs repair

* Non functional

The model is designed to support preventive maintenance planning, enabling early intervention before pumps become completely non-operational.

### Evaluation Metric

Macro F1 Score, selected to:

* Handle class imbalance

* Ensure fair performance across all classes, especially minority failure categories

## 3. Data Overview

* Source: Kaggle – Tanzania Water Pump Dataset

* Observations: 59,400 waterpoints

* Original Features: 41

* Final Features (after cleaning): 23

**Feature Groups**

* Geospatial: latitude, longitude, region

* Pump Technical: extraction type, waterpoint type, construction year

* Water Condition: quantity, quality, source

* Socio-economic: population, payment type, management
 
Target Variable: status_group

## 4. Key Data Challenges & Solutions
### Challenges

* Highly imbalanced multi-class target

* High-cardinality categorical features

* Missing values and placeholder entries

* Noisy and inconsistent GPS information

### Solutions

* Context-aware handling of missing and placeholder values

* Removal of redundant and low-informative features

* Feature consolidation for high-cardinality categories

* Pipeline-based preprocessing to ensure consistency between training and deployment

All preprocessing steps were implemented inside a single machine learning pipeline.

## 5. Exploratory Data Analysis (Key Insights)

Insights that directly inform maintenance and operational decisions:

* Older pumps show a significantly higher failure probability

* Water quantity labeled as “dry” is a strong indicator of non-functional pumps

* Pump failures cluster geographically in specific regions

* Community-managed pumps (VWC) tend to be more reliable

* Gravity-based extraction systems outperform mechanical pumps

These findings highlight the combined importance of geospatial context and technical pump characteristics.

## 6. Modeling Approach & Performance

### Baseline Models

* Logistic Regression (balanced class weights)

* Random Forest (balanced class weights)

  
| Model                    | Accuracy | Macro F1 |
| ------------------------ | -------- | -------- |
| Logistic Regression      | 0.626    | 0.550    |
| Random Forest (Balanced) | 0.804    | 0.689    |
| SMOTE + Random Forest    | 0.789    | 0.689    |

**Best baseline: **Random Forest with balanced class weights

## 7. Hyperparameter Tuning

* Method: RandomizedSearchCV

* Subset: 20,000 samples (efficiency-focused)

* Tuned Parameters:

  * n_estimators

  * max_depth

  * min_samples_split

### Final Model Performance

* Accuracy: 0.866

* Macro F1: 0.813

The tuned model shows significant improvement in minority class prediction, making it suitable for imbalanced multi-class classification.

## 8. Feature Importance & Interpretability

Top predictive features include:

* Longitude & Latitude

* Pump Age

* Water Quantity (Dry)

* GPS Height

* Population

**Interpretation:**
Water pump failure risk is primarily driven by location-specific and physical pump indicators, reinforcing the need for region-based maintenance strategies.


## 9. Deployment: Decision Support System

The trained machine learning pipeline was deployed as an interactive Streamlit web application.

### Application Features

* Single waterpoint prediction via form input

* Probability distribution across all three status categories

* Interpretative explanation based on input characteristics

**How the Prediction System Works**

1. User inputs waterpoint characteristics (location, pump age, water condition, management type, etc.)

2. Inputs are converted into a single-row DataFrame

3. Data is passed directly into the trained ML pipeline

4. The pipeline automatically:

  * Applies encoding and transformations

  * Generates predictions

  * Outputs class probabilities

No manual preprocessing occurs in Streamlit.

**Design Principles**

  * Reproducibility

  * Deployment safety

  * Consistency between training and inference

## 10. Business Recommendations

### Based on model insights:

* Prioritize maintenance in regions with high predicted failure risk

* Replace or rehabilitate pumps older than 10 years

* Closely monitor areas frequently classified as “water quantity = dry”

* Use predictions as an early warning system for preventive maintenance planning

## 11. Key Learnings

* Importance of Macro F1 Score for imbalanced multi-class problems

* Value of pipeline-based preprocessing for production-ready ML

* Translating model outputs into actionable infrastructure decisions

* End-to-end data science thinking: from raw data to real-world impact

## **Final Notes**

This project demonstrates the application of machine learning not only as a predictive tool, but as a decision support system for real-world infrastructure challenges.
