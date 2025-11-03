# EyeSight
Integrative AI for Non-Invasive Diabetic Retinopathy Detection Using Pupillometry and Ensemble Deep Learning

## What the Project Does

This project presents **EyeSight**, a non-invasive, AI-powered diagnostic system for early detection and classification of **Diabetic Retinopathy (DR)**. By using **pupillometry**—analyzing the eye's pupil response to light—combined with ensemble deep learning models (**EfficientNetB3**, **DenseNet169**, and **ResNet50**), the system accurately predicts the stage of DR. A **Streamlit-based web interface** enables clinicians to interact with the model in real-time for swift and accessible diagnosis.

This system is designed to detect and classify DR severity into five stages:  
- Healthy  
- Mild  
- Moderate  
- Severe  
- Proliferative DR

By preprocessing pupil images and analyzing them through transfer-learned CNNs, the project builds a robust classification pipeline that also generates automated medical reports with recommendations.

## Why the Project is Useful

Diabetic Retinopathy is a leading cause of vision loss globally, particularly among diabetic patients. Traditional diagnostic methods require expensive retinal imaging equipment, which limits accessibility in rural and resource-constrained areas. EyeSight offers a **low-cost, real-time screening alternative** by analyzing readily accessible pupillometry data.

- Enables early intervention and monitoring  
- Reduces reliance on costly fundus cameras  
- Helps healthcare professionals make informed decisions quickly  
- Designed for low-resource clinical settings, enhancing rural accessibility  

## Model Accuracy

The project evaluated multiple deep learning architectures and ensemble methods for best performance:

| Model             | Accuracy |
|------------------|----------|
| EfficientNetB3    | 0.95     |
| DenseNet169       | 0.85     |
| ResNet50          | 0.81     |
| Stacked Ensemble  | 0.72     |

EfficientNetB3 was selected as the final model due to its superior performance.

## How Users Can Get Started with the Project

### Prerequisites

Ensure you have Python installed, along with the following packages:

- numpy  
- pandas  
- matplotlib  
- seaborn  
- scikit-learn  
- tensorflow  
- streamlit  
- opencv-python  

### Dataset

Ensure the **pupillometry dataset** is present in the working directory.

## Running the Project

1. Clone this repository.
2. Open and run the notebook: `Diabetic_retinopathy_detection.ipynb` to:  
   - Train models using pupillometry features  
   - Evaluate performance (Accuracy, Precision, Confusion Matrix, etc.)  
   - Export the best-performing model (EfficientNetB3)  
3. Launch the Streamlit web app:
   ```bash
   streamlit run .\app.py
   ```
4. Use the GUI to upload pupil images and predict DR stages.

## Usage

The system classifies the user input into one of five DR stages (0–4). The user uploads a pupil image and the model outputs:

- **Predicted DR Stage**  
- **Preventive & Precautionary Measures**
