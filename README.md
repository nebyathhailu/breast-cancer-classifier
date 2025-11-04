# Breast Cancer Classifier

This project is a Breast Cancer Classifier implemented as an interactive Jupyter notebook, optimized for use in Google Colab. Use the notebook to explore data, train a classification model (benign vs malignant), evaluate performance, and save results without installing anything locally.

## Features

- **Breast Tumor Classification:** Trains a model to distinguish benign vs malignant tumors using common datasets (e.g., Wisconsin Breast Cancer).
- **Colab Integration:** Open and run the notebook in Google Colab—no local setup required.
- **Interactive Notebooks:** Inspect preprocessing, model training, evaluation, and visualizations cell-by-cell.
- **Modular Design:** Easily adapt preprocessing and model code to different datasets or classifiers.

## Getting Started

### Run in Google Colab

Click the badge below to open the notebook in Colab:

<a href="https://colab.research.google.com/github/nebyathhailu/breast-cancer-classifier/blob/main/notebooks/breast_cancer_classifier.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

(If your notebook uses a different filename or path, update the URL above to point to the actual notebook file.)

### Prerequisites (for Colab)

- A Google account
- Your dataset uploaded to Google Drive (e.g., the Wisconsin Breast Cancer dataset or a CSV with features and labels)

### Steps

1. **Open the Notebook in Colab:**  
   Click the "Open in Colab" badge above or open the notebook directly:
   https://colab.research.google.com/github/nebyathhailu/breast-cancer-classifier/blob/main/notebooks/breast_cancer_classifier.ipynb

2. **Install Required Libraries (if needed):**  
   The notebook may install packages automatically; otherwise run a cell like:
   ```python
   !pip install pandas scikit-learn matplotlib seaborn joblib
   ```

3. **Mount Google Drive:**  
   If you store data/models in Drive, run:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
   Place your dataset in Drive (for example, `/content/drive/MyDrive/breast_cancer/wisconsin_breast_cancer.csv`) or update the path in the notebook.

4. **Update Paths (if necessary):**  
   Adjust file paths in the notebook to point to your uploaded dataset or desired save location.

5. **Run All Cells:**  
   Use `Runtime > Run all` in Colab to execute the full workflow and interact with the training & evaluation steps.

## Notebook Workflow

- Data loading and basic exploration
- Preprocessing (missing values, scaling, encoding if needed)
- Train/test split and model training (examples with scikit-learn classifiers)
- Model evaluation: accuracy, precision, recall, F1-score, ROC AUC, and confusion matrix
- Visualizations of feature distributions and model performance
- Save trained model and results to Google Drive

## Repository Structure

- `notebooks/breast_cancer_classifier.ipynb` : Main Colab notebook (interactive walkthrough).
- `README.md` 
## Notes

- Recommended metrics to inspect: Accuracy, Precision, Recall, F1-score, ROC AUC.
- Save trained models (e.g., with joblib) and evaluation reports to Google Drive to persist across sessions.
- Do not add or commit private patient data to the repository — keep sensitive data in Drive and reference it from the notebook.

## Contributing

Contributions are welcome! Please fork the repository, create a new branch for your feature or fix, and open a pull request. Notebook improvements, clearer explanations, and reproducibility fixes are especially helpful.

## Contact

- Repository: https://github.com/nebyathhailu/breast-cancer-classifier  
- Owner: https://github.com/nebyathhailu
