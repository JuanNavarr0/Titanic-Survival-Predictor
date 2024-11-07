# Titanic Survival Prediction

This project is a machine learning model designed to predict the survival chances of passengers on the Titanic based on various features like age, class, fare, and family relationships. The model is built using PyTorch and trained on the well-known Titanic dataset.

## Project Structure
- **main.ipynb**: Jupyter notebook containing the data analysis, feature engineering, model training, and evaluation steps.
- **app.py**: Streamlit app for deploying the model as a web application where users can input passenger details and get predictions on survival.
- **data/**: Directory containing `train.csv` and `test.csv` with training and testing data.
- **requirements.txt**: Lists all dependencies needed to run the notebook and app.
- **README.md**: Overview of the project, including setup and usage instructions.




## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/titanic-survival-prediction.git

2. Navigate to the project directory:
   ```bash
   cd titanic-survival-prediction
   
2.	Install the requiered packages:
   ```bash
   pip install -r requirements.txt
   ```

## Data

The dataset is derived from the Titanic survival dataset on Kaggle. It includes:

•	train.csv: Training data with labeled survival outcomes.
+•	test.csv: Test data to evaluate the model’s generalization.

## Usage

### Running the Jupyter Notebook

1.	Open main.ipynb in Jupyter Notebook or JupyterLab.
2.	Run through the cells to perform data analysis, preprocessing, model training, and evaluation.

### Running the Streamlit App

1.	Launch the app with the following command:
   ```bash
   streamlit run app.py
   ```

2.	Open the provided URL to interact with the app.
