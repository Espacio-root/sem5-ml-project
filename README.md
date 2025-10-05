# Prerequisites

### Set up virtual environment
```bash
python3 -m venv ./venv
source ./venv/bin/activate
pip install -r requirements.txt
```

### Get the datasets
1. Download the [first dataset](https://www.kaggle.com/datasets/techsash/waste-classification-data) and place it in `./data/d1.zip`  
2. Download the [second dataset](https://www.kaggle.com/datasets/rayhanzamzamy/non-and-biodegradable-waste-dataset) and place it in `./data/d2.zip`  
3. Download the [third dataset](https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification) and place it in `./data/d3.zip`  
4. Run the following commands from the project root (`./`):
```bash
unzip ./data/d1.zip -d ./data/d1
unzip ./data/d2.zip -d ./data/d2
unzip ./data/d3.zip -d ./data/d3
```

---

# Project Structure

### Data Cleaning and Preprocessing

- **`preprocess.ipynb`** — Processes and combines the three datasets into one and stores it in CSV format as `trash.csv`.  
- **`feature_extraction.ipynb`** — Uses a ResNet (CNN) model to extract features from each image in `trash.csv` and stores them in `final.csv`.

---

### Models

- **`random_forest_model.ipynb`** — Uses a Random Forest classifier and achieves **99.17%** accuracy.  
- **`xgboost_model.ipynb`** — Uses the XGBoost algorithm and achieves **99.61%** accuracy.  
- **`lightgbm_model.ipynb`** — Uses the LightGBM algorithm and achieves **99.63%** accuracy.

---

**Note:**
Since it might take several hours to generate `final.csv` in `feature_extraction.ipynb`, you can directly download it from [Google Drive](https://drive.google.com/file/d/1afvtsZXpUFNodYUS1kl6pmtIf6xvAZgy/view) and place it in `./data/final.csv`.

---

# Inference

Now you can test the model on your own images!
To do this:

1. Download or provide an image.
2. Run the following command with the correct path to your image:
   ```bash
   python3 inference.py path/to/image
   ```
