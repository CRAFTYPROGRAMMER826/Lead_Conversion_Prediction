# Lead Conversion Prediction using XGBoost 🚀

Welcome to the Lead Conversion Prediction repository! 

This project tackles a classic business problem: predicting whether a potential lead will convert into a successful conversion (1) or not (0). It serves as a practical implementation of data generation, feature engineering, and handling imbalanced datasets using ensemble methods like Random Forest and XGBoost.

## 🧠 What's Actually Happening in the Code?

If you are exploring the Jupyter Notebook (`LEADS_XGboost.ipynb`), here is a simple breakdown of the logic and flow:

### 1. Generating a Synthetic Dataset
Since real-world CRM data is often private, the first step is building our own playground. 
* Using the `Faker` library and `numpy`, the code simulates 500 random leads. 
* It assigns realistic features to each lead: `name`, `budget`, `interest_level` (1-10), `lead_source` (Website, Referral, etc.), and `follow_up_count`.
* We deliberately generate the target variable (`converted`) with a realistic 70/30 split—meaning 70% of leads don't convert, and 30% do. 

### 2. Data Preprocessing (Making it Machine-Readable)
Machine learning models speak numbers, not text. 
* The `lead_source` column contains categorical data (e.g., 'Email', 'Website. 
* We use One-Hot Encoding (`pd.get_dummies`) to split these sources into their own independent binary columns (True/False or 1/0) so the model can correctly weigh the success rate of each specific source.

### 3. The Baseline Model (Random Forest)
* The data is split into an 80/20 train-test ratio. 
* A `RandomForestClassifier` is trained first as a baseline. 
* **The Catch:** When we test it, the overall accuracy might look okay (around 65%), but looking closely at the `classification_report`, the model is heavily biased[cite: 1]. Because there are way more `0`s (no conversion) than `1`s (conversion), the model takes the easy way out and mostly predicts `0`. 

### 4. The Solution: XGBoost & Handling Class Imbalance
To fix the bias, we bring in `XGBClassifier`. 
* Instead of letting the model ignore the minority class (the successful conversions), we calculate the exact ratio of failures to successes.
* We pass this ratio into XGBoost using the `scale_pos_weight` parameter. 
* **The Logic:** This acts as a penalty multiplier. We are essentially telling the decision trees: *"Pay X times more attention to the minority class so you don't just guess 0 every time!"*.

## 🛠️ Tech Stack
* **Python** (Core logic)
* **Pandas & NumPy** (Data manipulation and mathematical operations)
* **Faker** (Generating realistic synthetic data)
* **Scikit-Learn** (Preprocessing, Random Forest, and Evaluation Metrics)
* **XGBoost** (Advanced gradient boosting for the final model)

## 🚀 How to Run the Notebook
1. Clone the repository to your local machine.
2. Ensure you have the required libraries installed:
   ```bash
      pip install pandas numpy scikit-learn xgboost faker

Open LEADS_XGboost.ipynb in Jupyter Notebook or Google Colab and run the cells sequentially to generate the data and train the models.

Feel free to explore the code, fork the repo, or reach out if you have any suggestions!
