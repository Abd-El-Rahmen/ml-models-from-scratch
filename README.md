# Machine Learning Models from Scratch

This repository provides Python implementations of several fundamental machine learning algorithms built from scratch. The goal is to demonstrate the core principles behind these algorithms and help users understand how they work under the hood, without relying on high-level libraries like scikit-learn, TensorFlow, or PyTorch.

The project includes implementations of the following models:
- Decision Tree
- Neural Network
- Random Forest

Each model is encapsulated in its own directory with relevant code, utilities, and test cases. You can find sample datasets and Jupyter notebooks for demonstrations and hands-on learning in the `data/` and `notebooks/` directories, respectively.


## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/ml-models-from-scratch.git
    cd ml-models-from-scratch
    ```

2. Create and activate a virtual environment (optional but recommended):

    ```bash
    python -m venv env
    source env/bin/activate   # For Windows use: env\Scripts\activate
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Done! You can now start exploring the models.


## Usage

### Example: Decision Tree

```python
from decision_tree import DecisionTree

# Load your dataset
X_train, y_train = load_data()

# Create and train the model
model = DecisionTree()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)