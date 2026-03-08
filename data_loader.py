"""
Data Loader Module for Heart Disease Dataset
Loads and preprocesses the UCI Heart Disease dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_heart_disease_data():
    """
    Load Heart Disease dataset from UCI ML Repository
    
    Returns:
        X_train, X_test, y_train, y_test: Split datasets
        feature_names: List of feature names
        scaler: Fitted StandardScaler object
    """
    try:
        # Try loading from ucimlrepo first
        from ucimlrepo import fetch_ucirepo
        
        # Fetch dataset
        heart_disease = fetch_ucirepo(id=45)
        
        # Extract features and targets
        X = heart_disease.data.features
        y = heart_disease.data.targets
        
        # Convert target to binary (0: no disease, 1: disease present)
        # Original dataset has 5 classes (0-4), we convert to binary
        y = (y > 0).astype(int).values.ravel()
        
        print("Dataset loaded successfully from UCI ML Repository")
        print(f"Dataset shape: {X.shape}")
        print(f"Number of features: {X.shape[1]}")
        print(f"Number of samples: {X.shape[0]}")
        
    except Exception as e:
        print(f"Could not load from ucimlrepo: {e}")
        print("Loading from fallback CSV...")
        
        # Fallback: Load from a local CSV or create sample data
        # Using Cleveland Heart Disease dataset structure
        column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                       'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 
                       'ca', 'thal', 'target']
        
        # For demo purposes, create a URL to the Cleveland dataset
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        
        try:
            X = pd.read_csv(url, names=column_names, na_values='?')
            y = X['target'].values
            X = X.drop('target', axis=1)
            
            # Handle missing values
            X = X.fillna(X.median())
            
            # Convert target to binary
            y = (y > 0).astype(int)
            
            print("Dataset loaded successfully from UCI archive")
            print(f"Dataset shape: {X.shape}")
            
        except Exception as e2:
            print(f"Could not load from URL: {e2}")
            print("Creating synthetic dataset for demonstration...")
            
            # Create synthetic data that mimics heart disease dataset structure
            np.random.seed(42)
            n_samples = 303
            
            X = pd.DataFrame({
                'age': np.random.randint(29, 78, n_samples),
                'sex': np.random.randint(0, 2, n_samples),
                'cp': np.random.randint(0, 4, n_samples),
                'trestbps': np.random.randint(94, 200, n_samples),
                'chol': np.random.randint(126, 564, n_samples),
                'fbs': np.random.randint(0, 2, n_samples),
                'restecg': np.random.randint(0, 3, n_samples),
                'thalach': np.random.randint(71, 202, n_samples),
                'exang': np.random.randint(0, 2, n_samples),
                'oldpeak': np.random.uniform(0, 6.2, n_samples),
                'slope': np.random.randint(0, 3, n_samples),
                'ca': np.random.randint(0, 4, n_samples),
                'thal': np.random.randint(0, 4, n_samples)
            })
            
            # Create target with some correlation to features
            y_prob = 0.3 + 0.01 * X['age'] - 0.05 * X['thalach'] + 0.1 * X['cp']
            y = (y_prob > y_prob.median()).astype(int).values
            
            print("Synthetic dataset created for demonstration")
            print(f"Dataset shape: {X.shape}")
    
    # Get feature names
    feature_names = X.columns.tolist()
    
    # Handle any remaining missing values
    if isinstance(X, pd.DataFrame):
        X = X.fillna(X.median())
        X = X.values
    
    # Check class distribution
    unique, counts = np.unique(y, return_counts=True)
    class_dist = dict(zip(unique, counts))
    print(f"\nClass distribution:")
    print(f"  No disease (0): {class_dist.get(0, 0)} samples ({class_dist.get(0, 0)/len(y)*100:.1f}%)")
    print(f"  Disease (1): {class_dist.get(1, 0)} samples ({class_dist.get(1, 0)/len(y)*100:.1f}%)")
    
    # Split the data (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"\nTrain set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test, feature_names, scaler


def get_feature_descriptions():
    """
    Return descriptions of features in the Heart Disease dataset
    """
    descriptions = {
        'age': 'Age in years',
        'sex': 'Sex (1 = male; 0 = female)',
        'cp': 'Chest pain type (0-3)',
        'trestbps': 'Resting blood pressure (mm Hg)',
        'chol': 'Serum cholesterol (mg/dl)',
        'fbs': 'Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)',
        'restecg': 'Resting electrocardiographic results (0-2)',
        'thalach': 'Maximum heart rate achieved',
        'exang': 'Exercise induced angina (1 = yes; 0 = no)',
        'oldpeak': 'ST depression induced by exercise',
        'slope': 'Slope of peak exercise ST segment (0-2)',
        'ca': 'Number of major vessels colored by fluoroscopy (0-3)',
        'thal': 'Thalassemia (0-3)'
    }
    return descriptions
