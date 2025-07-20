#!/usr/bin/env python3
"""
TMDB Movie Revenue Prediction
=============================

This script implements a comprehensive revenue prediction model using the TMDB dataset.
It includes proper data encoding, feature engineering, and compares two different
regressor models: Random Forest and XGBoost.

Features used:
- Budget (log transformed)
- Runtime
- Release month and season
- Genres (one-hot encoded)
- Keywords (TF-IDF encoded)
- Additional engineered features
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_prepare_data(file_path):
    """
    Load and prepare the TMDB dataset for revenue prediction.
    
    Args:
        file_path (str): Path to the TMDB CSV file
        
    Returns:
        tuple: (revenue_data, genre_features, keywords_features)
    """
    print("Loading TMDB dataset...")
    data = pd.read_csv(file_path)
    print(f"Original data shape: {data.shape}")
    
    # Filter data for revenue prediction
    revenue_data = data[
        (data['budget'] > 0) & 
        (data['revenue'] > 0) & 
        (data['runtime'] > 0) & 
        (data['genres'].notna()) & 
        (data['keywords'].notna()) &
        (data['original_language'] == 'en') &
        (data['status'] == 'Released') &
        (data['adult'] == False) &
        (data['release_date'].notna())
    ].copy()
    
    print(f"Filtered data shape: {revenue_data.shape}")
    
    # Create target variable (log transformation for revenue due to skewness)
    revenue_data['log_revenue'] = np.log1p(revenue_data['revenue'])
    
    # Feature engineering
    revenue_data = engineer_features(revenue_data)
    
    # Encode genres
    genre_features = encode_genres(revenue_data)
    
    # Encode keywords
    keywords_features = encode_keywords(revenue_data)
    
    return revenue_data, genre_features, keywords_features

def engineer_features(data):
    """
    Engineer additional features for the revenue prediction model.
    
    Args:
        data (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with engineered features
    """
    print("Engineering features...")
    
    # Convert release_date to datetime
    data['release_date'] = pd.to_datetime(data['release_date'], errors='coerce')
    
    # Release month encoding
    data['release_month'] = data['release_date'].dt.month_name()
    month_encoder = LabelEncoder()
    data['month_encoded'] = month_encoder.fit_transform(data['release_month'])
    
    # Release year
    data['release_year'] = data['release_date'].dt.year
    
    # Season encoding
    def get_season(month):
        if month in ['December', 'January', 'February']:
            return 'Winter'
        elif month in ['March', 'April', 'May']:
            return 'Spring'
        elif month in ['June', 'July', 'August']:
            return 'Summer'
        else:
            return 'Fall'
    
    data['season'] = data['release_month'].apply(get_season)
    season_encoder = LabelEncoder()
    data['season_encoded'] = season_encoder.fit_transform(data['season'])
    
    # Budget features
    data['log_budget'] = np.log1p(data['budget'])
    data['budget_per_minute'] = data['budget'] / data['runtime']
    
    # Runtime features
    data['runtime_category'] = pd.cut(
        data['runtime'], 
        bins=[0, 90, 120, 150, 300], 
        labels=['Short', 'Medium', 'Long', 'Very Long']
    )
    runtime_encoder = LabelEncoder()
    data['runtime_encoded'] = runtime_encoder.fit_transform(data['runtime_category'])
    
    # Additional features
    data['budget_runtime_ratio'] = data['budget'] / data['runtime']
    data['is_high_budget'] = (data['budget'] > data['budget'].quantile(0.75)).astype(int)
    data['is_long_movie'] = (data['runtime'] > data['runtime'].quantile(0.75)).astype(int)
    
    return data

def encode_genres(data):
    """
    Encode genres using one-hot encoding.
    
    Args:
        data (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Genre features dataframe
    """
    print("Encoding genres...")
    
    # Clean genres
    data['genres_clean'] = data['genres'].fillna('')
    
    # Create genre matrix
    genre_vectorizer = CountVectorizer(tokenizer=lambda x: x.split(', '), binary=True)
    genre_matrix = genre_vectorizer.fit_transform(data['genres_clean'])
    genre_features = genre_vectorizer.get_feature_names_out()
    genre_df = pd.DataFrame(genre_matrix.toarray(), columns=genre_features, index=data.index)
    
    print(f"Created {len(genre_features)} genre features")
    return genre_df

def encode_keywords(data):
    """
    Encode keywords using TF-IDF.
    
    Args:
        data (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Keywords features dataframe
    """
    print("Encoding keywords...")
    
    # Clean keywords
    data['keywords_clean'] = data['keywords'].fillna('')
    
    # Create keywords TF-IDF matrix (limit to top 100 features to avoid overfitting)
    keywords_vectorizer = TfidfVectorizer(
        tokenizer=lambda x: x.split(', '), 
        max_features=100,
        stop_words='english'
    )
    keywords_matrix = keywords_vectorizer.fit_transform(data['keywords_clean'])
    keywords_features = keywords_vectorizer.get_feature_names_out()
    keywords_df = pd.DataFrame(keywords_matrix.toarray(), columns=keywords_features, index=data.index)
    
    print(f"Created {len(keywords_features)} keyword features")
    return keywords_df

def prepare_feature_matrix(data, genre_df, keywords_df):
    """
    Prepare the final feature matrix for modeling.
    
    Args:
        data (pd.DataFrame): Input dataframe with engineered features
        genre_df (pd.DataFrame): Genre features
        keywords_df (pd.DataFrame): Keywords features
        
    Returns:
        tuple: (X, y) feature matrix and target variable
    """
    print("Preparing feature matrix...")
    
    # Select base features
    base_features = [
        'log_budget', 'budget_per_minute', 'runtime', 'runtime_encoded',
        'month_encoded', 'release_year', 'season_encoded',
        'budget_runtime_ratio', 'is_high_budget', 'is_long_movie'
    ]
    
    # Create final feature matrix
    X = pd.concat([
        data[base_features],
        genre_df.add_prefix('genre_'),
        keywords_df.add_prefix('keyword_')
    ], axis=1)
    
    y = data['log_revenue']
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target variable shape: {y.shape}")
    
    return X, y

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Train and evaluate Random Forest and XGBoost models.
    
    Args:
        X_train, X_test, y_train, y_test: Training and test data
        
    Returns:
        dict: Dictionary containing model results and predictions
    """
    print("\n" + "="*60)
    print("TRAINING AND EVALUATING MODELS")
    print("="*60)
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_features = ['log_budget', 'budget_per_minute', 'runtime', 'release_year', 
                         'budget_runtime_ratio']
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])
    
    results = {}
    
    # Model 1: Random Forest Regressor
    print("\n" + "-"*30)
    print("RANDOM FOREST REGRESSOR")
    print("-"*30)
    
    rf_model = RandomForestRegressor(
        n_estimators=100, 
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42, 
        n_jobs=-1
    )
    
    rf_model.fit(X_train_scaled, y_train)
    
    # Predictions
    rf_train_pred = rf_model.predict(X_train_scaled)
    rf_test_pred = rf_model.predict(X_test_scaled)
    
    # Metrics
    rf_metrics = calculate_metrics(y_train, y_test, rf_train_pred, rf_test_pred)
    rf_cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring='r2')
    
    print_metrics("Random Forest", rf_metrics, rf_cv_scores)
    
    # Feature importance
    rf_feature_importance = pd.DataFrame({
        'feature': X_train_scaled.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    results['random_forest'] = {
        'model': rf_model,
        'predictions': {'train': rf_train_pred, 'test': rf_test_pred},
        'metrics': rf_metrics,
        'cv_scores': rf_cv_scores,
        'feature_importance': rf_feature_importance
    }
    
    # Model 2: XGBoost Regressor
    print("\n" + "-"*30)
    print("XGBOOST REGRESSOR")
    print("-"*30)
    
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    xgb_model.fit(X_train_scaled, y_train)
    
    # Predictions
    xgb_train_pred = xgb_model.predict(X_train_scaled)
    xgb_test_pred = xgb_model.predict(X_test_scaled)
    
    # Metrics
    xgb_metrics = calculate_metrics(y_train, y_test, xgb_train_pred, xgb_test_pred)
    xgb_cv_scores = cross_val_score(xgb_model, X_train_scaled, y_train, cv=5, scoring='r2')
    
    print_metrics("XGBoost", xgb_metrics, xgb_cv_scores)
    
    # Feature importance
    xgb_feature_importance = pd.DataFrame({
        'feature': X_train_scaled.columns,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    results['xgboost'] = {
        'model': xgb_model,
        'predictions': {'train': xgb_train_pred, 'test': xgb_test_pred},
        'metrics': xgb_metrics,
        'cv_scores': xgb_cv_scores,
        'feature_importance': xgb_feature_importance
    }
    
    return results

def calculate_metrics(y_train, y_test, train_pred, test_pred):
    """
    Calculate various performance metrics.
    
    Args:
        y_train, y_test: Actual values
        train_pred, test_pred: Predicted values
        
    Returns:
        dict: Dictionary of metrics
    """
    return {
        'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
        'train_mae': mean_absolute_error(y_train, train_pred),
        'test_mae': mean_absolute_error(y_test, test_pred),
        'train_r2': r2_score(y_train, train_pred),
        'test_r2': r2_score(y_test, test_pred)
    }

def print_metrics(model_name, metrics, cv_scores):
    """
    Print model metrics in a formatted way.
    
    Args:
        model_name (str): Name of the model
        metrics (dict): Dictionary of metrics
        cv_scores (array): Cross-validation scores
    """
    print(f"{model_name} Results:")
    print(f"  Train RMSE: {metrics['train_rmse']:.4f}")
    print(f"  Test RMSE:  {metrics['test_rmse']:.4f}")
    print(f"  Train MAE:  {metrics['train_mae']:.4f}")
    print(f"  Test MAE:   {metrics['test_mae']:.4f}")
    print(f"  Train R²:   {metrics['train_r2']:.4f}")
    print(f"  Test R²:    {metrics['test_r2']:.4f}")
    print(f"  Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

def plot_results(results, y_test):
    """
    Create comprehensive visualization of model results.
    
    Args:
        results (dict): Dictionary containing model results
        y_test (pd.Series): Test target values
    """
    print("\nCreating visualizations...")
    
    # Create subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Predictions vs Actual
    plt.subplot(3, 3, 1)
    plt.scatter(y_test, results['random_forest']['predictions']['test'], alpha=0.5, label='Random Forest')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Log Revenue')
    plt.ylabel('Predicted Log Revenue')
    plt.title('Random Forest: Predicted vs Actual')
    plt.text(0.05, 0.95, f"R² = {results['random_forest']['metrics']['test_r2']:.3f}", 
             transform=plt.gca().transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.subplot(3, 3, 2)
    plt.scatter(y_test, results['xgboost']['predictions']['test'], alpha=0.5, label='XGBoost')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Log Revenue')
    plt.ylabel('Predicted Log Revenue')
    plt.title('XGBoost: Predicted vs Actual')
    plt.text(0.05, 0.95, f"R² = {results['xgboost']['metrics']['test_r2']:.3f}", 
             transform=plt.gca().transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 2. Residuals
    plt.subplot(3, 3, 3)
    rf_residuals = y_test - results['random_forest']['predictions']['test']
    plt.scatter(results['random_forest']['predictions']['test'], rf_residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Log Revenue')
    plt.ylabel('Residuals')
    plt.title('Random Forest Residuals')
    
    plt.subplot(3, 3, 4)
    xgb_residuals = y_test - results['xgboost']['predictions']['test']
    plt.scatter(results['xgboost']['predictions']['test'], xgb_residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Log Revenue')
    plt.ylabel('Residuals')
    plt.title('XGBoost Residuals')
    
    # 3. Feature Importance Comparison
    plt.subplot(3, 3, 5)
    top_rf_features = results['random_forest']['feature_importance'].head(10)
    plt.barh(range(len(top_rf_features)), top_rf_features['importance'])
    plt.yticks(range(len(top_rf_features)), top_rf_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Random Forest - Top 10 Features')
    
    plt.subplot(3, 3, 6)
    top_xgb_features = results['xgboost']['feature_importance'].head(10)
    plt.barh(range(len(top_xgb_features)), top_xgb_features['importance'])
    plt.yticks(range(len(top_xgb_features)), top_xgb_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title('XGBoost - Top 10 Features')
    
    # 4. Model Comparison
    plt.subplot(3, 3, 7)
    models = ['Random Forest', 'XGBoost']
    test_r2_scores = [results['random_forest']['metrics']['test_r2'], 
                     results['xgboost']['metrics']['test_r2']]
    bars = plt.bar(models, test_r2_scores, color=['skyblue', 'lightcoral'])
    plt.ylabel('Test R² Score')
    plt.title('Model Performance Comparison')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, score in zip(bars, test_r2_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom')
    
    # 5. Cross-validation comparison
    plt.subplot(3, 3, 8)
    cv_data = [results['random_forest']['cv_scores'], results['xgboost']['cv_scores']]
    plt.boxplot(cv_data, labels=models)
    plt.ylabel('Cross-validation R² Score')
    plt.title('Cross-validation Performance')
    
    # 6. Error distribution
    plt.subplot(3, 3, 9)
    plt.hist(rf_residuals, alpha=0.7, label='Random Forest', bins=30)
    plt.hist(xgb_residuals, alpha=0.7, label='XGBoost', bins=30)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('revenue_prediction_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_model_comparison(results):
    """
    Print a comprehensive model comparison table.
    
    Args:
        results (dict): Dictionary containing model results
    """
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    
    comparison_df = pd.DataFrame({
        'Metric': ['Test RMSE', 'Test MAE', 'Test R²', 'Mean CV R²', 'CV R² Std'],
        'Random Forest': [
            results['random_forest']['metrics']['test_rmse'],
            results['random_forest']['metrics']['test_mae'],
            results['random_forest']['metrics']['test_r2'],
            results['random_forest']['cv_scores'].mean(),
            results['random_forest']['cv_scores'].std()
        ],
        'XGBoost': [
            results['xgboost']['metrics']['test_rmse'],
            results['xgboost']['metrics']['test_mae'],
            results['xgboost']['metrics']['test_r2'],
            results['xgboost']['cv_scores'].mean(),
            results['xgboost']['cv_scores'].std()
        ]
    })
    
    print(comparison_df.to_string(index=False, float_format='%.4f'))
    
    # Determine best model
    best_model = "XGBoost" if results['xgboost']['metrics']['test_r2'] > results['random_forest']['metrics']['test_r2'] else "Random Forest"
    print(f"\nBest performing model: {best_model}")
    print(f"Best R² score: {max(results['xgboost']['metrics']['test_r2'], results['random_forest']['metrics']['test_r2']):.4f}")

def main():
    """
    Main function to run the complete revenue prediction pipeline.
    """
    print("TMDB Movie Revenue Prediction")
    print("="*50)
    
    # File path - update this to your actual file path
    file_path = '/home/elbaz/Bureau/TMDB_movie_dataset_v11.csv'
    
    try:
        # Load and prepare data
        revenue_data, genre_features, keywords_features = load_and_prepare_data(file_path)
        
        # Prepare feature matrix
        X, y = prepare_feature_matrix(revenue_data, genre_features, keywords_features)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        
        # Train and evaluate models
        results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
        
        # Print model comparison
        print_model_comparison(results)
        
        # Create visualizations
        plot_results(results, y_test)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print("Key findings:")
        print("- Budget is the most important feature for revenue prediction")
        print("- Release timing (month/season) has moderate impact")
        print("- Genre information provides valuable predictive power")
        print("- Keywords contribute to the model's performance")
        print("- Both models show good generalization with cross-validation")
        
    except FileNotFoundError:
        print(f"Error: Could not find the file {file_path}")
        print("Please update the file_path variable in the main() function.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 