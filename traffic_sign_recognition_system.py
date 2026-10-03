"""
Traffic Road Sign Recognition and Classification System
Recognizes and classifies traffic signs using computer vision and deep learning
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

def generate_traffic_sign_features(n_samples=2000, random_state=42):
    """Generate synthetic traffic sign features"""
    np.random.seed(random_state)
    
    sign_types = ['Warning', 'Regulatory', 'Mandatory', 'Informational']
    data = []
    
    for i in range(n_samples):
        sign_type = np.random.choice(sign_types)
        
        # Generate features based on sign type
        if sign_type == 'Warning':
            # Yellow triangular signs
            red_intensity = np.random.uniform(0.7, 1.0)
            green_intensity = np.random.uniform(0.6, 0.9)
            blue_intensity = np.random.uniform(0.0, 0.3)
            shape_complexity = np.random.uniform(0.4, 0.7)
            edge_density = np.random.uniform(0.5, 0.8)
            
        elif sign_type == 'Regulatory':
            # Red and white circular signs
            red_intensity = np.random.uniform(0.8, 1.0)
            green_intensity = np.random.uniform(0.0, 0.3)
            blue_intensity = np.random.uniform(0.0, 0.3)
            shape_complexity = np.random.uniform(0.2, 0.5)
            edge_density = np.random.uniform(0.6, 0.9)
            
        elif sign_type == 'Mandatory':
            # Blue circular signs
            red_intensity = np.random.uniform(0.0, 0.3)
            green_intensity = np.random.uniform(0.2, 0.5)
            blue_intensity = np.random.uniform(0.7, 1.0)
            shape_complexity = np.random.uniform(0.3, 0.6)
            edge_density = np.random.uniform(0.5, 0.8)
            
        else:  # Informational
            # Green and white signs
            red_intensity = np.random.uniform(0.0, 0.3)
            green_intensity = np.random.uniform(0.7, 1.0)
            blue_intensity = np.random.uniform(0.2, 0.5)
            shape_complexity = np.random.uniform(0.2, 0.5)
            edge_density = np.random.uniform(0.4, 0.7)
        
        # Additional features
        aspect_ratio = np.random.uniform(0.5, 2.0)
        contour_count = np.random.randint(3, 15)
        texture_variance = np.random.uniform(0.1, 0.9)
        brightness = (red_intensity + green_intensity + blue_intensity) / 3
        saturation = max(red_intensity, green_intensity, blue_intensity) - min(red_intensity, green_intensity, blue_intensity)
        
        data.append({
            'Red_Intensity': red_intensity,
            'Green_Intensity': green_intensity,
            'Blue_Intensity': blue_intensity,
            'Shape_Complexity': shape_complexity,
            'Edge_Density': edge_density,
            'Aspect_Ratio': aspect_ratio,
            'Contour_Count': contour_count,
            'Texture_Variance': texture_variance,
            'Brightness': brightness,
            'Saturation': saturation,
            'Sign_Type': sign_type
        })
    
    return pd.DataFrame(data)

def prepare_features(df):
    """Prepare features for modeling"""
    df_copy = df.copy()
    
    # Encode sign types
    sign_type_mapping = {'Warning': 0, 'Regulatory': 1, 'Mandatory': 2, 'Informational': 3}
    df_copy['Sign_Type_Encoded'] = df_copy['Sign_Type'].map(sign_type_mapping)
    
    feature_cols = ['Red_Intensity', 'Green_Intensity', 'Blue_Intensity', 'Shape_Complexity',
                   'Edge_Density', 'Aspect_Ratio', 'Contour_Count', 'Texture_Variance',
                   'Brightness', 'Saturation']
    
    X = df_copy[feature_cols]
    y = df_copy['Sign_Type_Encoded']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
    
    return X_scaled, y, scaler

def train_models(X_train, X_test, y_train, y_test):
    """Train multiple classification models"""
    from sklearn.model_selection import train_test_split
    
    models = {}
    results = []
    predictions = {}
    
    # Logistic Regression
    print("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_proba_lr = lr.predict_proba(X_test).max(axis=1)
    
    models['Logistic Regression'] = lr
    predictions['Logistic Regression'] = y_pred_lr
    results.append({
        'Model': 'Logistic Regression',
        'Accuracy': accuracy_score(y_test, y_pred_lr),
        'Precision': precision_score(y_test, y_pred_lr, average='weighted', zero_division=0),
        'Recall': recall_score(y_test, y_pred_lr, average='weighted', zero_division=0),
        'F1_Score': f1_score(y_test, y_pred_lr, average='weighted', zero_division=0)
    })
    
    # Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test).max(axis=1)
    
    models['Random Forest'] = rf
    predictions['Random Forest'] = y_pred_rf
    results.append({
        'Model': 'Random Forest',
        'Accuracy': accuracy_score(y_test, y_pred_rf),
        'Precision': precision_score(y_test, y_pred_rf, average='weighted', zero_division=0),
        'Recall': recall_score(y_test, y_pred_rf, average='weighted', zero_division=0),
        'F1_Score': f1_score(y_test, y_pred_rf, average='weighted', zero_division=0)
    })
    
    # SVM
    print("Training SVM...")
    svm = SVC(kernel='rbf', random_state=42, probability=True)
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    y_proba_svm = svm.predict_proba(X_test).max(axis=1)
    
    models['SVM'] = svm
    predictions['SVM'] = y_pred_svm
    results.append({
        'Model': 'SVM',
        'Accuracy': accuracy_score(y_test, y_pred_svm),
        'Precision': precision_score(y_test, y_pred_svm, average='weighted', zero_division=0),
        'Recall': recall_score(y_test, y_pred_svm, average='weighted', zero_division=0),
        'F1_Score': f1_score(y_test, y_pred_svm, average='weighted', zero_division=0)
    })
    
    return models, pd.DataFrame(results), predictions, y_test

def plot_sign_distribution(df):
    """Plot traffic sign distribution"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Sign type distribution
    sign_counts = df['Sign_Type'].value_counts()
    colors_sign = ['#FFD700', '#FF0000', '#0000FF', '#00AA00']
    axes[0, 0].bar(sign_counts.index, sign_counts.values, color=colors_sign, alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[0, 0].set_ylabel('Count', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Traffic Sign Type Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Color intensity distribution by sign type
    for idx, sign_type in enumerate(df['Sign_Type'].unique()):
        sign_data = df[df['Sign_Type'] == sign_type]
        axes[0, 1].scatter(sign_data['Red_Intensity'], sign_data['Green_Intensity'], 
                          label=sign_type, alpha=0.6, s=50, edgecolor='black', linewidth=0.5)
    axes[0, 1].set_xlabel('Red Intensity', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Green Intensity', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Color Intensity Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Shape complexity by sign type
    df.boxplot(column='Shape_Complexity', by='Sign_Type', ax=axes[1, 0])
    axes[1, 0].set_xlabel('Sign Type', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Shape Complexity', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Shape Complexity by Sign Type', fontsize=12, fontweight='bold')
    plt.sca(axes[1, 0])
    plt.xticks(rotation=45)
    
    # Edge density by sign type
    df.boxplot(column='Edge_Density', by='Sign_Type', ax=axes[1, 1])
    axes[1, 1].set_xlabel('Sign Type', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Edge Density', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Edge Density by Sign Type', fontsize=12, fontweight='bold')
    plt.sca(axes[1, 1])
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/sign_distribution.png', dpi=300, bbox_inches='tight')
    print("Saved: sign_distribution.png")
    plt.close()

def plot_model_comparison(results_df):
    """Plot model performance comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score']
    
    for idx, metric in enumerate(metrics):
        row = idx // 2
        col = idx % 2
        
        axes[row, col].bar(results_df['Model'], results_df[metric], color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        axes[row, col].set_ylabel(metric, fontsize=11, fontweight='bold')
        axes[row, col].set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
        axes[row, col].set_ylim([0, 1.1])
        axes[row, col].grid(axis='y', alpha=0.3)
        axes[row, col].tick_params(axis='x', rotation=45)
        
        for i, v in enumerate(results_df[metric]):
            axes[row, col].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/sign_model_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: sign_model_comparison.png")
    plt.close()

def plot_confusion_matrices(models, X_test, y_test):
    """Plot confusion matrices"""
    from sklearn.metrics import confusion_matrix
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    sign_types = ['Warning', 'Regulatory', 'Mandatory', 'Informational']
    
    for idx, (model_name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], 
                   xticklabels=sign_types, yticklabels=sign_types, cbar=False)
        axes[idx].set_title(f'{model_name} Confusion Matrix', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('True Label', fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/sign_confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("Saved: sign_confusion_matrices.png")
    plt.close()

def plot_feature_importance(models):
    """Plot feature importance"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Random Forest
    rf_model = models['Random Forest']
    feature_names = ['Red_Intensity', 'Green_Intensity', 'Blue_Intensity', 'Shape_Complexity',
                    'Edge_Density', 'Aspect_Ratio', 'Contour_Count', 'Texture_Variance',
                    'Brightness', 'Saturation']
    importances = rf_model.feature_importances_
    
    sorted_idx = np.argsort(importances)[::-1][:8]
    axes[0].barh(np.array(feature_names)[sorted_idx], importances[sorted_idx], color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[0].set_xlabel('Importance', fontsize=11, fontweight='bold')
    axes[0].set_title('Random Forest Feature Importance', fontsize=12, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    
    # Logistic Regression coefficients
    lr_model = models['Logistic Regression']
    coef_mean = np.abs(lr_model.coef_).mean(axis=0)
    sorted_idx = np.argsort(coef_mean)[::-1][:8]
    
    axes[1].barh(np.array(feature_names)[sorted_idx], coef_mean[sorted_idx], color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[1].set_xlabel('Coefficient Magnitude', fontsize=11, fontweight='bold')
    axes[1].set_title('Logistic Regression Feature Importance', fontsize=12, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/sign_feature_importance.png', dpi=300, bbox_inches='tight')
    print("Saved: sign_feature_importance.png")
    plt.close()

def main():
    print("="*80)
    print("TRAFFIC ROAD SIGN RECOGNITION AND CLASSIFICATION SYSTEM")
    print("="*80)
    
    # Generate dataset
    print("\n[1] Generating traffic sign features...")
    df = generate_traffic_sign_features(n_samples=2000)
    df.to_csv('/home/ubuntu/traffic_sign_data.csv', index=False)
    print(f"Dataset generated: {len(df)} samples")
    print(f"Sign types: {df['Sign_Type'].unique()}")
    
    # Prepare features
    print("\n[2] Preparing features...")
    X, y, scaler = prepare_features(df)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
    
    # Train models
    print("\n[3] Training classification models...")
    models, results_df, predictions, y_test = train_models(X_train, X_test, y_train, y_test)
    results_df.to_csv('/home/ubuntu/sign_model_results.csv', index=False)
    print("\nModel Results:")
    print(results_df.to_string(index=False))
    
    # Generate visualizations
    print("\n[4] Generating visualizations...")
    plot_sign_distribution(df)
    plot_model_comparison(results_df)
    plot_confusion_matrices(models, X_test, y_test)
    plot_feature_importance(models)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE - All visualizations saved")
    print("="*80)

if __name__ == "__main__":
    main()
