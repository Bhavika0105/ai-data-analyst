def suggest_model(df):
    print("\n🧠 Analyzing data to suggest a model...")

    if df.isnull().sum().sum() > 0:
        return "Preprocessing needed before modeling (missing values found)"
    
    if df.shape[0] > 1000:
        return "Suggested Model: Random Forest Classifier (good for large datasets)"
    
    if 'target' in df.columns:
        if df['target'].nunique() == 2:
            return "Suggested Model: Logistic Regression (Binary Classification)"
        else:
            return "Suggested Model: Decision Tree Classifier (Multi-Class Classification)"
    
    return "Suggested Model: KMeans Clustering (Unsupervised Learning)"
