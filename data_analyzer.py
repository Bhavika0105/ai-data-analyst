import pandas as pd

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print("✅ Data loaded successfully!")
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def basic_analysis(df):  # <--- this function is very important
    print("\n📊 Basic Information:")
    print(df.info())
    print("\n📈 Statistical Summary:")
    print(df.describe())
    print("\n🧹 Checking for null values:")
    print(df.isnull().sum())
