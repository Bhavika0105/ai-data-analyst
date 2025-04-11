from data_analyzer import load_data, basic_analysis
from model_suggester import suggest_model
from data_cleaner import clean_data
from model_trainer import train_model  # 👈 new import

def main():
    csv_path = input("Enter the path to your CSV file: ")
    df = load_data(csv_path)
    if df is not None:
        basic_analysis(df)
        suggest_model(df)
        
        # Clean data
        df = clean_data(df)
        print("\n✅ Data cleaned successfully! No missing values now.\n")

        # Train model
        target = input("Enter the name of the Target Column (the column you want to predict): ")
        train_model(df, target)

if __name__ == "__main__":
    main()
