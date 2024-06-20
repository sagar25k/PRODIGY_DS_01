import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def clean_and_visualize_titanic_data(file_path):
    # Load the Titanic dataset
    titanic = pd.read_csv(file_path)

    # Initial data types and NaN summary
    print("\nInitial data types and NaN summary:")
    print(titanic.dtypes)
    print(titanic.isna().sum())

    # 1. Handle missing values using central tendency
    for column in titanic.columns:
        if titanic[column].isna().sum() > 0:
            if titanic[column].dtype in ['float64', 'int64']:
                median_value = titanic[column].median()
                titanic[column].fillna(median_value, inplace=True)
                print(f"Imputed NaN in column '{column}' with median value: {median_value}")
            else:
                mode_value = titanic[column].mode()[0]
                titanic[column].fillna(mode_value, inplace=True)
                print(f"Imputed NaN in column '{column}' with mode value: {mode_value}")

    # 2. Remove duplicate rows
    initial_row_count = titanic.shape[0]
    titanic.drop_duplicates(inplace=True)
    final_row_count = titanic.shape[0]
    print(f"\nRemoved {initial_row_count - final_row_count} duplicate rows.")

    # 3. Correct data types
    titanic['Age'] = pd.to_numeric(titanic['Age'], errors='coerce')
    titanic['Fare'] = pd.to_numeric(titanic['Fare'], errors='coerce')
    titanic['Pclass'] = titanic['Pclass'].astype(int)
    titanic['Sex'] = titanic['Sex'].astype('category')
    titanic['Embarked'] = titanic['Embarked'].astype('category')
    titanic['Name'] = titanic['Name'].astype(str)
    titanic['Ticket'] = titanic['Ticket'].astype(str)
    titanic['Cabin'] = titanic['Cabin'].astype(str)

    # Correct 'Age' column infinite values
    titanic['Age'] = titanic['Age'].replace([float('inf'), float('-inf')], pd.NA)

    print("\nCorrected data types:")
    print(titanic.dtypes)

    # 4. Visualize distribution of 'Sex'
    plt.figure(figsize=(10, 6))
    sns.countplot(data=titanic, x='Sex', palette='pastel')
    plt.title('Distribution of Passengers by Sex')
    plt.xlabel('Sex')
    plt.ylabel('Count')
    plt.show()

    # 5. Visualize distribution of 'Age'
    plt.figure(figsize=(10, 6))
    sns.histplot(data=titanic, x='Age', kde=True, bins=30, color='skyblue')
    plt.title('Distribution of Passenger Ages')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.show()

    return titanic


# Example usage:
cleaned_titanic = clean_and_visualize_titanic_data('D:/titanic.csv')

# Print final NaN count and sample data
print("\nFinal NaN count per column:")
print(cleaned_titanic.isna().sum())

print("\nSample data from cleaned DataFrame:")
print(cleaned_titanic.head())
