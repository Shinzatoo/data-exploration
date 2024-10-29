import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Define the dataset
data = {
    "Country": ["Afghanistan", "Albania", "Algeria", "Angola", "Argentina", "Australia", "Austria", "Bahamas", "Bangladesh", 
                "Belarus", "Belgium", "Belize", "Bolivia", "Bosnia and Herzegovina", "Brazil", "Bulgaria", "Cambodia", 
                "Canada", "Chile", "China", "Colombia", "Costa Rica", "Croatia", "Cuba", "Czech Republic", "Denmark", 
                "Dominican Republic", "Ecuador", "Egypt", "El Salvador", "Estonia", "Finland", "France", "Georgia", "Germany", 
                "Ghana", "Greece", "Greenland", "Guatemala", "Honduras", "Hungary", "Iceland", "India", "Indonesia", "Iran", 
                "Iraq", "Ireland", "Israel", "Italy", "Jamaica", "Japan", "Jordan", "Kazakhstan", "Kenya", "Kuwait", 
                "Kyrgyzstan", "Laos", "Latvia", "Lebanon", "Lithuania", "Luxembourg", "Malaysia"],
    "IQ": [81, 82, 77, 72, 87, 99, 99, 84, 75, 99, 98, 67, 79, 87, 83, 91, 97, 100, 88, 104, 83, 88, 95, 84, 96, 98, 
           82, 79, 78, 72, 100, 101, 97, 86, 100, 61, 91, 99, 55, 67, 99, 99, 77, 80, 83, 88, 95, 93, 95, 74, 106, 
           81, 89, 74, 80, 64, 83, 95, 82, 95, 100, 89],
    "GDP": [473, 3513, 3789, 2396, 8795, 42959, 42634, 28622, 936, 4661, 40525, 4209, 1912, 4118, 7586, 5702, 776, 40205, 
            10195, 4654, 5037, 7480, 11659, 5538, 15552, 53149, 4964, 4162, 2175, 2912, 13774, 42706, 37610, 2921, 
            39920, 1166, 21101, 24486, 2830, 1665, 11391, 47758, 1164, 2355, 4545, 3758, 43914, 28975, 32103, 4356, 
            40964, 3091, 6380, 939, 37126, 1445, 1157, 11243, 6428, 11331, 71380, 7665],
    "HomicideRate": [4.022, 1.654, 1.77, 4.102, 4.309, 0.833, 0.884, 31.221, 2.34, 2.326, 1.077, 27.882, 4.318, 1.082, 
                     20.606, 1.121, 1.839, 2.273, 6.744, 0.502, 25.269, 17.382, 0.769, 4.418, 0.839, 0.986, 12.37, 
                     26.993, 1.336, 7.828, 1.508, 1.245, 1.56, 2.042, 0.823, 1.813, 1.127, 5.365, 19.99, 35.072, 0.973, 
                     1.073, 2.832, 0.307, 2.421, 9.41, 0.876, 1.626, 0.545, 53.336, 0.233, 0.997, 2.634, 4.892, 
                     0.252, 4.1, 2.5, 3.62, 2.26, 2.436, 0.0, 0.724]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Correlation matrix
correlation_matrix = df[["IQ", "GDP", "HomicideRate"]].corr()

# Regression analysis
X = df[["IQ", "GDP"]]  # Independent variables
y = df["HomicideRate"]  # Dependent variable
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

# Extract regression coefficients and p-values
coefficients = model.params[1:]  # Exclude intercept
p_values = model.pvalues[1:]  # Exclude intercept

# Combine correlations, coefficients, and p-values into one DataFrame
results_matrix = pd.DataFrame({
    "Correlation with Homicide Rate": correlation_matrix["HomicideRate"].drop("HomicideRate"),
    "Regression Coefficient": coefficients,
    "P-value": p_values
})

# Display the results matrix
print("Results Matrix with Correlations, Regression Coefficients, and P-values:")
print(results_matrix)

# Plot the heatmap for the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix of IQ, GDP, and Homicide Rate")
plt.show()
