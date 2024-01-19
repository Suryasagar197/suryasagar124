## The dataset is taken from Worldbank repository

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit

school_path = 'mean-years-of-schooling-long-run.csv'
GDP_path = 'gdp_per_capita.csv'

school_years = pd.read_csv(school_path, sep=';')
gdp = pd.read_csv(GDP_path)

#Reducing the GDP dataframe to the required columns and removing the data with empty year
gdp = pd.melt(frame = gdp, id_vars=['Country Name', 'Code'], var_name= 'year', value_name='gdp_per_capita')
gdp = gdp.query('year != "Unnamed: 65"')

gdp['year'] = gdp['year'].astype(int)
gdp.isnull().sum()

gdp = gdp.dropna()
gdp.shape

df = pd.merge(school_years, gdp, left_on = ['Entity', 'Code', 'Year'],\
              how='inner',\
              right_on=['Country Name', 'Code', 'year'],\
             validate = 'm:m')

df = df[['Country Name', 'year', 'gdp_per_capita', 'avg_years_of_schooling']]

fig, ax = plt.subplots(figsize=(9, 6), sharex = True)
plt.title('Correlation between GDP Per Capita and Years of Schooling')
ax2 = ax.twinx()

ax=df.groupby('year')['gdp_per_capita'].mean().plot(lw=2, ax=ax, color='red')
ax.set_ylabel('GDP per Capita', fontsize= 13, color= 'red')
ax.grid(False)


ax2= df.groupby('year')['avg_years_of_schooling'].mean().plot(lw=2, ax=ax2, color='blue')
ax2.set_ylabel('Average year of Schooling', fontsize = 13, color='blue')
ax2.grid(False)
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=0)
plt.savefig('GDP_SchoolingVsYear.png')

""" Tools to support clustering: correlation heatmap, normaliser and scale
(cluster centres) back to original scale, check for mismatching entries """


def scaler(df):
    """ Expects a dataframe and normalises all
        columnsto the 0-1 range. It also returns
        dataframes with minimum and maximum for
        transforming the cluster centres"""

    # Uses the pandas methods
    df_min = df.min()
    df_max = df.max()

    df = (df-df_min) / (df_max - df_min)

    return df, df_min, df_max


def error_prop(x, func, parameter, covar):
    """
    Calculates 1 sigma error ranges for number or array. It uses error
    propagation with variances and covariances taken from the covar matrix.
    Derivatives are calculated numerically.

    """

    # initiate sigma the same shape as parameter

    var = np.zeros_like(x)   # initialise variance vektor
    # Nested loop over all combinations of the parameters
    for i in range(len(parameter)):
        # derivative with respect to the ith parameter
        deriv1 = deriv(x, func, parameter, i)

        for j in range(len(parameter)):
            # derivative with respct to the jth parameter
            deriv2 = deriv(x, func, parameter, j)


            # multiplied with the i-jth covariance
            # variance vektor
            var = var + deriv1*deriv2*covar[i, j]

    sigma = np.sqrt(var)
    return sigma


def deriv(x, func, parameter, ip):
    """
    Calculates numerical derivatives from function
    values at parameter +/- delta.  Parameter is the vector with parameter
    values. ip is the index of the parameter to derive the derivative.

    """

    # print("in", ip, parameter[ip])
    # create vector with zeros and insert delta value for relevant parameter
    # delta is calculated as a small fraction of the parameter value
    scale = 1e-6   # scale factor to calculate the derivative
    delta = np.zeros_like(parameter, dtype=float)
    val = scale * np.abs(parameter[ip])
    delta[ip] = val  #scale * np.abs(parameter[ip])

    diff = 0.5 * (func(x, *parameter+delta) - func(x, *parameter-delta))
    dfdx = diff / val

    return dfdx


def covar_to_corr(covar):
    """ Converts the covariance matrix into a correlation matrix """

    # extract variances from the diagonal and calculate std. dev.
    sigma = np.sqrt(np.diag(covar))
    # construct matrix containing the sigma values
    matrix = np.outer(sigma, sigma)
    # and divide by it
    corr = covar/matrix

    return corr


pike = df.query('year > 1990').copy()
pike3 = pike.groupby('Country Name')['gdp_per_capita', 'avg_years_of_schooling']\
.mean()\
.sort_values('gdp_per_capita', ascending=False)\
.reset_index()\
.head(3).copy()

#Getting just the Top 10 Contries based on the slice I did before
top3 = df[df['Country Name'].isin(pike3['Country Name'])]
top3 = top3.query('year > 1990')


fig, axs = plt.subplots(3, 1, figsize=(4,8), sharex = True)
plt.suptitle('Top-3 GDP-Per Capita - After 1990', fontsize=12, weight = 'bold')
axs = axs.flatten()
i = 0

for country in top3['Country Name'].unique():

    ax2 = axs[i].twinx()

    top3[top3['Country Name'] == country].\
    plot(x='year', y='gdp_per_capita',\
    ax=axs[i],color='red')

    top3[top3['Country Name'] == country]\
    .plot(x='year', y='avg_years_of_schooling',\
    ax=ax2, color='blue')

    axs[i].set_ylabel('GDP per Capita', fontsize=9, color = 'red')
    ax2.set_ylabel('Years of Schooling', fontsize = 9, color = 'blue')
    axs[i].set_title(country)
    axs[i].grid(False)
    ax2.grid(False)
    axs[i].legend().set_visible(False)
    ax2.legend().set_visible(False)
    ax2.tick_params(axis='both', which='both', labelsize=7)

    i = i + 1
    if i==3:
      break

for ax in axs:
    ax.tick_params(axis='both', which='both', labelsize=7)


plt.tight_layout()
plt.savefig('Top3_Countries.png')

bottom = pike.groupby('Country Name')['gdp_per_capita', 'avg_years_of_schooling']\
.mean()\
.sort_values('gdp_per_capita', ascending=True)\
.reset_index()\
.head(3).copy()

bottom3 = df[df['Country Name'].isin(bottom['Country Name'])]
bottom3 = bottom3.query('year > 1990')

bottom

fig, axs = plt.subplots(3, 1, figsize=(4,8), sharex = True)
plt.suptitle('Bottom-3 GDP-Per Capita - After 1990', fontsize=12, weight = 'bold')
axs = axs.flatten()
i = 0

for country in bottom3['Country Name'].unique():
    ax2 = axs[i].twinx()

    bottom3[bottom3['Country Name'] == country]\
    .plot(x='year', y='gdp_per_capita',color='red', ax=axs[i])

    bottom3[bottom3['Country Name'] == country]\
    .plot(x='year', y='avg_years_of_schooling', color='blue',ax=ax2)

    axs[i].set_ylabel('GDP per Capita', fontsize=9, color='red')
    ax2.set_ylabel('Years of Schooling', fontsize=9, color='blue')
    axs[i].grid(False)
    ax2.grid(False)
    axs[i].legend().set_visible(False)
    ax2.legend().set_visible(False)
    ax2.tick_params(axis='both', which='both', labelsize=7)

    i = i + 1
    if i==3:
      break

for ax in axs:
    ax.tick_params(axis='both', which='both', labelsize=7)

plt.tight_layout()
plt.savefig('Bottom3_Countries.png')


data = df[df['year']==2017]
data = data[['gdp_per_capita', 'avg_years_of_schooling']]

# Perform clustering using KMeans
kmeans = KMeans(n_clusters=3)  # You can adjust the number of clusters as needed
datas, df_min, df_max = scaler(data)
data['Cluster'] = kmeans.fit_predict(datas)

# Plot the cluster plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='gdp_per_capita', y='avg_years_of_schooling', hue='Cluster', data=data, palette='viridis', s=100)
plt.title('Cluster Plot of GDP vs. Average Schooling')
plt.xlabel('GDP')
plt.ylabel('Average Schooling')
plt.savefig('Cluster_GDPVsSchooling.png')


data = df
# Function for the model (polynomial for demonstration)
def model_function(x, a, b, c):
    return a * x**2 + b * x + c

# Function to estimate confidence range using err_ranges
def err_ranges(x, func, params, covar):
    return error_prop(x, func, params, covar)

# Separate data for the UK
uk_data = data[data['Country Name'] == 'United Kingdom']

# Fit the model to the UK's data for average years of schooling
params_uk_schooling, covariance_uk_schooling = curve_fit(model_function, uk_data['year'], uk_data['avg_years_of_schooling'])

# Extend years for plotting
years_for_prediction = np.linspace(1960, 2030, 100)

# Predict values for average years of schooling in the UK until 2030
uk_predicted_schooling_extended = model_function(years_for_prediction, *params_uk_schooling)

# Estimate confidence range
confidence_range_schooling = err_ranges(years_for_prediction, model_function, params_uk_schooling, covariance_uk_schooling)

# Plot for the UK's average years of schooling with confidence range
plt.figure(figsize=(10, 6))
plt.scatter(uk_data['year'], uk_data['avg_years_of_schooling'], label='United Kingdom Data')
plt.plot(years_for_prediction, uk_predicted_schooling_extended, label='United Kingdom Prediction', color='green')

# Plot confidence range
plt.fill_between(
    years_for_prediction,
    uk_predicted_schooling_extended - confidence_range_schooling,
    uk_predicted_schooling_extended + confidence_range_schooling,
    color='green',
    alpha=0.2,
    label='Confidence Range'
)

plt.xlabel('Year')
plt.ylabel('Average Years of Schooling')
plt.title('Average Years of Schooling Prediction for the UK (Extended to 2030)')
plt.legend()
plt.savefig('Prediction_Avg_Schooling.png')



# Fit the model to the UK's data for GDP per capita
params_uk_gdp, covariance_uk_gdp = curve_fit(model_function, uk_data['year'], uk_data['gdp_per_capita'])

# Predict values for GDP per capita in the UK until 2030
uk_predicted_gdp_extended = model_function(years_for_prediction, *params_uk_gdp)

# Estimate confidence range
confidence_range_gdp = err_ranges(years_for_prediction, model_function, params_uk_gdp, covariance_uk_gdp)

# Plot for the UK's GDP per capita with confidence range
plt.figure(figsize=(10, 6))
plt.scatter(uk_data['year'], uk_data['gdp_per_capita'], label='United Kingdom Data')
plt.plot(years_for_prediction, uk_predicted_gdp_extended, label='United Kingdom Prediction', color='red')

# Plot confidence range
plt.fill_between(
    years_for_prediction,
    uk_predicted_gdp_extended - confidence_range_gdp,
    uk_predicted_gdp_extended + confidence_range_gdp,
    color='green',
    alpha=0.2,
    label='Confidence Range'
)

plt.xlabel('Year')
plt.ylabel('GDP per Capita')
plt.title('GDP per Capita Prediction for the UK (Extended to 2030)')
plt.legend()
plt.savefig('Prediction_GDP.png')

predicted_schooling_2030 = model_function(2030, *params_uk_schooling)
predicted_gdp_2030 = model_function(2030, *params_uk_gdp)

print(f"Predicted Average Years of Schooling in the UK for 2030: {predicted_schooling_2030:.2f}")
print(f"Predicted GDP per Capita in the UK for 2030: {predicted_gdp_2030:.2f}")