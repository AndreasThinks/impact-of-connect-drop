import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.seasonal import STL
import glob
from tqdm import tqdm

# Data Loading and Preprocessing
def product_monthly_crime_count(monthly_df):
    """
    Process monthly crime data to aggregate crime counts and detection rates.

    Args:
        monthly_df (DataFrame): Monthly crime data.

    Returns:
        DataFrame: Aggregated crime counts and detection rates.
    """
    positive_outcomes = [
        'Local resolution', 'Offender given a caution',
        'Awaiting court outcome', 'Offender given penalty notice',
        'Offender given a drugs possession warning'
    ]

    monthly_df['is_detected'] = monthly_df['Last outcome category'].isin(positive_outcomes)
    count_of_crime = monthly_df.groupby(['Falls within', 'Crime type']).agg({
        'Crime ID': 'count', 
        'is_detected': 'sum'
    }).reset_index()

    count_of_crime['month'] = monthly_df['Month'].unique()[0]
    count_of_crime['monthly_total_crime'] = count_of_crime['Crime ID'].sum()
    count_of_crime['detection_rate'] = count_of_crime['is_detected'] / count_of_crime['Crime ID']

    return count_of_crime

# Load and process all crime data filessynthetic_control/data/8f5296bfa6b53f4ff67f1662191313c722ef0a9c/2022-05
all_files = glob.glob('data/*/*/*-street.csv')
crime_data_list = [product_monthly_crime_count(pd.read_csv(filename)) for filename in all_files]

# Concatenate all dataframes and rename columns
all_crime_df = pd.concat(crime_data_list, axis=0, ignore_index=True).rename(columns={
    'Crime ID': 'crime_count', 
    'is_detected': 'detection_count'
})

# Filter relevant police forces
relevant_forces = [
    'West Midlands Police', 'Metropolitan Police Service', 'Merseyside Police', 
    'Thames Valley Police', 'West Mercia Police', 'West Yorkshire Police', 
    'Avon and Somerset Constabulary'
]

synthetic_control_df = all_crime_df[
    ~all_crime_df['Crime type'].str.contains('Anti-social behaviour') &
    all_crime_df['Falls within'].isin(relevant_forces)
]

# Save preprocessed data
synthetic_control_df.to_csv('synthetic_control.csv', index=False)

# Data Preparation
df = pd.read_csv('synthetic_control.csv')
df['month'] = pd.to_datetime(df['month'])
crime_types = df['Crime type'].unique()
police_forces = df['Falls within'].unique()
date_range = pd.date_range(start=df['month'].min(), end=df['month'].max(), freq='MS')

prepared_data = pd.DataFrame()

# Reshape data to have columns for each crime type and detection rate per police force
for force in police_forces:
    for crime in crime_types:
        force_crime_data = df[(df['Falls within'] == force) & (df['Crime type'] == crime)]
        crime_count = force_crime_data.set_index('month')['crime_count'].reindex(date_range).fillna(0)
        detection_rate = force_crime_data.set_index('month')['detection_rate'].reindex(date_range).fillna(0)
        prepared_data[f'{force}_{crime}_count'] = crime_count
        prepared_data[f'{force}_{crime}_detection_rate'] = detection_rate

prepared_data = prepared_data.reset_index().rename(columns={'index': 'date'})
prepared_data['post_implementation'] = (prepared_data['date'] > '2022-11-29').astype(int)
prepared_data.to_csv('prepared_crime_data.csv', index=False)

# Load prepared data
crime_data = pd.read_csv('prepared_crime_data.csv')
crime_data['date'] = pd.to_datetime(crime_data['date'])

# Filter columns related to Metropolitan Police and other forces
metropolitan_columns = [col for col in crime_data.columns if 'Metropolitan Police Service' in col]
other_forces_columns = [
    col for col in crime_data.columns if 'Metropolitan Police Service' not in col and col not in ['date', 'post_implementation']
]

metropolitan_data = crime_data[['date', 'post_implementation'] + metropolitan_columns]
other_forces_data = crime_data[['date', 'post_implementation'] + other_forces_columns]

# Helper Functions
def standardize(series):
    return (series - series.mean()) / series.std()

def remove_seasonality(series):
    stl = STL(series, period=12)
    result = stl.fit()
    return result.trend + result.resid

# Create Synthetic Control
def create_synthetic_control(crime_type):
    met_crime_count_col = f'Metropolitan Police Service_{crime_type}_count'
    met_detection_rate_col = f'Metropolitan Police Service_{crime_type}_detection_rate'

    pre_implementation_data = metropolitan_data[metropolitan_data['post_implementation'] == 0]

    Y_pre_count = remove_seasonality(pre_implementation_data[met_crime_count_col])
    Y_pre_rate = remove_seasonality(pre_implementation_data[met_detection_rate_col])

    X_pre = other_forces_data[other_forces_data['post_implementation'] == 0].drop(columns=['date', 'post_implementation'])
    X_pre = X_pre.apply(remove_seasonality)

    Y_pre_count_std = standardize(Y_pre_count)
    Y_pre_rate_std = standardize(Y_pre_rate)
    X_pre_std = X_pre.apply(standardize)

    reg_count = LinearRegression().fit(X_pre_std, Y_pre_count_std)
    reg_rate = LinearRegression().fit(X_pre_std, Y_pre_rate_std)

    X_all = other_forces_data.drop(columns=['date', 'post_implementation'])
    X_all = X_all.apply(remove_seasonality)
    X_all_std = X_all.apply(standardize)

    synthetic_control_count_std = reg_count.predict(X_all_std)
    synthetic_control_rate_std = reg_rate.predict(X_all_std)

    synthetic_control_count = (synthetic_control_count_std * Y_pre_count.std()) + Y_pre_count.mean()
    synthetic_control_rate = (synthetic_control_rate_std * Y_pre_rate.std()) + Y_pre_rate.mean()

    return synthetic_control_count, synthetic_control_rate

# Test Parallel Trends
def test_parallel_trends(target_series, predictor_df, implementation_date):
    pre_implementation_mask = crime_data['date'] < implementation_date
    target_pre = target_series[pre_implementation_mask]
    predictor_pre = predictor_df[pre_implementation_mask]
    
    target_trend = np.arange(len(target_pre))
    predictor_trends = pd.DataFrame({col: np.arange(len(predictor_pre)) for col in predictor_pre.columns})
    
    target_model = LinearRegression().fit(target_trend.reshape(-1, 1), target_pre)
    predictor_models = {col: LinearRegression().fit(predictor_trends[col].values.reshape(-1, 1), predictor_pre[col]) 
                        for col in predictor_pre.columns}
    
    target_slope = target_model.coef_[0]
    predictor_slopes = [model.coef_[0] for model in predictor_models.values()]
    
    _, p_value = stats.ttest_ind([target_slope], predictor_slopes)
    
    return p_value

# Plot with Confidence Interval
def plot_with_ci(ax, x, y, label):
    ax.plot(x, y, label=label)
    ci = 1.96 * y.std() / np.sqrt(len(y))
    ax.fill_between(x, (y-ci), (y+ci), alpha=.1)

# Plot all Crime Types
def plot_all_crime_types(synthetic_controls, crime_types, metropolitan_data):
    fig, axes = plt.subplots(len(crime_types), 2, figsize=(15, len(crime_types) * 5))
    
    for i, crime_type in enumerate(crime_types):
        met_crime_count_col = f'Metropolitan Police Service_{crime_type}_count'
        met_detection_rate_col = f'Metropolitan Police Service_{crime_type}_detection_rate'

        actual_count = metropolitan_data[met_crime_count_col]
        actual_rate = metropolitan_data[met_detection_rate_col]

        synthetic_count, synthetic_rate = synthetic_controls[crime_type]

        plot_with_ci(axes[i, 0], metropolitan_data['date'], actual_count, 'Actual')
        plot_with_ci(axes[i, 0], metropolitan_data['date'], synthetic_count, 'Synthetic Control')
        axes[i, 0].axvline(x=pd.to_datetime('2022-11-29'), color='red', linestyle=':', label='Connect Drop 1')
        axes[i, 0].set_xlabel('Date')
        axes[i, 0].set_ylabel(f'{crime_type} Count')
        axes[i, 0].set_title(f'{crime_type} Count')
        axes[i, 0].legend()
        axes[i, 0].tick_params(axis='x', rotation=45)

        plot_with_ci(axes[i, 1], metropolitan_data['date'], actual_rate, 'Actual')
        plot_with_ci(axes[i, 1], metropolitan_data['date'], synthetic_rate, 'Synthetic Control')
        axes[i, 1].axvline(x=pd.to_datetime('2022-11-29'), color='red', linestyle=':', label='Connect Drop 1')
        axes[i, 1].set_xlabel('Date')
        axes[i, 1].set_ylabel('Detection Rate')
        axes[i, 1].set_title(f'Detection Rate - {crime_type}')
        axes[i, 1].legend()
        axes[i, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    # save 
    plt.savefig('crime_types_seasonally_adjusted.png')
    plt.show()

# Plot Residuals
def plot_residuals(crime_type):
    actual_count = metropolitan_data[f'Metropolitan Police Service_{crime_type}_count']
    synthetic_count = synthetic_controls[crime_type][0]
    residuals = actual_count - synthetic_count

    plt.figure(figsize=(10, 5))
    plt.plot(metropolitan_data['date'], residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.axvline(x=pd.to_datetime('2022-11-29'), color='g', linestyle=':', label='Connect Drop 1')
    plt.title(f'Residuals for {crime_type}')
    plt.xlabel('Date')
    plt.ylabel('Residuals')
    plt.legend()
    plt.show()

# Cross-Validation for Synthetic Control
def cross_validate_synthetic_control(crime_type, n_splits=5):
    met_crime_count_col = f'Metropolitan Police Service_{crime_type}_count'
    pre_implementation_data = metropolitan_data[metropolitan_data['post_implementation'] == 0]
    Y_pre_count = pre_implementation_data[met_crime_count_col]
    X_pre = other_forces_data[other_forces_data['post_implementation'] == 0].drop(columns=['date', 'post_implementation'])

    tscv = TimeSeriesSplit(n_splits=n_splits)
    mse_scores = []

    for train_index, test_index in tscv.split(pre_implementation_data):
        X_train, X_test = X_pre.iloc[train_index], X_pre.iloc[test_index]
        y_train, y_test = Y_pre_count.iloc[train_index], Y_pre_count.iloc[test_index]

        reg = LinearRegression().fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mse_scores.append(mse)

    return np.mean(mse_scores)

# Apply Bonferroni Correction
def apply_bonferroni_correction(p_values, alpha=0.05):
    n_tests = len(p_values)
    bonferroni_threshold = alpha / n_tests
    significant = p_values < bonferroni_threshold
    adjusted_p_values = np.minimum(p_values * n_tests, 1.0)
    return significant, adjusted_p_values

# Cohen's d Calculation
def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_se

# Calculate Trend
def calculate_trend(data):
    x = np.arange(len(data))
    slope, _, _, _, _ = stats.linregress(x, data)
    return slope

# Create Final Summary Table
def create_final_summary_table(crime_types, synthetic_controls, metropolitan_data, cv_scores):
    summary_table = []

    for crime_type in crime_types:
        met_crime_count_col = f'Metropolitan Police Service_{crime_type}_count'
        met_detection_rate_col = f'Metropolitan Police Service_{crime_type}_detection_rate'

        actual_count = metropolitan_data[met_crime_count_col]
        actual_rate = metropolitan_data[met_detection_rate_col]

        synthetic_count, synthetic_rate = synthetic_controls[crime_type]

        post_implementation = metropolitan_data['post_implementation'] == 1
        pre_implementation = metropolitan_data['post_implementation'] == 0

        avg_actual_count_pre = np.mean(actual_count[pre_implementation])
        avg_actual_count_post = np.mean(actual_count[post_implementation])
        avg_synthetic_count_post = np.mean(synthetic_count[post_implementation])

        avg_actual_rate_pre = np.mean(actual_rate[pre_implementation])
        avg_actual_rate_post = np.mean(actual_rate[post_implementation])
        avg_synthetic_rate_post = np.mean(synthetic_rate[post_implementation])

        count_change_actual = (avg_actual_count_post - avg_actual_count_pre) / avg_actual_count_pre * 100
        count_change_synthetic = (avg_synthetic_count_post - avg_actual_count_pre) / avg_actual_count_pre * 100
        rate_change_actual = (avg_actual_rate_post - avg_actual_rate_pre) / avg_actual_rate_pre * 100
        rate_change_synthetic = (avg_synthetic_rate_post - avg_actual_rate_pre) / avg_actual_rate_pre * 100

        t_stat_count, p_value_count = stats.ttest_ind(actual_count[post_implementation], synthetic_count[post_implementation])
        t_stat_rate, p_value_rate = stats.ttest_ind(actual_rate[post_implementation], synthetic_rate[post_implementation])

        effect_size_count = cohens_d(actual_count[post_implementation], synthetic_count[post_implementation])
        effect_size_rate = cohens_d(actual_rate[post_implementation], synthetic_rate[post_implementation])

        trend_actual_pre = calculate_trend(actual_count[pre_implementation])
        trend_actual_post = calculate_trend(actual_count[post_implementation])
        trend_synthetic_post = calculate_trend(synthetic_count[post_implementation])

        summary_table.append({
            'Crime Type': crime_type,
            'Avg Count (Pre-Implementation)': avg_actual_count_pre,
            'Avg Count (Post-Implementation, Actual)': avg_actual_count_post,
            'Avg Count (Post-Implementation, Synthetic)': avg_synthetic_count_post,
            'Count % Change (Actual)': count_change_actual,
            'Count % Change (Synthetic)': count_change_synthetic,
            'Count T-statistic': t_stat_count,
            'Count P-value': p_value_count,
            'Count Effect Size (Cohen\'s d)': effect_size_count,
            'Avg Rate (Pre-Implementation)': avg_actual_rate_pre,
            'Avg Rate (Post-Implementation, Actual)': avg_actual_rate_post,
            'Avg Rate (Post-Implementation, Synthetic)': avg_synthetic_rate_post,
            'Rate % Change (Actual)': rate_change_actual,
            'Rate % Change (Synthetic)': rate_change_synthetic,
            'Rate T-statistic': t_stat_rate,
            'Rate P-value': p_value_rate,
            'Rate Effect Size (Cohen\'s d)': effect_size_rate,
            'Cross-validation MSE': cv_scores[crime_type],
            'Trend (Pre-Implementation)': trend_actual_pre,
            'Trend (Post-Implementation, Actual)': trend_actual_post,
            'Trend (Post-Implementation, Synthetic)': trend_synthetic_post
        })
    
    summary_df = pd.DataFrame(summary_table)

    count_significant, count_adjusted_p = apply_bonferroni_correction(summary_df['Count P-value'])
    rate_significant, rate_adjusted_p = apply_bonferroni_correction(summary_df['Rate P-value'])

    summary_df['Count Significant (Bonferroni)'] = count_significant
    summary_df['Count Adjusted P-value'] = count_adjusted_p
    summary_df['Rate Significant (Bonferroni)'] = rate_significant
    summary_df['Rate Adjusted P-value'] = rate_adjusted_p

    return summary_df

# Run Robustness Checks for a Crime Type
def run_robustness_checks(crime_type, is_detection_rate=False):
    target_series = crime_data[
        f'Metropolitan Police Service_{crime_type}_detection_rate' if is_detection_rate 
        else f'Metropolitan Police Service_{crime_type}_count'
    ]
    
    predictor_df = crime_data[
        [col for col in crime_data.columns if 'Metropolitan Police' not in col and col not in ['date', 'post_implementation']]
    ]
    
    results = {
        'Crime Type': crime_type,
        'Is Detection Rate': is_detection_rate,
        'Placebo Test p-value': placebo_test(target_series, predictor_df, implementation_date),
        'Sensitivity': sensitivity_analysis(target_series, predictor_df, implementation_date),
        'Pre-implementation Fit (RMSE)': assess_preimplementation_fit(target_series, predictor_df, implementation_date),
        'Prediction Interval Violation Rate': calculate_prediction_intervals(target_series, predictor_df, implementation_date),
        'Parallel Trends p-value': test_parallel_trends(target_series, predictor_df, implementation_date)
    }
    
    return results

# Create Synthetic Control for Robustness Checks
def create_synthetic_control_for_robustness(target_series, predictor_df, implementation_date, placebo_date=None):
    if placebo_date is None:
        placebo_date = implementation_date
    
    pre_implementation_mask = crime_data['date'] < placebo_date
    Y_pre = remove_seasonality(target_series[pre_implementation_mask])
    X_pre = predictor_df[pre_implementation_mask].apply(remove_seasonality)
    
    if len(Y_pre) == 0 or len(X_pre) == 0:
        return None
    
    reg = LinearRegression().fit(X_pre, Y_pre)
    X_all = predictor_df.apply(remove_seasonality)
    synthetic_control = reg.predict(X_all)
    
    return synthetic_control

# Calculate RMSE
def calculate_rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))

# Placebo Test for Robustness Checks
def placebo_test(target_series, predictor_df, implementation_date, n_placebos=100):
    actual_synthetic = create_synthetic_control_for_robustness(target_series, predictor_df, implementation_date)
    if actual_synthetic is None:
        return None
    
    actual_effect = calculate_rmse(target_series[crime_data['date'] >= implementation_date],
                                   actual_synthetic[crime_data['date'] >= implementation_date])
    
    placebo_effects = []
    attempts = 0
    while len(placebo_effects) < n_placebos and attempts < n_placebos * 2:
        placebo_date = pd.Timestamp(np.random.choice(crime_data[crime_data['date'] < implementation_date]['date']))
        placebo_synthetic = create_synthetic_control_for_robustness(target_series, predictor_df, implementation_date, placebo_date)
        if placebo_synthetic is not None:
            placebo_effect = calculate_rmse(target_series[crime_data['date'] >= placebo_date],
                                            placebo_synthetic[crime_data['date'] >= placebo_date])
            placebo_effects.append(placebo_effect)
        attempts += 1
    
    if len(placebo_effects) == 0:
        return None
    
    p_value = sum(pe >= actual_effect for pe in placebo_effects) / len(placebo_effects)
    return p_value

# Sensitivity Analysis for Robustness Checks
def sensitivity_analysis(target_series, predictor_df, implementation_date):
    full_synthetic = create_synthetic_control_for_robustness(target_series, predictor_df, implementation_date)
    if full_synthetic is None:
        return None
    
    full_rmse = calculate_rmse(target_series[crime_data['date'] >= implementation_date],
                               full_synthetic[crime_data['date'] >= implementation_date])
    
    sensitivities = []
    for col in predictor_df.columns:
        reduced_predictors = predictor_df.drop(columns=[col])
        reduced_synthetic = create_synthetic_control_for_robustness(target_series, reduced_predictors, implementation_date)
        if reduced_synthetic is not None:
            reduced_rmse = calculate_rmse(target_series[crime_data['date'] >= implementation_date],
                                          reduced_synthetic[crime_data['date'] >= implementation_date])
            sensitivity = (reduced_rmse - full_rmse) / full_rmse
            sensitivities.append(sensitivity)
    
    return np.mean(sensitivities) if sensitivities else None

# Assess Pre-implementation Fit for Robustness Checks
def assess_preimplementation_fit(target_series, predictor_df, implementation_date):
    pre_implementation_mask = crime_data['date'] < implementation_date
    synthetic_control = create_synthetic_control_for_robustness(target_series, predictor_df, implementation_date)
    
    if synthetic_control is None:
        return None
    
    pre_rmse = calculate_rmse(target_series[pre_implementation_mask],
                              synthetic_control[pre_implementation_mask])
    return pre_rmse

# Calculate Prediction Intervals for Robustness Checks
def calculate_prediction_intervals(target_series, predictor_df, implementation_date):
    pre_implementation_mask = crime_data['date'] < implementation_date
    post_implementation_mask = crime_data['date'] >= implementation_date
    
    synthetic_control = create_synthetic_control_for_robustness(target_series, predictor_df, implementation_date)
    
    if synthetic_control is None:
        return None
    
    pre_residuals = target_series[pre_implementation_mask] - synthetic_control[pre_implementation_mask]
    residual_std = np.std(pre_residuals)
    
    post_synthetic = synthetic_control[post_implementation_mask]
    post_actual = target_series[post_implementation_mask]
    
    lower_bound = post_synthetic - 1.96 * residual_std
    upper_bound = post_synthetic + 1.96 * residual_std
    
    outside_interval = ((post_actual < lower_bound) | (post_actual > upper_bound))
    
    return np.mean(outside_interval)

# Main execution
crime_types = [
    'Bicycle theft', 'Burglary', 'Criminal damage and arson', 'Drugs', 'Other crime', 
    'Other theft', 'Possession of weapons', 'Public order', 'Robbery', 
    'Shoplifting', 'Theft from the person', 'Vehicle crime', 
    'Violence and sexual offences'
]

synthetic_controls = {}
for crime_type in crime_types:
    synthetic_controls[crime_type] = create_synthetic_control(crime_type)

plot_all_crime_types(synthetic_controls, crime_types, metropolitan_data)

cv_scores = {crime_type: cross_validate_synthetic_control(crime_type) for crime_type in crime_types}

final_summary_df = create_final_summary_table(crime_types, synthetic_controls, metropolitan_data, cv_scores)
final_summary_df.to_csv('summary_table_seasonally_adjusted.csv', index=False)
print("\nFinal summary table has been saved to 'summary_table_seasonally_adjusted.csv'")

# Robustness Checks
implementation_date = pd.Timestamp('2022-11-29')

robustness_results = []
for crime_type in tqdm(crime_types):
    robustness_results.append(run_robustness_checks(crime_type, is_detection_rate=False))
    robustness_results.append(run_robustness_checks(crime_type, is_detection_rate=True))

# Convert results to DataFrame
robustness_df = pd.DataFrame(robustness_results)
robustness_df.to_csv('improved_robustness_check_results.csv', index=False)
print("\nImproved robustness check results have been saved to 'improved_robustness_check_results.csv'")

# Display results
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', '{:.4f}'.format)
robustness_df
