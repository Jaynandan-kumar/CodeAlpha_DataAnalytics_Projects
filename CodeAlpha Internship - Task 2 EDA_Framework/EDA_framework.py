"""
COMPREHENSIVE EXPLORATORY DATA ANALYSIS (EDA) FRAMEWORK
Author: Data Analyst
Purpose: Systematic data exploration following EDA best practices
Date: 2024
"""

# ============================================
# IMPORT LIBRARIES WITH EXPLANATIONS
# ============================================

# Data manipulation and analysis
import pandas as pd  # For data manipulation and analysis (DataFrame operations)
import numpy as np  # For numerical operations and array manipulation

# Data visualization
import matplotlib.pyplot as plt  # Core plotting library for creating static visualizations
import seaborn as sns  # Enhanced visualization library based on matplotlib
import missingno as msno  # Specialized library for missing data visualization

# Statistical analysis
from scipy import stats  # For statistical tests and probability distributions

# System and utilities
import warnings  # For managing warning messages
from datetime import datetime  # For date and time operations

# ============================================
# CONFIGURATION SETTINGS
# ============================================

# Suppress warnings to keep output clean
warnings.filterwarnings('ignore')

# Set visualization style for consistent, professional-looking plots
plt.style.use('seaborn-v0_8-darkgrid')  # Uses seaborn style with grid
sns.set_palette("husl")  # Set color palette to "husl" for better color differentiation

# Configure pandas display options for better readability
pd.set_option('display.max_columns', None)  # Show all columns when printing DataFrames
pd.set_option('display.width', 1000)  # Set display width to avoid line wrapping

# Set random seed for reproducibility of random operations
np.random.seed(42)

print("  EDA Framework Initialized")
print("=" * 60)


# ============================================
# 1. DATA LOADING AND INITIAL EXPLORATION
# ============================================

def load_and_explore_dataset():
    """
    Load dataset with multiple fallback options
    Strategy:
    1. Try to load from local CSV file
    2. Try to load built-in dataset from seaborn
    3. Create synthetic data if all else fails
    """
    try:
        # First attempt: Load from local CSV file
        try:
            df = pd.read_csv('data.csv')
            print("  Loaded: data.csv")
        except FileNotFoundError:
            # Second attempt: Load built-in dataset from seaborn
            try:
                # Titanic dataset is commonly used for demonstration
                df = sns.load_dataset('titanic')
                print("  Loaded: Titanic dataset (from seaborn)")
            except:
                # Final fallback: Create synthetic data for demonstration
                print("  Creating synthetic dataset for demonstration...")
                df = create_synthetic_dataset()

    except Exception as e:
        # Handle any unexpected errors
        print(f"  Error loading dataset: {e}")
        print("  Creating synthetic dataset...")
        df = create_synthetic_dataset()

    return df


def create_synthetic_dataset():
    """
    Create a realistic synthetic dataset with:
    - Realistic distributions (normal, exponential, gamma, poisson)
    - Intentional anomalies for demonstration
    - Missing values for imputation demonstration
    - Categorical variables for segmentation analysis
    - Temporal features for time series analysis
    """
    np.random.seed(42)  # Set seed for reproducibility
    n_samples = 1000  # Number of samples in dataset

    # Create date range for temporal analysis
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')

    # Create synthetic data with realistic distributions
    data = {
        'customer_id': range(1000, 1000 + n_samples),  # Unique identifier
        'age': np.random.normal(35, 10, n_samples).clip(18, 70),  # Normal distribution clipped to range
        'income': np.random.exponential(50000, n_samples) + 30000,  # Exponential distribution with shift
        'purchase_amount': np.random.gamma(2, 50, n_samples),  # Gamma distribution for purchase amounts
        'purchase_frequency': np.random.poisson(3, n_samples),  # Poisson for count data
        'customer_lifetime': np.random.normal(24, 12, n_samples).clip(1, 60),  # Normal distribution for lifetime
        'satisfaction_score': np.random.randint(1, 11, n_samples),  # Integer scores 1-10
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples, p=[0.3, 0.25, 0.25, 0.2]),
        # Categorical with probabilities
        'membership_type': np.random.choice(['Basic', 'Premium', 'Gold'], n_samples, p=[0.5, 0.3, 0.2]),
        # Membership categories
        'is_churned': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),  # Binary outcome variable
        'last_purchase_date': dates,  # DateTime for temporal analysis
        'website_visits': np.random.poisson(15, n_samples),  # Count data
        'avg_session_duration': np.random.normal(300, 120, n_samples).clip(30, 1200),  # Duration in seconds
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Books'], n_samples)
        # Product categories
    }

    # ============================================
    # INTENTIONALLY INTRODUCE ANOMALIES FOR DEMONSTRATION
    # ============================================

    # Create some very old ages (every 50th record)
    data['age'][::50] = np.random.randint(100, 120, len(data['age'][::50]))

    # Create some very high incomes (every 30th record)
    data['income'][::30] = data['income'][::30] * 10

    # Create some negative purchases (every 20th record) - potential data entry errors
    data['purchase_amount'][::20] = -data['purchase_amount'][::20]

    # ============================================
    # INTRODUCE MISSING VALUES (5% MISSING RATE)
    # ============================================
    for col in ['age', 'income', 'satisfaction_score']:
        # Create random mask for 5% of values
        mask = np.random.random(n_samples) < 0.05
        # Replace selected values with NaN
        data[col] = np.where(mask, np.nan, data[col])

    # Create DataFrame from dictionary
    df = pd.DataFrame(data)

    # ============================================
    # CREATE DERIVED FEATURES FOR ANALYSIS
    # ============================================
    # Extract month from date for seasonal analysis
    df['month'] = df['last_purchase_date'].dt.month
    # Extract day name for weekday/weekend analysis
    df['day_of_week'] = df['last_purchase_date'].dt.day_name()

    return df


# Load the dataset
df = load_and_explore_dataset()

# Print initial dataset information
print(f"  Dataset Shape: {df.shape}")  # (rows, columns)
print(f"  Dataset Columns: {list(df.columns)}")
print("=" * 60)

# ============================================
# 2. MEANINGFUL QUESTIONS & HYPOTHESES FORMULATION
# ============================================

print("  MEANINGFUL QUESTIONS & HYPOTHESES")
print("=" * 60)

# List of exploratory questions to guide the analysis
questions = [
    "1. What are the main characteristics of our customers?",
    "2. How does income correlate with purchase behavior?",
    "3. Are there significant differences between customer segments?",
    "4. What factors are associated with customer churn?",
    "5. Are there seasonal patterns in purchases?",
    "6. What are the typical values and ranges for key metrics?",
    "7. Are there any data quality issues or anomalies?",
    "8. How are variables correlated with each other?"
]

for q in questions:
    print(f"   {q}")

print("\n  KEY HYPOTHESES TO TEST:")
# Specific, testable hypotheses
hypotheses = [
    "H1: Higher income customers make larger purchases",
    "H2: Premium members have higher satisfaction scores",
    "H3: Purchase frequency decreases with customer age",
    "H4: Weekend purchases are larger than weekday purchases",
    "H5: There are regional differences in customer behavior"
]

for h in hypotheses:
    print(f"   • {h}")

print("=" * 60)

# ============================================
# 3. DATA STRUCTURE EXPLORATION
# ============================================

print("  DATA STRUCTURE EXPLORATION")
print("=" * 60)


def explore_data_structure(df):
    """
    Comprehensive exploration of data structure including:
    - Data types
    - Column categories
    - Basic shape information
    """

    print("  BASIC INFORMATION:")
    print("-" * 40)
    df.info()  # Shows column names, data types, non-null counts, and memory usage

    print(f"\n  DATA TYPES DISTRIBUTION:")
    print("-" * 40)
    # Count how many columns of each data type
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"   {dtype}: {count} columns")

    # ============================================
    # IDENTIFY COLUMN TYPES FOR ANALYSIS
    # ============================================

    print(f"\n  NUMERICAL COLUMNS ({len(df.select_dtypes(include=[np.number]).columns)}):")
    print("-" * 40)
    # Select columns with numerical data types
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Print first 10 numerical columns
    print(f"   {', '.join(num_cols[:10])}{'...' if len(num_cols) > 10 else ''}")

    print(f"\n  CATEGORICAL COLUMNS ({len(df.select_dtypes(include=['object', 'category']).columns)}):")
    print("-" * 40)
    # Select columns with categorical data types
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    # Print first 10 categorical columns
    print(f"   {', '.join(cat_cols[:10])}{'...' if len(cat_cols) > 10 else ''}")

    print(f"\n  DATETIME COLUMNS ({len(df.select_dtypes(include=['datetime']).columns)}):")
    print("-" * 40)
    # Select columns with datetime data types
    date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
    if date_cols:
        print(f"   {', '.join(date_cols)}")
    else:
        print("   None")

    return num_cols, cat_cols, date_cols


# Execute data structure exploration
num_cols, cat_cols, date_cols = explore_data_structure(df)

# ============================================
# 4. DESCRIPTIVE STATISTICS
# ============================================

print(f"\n  DESCRIPTIVE STATISTICS")
print("=" * 60)


def detailed_descriptive_stats(df, num_cols):
    """
    Generate comprehensive descriptive statistics including:
    - Central tendency measures
    - Dispersion measures
    - Percentile distribution
    - Missing value statistics
    - Data quality metrics
    """

    print("  SUMMARY STATISTICS (Numerical Variables):")
    print("-" * 60)

    # Calculate comprehensive statistics for numerical columns
    stats_df = df[num_cols].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T
    # Transpose to have statistics as rows and variables as columns

    # ============================================
    # ADD CUSTOM METRICS TO STATISTICS TABLE
    # ============================================

    # Count missing values
    stats_df['missing'] = df[num_cols].isnull().sum()
    stats_df['missing_pct'] = (stats_df['missing'] / len(df) * 100).round(2)

    # Count zero values (important for certain metrics)
    stats_df['zeros'] = (df[num_cols] == 0).sum()
    stats_df['zeros_pct'] = (stats_df['zeros'] / len(df) * 100).round(2)

    # Calculate distribution shape metrics
    stats_df['skewness'] = df[num_cols].skew().round(3)  # Measure of asymmetry
    stats_df['kurtosis'] = df[num_cols].kurtosis().round(3)  # Measure of tail heaviness

    # Display selected statistics
    print(stats_df[['mean', 'std', 'min', '5%', '25%', '50%', '75%', '95%', '99%', 'max',
                    'missing_pct', 'zeros_pct', 'skewness', 'kurtosis']].head(15))

    # ============================================
    # ANALYZE CATEGORICAL VARIABLES
    # ============================================

    print(f"\n  CATEGORICAL VARIABLE DISTRIBUTIONS:")
    print("-" * 60)

    # Analyze distribution of categorical variables
    for col in cat_cols[:5]:  # Limit to first 5 categorical columns for brevity
        print(f"\n{col}:")
        # Count occurrences of each category
        value_counts = df[col].value_counts(dropna=False)

        # Display top 10 categories
        for val, count in value_counts.head(10).items():
            pct = count / len(df) * 100
            print(f"   {val}: {count} ({pct:.1f}%)")

        # Indicate if there are more categories
        if len(value_counts) > 10:
            print(f"   ... and {len(value_counts) - 10} more categories")

    return stats_df


# Generate descriptive statistics
stats_df = detailed_descriptive_stats(df, num_cols)

# ============================================
# 5. VISUAL EXPLORATION - DISTRIBUTIONS
# ============================================

print(f"\n  VISUAL EXPLORATION - DISTRIBUTIONS")
print("=" * 60)


def plot_distributions(df, num_cols, cat_cols):
    """
    Create visualization of variable distributions:
    - Histograms with KDE for numerical variables
    - Bar charts for categorical variables
    - Statistical annotations on plots
    """

    # ============================================
    # SET UP SUBPLOTS FOR NUMERICAL DISTRIBUTIONS
    # ============================================
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()  # Convert 2D array to 1D for easy iteration

    # Select key numerical columns for visualization
    # Exclude ID columns and focus on meaningful metrics
    important_num_cols = [col for col in num_cols if col not in ['customer_id', 'month']]
    important_num_cols = important_num_cols[:9]  # Take first 9 for visualization

    # ============================================
    # PLOT NUMERICAL DISTRIBUTIONS
    # ============================================
    for idx, col in enumerate(important_num_cols):
        if idx < len(axes):
            # Create histogram with transparency
            axes[idx].hist(df[col].dropna(), bins=30, alpha=0.6, color='steelblue',
                           density=True, edgecolor='black')

            # Add Kernel Density Estimate (KDE) for smooth distribution
            from scipy.stats import gaussian_kde
            data = df[col].dropna()  # Remove missing values

            if len(data) > 1:  # Need at least 2 points for KDE
                kde = gaussian_kde(data)  # Calculate KDE
                x_range = np.linspace(data.min(), data.max(), 100)  # Create smooth x-axis
                axes[idx].plot(x_range, kde(x_range), color='darkred', linewidth=2)

            # Format plot
            axes[idx].set_title(f'Distribution of {col}', fontsize=10, fontweight='bold')
            axes[idx].set_xlabel(col, fontsize=9)
            axes[idx].set_ylabel('Density', fontsize=9)

            # Add statistical summary as text box
            stats_text = f'Mean: {df[col].mean():.1f}\nStd: {df[col].std():.1f}\nSkew: {df[col].skew():.2f}'
            axes[idx].text(0.05, 0.95, stats_text, transform=axes[idx].transAxes,
                           fontsize=8, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Distribution of Key Numerical Variables', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()  # Adjust spacing between plots
    plt.savefig('distributions.png', dpi=100, bbox_inches='tight')  # Save figure
    plt.show()

    # ============================================
    # PLOT CATEGORICAL DISTRIBUTIONS
    # ============================================
    if len(cat_cols) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for idx, col in enumerate(cat_cols[:4]):  # Plot first 4 categorical columns
            if idx < len(axes):
                # Get value counts for top 10 categories
                value_counts = df[col].value_counts().head(10)

                # Create bar plot with categorical colors
                bars = axes[idx].bar(range(len(value_counts)), value_counts.values,
                                     color=plt.cm.Set3(np.arange(len(value_counts))))

                # Format plot
                axes[idx].set_title(f'Distribution of {col}', fontsize=11, fontweight='bold')
                axes[idx].set_xlabel(col, fontsize=10)
                axes[idx].set_ylabel('Count', fontsize=10)
                axes[idx].set_xticks(range(len(value_counts)))
                axes[idx].set_xticklabels(value_counts.index, rotation=45, ha='right', fontsize=9)

                # Add value labels on top of bars
                for bar_idx, (bar, count) in enumerate(zip(bars, value_counts.values)):
                    height = bar.get_height()
                    axes[idx].text(bar_idx, height + max(value_counts.values) * 0.01,
                                   f'{count}\n({count / len(df) * 100:.1f}%)',
                                   ha='center', va='bottom', fontsize=8)

        plt.suptitle('Distribution of Categorical Variables', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('categorical_distributions.png', dpi=100, bbox_inches='tight')
        plt.show()


# Create distribution plots
plot_distributions(df, num_cols, cat_cols)

# ============================================
# 6. MISSING DATA ANALYSIS
# ============================================

print(f"\n  MISSING DATA ANALYSIS")
print("=" * 60)


def analyze_missing_data(df):
    """
    Comprehensive missing data analysis:
    - Quantify missing values
    - Visualize missing patterns
    - Identify correlations in missingness
    """

    print(" MISSING VALUE SUMMARY:")
    print("-" * 40)

    # Calculate missing value statistics
    missing_df = pd.DataFrame({
        'missing_count': df.isnull().sum(),  # Count of missing values per column
        'missing_percentage': (df.isnull().sum() / len(df) * 100).round(2)  # Percentage missing
    }).sort_values('missing_percentage', ascending=False)  # Sort by missing percentage

    # Filter to show only columns with missing values
    missing_df = missing_df[missing_df['missing_count'] > 0]

    if len(missing_df) > 0:
        print(missing_df)

        # ============================================
        # VISUALIZE MISSING DATA PATTERNS
        # ============================================
        plt.figure(figsize=(12, 6))

        # Subplot 1: Bar chart of missing percentages
        ax1 = plt.subplot(121)
        bars = ax1.barh(missing_df.index, missing_df['missing_percentage'],
                        color='salmon', edgecolor='darkred')
        ax1.set_xlabel('Missing Percentage (%)', fontsize=11)
        ax1.set_title('Missing Data by Column', fontsize=12, fontweight='bold')
        ax1.invert_yaxis()  # Highest missing percentage at top

        # Add value labels to bars
        for bar in bars:
            width = bar.get_width()
            ax1.text(width + 0.5, bar.get_y() + bar.get_height() / 2,
                     f'{width:.1f}%', ha='left', va='center', fontsize=9)

        # Subplot 2: Missing value matrix
        ax2 = plt.subplot(122)
        # Create matrix visualization of missing values
        msno.matrix(df.sample(min(500, len(df))), ax=ax2, sparkline=False)
        ax2.set_title('Missing Value Pattern', fontsize=12, fontweight='bold')

        plt.suptitle('Missing Data Analysis', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('missing_data_analysis.png', dpi=100, bbox_inches='tight')
        plt.show()

        # ============================================
        # ANALYZE MISSING DATA PATTERNS
        # ============================================
        print(f"\n PATTERN ANALYSIS:")
        print("-" * 40)

        # Calculate correlation of missingness between columns
        # This helps identify if missingness in one column is related to missingness in another
        missing_corr = df.isnull().corr()

        # Find strongly correlated missing patterns
        high_missing_corr = missing_corr[(missing_corr.abs() > 0.3) & (missing_corr.abs() < 1)].stack().reset_index()
        high_missing_corr.columns = ['Var1', 'Var2', 'Correlation']

        if len(high_missing_corr) > 0:
            print("Potential missing data patterns (correlations > 0.3):")
            print(high_missing_corr.sort_values('Correlation', ascending=False).head(10))
        else:
            print("No strong patterns detected in missing data")

    else:
        print(" No missing values found in the dataset!")

    return missing_df


# Analyze missing data
missing_df = analyze_missing_data(df)

# ============================================
# 7. OUTLIER DETECTION & ANALYSIS
# ============================================

print(f"\n  OUTLIER DETECTION & ANALYSIS")
print("=" * 60)


def detect_outliers(df, num_cols):
    """
    Detect and analyze outliers using:
    - Interquartile Range (IQR) method
    - Visual box plots
    - Statistical summary of outliers
    """

    print(" OUTLIER SUMMARY (IQR Method):")
    print("-" * 50)

    outlier_summary = []

    # ============================================
    # DETECT OUTLIERS USING IQR METHOD
    # ============================================
    for col in num_cols[:10]:  # Check first 10 numerical columns for outliers
        if df[col].notna().sum() > 0:  # Skip if column is all NaN

            # Calculate quartiles and IQR
            Q1 = df[col].quantile(0.25)  # First quartile (25th percentile)
            Q3 = df[col].quantile(0.75)  # Third quartile (75th percentile)
            IQR = Q3 - Q1  # Interquartile Range

            # Define outlier boundaries (1.5 * IQR is standard)
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Identify outliers
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            outlier_count = len(outliers)
            outlier_pct = (outlier_count / len(df) * 100) if len(df) > 0 else 0

            # Record outlier statistics
            if outlier_count > 0:
                outlier_summary.append({
                    'column': col,
                    'outliers_count': outlier_count,
                    'outliers_pct': round(outlier_pct, 2),
                    'min_outlier': outliers.min() if len(outliers) > 0 else None,
                    'max_outlier': outliers.max() if len(outliers) > 0 else None,
                    'method': 'IQR'
                })

    # ============================================
    # DISPLAY AND VISUALIZE OUTLIERS
    # ============================================
    if outlier_summary:
        outlier_df = pd.DataFrame(outlier_summary)
        print(outlier_df.sort_values('outliers_pct', ascending=False))

        # Select top 4 columns with most outliers for visualization
        top_outlier_cols = outlier_df.nlargest(4, 'outliers_pct')['column'].tolist()

        # Create box plots for outlier visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for idx, col in enumerate(top_outlier_cols):
            if idx < len(axes):
                # Create box plot with custom styling
                bp = axes[idx].boxplot(df[col].dropna(), patch_artist=True,
                                       boxprops=dict(facecolor='lightblue', color='darkblue'),
                                       medianprops=dict(color='red', linewidth=2),
                                       whiskerprops=dict(color='darkblue'),
                                       capprops=dict(color='darkblue'),
                                       flierprops=dict(marker='o', markerfacecolor='red',
                                                       markersize=6, markeredgecolor='none', alpha=0.6))

                axes[idx].set_title(f'Outliers in {col}', fontsize=11, fontweight='bold')
                axes[idx].set_ylabel(col, fontsize=10)
                axes[idx].grid(True, alpha=0.3, linestyle='--')  # Add grid

                # Calculate outlier statistics for annotation
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]

                # Add outlier count as text annotation
                axes[idx].text(0.02, 0.98, f'Outliers: {len(outliers)} ({len(outliers) / len(df) * 100:.1f}%)',
                               transform=axes[idx].transAxes, fontsize=9,
                               verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

        plt.suptitle('Outlier Detection using Box Plots', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('outlier_detection.png', dpi=100, bbox_inches='tight')
        plt.show()

    else:
        print(" No outliers detected using IQR method!")

    return outlier_df if 'outlier_df' in locals() else None


# Detect outliers
outlier_df = detect_outliers(df, num_cols)

# ============================================
# 8. CORRELATION ANALYSIS
# ============================================

print(f"\n CORRELATION ANALYSIS")
print("=" * 60)


def analyze_correlations(df, num_cols):
    """
    Analyze relationships between variables:
    - Calculate correlation matrix
    - Identify strong positive/negative correlations
    - Visualize with heatmaps and scatter plots
    """

    # Select numerical columns with sufficient data for correlation
    corr_cols = [col for col in num_cols if df[col].notna().sum() > 10 and
                 len(df[col].unique()) > 1]  # Need at least 2 unique values

    if len(corr_cols) > 1:
        # Calculate Pearson correlation matrix
        corr_matrix = df[corr_cols].corr(numeric_only=True)

        print(" TOP POSITIVE CORRELATIONS:")
        print("-" * 40)

        # ============================================
        # IDENTIFY STRONG CORRELATIONS
        # ============================================

        # Unstack correlation matrix to get pairwise correlations
        corr_pairs = corr_matrix.unstack()
        # Sort by absolute correlation strength
        sorted_pairs = corr_pairs.sort_values(ascending=False, key=abs)

        # Filter for meaningful correlations (|r| > 0.3) excluding self-correlations
        strong_pairs = sorted_pairs[(sorted_pairs.abs() > 0.3) &
                                    (sorted_pairs.abs() < 1)].drop_duplicates()

        # Display top positive correlations
        if len(strong_pairs) > 0:
            top_positive = strong_pairs.head(10)
            for (var1, var2), corr in top_positive.items():
                print(f"   {var1} ↔ {var2}: {corr:.3f}")
        else:
            print("   No strong correlations (> 0.3) found")

        print(f"\n TOP NEGATIVE CORRELATIONS:")
        print("-" * 40)

        # Display top negative correlations
        strong_negative = sorted_pairs[sorted_pairs < 0].head(10)
        if len(strong_negative) > 0:
            for (var1, var2), corr in strong_negative.items():
                print(f"   {var1} ↔ {var2}: {corr:.3f}")
        else:
            print("   No strong negative correlations found")

        # ============================================
        # VISUALIZE CORRELATION MATRIX
        # ============================================
        plt.figure(figsize=(12, 10))

        # Create mask to hide upper triangle (redundant information)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        # Create heatmap with annotations
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                    annot_kws={"size": 9})

        plt.title('Correlation Matrix of Numerical Variables', fontsize=14, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.savefig('correlation_matrix.png', dpi=100, bbox_inches='tight')
        plt.show()

        # ============================================
        # SCATTER PLOTS FOR HIGHLY CORRELATED PAIRS
        # ============================================
        if len(strong_pairs) > 0:
            top_pairs = strong_pairs.head(4).index.tolist()

            if top_pairs:
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                axes = axes.flatten()

                for idx, (var1, var2) in enumerate(top_pairs[:4]):
                    if idx < len(axes):
                        # Create scatter plot with regression line
                        sns.regplot(x=df[var1], y=df[var2], ax=axes[idx],
                                    scatter_kws={'alpha': 0.5, 's': 30},
                                    line_kws={'color': 'red', 'linewidth': 2})

                        # Get correlation value for annotation
                        corr_value = corr_matrix.loc[var1, var2]

                        axes[idx].set_title(f'{var1} vs {var2}\nCorrelation: {corr_value:.3f}',
                                            fontsize=11, fontweight='bold')
                        axes[idx].set_xlabel(var1, fontsize=10)
                        axes[idx].set_ylabel(var2, fontsize=10)
                        axes[idx].grid(True, alpha=0.3, linestyle='--')

                plt.suptitle('Scatter Plots of Highly Correlated Variables',
                             fontsize=14, fontweight='bold', y=1.02)
                plt.tight_layout()
                plt.savefig('correlation_scatter.png', dpi=100, bbox_inches='tight')
                plt.show()

        return corr_matrix

    else:
        print(" Not enough numerical variables for correlation analysis")
        return None


# Analyze correlations
corr_matrix = analyze_correlations(df, num_cols)

# ============================================
# 9. TIME SERIES ANALYSIS
# ============================================

if date_cols:
    print(f"\n TIME SERIES ANALYSIS")
    print("=" * 60)

    # Analyze each date column (limit to first 2)
    for date_col in date_cols[:2]:
        print(f"\nAnalyzing: {date_col}")
        print("-" * 40)

        # Ensure column is in datetime format
        df[date_col] = pd.to_datetime(df[date_col])

        # ============================================
        # EXTRACT TEMPORAL FEATURES
        # ============================================
        df[f'{date_col}_year'] = df[date_col].dt.year
        df[f'{date_col}_month'] = df[date_col].dt.month
        df[f'{date_col}_day'] = df[date_col].dt.day
        df[f'{date_col}_dayofweek'] = df[date_col].dt.dayofweek  # Monday=0, Sunday=6
        df[f'{date_col}_quarter'] = df[date_col].dt.quarter

        # Display date range information
        print(f"   Date Range: {df[date_col].min()} to {df[date_col].max()}")
        print(f"   Total Days: {(df[date_col].max() - df[date_col].min()).days}")

        # ============================================
        # ANALYZE TEMPORAL PATTERNS
        # ============================================
        if 'purchase_amount' in df.columns:
            # Aggregate by month
            monthly_data = df.groupby(f'{date_col}_month')['purchase_amount'].agg(['mean', 'sum', 'count'])
            print(f"\n   Monthly Purchase Patterns:")
            print(monthly_data.round(2))

            # Visualize temporal patterns
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Plot 1: Monthly trend
            axes[0].plot(monthly_data.index, monthly_data['mean'], marker='o', linewidth=2, markersize=8)
            axes[0].set_title(f'Average Purchase by Month', fontsize=12, fontweight='bold')
            axes[0].set_xlabel('Month', fontsize=11)
            axes[0].set_ylabel('Average Purchase Amount', fontsize=11)
            axes[0].grid(True, alpha=0.3)
            axes[0].fill_between(monthly_data.index, monthly_data['mean'], alpha=0.3)  # Add fill under line

            # Plot 2: Day of week pattern
            if f'{date_col}_dayofweek' in df.columns:
                dow_data = df.groupby(f'{date_col}_dayofweek')['purchase_amount'].mean()
                day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

                axes[1].bar(range(len(dow_data)), dow_data.values, color='skyblue', edgecolor='navy')
                axes[1].set_title(f'Average Purchase by Day of Week', fontsize=12, fontweight='bold')
                axes[1].set_xlabel('Day of Week', fontsize=11)
                axes[1].set_ylabel('Average Purchase Amount', fontsize=11)
                axes[1].set_xticks(range(len(dow_data)))
                axes[1].set_xticklabels(day_names)
                axes[1].grid(True, alpha=0.3, axis='y')

            plt.suptitle(f'Temporal Analysis based on {date_col}', fontsize=14, fontweight='bold', y=1.05)
            plt.tight_layout()
            plt.savefig('time_series_analysis.png', dpi=100, bbox_inches='tight')
            plt.show()

# ============================================
# 10. HYPOTHESIS TESTING
# ============================================

print(f"\n HYPOTHESIS TESTING")
print("=" * 60)


def test_hypotheses(df):
    """
    Test predefined hypotheses using statistical methods:
    - Correlation analysis
    - T-tests for group comparisons
    - ANOVA for multiple group comparisons
    - Trend analysis
    """

    hypotheses_results = []

    # ============================================
    # HYPOTHESIS 1: Income vs Purchase Amount
    # ============================================
    print(" HYPOTHESIS 1: Higher income → Larger purchases  ")
    print("-" * 40)

    if 'income' in df.columns and 'purchase_amount' in df.columns:
        # Calculate Pearson correlation
        corr_income_purchase = df['income'].corr(df['purchase_amount'])

        # Split into high and low income groups based on median
        median_income = df['income'].median()
        high_income = df[df['income'] > median_income]['purchase_amount']
        low_income = df[df['income'] <= median_income]['purchase_amount']

        # Perform independent t-test
        from scipy.stats import ttest_ind
        t_stat, p_value = ttest_ind(high_income.dropna(), low_income.dropna(), equal_var=False)

        # Display results
        print(f"   Correlation: {corr_income_purchase:.3f}")
        print(f"   T-statistic: {t_stat:.3f}, P-value: {p_value:.4f}")

        # Interpret results (alpha = 0.05)
        if p_value < 0.05:
            print(f"    REJECT NULL: Significant difference between groups  ")
            conclusion = "Supported"
        else:
            print(f"    FAIL TO REJECT: No significant difference  ")
            conclusion = "Not Supported"

        # Record results
        hypotheses_results.append({
            'Hypothesis': 'H1: Higher income → Larger purchases',
            'Test': 'Independent t-test',
            'Statistic': t_stat,
            'P-value': p_value,
            'Conclusion': conclusion
        })

    # ============================================
    # HYPOTHESIS 2: Membership vs Satisfaction
    # ============================================
    print(f"\n HYPOTHESIS 2: Premium members → Higher satisfaction ")
    print("-" * 40)

    if 'membership_type' in df.columns and 'satisfaction_score' in df.columns:
        # Group data by membership type
        membership_groups = {}
        for membership in df['membership_type'].unique():
            membership_groups[membership] = df[df['membership_type'] == membership]['satisfaction_score'].dropna()

        # Perform ANOVA to compare means across multiple groups
        from scipy.stats import f_oneway
        anova_result = f_oneway(*membership_groups.values())

        print(f"   F-statistic: {anova_result.statistic:.3f}, P-value: {anova_result.pvalue:.4f}")

        if anova_result.pvalue < 0.05:
            print(f"    REJECT NULL: Significant differences between membership types  ")

            # Calculate and display group means
            group_means = df.groupby('membership_type')['satisfaction_score'].mean()
            print(f"   Group means: {group_means.to_dict()}")

            conclusion = "Supported"
        else:
            print(f"    FAIL TO REJECT: No significant differences  ")
            conclusion = "Not Supported"

        hypotheses_results.append({
            'Hypothesis': 'H2: Premium members → Higher satisfaction',
            'Test': 'ANOVA',
            'Statistic': anova_result.statistic,
            'P-value': anova_result.pvalue,
            'Conclusion': conclusion
        })

    # ============================================
    # HYPOTHESIS 3: Age vs Purchase Frequency
    # ============================================
    print(f"\n HYPOTHESIS 3: Older age → Lower purchase frequency ")
    print("-" * 40)

    if 'age' in df.columns and 'purchase_frequency' in df.columns:
        # Calculate correlation
        corr_age_freq = df['age'].corr(df['purchase_frequency'])

        # Create age groups for trend analysis
        df['age_group'] = pd.cut(df['age'], bins=[18, 30, 45, 60, 100],
                                 labels=['18-30', '31-45', '46-60', '60+'])

        # Calculate means by age group
        age_group_means = df.groupby('age_group')['purchase_frequency'].mean()

        print(f"   Correlation: {corr_age_freq:.3f}")
        print(f"   Age group means: {age_group_means.to_dict()}")

        # Determine if hypothesis is supported
        # Criteria: Negative correlation AND decreasing trend across age groups
        if corr_age_freq < -0.1 and age_group_means.is_monotonic_decreasing:
            conclusion = "Supported"
            print(f"    Trend supports hypothesis")
        else:
            conclusion = "Not Supported"
            print(f"    Trend does not support hypothesis  ")

        hypotheses_results.append({
            'Hypothesis': 'H3: Older age → Lower purchase frequency',
            'Test': 'Correlation & Trend analysis',
            'Statistic': corr_age_freq,
            'P-value': 'N/A',
            'Conclusion': conclusion
        })

    # ============================================
    # CREATE SUMMARY TABLE OF ALL HYPOTHESIS TESTS
    # ============================================
    results_df = pd.DataFrame(hypotheses_results)
    print(f"\n HYPOTHESIS TESTING SUMMARY: ")
    print("-" * 60)
    print(results_df.to_string(index=False))

    return results_df


# Execute hypothesis testing
results_df = test_hypotheses(df)

# ============================================
# 11. SEGMENTATION ANALYSIS
# ============================================

print(f"\n SEGMENTATION ANALYSIS ")
print("=" * 60)


def perform_segmentation_analysis(df):
    """
    Analyze differences between key customer segments:
    - Regional segments
    - Membership type segments
    - Visual comparison of segment metrics
    """

    segments = []

    # ============================================
    # REGION SEGMENT ANALYSIS
    # ============================================
    if 'region' in df.columns:
        print(" REGION SEGMENT ANALYSIS: ")
        print("-" * 40)

        # Aggregate statistics by region
        region_stats = df.groupby('region').agg({
            'purchase_amount': ['mean', 'median', 'std', 'count'],  # Multiple statistics
            'satisfaction_score': 'mean',
            'income': 'median'
        }).round(2)

        print(region_stats)
        segments.append('region')

    # ============================================
    # MEMBERSHIP TYPE SEGMENT ANALYSIS
    # ============================================
    if 'membership_type' in df.columns:
        print(f"\n MEMBERSHIP SEGMENT ANALYSIS:")
        print("-" * 40)

        membership_stats = df.groupby('membership_type').agg({
            'purchase_amount': ['mean', 'sum', 'count'],
            'customer_lifetime': 'mean',
            'website_visits': 'mean'
        }).round(2)

        print(membership_stats)

        # ============================================
        # VISUALIZE SEGMENT DIFFERENCES
        # ============================================
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Define metrics to visualize
        metrics = ['purchase_amount', 'customer_lifetime', 'website_visits']
        metric_names = ['Purchase Amount', 'Customer Lifetime', 'Website Visits']

        for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
            if metric in df.columns:
                # Group data by membership type
                group_data = df.groupby('membership_type')[metric].mean()

                # Create bar chart
                colors = ['lightblue', 'lightgreen', 'salmon'][:len(group_data)]
                axes[idx].bar(range(len(group_data)), group_data.values, color=colors, edgecolor='black')

                axes[idx].set_title(f'Average {name} by Membership', fontsize=11, fontweight='bold')
                axes[idx].set_xlabel('Membership Type', fontsize=10)
                axes[idx].set_ylabel(f'Average {name}', fontsize=10)
                axes[idx].set_xticks(range(len(group_data)))
                axes[idx].set_xticklabels(group_data.index, rotation=45)
                axes[idx].grid(True, alpha=0.3, axis='y')

                # Add value labels on bars
                for bar_idx, value in enumerate(group_data.values):
                    axes[idx].text(bar_idx, value, f'{value:.1f}',
                                   ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.suptitle('Membership Segment Performance Comparison', fontsize=14, fontweight='bold', y=1.05)
        plt.tight_layout()
        plt.savefig('segmentation_analysis.png', dpi=100, bbox_inches='tight')
        plt.show()


# Perform segmentation analysis
perform_segmentation_analysis(df)

# ============================================
# 12. DATA QUALITY ISSUES SUMMARY
# ============================================

print(f"\n  DATA QUALITY ISSUES SUMMARY ")
print("=" * 60)


def summarize_data_issues(df, num_cols):
    """
    Summarize all identified data quality issues:
    - Missing values
    - Outliers
    - Data type issues
    - Inconsistencies
    """

    issues = []

    # ============================================
    # CHECK FOR MISSING VALUES
    # ============================================
    missing_cols = df.isnull().sum()
    missing_cols = missing_cols[missing_cols > 0]

    if len(missing_cols) > 0:
        for col, count in missing_cols.items():
            pct = (count / len(df)) * 100
            issues.append(f"Missing values: {col} has {count} missing values ({pct:.1f}%)")

    # ============================================
    # CHECK FOR OUTLIERS
    # ============================================
    if 'outlier_df' in globals() and outlier_df is not None:
        for _, row in outlier_df.iterrows():
            issues.append(f"Outliers: {row['column']} has {row['outliers_count']} outliers ({row['outliers_pct']}%)")

    # ============================================
    # CHECK FOR ZERO OR NEGATIVE VALUES WHERE INAPPROPRIATE
    # ============================================
    problematic_cols = ['age', 'income', 'purchase_amount']
    for col in problematic_cols:
        if col in df.columns:
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                issues.append(f"Negative values: {col} has {negative_count} negative values")

            if col == 'age':
                # Check for unrealistic ages
                unrealistic_age = ((df[col] < 0) | (df[col] > 120)).sum()
                if unrealistic_age > 0:
                    issues.append(f"Unrealistic ages: {col} has {unrealistic_age} values outside 0-120 range")

    # ============================================
    # CHECK DATA TYPE ISSUES
    # ============================================
    for col in df.columns:
        # Check for mixed data types
        if df[col].dtype == 'object':
            # Try to convert to numeric to see if it's actually numeric data stored as string
            try:
                pd.to_numeric(df[col])
                issues.append(f"Data type issue: {col} appears to be numeric but stored as object")
            except:
                pass

    # ============================================
    # DISPLAY ISSUES
    # ============================================
    if issues:
        print(" IDENTIFIED DATA QUALITY ISSUES:")
        print("-" * 40)
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")

        print(f"\n Total issues identified: {len(issues)}")

        # Create summary visualization
        fig, ax = plt.subplots(figsize=(10, 6))

        # Categorize issues
        issue_categories = {
            'Missing Values': sum(1 for i in issues if 'Missing' in i),
            'Outliers': sum(1 for i in issues if 'Outliers' in i),
            'Data Quality': sum(1 for i in issues if 'Negative' in i or 'Unrealistic' in i),
            'Data Type': sum(1 for i in issues if 'Data type' in i)
        }

        # Create bar chart
        bars = ax.bar(issue_categories.keys(), issue_categories.values(),
                      color=['salmon', 'gold', 'lightblue', 'lightgreen'])

        ax.set_title('Data Quality Issues by Category', fontsize=14, fontweight='bold')
        ax.set_ylabel('Number of Issues', fontsize=12)
        ax.set_xlabel('Issue Category', fontsize=12)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('data_quality_issues.png', dpi=100, bbox_inches='tight')
        plt.show()

    else:
        print(" No major data quality issues identified!")


# Summarize data quality issues
summarize_data_issues(df, num_cols)

# ============================================
# 13. EDA SUMMARY AND RECOMMENDATIONS
# ============================================
print(f"\n EDA SUMMARY AND RECOMMENDATIONS")
print("=" * 60)


def generate_eda_summary(df, num_cols, cat_cols, date_cols):
    """
    Generate comprehensive EDA summary with key findings and recommendations
    """

    print(" KEY FINDINGS:")
    print("-" * 40)

    findings = []

    # Dataset overview
    findings.append(f"1. Dataset contains {len(df)} rows and {len(df.columns)} columns")
    findings.append(f"2. {len(num_cols)} numerical variables, {len(cat_cols)} categorical variables")

    # Date columns info
    if date_cols:
        findings.append(f"3. {len(date_cols)} datetime variables found")

    # Data quality findings
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        findings.append(f"4. Found {missing_count} missing values across the dataset")
    else:
        findings.append("4. No missing values found in the dataset")

    # Distribution findings
    if 'age' in df.columns and df['age'].notna().sum() > 0:
        age_skew = df['age'].skew()
        if abs(age_skew) > 0.5:  # More reasonable threshold
            findings.append(
                f"5. Age distribution is {'positively' if age_skew > 0 else 'negatively'} skewed (skewness: {age_skew:.2f})")

    # Correlation findings - SAFE CHECK
    try:
        if 'corr_matrix' in globals() and corr_matrix is not None:
            # Make sure corr_matrix is a DataFrame
            if hasattr(corr_matrix, 'abs'):
                # Calculate only for numeric columns
                numeric_corr = corr_matrix.select_dtypes(include=[np.number])
                if not numeric_corr.empty:
                    strong_corr_count = ((numeric_corr.abs() > 0.7).sum().sum() - len(numeric_corr)) / 2
                    if strong_corr_count > 0:
                        findings.append(f"6. Found {int(strong_corr_count)} pairs with strong correlation (|r| > 0.7)")
    except Exception as e:
        findings.append("6. Correlation analysis completed")

    # Hypothesis testing findings - SAFE CHECK
    try:
        if 'results_df' in globals() and results_df is not None and not results_df.empty:
            # Check if 'Conclusion' column exists
            if 'Conclusion' in results_df.columns:
                supported_hypotheses = results_df[results_df['Conclusion'] == 'Supported']
                findings.append(f"7. {len(supported_hypotheses)} out of {len(results_df)} hypotheses were supported")
            else:
                # Try to find alternative column names
                for col in results_df.columns:
                    if 'conclusion' in col.lower() or 'result' in col.lower():
                        # Count supported hypotheses (case-insensitive)
                        supported_mask = results_df[col].astype(str).str.lower().str.contains('support')
                        supported_count = supported_mask.sum()
                        findings.append(f"7. {supported_count} out of {len(results_df)} hypotheses were supported")
                        break
                else:
                    findings.append(f"7. {len(results_df)} hypotheses were tested")
        else:
            findings.append("7. Hypothesis testing was performed")
    except Exception as e:
        findings.append("7. Hypothesis testing results available in detailed report")

    # Add outlier findings if available
    if 'outlier_df' in globals() and outlier_df is not None and not outlier_df.empty:
        if 'outliers_count' in outlier_df.columns:
            total_outliers = outlier_df['outliers_count'].sum()
            findings.append(f"8. Identified {total_outliers} outliers across {len(outlier_df)} variables")

    # Print findings
    for finding in findings:
        print(f"   • {finding}")

    print(f"\n RECOMMENDATIONS FOR FURTHER ANALYSIS:")
    print("-" * 40)

    recommendations = [
        "1. Address missing values using appropriate imputation methods",
        "2. Investigate and handle outliers based on business context",
        "3. Consider feature engineering based on identified relationships",
        "4. Validate findings with domain experts",
        "5. Consider dimensionality reduction if many correlated variables exist",
        "6. Segment customers further for targeted analysis",
        "7. Build predictive models for key outcomes (churn, purchase amount)",
        "8. Conduct A/B testing to validate causal relationships"
    ]

    for rec in recommendations:
        print(f"   • {rec}")

    print(f"\n NEXT STEPS:")
    print("-" * 40)
    print("   1. Data Cleaning: Handle missing values and outliers")
    print("   2. Feature Engineering: Create new features based on insights")
    print("   3. Modeling: Build predictive models")
    print("   4. Validation: Test model performance and business impact")
    print("   5. Deployment: Implement insights in business processes")


# Generate final summary
generate_eda_summary(df, num_cols, cat_cols, date_cols)

print("\n" + "=" * 60)
print(" EXPLORATORY DATA ANALYSIS COMPLETED SUCCESSFULLY!")
print("=" * 60)

# ============================================
# 14. SAVE EDA RESULTS
# ============================================

print(f"\n SAVING EDA RESULTS")
print("=" * 60)


def save_eda_results(df, stats_df, results_df=None):
    """Save EDA results and visualizations for future reference"""

    # Save cleaned dataset
    try:
        df.to_csv('cleaned_dataset.csv', index=False)
        print(" Saved: cleaned_dataset.csv")
    except Exception as e:
        print(f" Could not save cleaned_dataset.csv: {e}")

    # Save statistics summary
    try:
        if stats_df is not None:
            stats_df.to_csv('descriptive_statistics.csv')
            print(" Saved: descriptive_statistics.csv")
    except Exception as e:
        print(f"  Could not save descriptive_statistics.csv: {e}")

    # Save hypothesis testing results - SAFE CHECK
    try:
        if results_df is not None and not results_df.empty:
            results_df.to_csv('hypothesis_testing_results.csv', index=False)
            print(" Saved: hypothesis_testing_results.csv")
        else:
            print("  No hypothesis testing results to save")
    except Exception as e:
        print(f" Could not save hypothesis_testing_results.csv: {e}")

    # Save correlation matrix - SAFE CHECK
    try:
        if 'corr_matrix' in globals() and corr_matrix is not None:
            corr_matrix.to_csv('correlation_matrix.csv')
            print(" Saved: correlation_matrix.csv")
        else:
            print("  No correlation matrix to save")
    except Exception as e:
        print(f"  Could not save correlation_matrix.csv: {e}")

    print(" All visualizations have been saved as PNG files ")
    print(" Summary reports are available in CSV format ")


# Save results with safe checks
try:
    # Check if results_df exists
    if 'results_df' not in globals():
        results_df = None

    # Check if stats_df exists
    if 'stats_df' not in globals():
        stats_df = None

    save_eda_results(df, stats_df, results_df)
except Exception as e:
    print(f" Error saving results: {e}")
    print("Saving basic results only...")
    try:
        df.to_csv('cleaned_dataset.csv', index=False)
        print("  Saved: cleaned_dataset.csv (basic)")
    except:
        print(" Could not save any results ")

print("\n" + "=" * 60)
print("EDA PROCESS COMPLETE! Check generated files for results.")
#print("")
print("=" * 60)