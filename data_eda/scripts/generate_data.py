import pandas as pd
import numpy as np
from pathlib import Path

def generate_kenyan_msme_data(n_samples=5000):
    """
    Generates a realistic synthetic dataset of Kenyan MSMEs.
    The 'distress_label' is logically derived from financial ratios, 
    not randomly assigned, to ensure the ML model has real patterns to learn.
    """
    np.random.seed(42)

    # 1. Define Kenyan Context Variables
    counties = ['Nairobi', 'Mombasa', 'Kisumu', 'Nakuru', 'Eldoret', 
                'Thika', 'Nyeri', 'Machakos', 'Meru', 'Kakamega']
    sectors = ['Retail', 'Agriculture', 'Transport', 'Manufacturing', 
               'Food Services', 'Construction', 'Digital Commerce', 'Services']

    # 2. Generate Base Features (Realistic KES scales using Lognormal distribution)
    business_age_months = np.random.randint(1, 120, n_samples) # 1 month to 10 years
    employees = np.random.choice([1, 2, 3, 5, 10, 15, 25, 50], n_samples, 
                                 p=[0.30, 0.25, 0.15, 0.10, 0.08, 0.05, 0.04, 0.03])
    
    # Monthly Revenue: Mean around ~150,000 KES
    monthly_revenue_kes = np.round(np.random.lognormal(mean=11.8, sigma=1.0, size=n_samples), 2)
    
    # Monthly Expenses: Typically 60% to 110% of revenue (some businesses operate at a loss)
    expense_multiplier = np.random.uniform(0.6, 1.15, n_samples)
    monthly_expenses_kes = np.round(monthly_revenue_kes * expense_multiplier, 2)
    
    # Assets & Liabilities
    total_assets_kes = np.round(np.random.lognormal(mean=13.5, sigma=1.2, size=n_samples), 2)
    total_liabilities_kes = np.round(total_assets_kes * np.random.uniform(0.1, 0.9, n_samples), 2)
    
    # Loan Amount: 60% of MSMEs have a loan
    has_loan = np.random.rand(n_samples) > 0.4
    loan_amount_kes = np.round(np.where(has_loan, np.random.lognormal(10.5, 1.2, n_samples), 0), 2)
    
    # M-Pesa Volume: Highly correlated with revenue in Kenya (50% to 90% of revenue flows through mobile money)
    mpesa_volume_kes = np.round(monthly_revenue_kes * np.random.uniform(0.5, 0.95, n_samples), 2)

    # 3. Create the DataFrame
    df = pd.DataFrame({
        'business_id': [f"MSME_{i:05d}" for i in range(n_samples)],
        'county': np.random.choice(counties, n_samples),
        'sector': np.random.choice(sectors, n_samples),
        'business_age_months': business_age_months,
        'employees': employees,
        'monthly_revenue_kes': monthly_revenue_kes,
        'monthly_expenses_kes': monthly_expenses_kes,
        'total_assets_kes': total_assets_kes,
        'total_liabilities_kes': total_liabilities_kes,
        'loan_amount_kes': loan_amount_kes,
        'mpesa_volume_kes': mpesa_volume_kes,
    })

    # 4. Feature Engineering (Creating the predictors for distress)
    # Add 1 to denominator to prevent division by zero
    df['expense_ratio'] = df['monthly_expenses_kes'] / (df['monthly_revenue_kes'] + 1)
    df['debt_to_asset_ratio'] = df['total_liabilities_kes'] / (df['total_assets_kes'] + 1)
    df['age_risk_factor'] = 1 / (df['business_age_months'] + 1) # Younger businesses are riskier
    df['mpesa_dependency'] = df['mpesa_volume_kes'] / (df['monthly_revenue_kes'] + 1)

    # 5. Calculate Distress Score (The "Ground Truth" Logic)
    # Weighted formula: High expenses (40%), High debt (30%), Young age (20%), Low M-Pesa velocity (10%)
    distress_score = (
        (df['expense_ratio'] * 0.40) + 
        (df['debt_to_asset_ratio'] * 0.30) + 
        (df['age_risk_factor'] * 0.20) + 
        ((1 - df['mpesa_dependency']) * 0.10)
    )

    # Add realistic noise so it's not perfectly predictable
    noise = np.random.normal(0, 0.08, n_samples)
    final_score = distress_score + noise

    # Convert to binary label: 1 = Distressed, 0 = Stable
    # We use the median as the threshold to ensure a roughly 50/50 split for balanced training
    df['distress_label'] = (final_score > np.median(final_score)).astype(int)

    # 6. Clean up helper columns before saving (optional, but keeps the raw data clean)
    # We will save them anyway so Denis can use them for EDA, but the model will use the base features.
    
    return df


if __name__ == "__main__":
    print("🚀 Generating realistic Kenyan MSME synthetic dataset...")
    df = generate_kenyan_msme_data(n_samples=5000)

    # Ensure target directories exist
    Path("data_eda/synthetic").mkdir(parents=True, exist_ok=True)
    Path("data_eda/raw").mkdir(parents=True, exist_ok=True)

    # Save to Denis's folder (so he can use it for EDA)
    df.to_csv("data_eda/synthetic/sample_msme_data.csv", index=False)
    df.to_csv("data_eda/raw/synthetic_msme_data.csv", index=False)
    
    # Also save a copy in Edusei's folder for quick ML testing
    Path("ml_models/data").mkdir(parents=True, exist_ok=True)
    df.to_csv("ml_models/data/synthetic_msme_data.csv", index=False)

    print("✅ Success! Dataset generated.")
    print(f"📊 Total Records: {len(df)}")
    print(f"📉 Distress Rate: {df['distress_label'].mean():.2%}")
    print(f"💰 Avg Monthly Revenue: KES {df['monthly_revenue_kes'].mean():,.2f}")
    print(f"📁 Saved to: data_eda/synthetic/ and ml_models/data/")