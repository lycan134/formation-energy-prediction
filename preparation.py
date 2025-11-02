import pandas as pd

def classify_stability(e_hull):
    """Classify materials into stability categories based on energy above hull."""
    if e_hull <= 0.025:
        return "Stable"
    elif e_hull <= 0.100:
        return "Metastable"
    else:
        return "Unstable"


def load_and_filter_data(csv_path):
    """Load data, remove outliers, and add stability labels."""
    df = pd.read_csv(csv_path)

    # === Filter based on formation energy (±5σ) ===
    mean = df['formation_energy_per_atom'].mean()
    std = df['formation_energy_per_atom'].std()
    lower, upper = mean - 5 * std, mean + 5 * std

    df_filtered = df[(df['formation_energy_per_atom'] >= lower) & 
                     (df['formation_energy_per_atom'] <= upper)].copy()
    df_filtered.reset_index(drop=True, inplace=True)

    # === Add stability label ===
    df_filtered["stability_label"] = df_filtered["energy_above_hull"].apply(classify_stability)

    # === Keep relevant columns ===
    columns_to_keep = [
        'material_id', 'formula_pretty', 'formation_energy_per_atom', 'energy_above_hull',
        'crystal_system', 'number', 'symbol', 'point_group',
        'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si',
        'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni',
        'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo',
        'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba',
        'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
        'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po',
        'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf',
        'Es', 'Fm', 'Md', 'No', 'Lr', 'n_atoms', 'n_elements', 'avg_atomic_mass', 'en_mean',
        'en_max', 'en_min', 'en_range', 'avg_covalent_radius', 'ea_mean', 'ea_max', 'ea_min', 'ea_range', 'stability_label'
    ]

    df_filtered = df_filtered[columns_to_keep]

    # === Keep only lowest-energy duplicates ===
    df_lowest_energy = (
        df_filtered.sort_values('formation_energy_per_atom')
        .drop_duplicates(subset=['formula_pretty', 'number', 'stability_label'], keep='first')
        .reset_index(drop=True)
    )

    return df_lowest_energy


def preprocess_material_data(df, fill_strategy="zero"):
    """Preprocess materials data for ML training."""
    # Drop rows with missing target
    df = df.dropna(subset=['formation_energy_per_atom']).copy()

    # One-hot encode space group & stability label
    df_encoded = pd.get_dummies(df, columns=["number", "stability_label"], prefix=["sg", "stab"])

    # 🔧 Convert bool columns (from one-hot encoding) to int
    for col in df_encoded.select_dtypes(include=['bool']).columns:
        df_encoded[col] = df_encoded[col].astype(int)

    # Identify dummy columns
    dummy_cols = [col for col in df_encoded.columns if col.startswith("sg_") or col.startswith("stab_")]

    # Define numerical feature columns
    feature_cols = [
        'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si',
        'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni',
        'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo',
        'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba',
        'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
        'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po',
        'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf',
        'Es', 'Fm', 'Md', 'No', 'Lr', 'n_atoms', 'n_elements', 'avg_atomic_mass', 'en_mean',
        'en_max', 'en_min', 'en_range', 'avg_covalent_radius', 'ea_mean', 'ea_max', 'ea_min', 'ea_range'
    ]

    # Combine all features
    all_features = feature_cols + dummy_cols

    # Handle missing values
    if fill_strategy == "zero":
        X = df_encoded[all_features].fillna(0)
    elif fill_strategy == "mean":
        X = df_encoded[all_features].fillna(df_encoded[all_features].mean())
    else:
        raise ValueError("fill_strategy must be 'zero' or 'mean'")

    y = df_encoded['formation_energy_per_atom']

    return X, y


# ===========================
# 🚀 MAIN EXECUTION PIPELINE
# ===========================
if __name__ == "__main__":
    data_path = "data/MP_queried_data_featurized_w_additional_acr_ae_en.csv"

    # Step 1: Filter + Label + Deduplicate
    df_clean = load_and_filter_data(data_path)
    print(f"✅ Cleaned dataset shape: {df_clean.shape}")

    # Step 2: Preprocess into ML-ready tensors
    X, y = preprocess_material_data(df_clean, fill_strategy="zero")
    print(f"✅ Features shape: {X.shape}, Target shape: {y.shape}")

    # Optional: Save preprocessed data
    X.to_csv("data/X_preprocessed.csv", index=False)
    y.to_csv("data/y_preprocessed.csv", index=False)
    print("💾 Saved preprocessed data to CSV.")
