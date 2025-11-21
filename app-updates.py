import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt 
import re
import time

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Plant Breeding Analytics",
    page_icon="🧬",
    layout="wide"
)

# --- HELPER FUNCTIONS ---

def extract_genotype_name(key, prefix="Genotype"):
    """
    Extracts clean genotype name from mixed model random effect keys.
    Handles formats like: "Genotype[C(Hybrid)[HyTTec Velocity]]"
    Returns: "HyTTec Velocity"
    """
    # First, check if this key starts with the expected prefix
    if not key.startswith(f"{prefix}["):
        return key

    # Remove the prefix part
    key_content = key[len(prefix)+1:]  # Remove "Genotype[" or "Male_GCA[" etc.

    # Look for pattern "C(column_name)[actual_value]"
    # This handles the statsmodels categorical variable encoding
    match = re.search(r'C\([^)]+\)\[([^\]]+)\]', key_content)
    if match:
        return match.group(1)

    # Fallback: just extract content from the outermost brackets
    if key_content.endswith(']'):
        key_content = key_content[:-1]

    # If there's still a bracket structure, get the innermost value
    inner_match = re.search(r'\[([^\]]+)\]$', key_content)
    if inner_match:
        return inner_match.group(1)

    return key_content

def clean_curveballs(df, traits):
    log = []
    df_clean = df.copy()

    for trait in traits:
        # 1. Force Numeric
        if not pd.api.types.is_numeric_dtype(df_clean[trait]):
            df_clean[trait] = pd.to_numeric(df_clean[trait], errors='coerce')
            log.append(f"⚠️ Column '{trait}' contained non-numeric data. Values converted to NA.")

        # 2. Negative Value Detection
        neg_mask = df_clean[trait] < 0
        neg_count = neg_mask.sum()
        if neg_count > 0:
            df_clean.loc[neg_mask, trait] = np.nan
            log.append(f"🚨 **CRITICAL:** Found {neg_count} negative values in '{trait}'. Set to NA.")

    return df_clean, log

def run_hybrid_model(df, trait, gen_col, row_col, col_col, expt_col, analyze_separate):
    """
    Runs LMM for Genotype Performance (BLUPs), including spatial correction.
    """
    progress_bar = st.progress(0, text="Initializing Model...")

    results_container = []
    summary_stats = []
    debug_log = []

    # Define datasets to iterate over
    if analyze_separate:
        datasets = [(expt, df[df[expt_col] == expt].copy()) for expt in df[expt_col].unique()]
        total_runs = len(datasets)
    else:
        datasets = [("Combined Analysis", df.copy())]
        total_runs = 1

    for i, (run_name, model_data) in enumerate(datasets):
        step_val = int((i / total_runs) * 100)

        progress_bar.progress(step_val, text=f"Processing **{run_name}**: Preparing Matrix...")

        # Initialize status tracking
        convergence_status = "❌ Failed"
        convergence_message = ""
        optimizer_used = "None"

        try:
            model_data = model_data.dropna(subset=[trait, gen_col, row_col, col_col]).copy()

            # Ensure enough data for modeling
            if model_data[gen_col].nunique() < 2:
                 debug_log.append(f"ERROR in {run_name}: Not enough unique genotypes ({model_data[gen_col].nunique()}) to run the model.")
                 convergence_message = f"Only {model_data[gen_col].nunique()} unique genotype(s). Need at least 2."
                 summary_stats.append({
                     'Experiment ID': run_name,
                     'Status': convergence_status,
                     'Message': convergence_message,
                     'Optimizer': optimizer_used,
                     'Mean': 'N/A',
                     'CV (%)': 'N/A',
                     'Genotype Var (Vg)': 'N/A',
                     'Residual Var (Ve)': 'N/A',
                     'Est. Heritability (H2, %)': 'N/A',
                     'Plots': len(model_data),
                     'Unique Genotypes': model_data[gen_col].nunique()
                 })
                 continue

            # Create Unique Spatial IDs
            model_data["Unique_Row"] = model_data[expt_col].astype(str) + "_" + model_data[row_col].astype(str)
            model_data["Unique_Col"] = model_data[expt_col].astype(str) + "_" + model_data[col_col].astype(str)
            model_data["Global_Group"] = 1

            # Formula Logic
            if not analyze_separate and model_data[expt_col].nunique() > 1:
                # Fixed Effect: Experiment ID (used when combining multiple experiments)
                formula = f"{trait} ~ C({expt_col})"
                fixed_effects_formula = f"Fixed Effects: {trait} ~ C({expt_col}) (Corrects for baseline differences between sites/years)"
            else:
                # Fixed Effect: Intercept (Overall Mean of the single experiment)
                formula = f"{trait} ~ 1"
                fixed_effects_formula = f"Fixed Effects: {trait} ~ Intercept (Overall average trait performance)"

            # Variance Components (Random Effects)
            vc = {
                "Genotype": f"0 + C({gen_col})",
                "SpatialRow": f"0 + C(Unique_Row)",
                "SpatialCol": f"0 + C(Unique_Col)"
            }
            random_effects_formula = f"Random Effects: (1|{gen_col}) + (1|Unique_Row) + (1|Unique_Col)"

            debug_log.append(f"--- {run_name} Complete Model Formula ---\n{fixed_effects_formula}\n{random_effects_formula}")

            progress_bar.progress(
                min(step_val + 10, 100),
                text=f"Processing **{run_name}**: Fitting Spatial Model (REML)..."
            )


            # Try multiple optimizers for robustness
            model = smf.mixedlm(formula, model_data, groups="Global_Group", vc_formula=vc)
            result = None
            optimizers = ['powell', 'lbfgs', 'bfgs']

            for opt in optimizers:
                try:
                    result = model.fit(method=opt, reml=True)
                    optimizer_used = opt
                    # Check if convergence was successful
                    if hasattr(result, 'converged'):
                        if result.converged:
                            convergence_status = "✓ Good"
                            convergence_message = "Model converged successfully"
                            break
                        else:
                            convergence_status = "⚠️ Caution"
                            convergence_message = "Weak convergence - results may be less reliable"
                    else:
                        # If no converged attribute, assume success if no error
                        convergence_status = "✓ Good"
                        convergence_message = "Model fitted successfully"
                        break
                except Exception as opt_error:
                    debug_log.append(f"Optimizer {opt} failed for {run_name}: {str(opt_error)}")
                    if opt == optimizers[-1]:  # Last optimizer failed
                        raise Exception(f"All optimizers failed. Last error: {str(opt_error)}")
                    continue

            if result is None:
                raise Exception("Model fitting failed with all optimizers")

            progress_bar.progress(
                min(step_val + 20, 100),
                text=f"Processing **{run_name}**: Extracting BLUPs..."
            )


            # Extract & Clean Genotype BLUPs
            re_dict = result.random_effects[1]
            geno_blups = {}

            for key, val in re_dict.items():
                if key.startswith("Genotype["):
                    # Use improved extraction function
                    clean_name = extract_genotype_name(key, "Genotype")
                    geno_blups[clean_name] = val

            # Create DataFrame
            temp_df = pd.DataFrame.from_dict(geno_blups, orient='index', columns=[f'BLUP_{trait}'])
            temp_df.index.name = 'Genotype'

            # Add Intercept (Fixed Effect Mean)
            intercept = result.params.get('Intercept', 0)
            temp_df[f'Predicted_{trait}'] = temp_df[f'BLUP_{trait}'] + intercept
            temp_df['Analysis_Group'] = run_name

            results_container.append(temp_df)
            debug_log.append(f"--- {run_name} Summary ---\n{result.summary().as_text()}")

            # Calculate Experiment-Wide Statistics
            # variance components returned in same order as vc dict
            v_g = None
            try:
                if hasattr(result, "vcomp") and len(result.vcomp) > 0:
                    v_g = result.vcomp[0]  # Genotype variance
            except:
                v_g = None

            v_e = result.scale  # residual variance

            h2 = 'N/A'
            h2_val = None

            # Check for problematic variance components
            if v_g is None or v_g <= 0:
                convergence_status = "⚠️ Caution"
                if v_g is None:
                    convergence_message = "Genetic variance could not be estimated"
                else:
                    convergence_message = "No genetic variation detected (Vg ≤ 0)"
                h2 = 'N/A'
            elif v_e is not None and model_data[gen_col].nunique() > 1:
                # Avg replication estimate based on total plots / total unique genotypes
                avg_plots_per_geno = model_data.groupby(gen_col).size().mean()

                # Formula: H^2 = Vg / ( Vg + Ve / n_plots)
                h2_val = v_g / (v_g + (v_e / avg_plots_per_geno))
                h2 = round(h2_val * 100, 1)

                # Update status based on heritability
                if convergence_status == "✓ Good":
                    if h2_val < 0.10:  # Less than 10%
                        convergence_status = "⚠️ Caution"
                        convergence_message = f"Low heritability ({h2}%) - weak genetic signal"
                    elif h2_val > 0.95:  # Greater than 95%
                        convergence_status = "⚠️ Caution"
                        convergence_message = f"Unusually high heritability ({h2}%) - check data quality"

            mean = model_data[trait].mean()
            std_dev = model_data[trait].std()
            cv = (std_dev / mean) * 100 if mean != 0 else 0

            summary_stats.append({
                'Experiment ID': run_name,
                'Status': convergence_status,
                'Message': convergence_message,
                'Optimizer': optimizer_used,
                'Mean': round(mean, 3),
                'CV (%)': round(cv, 1),
                'Genotype Var (Vg)': round(v_g, 3) if v_g is not None and v_g > 0 else 'N/A',
                'Residual Var (Ve)': round(v_e, 3) if v_e is not None else 'N/A',
                'Est. Heritability (H2, %)': h2,
                'Plots': len(model_data),
                'Unique Genotypes': model_data[gen_col].nunique()
            })

        except Exception as e:
            error_msg = str(e)
            # Make error messages more user-friendly
            if "singular" in error_msg.lower():
                user_msg = "Not enough variation in data or insufficient replication"
            elif "convergence" in error_msg.lower():
                user_msg = "Model couldn't find a stable solution"
            else:
                user_msg = error_msg[:100]  # Truncate long errors

            debug_log.append(f"ERROR in {run_name}: {error_msg}")

            summary_stats.append({
                'Experiment ID': run_name,
                'Status': "❌ Failed",
                'Message': user_msg,
                'Optimizer': optimizer_used if 'optimizer_used' in locals() else 'None',
                'Mean': 'N/A',
                'CV (%)': 'N/A',
                'Genotype Var (Vg)': 'N/A',
                'Residual Var (Ve)': 'N/A',
                'Est. Heritability (H2, %)': 'N/A',
                'Plots': len(model_data) if 'model_data' in locals() else 0,
                'Unique Genotypes': model_data[gen_col].nunique() if 'model_data' in locals() else 0
            })

    progress_bar.progress(100, text="Finalizing Results...")
    time.sleep(0.5)
    progress_bar.empty()

    stats_df = pd.DataFrame(summary_stats)

    if results_container:
        final_df = pd.concat(results_container)
        return True, final_df, stats_df, "\n".join(debug_log)
    else:
        return False, None, stats_df, "\n".join(debug_log)

def run_parental_model(df, trait, male_col, female_col, row_col, col_col, expt_col):
    """
    REPLACED FUNCTION — Runs GCA per experiment and returns stacked results.

    Returns:
        success (bool),
        df_male (pd.DataFrame) or None,
        df_female (pd.DataFrame) or None,
        debug_log (str)
    """

    progress_bar = st.progress(0, text="Calculating GCA (Parental BLUPs)...")

    male_results = []
    female_results = []
    debug_messages = []

    # Determine experiments present in the provided dataframe
    experiments = df[expt_col].dropna().unique().tolist()
    total = max(1, len(experiments))

    for i, expt in enumerate(experiments):
        progress_bar.progress(int((i / total) * 100), text=f"Running GCA for {expt}...")

        # Work on a per-experiment subset, require non-missing essential cols
        expt_data = df[df[expt_col] == expt].dropna(
            subset=[trait, male_col, female_col, row_col, col_col]
        ).copy()

        if expt_data.empty:
            debug_messages.append(f"[{expt}] Skipped: no valid rows after dropping NA.")
            continue

        # Create Unique Spatial IDs (local to experiment)
        expt_data["Unique_Row"] = expt_data[row_col].astype(str)
        expt_data["Unique_Col"] = expt_data[col_col].astype(str)
        expt_data["Global_Group"] = 1

        # Since we're analyzing a single experiment at a time, use intercept-only fixed effect
        formula = f"{trait} ~ 1"
        vc = {
            "Male_GCA": f"0 + C({male_col})",
            "Female_GCA": f"0 + C({female_col})",
            "SpatialRow": "0 + C(Unique_Row)",
            "SpatialCol": "0 + C(Unique_Col)"
        }
        random_effects_formula = f"Random Effects: (1|{male_col}) + (1|{female_col}) + (1|Unique_Row) + (1|Unique_Col)"
        fixed_effects_formula = "Fixed Effects: Intercept"

        try:
            progress_bar.progress(min(int((i / total) * 100) + 10, 90), text=f"Fitting model for {expt}...")
            model = smf.mixedlm(formula, expt_data, groups="Global_Group", vc_formula=vc)
            result = model.fit(method='powell', reml=True)

            # random_effects[1] holds the dictionary of vc-level effects
            re_dict = result.random_effects[1]

            male_gca = {}
            female_gca = {}

            for key, val in re_dict.items():
                if key.startswith("Male_GCA["):
                    name = extract_genotype_name(key, "Male_GCA")
                    male_gca[name] = val
                elif key.startswith("Female_GCA["):
                    name = extract_genotype_name(key, "Female_GCA")
                    female_gca[name] = val

            # Convert to dataframes, reset index to avoid non-unique index issues
            if male_gca:
                df_m = pd.DataFrame.from_dict(male_gca, orient='index', columns=[f'GCA_{trait}'])
                df_m.index.name = 'Male Parent'
                df_m.reset_index(inplace=True)
                df_m['Group'] = expt
                male_results.append(df_m)
            else:
                debug_messages.append(f"[{expt}] No Male_GCA values extracted.")

            if female_gca:
                df_f = pd.DataFrame.from_dict(female_gca, orient='index', columns=[f'GCA_{trait}'])
                df_f.index.name = 'Female Parent'
                df_f.reset_index(inplace=True)
                df_f['Group'] = expt
                female_results.append(df_f)
            else:
                debug_messages.append(f"[{expt}] No Female_GCA values extracted.")

            debug_messages.append(f"\n--- GCA Model Summary ({expt}) ---\n{result.summary().as_text()}\n")

        except Exception as e:
            debug_messages.append(f"[{expt}] ERROR: {str(e)}")
            # continue to next experiment rather than fail everything
            continue

    progress_bar.progress(100, text="Finalizing GCA Results...")
    time.sleep(0.2)
    progress_bar.empty()

    # Concatenate per-experiment results (if any) and return
    male_df = pd.concat(male_results, ignore_index=True) if male_results else None
    female_df = pd.concat(female_results, ignore_index=True) if female_results else None

    # Final return structure matches original expectations
    success_flag = True if (male_df is not None or female_df is not None) else False
    return success_flag, male_df, female_df, "\n".join(debug_messages)


# --- MAIN APP ---

def main():
    # Streamlit state to store results
    if 'results_df' not in st.session_state:
        st.session_state.results_df = None
    if 'stats_df' not in st.session_state:
        st.session_state.stats_df = None
    if 'debug_log' not in st.session_state:
        st.session_state.debug_log = None
    if 'analysis_mode' not in st.session_state:
        st.session_state.analysis_mode = "Analyze experiments separately"
    if 'trait_ran' not in st.session_state:
        st.session_state.trait_ran = None


    st.title("🧬 Plant Breeding Trial Analytics")
    st.write("Upload your trial CSV. Perform spatial correction and genetic analysis.")

    # --- SIDEBAR ---
    st.sidebar.header("1. Upload Data")
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])
    df = None

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("File Uploaded")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")
            return

    # --- 2. COLUMN MAPPING ---
    col_map = {}
    gca_enabled = False
    if df is not None:
        st.sidebar.divider()
        st.sidebar.header("2. Map Columns")
        st.sidebar.info("Please identify the columns in your file.")

        all_cols = ["Select Column..."] + df.columns.tolist()

        col_map['expt'] = st.sidebar.selectbox("Experiment ID", all_cols)
        col_map['geno'] = st.sidebar.selectbox("Genotype", all_cols)
        col_map['row'] = st.sidebar.selectbox("Plot Row", all_cols)
        col_map['col'] = st.sidebar.selectbox("Plot Column", all_cols)

        # Optional Parent Columns
        with st.sidebar.expander("GCA Parental Lines (Optional)"):
            st.info("GCA analysis can be performed if parental lines are selected for hybrid crops.")
            col_map['male'] = st.selectbox("Male Parent (Required for GCA)", all_cols)
            col_map['female'] = st.selectbox("Female Parent (Required for GCA)", all_cols)

        gca_enabled = (col_map['male'] != "Select Column...") and (col_map['female'] != "Select Column...")

        # Check if mapping is complete (required columns only)
        required_cols = ['expt', 'geno', 'row', 'col']
        if any(col_map[key] == "Select Column..." for key in required_cols):
            st.sidebar.warning("⚠️ You must map the required columns to proceed.")
            return 

        st.sidebar.divider()
        st.sidebar.header("3. Filter & Configure")
        unique_expts = df[col_map['expt']].unique().tolist()
        selected_expts = st.sidebar.multiselect("Select Experiments to Include", unique_expts, default=unique_expts)

        potential_traits = [c for c in df.columns if c not in col_map.values() or c == "Select Column..."]
        selected_traits = st.sidebar.multiselect("Select Phenotypes", potential_traits)

    # --- MAIN LOGIC ---
    if df is None:
        st.info("👈 Upload a CSV file to begin.")
        return

    if not selected_expts or not selected_traits:
        st.sidebar.warning("Please select experiments and traits.")
        return

    # Filter Data
    df_filtered = df[df[col_map['expt']].isin(selected_expts)].copy()
    df_clean, cleaning_log = clean_curveballs(df_filtered, selected_traits)

    # --- TABS ---
    gca_tab_title = f"👨‍👩‍👧 Parental GCA ({'Ready' if gca_enabled else 'Disabled'})"
    tab_qc, tab_spatial, tab_perf, tab_parents = st.tabs([
        "🛡️ QC & Cleaning", 
        "🗺️ Spatial Analysis", 
        "📊 Genotype Performance", 
        gca_tab_title
    ])

    # --- TAB 1: QC ---
    with tab_qc:
        st.header("Data Quality Control")
        if cleaning_log:
            for msg in cleaning_log:
                if "CRITICAL" in msg:
                    st.error(msg)
                else:
                    st.warning(msg)
        else:
            st.success("No statistical anomalies detected in selected traits.")

        st.subheader("Summary Statistics")
        c_stats, c_missing = st.columns(2)

        with c_stats:
            st.markdown(f"**Selected Experiments:** `{len(selected_expts)}`")
            st.markdown(f"**Total Plots:** `{len(df_clean)}`")
            st.markdown(f"**Unique Genotypes:** `{df_clean[col_map['geno']].nunique()}`")

        with c_missing:
            st.subheader("Missing Values Count")
            st.dataframe(df_clean[selected_traits].isnull().sum(), use_container_width=True)

        st.empty() 

    # --- TAB 2: SPATIAL ---
    with tab_spatial:
        st.header("Spatial Field Map")
        c1, c2 = st.columns([1, 3])
        map_trait = c1.selectbox("View Trait", selected_traits, key='map_trait')
        map_expt = c1.selectbox("View Experiment", selected_expts, key='map_expt')

        map_data = df_clean[df_clean[col_map['expt']] == map_expt]
        if not map_data.empty:
            pivot = map_data.pivot_table(index=col_map['row'], columns=col_map['col'], values=map_trait)
            fig = px.imshow(pivot, color_continuous_scale='Viridis', title=f"{map_expt}: {map_trait}", aspect="auto")
            fig.update_yaxes(autorange="reversed")
            c2.plotly_chart(fig, use_container_width=True)
        else:
            c2.warning(f"No data for experiment: {map_expt}")

    # --- TAB 3: GENOTYPE PERFORMANCE ---
    with tab_perf:
        st.header("Genotype Performance Analysis (BLUPs)")
        
        with st.expander("Understanding the Statistical Analysis: Fixed vs. Random Effects"):
            st.markdown("""
            The analysis uses a **Linear Mixed Model (LMM)** to calculate **Best Linear Unbiased Predictions (BLUPs)**. This method is the gold standard for separating true genetic merit from field noise. 
            
            ### Fixed Effects vs. Random Effects
            
            A Mixed Model splits the sources of variation into two types:
            
            * **Fixed Effects (Known, Non-Variable Factors):**
                These are effects we want to *estimate* precisely. They usually represent known, deliberate differences in the experimental design.
                * **Example:** The overall average performance (Intercept) or the average difference between two different testing **environments/experiments** (if combining trials). We assume all genotypes will be affected by these factors in the same way.
            
            * **Random Effects (Unknown, Variable Factors):**
                These are effects we want to *predict* (BLUPs) or account for in the error term. We assume they are drawn from a normal distribution and vary unpredictably.
                * **Genotype:** This is treated as a random effect because we want to predict the **true breeding value (BLUP)** for each genotype, which is a prediction of its performance if tested infinitely.
                * **Spatial Correction (Row/Column):** Field variation (e.g., a wet patch or soil gradient) is random across the field. We model this as a random effect to remove its influence, leading to cleaner genotype estimates.
            
            ### How BLUPs Work (The Prediction)
            
            1.  **Field Correction:** The model uses the spatial random effects (Plot Row and Plot Column) to map the field's environmental noise. This noise is mathematically subtracted from your raw plot data.
            2.  **Genetic Prediction:** The corrected data is then used to predict the value for the Genotype random effect. This prediction is the **BLUP**. The model 'shrinks' the estimates of poorly-replicated or highly variable genotypes toward the overall mean, making the high-performing, consistent genotypes stand out.
            3.  **Final Value:** The final **Predicted Trait Value** reported in the tables is the Genotype BLUP plus the relevant Fixed Effect (e.g., the overall experiment mean).
            """)
        
        col_opts, col_act = st.columns([2, 1])
        perf_trait = col_opts.selectbox("Trait to Analyze", selected_traits, key='perf_trait')
        
        def update_analysis_mode():
            st.session_state.analysis_mode = st.session_state.temp_analysis_mode
            st.session_state.results_df = None

        analysis_mode = col_opts.radio(
            "Analysis Strategy", 
            ["Analyze experiments separately", "Analyze all experiments as one group"],
            key='temp_analysis_mode',
            on_change=update_analysis_mode,
            help="Separate: Runs a spatial model for each trial loop. Group: Runs one model with Experiment ID as a fixed effect."
        )
        
        separate_flag = (st.session_state.analysis_mode == "Analyze experiments separately")
        
        col_act.write("") 
        col_act.write("") 
        run_btn = col_act.button("🚀 Run Genotype Analysis", type="primary")

        if run_btn:
            success, res_df, stats_df, debug = run_hybrid_model(
                df_clean, perf_trait, col_map['geno'], col_map['row'], col_map['col'], col_map['expt'], separate_flag
            )
            st.session_state.results_df = res_df
            st.session_state.stats_df = stats_df
            st.session_state.debug_log = debug
            st.session_state.trait_ran = perf_trait
            st.session_state.analysis_mode_ran = st.session_state.analysis_mode 
            
        
        # Display Results if available
        if st.session_state.results_df is not None and st.session_state.trait_ran is not None:
            res_df = st.session_state.results_df
            perf_trait_ran = st.session_state.trait_ran
            
            st.subheader(f"Results for: {perf_trait_ran}")

            # --- STATS TABLE - Now more prominent ---
            if st.session_state.stats_df is not None and not st.session_state.stats_df.empty:
                st.markdown("---")
                with st.container():
                    st.markdown("### 📊 Experiment Analysis Summary")
                    
                    # Color-code the dataframe based on status
                    def color_status(row):
                        if row['Status'] == '✓ Good':
                            return ['background-color: #d4edda']*len(row)
                        elif row['Status'] == '⚠️ Caution':
                            return ['background-color: #fff3cd']*len(row)
                        elif row['Status'] == '❌ Failed':
                            return ['background-color: #f8d7da']*len(row)
                        return ['']*len(row)
                    
                    styled_stats = st.session_state.stats_df.style.apply(color_status, axis=1)
                    st.dataframe(styled_stats, use_container_width=True)
                    
                    # Show any important messages
                    caution_rows = st.session_state.stats_df[st.session_state.stats_df['Status'].str.contains('Caution|Failed', na=False)]
                    if not caution_rows.empty:
                        with st.expander("ℹ️ What do these status messages mean?", expanded=False):
                            for _, row in caution_rows.iterrows():
                                if row['Status'] == '⚠️ Caution':
                                    st.warning(f"**{row['Experiment ID']}**: {row['Message']}")
                                    if 'Low heritability' in row['Message']:
                                        st.caption("💡 Low heritability means genotypes performed similarly or environmental variation was high. Use results cautiously and consider increasing replication.")
                                    elif 'genetic variation' in row['Message']:
                                        st.caption("💡 This usually means genotypes are too similar genetically, or field noise is masking genetic differences.")
                                elif row['Status'] == '❌ Failed':
                                    st.error(f"**{row['Experiment ID']}**: {row['Message']}")
                                    st.caption("💡 Try: (1) Check for data entry errors, (2) Ensure minimum 2 reps per genotype, (3) Remove experiments with very few plots (<20)")
                
                st.markdown("---")
            else:
                st.info("No analysis results yet. Click 'Run Genotype Analysis' above to begin.")

            current_view_df = res_df.copy()
            download_df = res_df.copy()
            selected_group = "All Combined"

            if st.session_state.analysis_mode_ran == "Analyze experiments separately" and len(res_df['Analysis_Group'].unique()) > 1:
                unique_groups = ["All Combined"] + res_df['Analysis_Group'].unique().tolist()
                selected_group = st.selectbox("View Results By Experiment", unique_groups)
                
                if selected_group != "All Combined":
                    current_view_df = res_df[res_df['Analysis_Group'] == selected_group].copy()
                    download_df = current_view_df.copy()
                    st.markdown(f"**Showing results for:** `{selected_group}`")
                else:
                    st.markdown("**Showing results for:** All experiments combined")
            else:
                st.markdown("**Showing results for:** Combined Analysis")

            # Final Display
            current_view_df = current_view_df.sort_values(by=f"Predicted_{perf_trait_ran}", ascending=False)
            
            st.markdown("##### Full Results Table")
            st.dataframe(current_view_df)
            
            st.download_button(
                "Download All Combined Results (CSV)", 
                res_df.to_csv().encode('utf-8'), 
                f"All_Genotype_BLUPs_{perf_trait_ran}.csv",
                key='dl_all_blups'
            )
            
            if selected_group != "All Combined" and st.session_state.analysis_mode_ran == "Analyze experiments separately":
                 st.download_button(
                    f"Download {selected_group} Results Only (CSV)", 
                    download_df.to_csv().encode('utf-8'), 
                    f"{selected_group}_BLUPs_{perf_trait_ran}.csv",
                    key='dl_single_blup'
                )

            with st.expander("Complete Statistical Model Output (Debug)"):
                st.text(st.session_state.debug_log)


    # --- TAB 4: PARENTAL GCA ---
    with tab_parents:
        if not gca_enabled:
            st.header("Parental GCA (General Combining Ability) Disabled")
            st.error("GCA analysis requires both the 'Male Parent' and 'Female Parent' columns to be selected in the sidebar map.")
            st.info("General Combining Ability (GCA) is the average performance of a parental line in hybrid combinations. It is used to predict which parents will produce the best offspring.")
        else:
            st.header("Parental GCA (General Combining Ability)")
            st.markdown("""
            **Statistical Rigor:** This module fits a Linear Mixed Model: `Trait ~ FixedEffects(e.g., ExperimentID) + RandomEffects(1|Male) + RandomEffects(1|Female) + SpatialCorrection`.
            This isolates the true genetic breeding value (GCA) of the parents.
            """)
            
            gca_trait = st.selectbox("Trait for GCA", selected_traits, key='gca_trait')
            
            if st.button("Calculate GCA", key='run_gca_btn', type='primary'):
                success, male_df, female_df, debug = run_parental_model(
                    df_clean, gca_trait, col_map['male'], col_map['female'], col_map['row'], col_map['col'], col_map['expt']
                )
                
                if success:
                    c_male, c_fem = st.columns(2)
                    
                    c_male.subheader(f"Male GCA (Trait: {gca_trait})")
                    if male_df is not None:
                        male_df = male_df.sort_values(by=f"GCA_{gca_trait}", ascending=False)
                        # index is reset inside run_parental_model so Styler won't error
                        c_male.dataframe(male_df.style.background_gradient(cmap="Blues"), use_container_width=True)
                    else:
                        c_male.info("No Male GCA results to show.")

                    c_fem.subheader(f"Female GCA (Trait: {gca_trait})")
                    if female_df is not None:
                        female_df = female_df.sort_values(by=f"GCA_{gca_trait}", ascending=False)
                        c_fem.dataframe(female_df.style.background_gradient(cmap="Reds"), use_container_width=True)
                    else:
                        c_fem.info("No Female GCA results to show.")

                    if male_df is not None:
                        st.download_button(
                            f"Download Male GCA for {gca_trait} (CSV)", 
                            male_df.to_csv(index=False).encode('utf-8'), 
                            f"Male_GCA_{gca_trait}.csv",
                            key='dl_male_gca'
                        )
                    if female_df is not None:
                        st.download_button(
                            f"Download Female GCA for {gca_trait} (CSV)", 
                            female_df.to_csv(index=False).encode('utf-8'), 
                            f"Female_GCA_{gca_trait}.csv",
                            key='dl_female_gca'
                        )
                        
                    with st.expander("Complete GCA Model Output (Debug)"):
                        st.text(debug)
                else:
                    st.error("GCA Model Failed")
                    st.text(debug)

if __name__ == "__main__":
    main()
