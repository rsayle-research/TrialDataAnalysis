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
        
        try:
            model_data = model_data.dropna(subset=[trait, gen_col, row_col, col_col]).copy()
            
            # --- Data Preprocessing ---
            model_data[gen_col] = model_data[gen_col].astype(str)
            model_data[row_col] = model_data[row_col].astype(str)
            model_data[col_col] = model_data[col_col].astype(str)

            # Check for sufficient unique genotypes
            if model_data[gen_col].nunique() < 2:
                 debug_log.append(f"ERROR in {run_name}: Not enough unique genotypes ({model_data[gen_col].nunique()}) to run the model.")
                 raise ValueError("Not enough unique genotypes.")

            # Create Unique Spatial IDs
            model_data["Unique_Row"] = model_data[expt_col].astype(str) + "_" + model_data[row_col].astype(str)
            model_data["Unique_Col"] = model_data[expt_col].astype(str) + "_" + model_data[col_col].astype(str)
            model_data["Global_Group"] = 1

            # Formula Logic
            if not analyze_separate and model_data[expt_col].nunique() > 1:
                formula = f"{trait} ~ C({expt_col})"
                fixed_effects_formula = f"Fixed Effects: {trait} ~ C({expt_col}) (Corrects for baseline differences between sites/years)"
            else:
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
            
            progress_bar.progress(step_val + 10, text=f"Processing **{run_name}**: Fitting Spatial Model (REML)...")
            
            model = smf.mixedlm(formula, model_data, groups="Global_Group", vc_formula=vc)
            result = model.fit(method='powell', reml=True) 
            
            # --- STEP 1: LOG SUMMARY ---
            debug_log.append(f"--- {run_name} Summary ---\n{result.summary().as_text()}")

            # --- STEP 2: BLUP EXTRACTION (The working part) ---
            if not result.random_effects or 1 not in result.random_effects:
                 raise ValueError("Model failed to estimate random effects (BLUPs).")

            progress_bar.progress(step_val + 20, text=f"Processing **{run_name}**: Extracting BLUPs...")
            
            re_dict = result.random_effects[1]
            geno_blups = {}
            expected_prefix = f"Genotype[C({gen_col})]" 
            
            for key, val in re_dict.items():
                if expected_prefix in key:
                    match = re.search(r'\[([^\[\]]+)\]$', key) 
                    name = match.group(1) if match and match.group(1) else key
                    if name.startswith('T.'):
                        name = name[2:]
                    geno_blups[name] = val

            # Create BLUP DataFrame
            temp_df = pd.DataFrame.from_dict(geno_blups, orient='index', columns=[f'BLUP_{trait}'])
            temp_df.index.name = 'Genotype'
            
            # Add Intercept (Fixed Effect Mean)
            intercept = result.params.get('Intercept', 0)
            temp_df[f'Predicted_{trait}'] = temp_df[f'BLUP_{trait}'] + intercept
            temp_df['Analysis_Group'] = run_name
            
            results_container.append(temp_df)
            
            # --- STEP 3: EXPERIMENT-WIDE STATS CALCULATION (The potentially failing part) ---
            
            # Default values in case the statistical extraction fails
            h2 = 'N/A'
            Vg_val = 'N/A'
            Ve_val = 'N/A'
            
            try:
                # --- This block needs to be robust against missing varmix ---
                
                # Try the full key first, then fall back to the simple key "Genotype" (the VC name)
                v_g = result.varmix.get(f"Genotype[C({gen_col})]", None)
                if v_g is None:
                     v_g = result.varmix.get("Genotype", None) # Often the key when using vc_formula
                
                v_e = result.scale # Residual variance
                
                Ve_val = round(v_e, 3)
                
                if v_g is not None and model_data[gen_col].nunique() > 1:
                    Vg_val = round(v_g, 3) 
                    
                    # Avg replication estimate based on total plots / total unique genotypes
                    avg_plots_per_geno = model_data.groupby(gen_col).size().mean()
                    
                    # Formula: H^2 = Vg / ( Vg + Ve / n_plots)
                    h2_val = v_g / (v_g + (v_e / avg_plots_per_geno))
                    h2 = round(h2_val * 100, 1)

            except Exception as e:
                # If varmix fails, we log it but continue since BLUPs are already extracted
                debug_log.append(f"🚨 WARNING in {run_name}: Failed to extract Variance Components (Vg/H2) from 'varmix': {str(e)}. BLUPs were still successfully calculated.")
                Vg_val = 'FAILED'
                Ve_val = 'FAILED'
                h2 = 'FAILED'

            # Calculate basic stats (these should always work)
            mean = model_data[trait].mean()
            std_dev = model_data[trait].std()
            cv = (std_dev / mean) * 100 if mean != 0 else 0
            
            # Append results to summary_stats (will contain N/A or FAILED if stats failed)
            summary_stats.append({
                'Experiment ID': run_name,
                'Mean': round(mean, 3),
                'CV (%)': round(cv, 1),
                'Genotype Var (Vg)': Vg_val,
                'Residual Var (Ve)': Ve_val,
                'Est. Heritability (H2, %)': h2,
                'Plots': len(model_data),
                'Unique Genotypes': model_data[gen_col].nunique()
            })
            
        except Exception as e:
            # This catch block is for critical failures (e.g., model fitting failed or not enough unique genotypes)
            error_message = f"MODEL FAILED: {str(e)}"
            debug_log.append(f"🚨 CRITICAL ERROR in {run_name}: {error_message}. Check raw data for NaNs, zero variance, or singular matrix.")
            
            summary_stats.append({
                'Experiment ID': run_name,
                'Mean': model_data[trait].mean() if 'model_data' in locals() and not model_data.empty else 'N/A',
                'CV (%)': 'N/A',
                'Genotype Var (Vg)': 'CRITICAL FAILED',
                'Residual Var (Ve)': 'CRITICAL FAILED',
                'Est. Heritability (H2, %)': 'CRITICAL FAILED',
                'Plots': len(model_data) if 'model_data' in locals() else 0,
                'Unique Genotypes': model_data[gen_col].nunique() if 'model_data' in locals() else 0
            })
            continue
    
    progress_bar.progress(100, text="Finalizing Results...")
    time.sleep(0.5)
    progress_bar.empty()
    
    stats_df = pd.DataFrame(summary_stats)
    
    if results_container:
        final_df = pd.concat(results_container)
        return True, final_df, stats_df, "\n".join(debug_log)
    else:
        return False, pd.DataFrame(columns=['Genotype', f'BLUP_{trait}', f'Predicted_{trait}', 'Analysis_Group']), stats_df, "\n".join(debug_log)

def run_parental_model(df, trait, male_col, female_col, row_col, col_col, expt_col):
    """
    Runs a dedicated GCA model.
    """
    progress_bar = st.progress(0, text="Calculating GCA (Parental BLUPs)...")
    
    debug_output = "" # Initialize output string
    
    try:
        model_data = df.dropna(subset=[trait, male_col, female_col, row_col, col_col]).copy()
        
        # --- Data Preprocessing ---
        model_data[male_col] = model_data[male_col].astype(str)
        model_data[female_col] = model_data[female_col].astype(str)
        model_data[row_col] = model_data[row_col].astype(str)
        model_data[col_col] = model_data[col_col].astype(str)

        # Create Unique Spatial IDs
        model_data["Unique_Row"] = model_data[expt_col].astype(str) + "_" + model_data[row_col].astype(str)
        model_data["Unique_Col"] = model_data[expt_col].astype(str) + "_" + model_data[col_col].astype(str)
        model_data["Global_Group"] = 1
        
        # Formula
        if model_data[expt_col].nunique() > 1:
            formula = f"{trait} ~ C({expt_col})"
            fixed_effects_formula = f"Fixed Effects: {trait} ~ C({expt_col})"
        else:
            formula = f"{trait} ~ 1"
            fixed_effects_formula = f"Fixed Effects: {trait} ~ Intercept"
            
        # Variance Components
        vc = {
            "Male_GCA": f"0 + C({male_col})",
            "Female_GCA": f"0 + C({female_col})",
            "SpatialRow": f"0 + C(Unique_Row)",
            "SpatialCol": f"0 + C(Unique_Col)"
        }
        random_effects_formula = f"Random Effects: (1|{male_col}) + (1|{female_col}) + (1|Unique_Row) + (1|Unique_Col)"

        progress_bar.progress(50, text="Fitting Parental Model...")
        model = smf.mixedlm(formula, model_data, groups="Global_Group", vc_formula=vc)
        result = model.fit(method='powell', reml=True)

        debug_output += f"--- GCA Model Formula ---\n{fixed_effects_formula}\n{random_effects_formula}\n\n"
        debug_output += f"--- GCA Model Summary ---\n{result.summary().as_text()}"

        if not result.random_effects or 1 not in result.random_effects:
             raise ValueError("Model failed to estimate random effects (GCA).")
        
        progress_bar.progress(80, text="Extracting GCA Values...")
        
        re_dict = result.random_effects[1]
        
        male_gca = {}
        female_gca = {}
        
        # --- Extract & Clean GCA Names Robustly ---
        male_prefix = f"Male_GCA[C({male_col})]"
        female_prefix = f"Female_GCA[C({female_col})]"
            
        for key, val in re_dict.items():
            match = re.search(r'\[([^\[\]]+)\]$', key) 
            name = match.group(1) if match and match.group(1) else key
                
            if name.startswith('T.'):
                name = name[2:]
                
            if male_prefix in key:
                male_gca[name] = val
            elif female_prefix in key:
                female_gca[name] = val
                
        df_male = pd.DataFrame.from_dict(male_gca, orient='index', columns=[f'GCA_{trait}'])
        df_male.index.name = 'Male Parent'
        df_female = pd.DataFrame.from_dict(female_gca, orient='index', columns=[f'GCA_{trait}'])
        df_female.index.name = 'Female Parent'
        
        progress_bar.empty()
        
        return True, df_male, df_female, debug_output
        
    except Exception as e:
        progress_bar.empty()
        # Log the full error to the output
        debug_output += f"🚨 CRITICAL GCA Model Failure: {str(e)}"
        return False, None, None, debug_output

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
        
        # --- Required Columns ---
        col_map['expt'] = st.sidebar.selectbox("Experiment ID", all_cols)
        col_map['geno'] = st.sidebar.selectbox("Genotype", all_cols)
        col_map['row'] = st.sidebar.selectbox("Plot Row", all_cols)
        col_map['col'] = st.sidebar.selectbox("Plot Column", all_cols)
        
        # --- Optional Parent Columns ---
        with st.sidebar.expander("GCA Parental Lines (Optional)"):
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
        
        # Ensure selected_traits doesn't include the mapped columns that are not 'Select Column...'
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
        if st.session_state.stats_df is not None:
            
            st.subheader(f"Results for: {st.session_state.trait_ran}")

            # Experiment-Wide Statistics Table (Guaranteed to show, even with failures)
            if not st.session_state.stats_df.empty:
                st.markdown("##### Experiment-Wide Statistical Summary")
                st.dataframe(st.session_state.stats_df, use_container_width=True)
                st.markdown(
                    """
                    *Note: Entries with 'FAILED' or 'CRITICAL FAILED' indicate the statistical model's post-estimation process (calculating $V_g$ and $H^2$) 
                    failed due to numerical instability, even if the model successfully extracted BLUPs.*
                    """
                )
                st.markdown("---")
            else:
                st.warning("No statistical data was generated. Check the Debug Log below.")


        if st.session_state.results_df is not None and not st.session_state.results_df.empty:
            res_df = st.session_state.results_df
            perf_trait_ran = st.session_state.trait_ran
            
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
            
            st.markdown(f"##### Top 20 Genotypes (Predicted {perf_trait_ran})")
            fig = px.bar(
                current_view_df.head(20),
                x=f"Predicted_{perf_trait_ran}",
                y=current_view_df.head(20).index,
                orientation='h',
                title=f"Top Genotypes: {perf_trait_ran}"
            )
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
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

        else:
             st.error("Model results table is empty. Check the 'Experiment-Wide Statistical Summary' above for 'FAILED' entries and review the Complete Statistical Model Output below for error messages.")


        if st.session_state.debug_log:
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
                    st.markdown("""
                    ### GCA Interpretation
                    
                    * **Positive GCA:** The parent contributes positively to the hybrid's performance. These are the best parents for that trait.
                    * **Negative GCA:** The parent contributes negatively.
                    """)

                    c_male, c_fem = st.columns(2)
                    
                    c_male.subheader(f"Male GCA (Trait: {gca_trait})")
                    male_df = male_df.sort_values(by=f"GCA_{gca_trait}", ascending=False)
                    c_male.dataframe(male_df.style.background_gradient(cmap="Blues"), use_container_width=True)
                    
                    c_fem.subheader(f"Female GCA (Trait: {gca_trait})")
                    female_df = female_df.sort_values(by=f"GCA_{gca_trait}", ascending=False)
                    c_fem.dataframe(female_df.style.background_gradient(cmap="Reds"), use_container_width=True)

                    st.download_button(
                        f"Download Male GCA for {gca_trait} (CSV)", 
                        male_df.to_csv().encode('utf-8'), 
                        f"Male_GCA_{gca_trait}.csv",
                        key='dl_male_gca'
                    )
                    st.download_button(
                        f"Download Female GCA for {gca_trait} (CSV)", 
                        female_df.to_csv().encode('utf-8'), 
                        f"Female_GCA_{gca_trait}.csv",
                        key='dl_female_gca'
                    )
                        
                    with st.expander("Complete GCA Model Output (Debug)"):
                        st.text(debug)
                else:
                    st.error("GCA Model Failed. Check the debug output.")
                    with st.expander("Complete GCA Model Output (Debug)"):
                        st.text(debug)

if __name__ == "__main__":
    main()
