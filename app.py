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
        
        # --- Point 2: Dynamic Progress Bar ---
        progress_bar.progress(step_val, text=f"Processing **{run_name}**: Preparing Matrix...")
        
        try:
            model_data = model_data.dropna(subset=[trait, gen_col, row_col, col_col]).copy()
            
            # Ensure enough data for modeling
            if model_data[gen_col].nunique() < 2:
                 debug_log.append(f"ERROR in {run_name}: Not enough unique genotypes ({model_data[gen_col].nunique()}) to run the model.")
                 continue

            # Create Unique Spatial IDs
            model_data["Unique_Row"] = model_data[expt_col].astype(str) + "_" + model_data[row_col].astype(str)
            model_data["Unique_Col"] = model_data[expt_col].astype(str) + "_" + model_data[col_col].astype(str)
            model_data["Global_Group"] = 1

            # Formula Logic
            if not analyze_separate and model_data[expt_col].nunique() > 1:
                formula = f"{trait} ~ C({expt_col})"
                fixed_effects_formula = f"Fixed Effects: {trait} ~ C({expt_col})"
            else:
                formula = f"{trait} ~ 1"
                fixed_effects_formula = f"Fixed Effects: {trait} ~ Intercept"

            # Variance Components (Random Effects)
            vc = {
                "Genotype": f"0 + C({gen_col})",
                "SpatialRow": f"0 + C(Unique_Row)",
                "SpatialCol": f"0 + C(Unique_Col)"
            }
            random_effects_formula = f"Random Effects: (1|{gen_col}) + (1|Unique_Row) + (1|Unique_Col)"

            # --- Point 4: Model Details in Log ---
            debug_log.append(f"--- {run_name} Model Formula ---\n{fixed_effects_formula}\n{random_effects_formula}")
            
            progress_bar.progress(step_val + 10, text=f"Processing **{run_name}**: Fitting Spatial Model (REML)...")
            
            model = smf.mixedlm(formula, model_data, groups="Global_Group", vc_formula=vc)
            result = model.fit(method='powell', reml=True) # Use Powell for more robust convergence
            
            progress_bar.progress(step_val + 20, text=f"Processing **{run_name}**: Extracting BLUPs...")
            
            # --- Point 3: Extract & Clean Genotype BLUPs ---
            re_dict = result.random_effects[1]
            geno_blups = {}
            
            for key, val in re_dict.items():
                if key.startswith("Genotype["):
                    # Use regex to extract only the genotype name, which is always in the last square brackets
                    match = re.search(r'\[(.*?)\]$', key)
                    name = match.group(1) if match else key
                    geno_blups[name] = val

            # Create DataFrame
            temp_df = pd.DataFrame.from_dict(geno_blups, orient='index', columns=[f'BLUP_{trait}'])
            temp_df.index.name = 'Genotype'
            
            # Add Intercept (Fixed Effect Mean)
            intercept = result.params.get('Intercept', 0)
            temp_df[f'Predicted_{trait}'] = temp_df[f'BLUP_{trait}'] + intercept
            temp_df['Analysis_Group'] = run_name
            
            results_container.append(temp_df)
            debug_log.append(f"--- {run_name} Summary ---\n{result.summary().as_text()}")

            # --- Point 6: Calculate Experiment-Wide Statistics ---
            v_g = result.varmix.get(f"Genotype[C({gen_col})]")
            v_e = result.scale # Residual variance
            
            h2 = 'N/A'
            if v_g is not None and v_e is not None and model_data[gen_col].nunique() > 1:
                # Avg replication estimate based on total plots / total unique genotypes
                avg_plots_per_geno = model_data.groupby(gen_col).size().mean()
                
                # Formula: H^2 = Vg / ( Vg + Ve / n_plots)
                h2_val = v_g / (v_g + (v_e / avg_plots_per_geno))
                h2 = round(h2_val * 100, 1)

            mean = model_data[trait].mean()
            std_dev = model_data[trait].std()
            cv = (std_dev / mean) * 100 if mean != 0 else 0
            
            summary_stats.append({
                'Experiment ID': run_name,
                'Mean': round(mean, 3),
                'CV (%)': round(cv, 1),
                'Genotype Var (Vg)': round(v_g, 3) if v_g is not None else 'N/A',
                'Residual Var (Ve)': round(v_e, 3),
                'Est. Heritability (H2, %)': h2,
                'Plots': len(model_data),
                'Unique Genotypes': model_data[gen_col].nunique()
            })
            
        except Exception as e:
            debug_log.append(f"ERROR in {run_name}: {str(e)}")
    
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
    Runs a dedicated GCA model.
    """
    progress_bar = st.progress(0, text="Calculating GCA (Parental BLUPs)...")
    
    try:
        model_data = df.dropna(subset=[trait, male_col, female_col, row_col, col_col]).copy()
        
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
        
        progress_bar.progress(80, text="Extracting GCA Values...")
        
        re_dict = result.random_effects[1]
        
        male_gca = {}
        female_gca = {}
        
        # --- Point 3: Extract & Clean GCA Names ---
        for key, val in re_dict.items():
            match = re.search(r'\[(.*?)\]$', key)
            name = match.group(1) if match else key
            
            if key.startswith("Male_GCA["):
                male_gca[name] = val
            elif key.startswith("Female_GCA["):
                female_gca[name] = val
                
        df_male = pd.DataFrame.from_dict(male_gca, orient='index', columns=[f'GCA_{trait}'])
        df_male.index.name = 'Male Parent'
        df_female = pd.DataFrame.from_dict(female_gca, orient='index', columns=[f'GCA_{trait}'])
        df_female.index.name = 'Female Parent'
        
        progress_bar.empty()
        
        # --- Point 4: Statistical Output ---
        debug_output = f"--- Model Formula ---\n{fixed_effects_formula}\n{random_effects_formula}\n\n"
        debug_output += f"--- Model Summary ---\n{result.summary().as_text()}"
        
        return True, df_male, df_female, debug_output
        
    except Exception as e:
        progress_bar.empty()
        return False, None, None, str(e)

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
        
        # --- Point 1: Change label to "Genotype" ---
        col_map['expt'] = st.sidebar.selectbox("Experiment ID", all_cols)
        col_map['geno'] = st.sidebar.selectbox("Genotype", all_cols)
        col_map['row'] = st.sidebar.selectbox("Plot Row", all_cols)
        col_map['col'] = st.sidebar.selectbox("Plot Column", all_cols)
        
        # --- Point 1: Optional Parent Columns with Info Bubble ---
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
    # --- Point 1: GCA Tab Label based on status ---
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
            # --- Point 1: Change to "Genotypes" ---
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
        
        # --- Point 5: Information Bubble ---
        with st.expander("Understanding the Statistical Analysis (BLUPs)"):
            st.markdown("""
            This analysis uses a Linear Mixed Model (LMM) with spatial correction to calculate Best Linear Unbiased Predictions (BLUPs).
            
            **Why use BLUPs?**
            In field trials, plot yield is affected by genetics (the genotype) and non-genetic factors (soil variability, pests, etc.). BLUPs are a statistical method that separates the true genetic performance of a genotype from these environmental 'noise' effects.
            
            **How it works (in simple terms):**
            1. **Spatial Correction:** The model looks at adjacent plots (rows and columns) and estimates a spatial trend across the entire field. It then removes this trend, correcting for local soil differences or field gradients.
            2. **Genotype Value:** After correcting for spatial variation, the model estimates the 'true' breeding value for each genotype (the BLUP). This value is adjusted based on how many times the genotype was tested (replicates) and how consistent its performance was across the trial.
            
            The final **Predicted Trait Value** is the Genotype BLUP added back to the overall mean of the trial.
            """)
        
        col_opts, col_act = st.columns([2, 1])
        perf_trait = col_opts.selectbox("Trait to Analyze", selected_traits, key='perf_trait')
        
        # Use session state for analysis mode so it persists
        def update_analysis_mode():
            st.session_state.analysis_mode = st.session_state.temp_analysis_mode
            st.session_state.results_df = None # Clear results when mode changes

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
            # Store results in session state
            st.session_state.results_df = res_df
            st.session_state.stats_df = stats_df
            st.session_state.debug_log = debug
            st.session_state.perf_trait = perf_trait
            st.session_state.analysis_mode_ran = st.session_state.analysis_mode # store the mode that produced the results
            
        
        # Display Results if available
        if st.session_state.results_df is not None:
            res_df = st.session_state.results_df
            perf_trait = st.session_state.perf_trait
            
            st.subheader(f"Results for: {perf_trait}")

            # --- Point 6: Experiment-Wide Statistics Table ---
            if not st.session_state.stats_df.empty:
                st.markdown("##### Experiment-Wide Statistical Summary")
                st.dataframe(st.session_state.stats_df, use_container_width=True)
                st.markdown("---")

            current_view_df = res_df.copy()
            download_df = res_df.copy()

            # --- Point 2: Toggling Results for Separate Analysis ---
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
            current_view_df = current_view_df.sort_values(by=f"Predicted_{perf_trait}", ascending=False)
            
            st.markdown(f"##### Top 20 Genotypes (Predicted {perf_trait})")
            fig = px.bar(
                current_view_df.head(20),
                x=f"Predicted_{perf_trait}",
                y=current_view_df.head(20).index,
                orientation='h',
                title=f"Top Genotypes: {perf_trait}"
            )
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("##### Full Results Table")
            st.dataframe(current_view_df)
            
            # --- Point 2: Download Options ---
            st.download_button(
                "Download All Combined Results (CSV)", 
                res_df.to_csv().encode('utf-8'), 
                f"All_Genotype_BLUPs_{perf_trait}.csv",
                key='dl_all_blups'
            )
            
            if selected_group != "All Combined" and st.session_state.analysis_mode_ran == "Analyze experiments separately":
                 st.download_button(
                    f"Download {selected_group} Results Only (CSV)", 
                    download_df.to_csv().encode('utf-8'), 
                    f"{selected_group}_BLUPs_{perf_trait}.csv",
                    key='dl_single_blup'
                )

            # --- Point 4: Statistical Output ---
            with st.expander("Complete Statistical Model Output (Debug)"):
                st.text(st.session_state.debug_log)


    # --- TAB 4: PARENTAL GCA ---
    with tab_parents:
        # --- Point 1: Greyed out logic and explanation ---
        if not gca_enabled:
            st.header("Parental GCA (General Combining Ability) Disabled")
            st.error("GCA analysis requires both the 'Male Parent' and 'Female Parent' columns to be selected in the sidebar map.")
            st.info("General Combining Ability (GCA) is the average performance of a parental line in hybrid combinations. It is used to predict which parents will produce the best offspring.")
        else:
            st.header("Parental GCA (General Combining Ability)")
            st.markdown("""
            **Statistical Rigor:** This module fits a Linear Mixed Model: `Trait ~ (1|Male) + (1|Female) + SpatialCorrection`.
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
                    st.error("GCA Model Failed")
                    st.text(debug)

if __name__ == "__main__":
    main()
