import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt # Explicit import to ensure backend is ready
import re
import time

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Plant Breeding Analytics",
    page_icon="🧬",
    layout="wide"
)

# --- HELPER FUNCTIONS ---

def validate_data(df):
    if df.empty:
        return False, ["The uploaded file is empty."]
    return True, []

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
    Runs LMM for Hybrid Performance (Genotypes).
    """
    progress_bar = st.progress(0, text="Initializing Model...")
    
    results_container = []
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
        progress_bar.progress(step_val, text=f"Processing {run_name}: Preparing Matrix...")
        
        try:
            model_data = model_data.dropna(subset=[trait, gen_col, row_col, col_col]).copy()
            
            # Create Unique Spatial IDs to prevent row overlap
            model_data["Unique_Row"] = model_data[expt_col].astype(str) + "_" + model_data[row_col].astype(str)
            model_data["Unique_Col"] = model_data[expt_col].astype(str) + "_" + model_data[col_col].astype(str)
            model_data["Global_Group"] = 1

            # Formula Logic
            if not analyze_separate and model_data[expt_col].nunique() > 1:
                formula = f"{trait} ~ C({expt_col})"
            else:
                formula = f"{trait} ~ 1"

            # Variance Components
            vc = {
                "Genotype": f"0 + C({gen_col})",
                "SpatialRow": f"0 + C(Unique_Row)",
                "SpatialCol": f"0 + C(Unique_Col)"
            }
            
            progress_bar.progress(step_val + 10, text=f"Processing {run_name}: Fitting Spatial Model (REML)...")
            
            model = smf.mixedlm(formula, model_data, groups="Global_Group", vc_formula=vc)
            result = model.fit()
            
            progress_bar.progress(step_val + 20, text=f"Processing {run_name}: Extracting BLUPs...")
            
            # Extract Genotype BLUPs
            re_dict = result.random_effects[1]
            geno_blups = {}
            
            for key, val in re_dict.items():
                if key.startswith("Genotype["):
                    match = re.search(r"\[C\(" + re.escape(gen_col) + r"\)\]\[(.*?)\]", key)
                    name = match.group(1) if match else key
                    geno_blups[name] = val

            # Create DataFrame
            temp_df = pd.DataFrame.from_dict(geno_blups, orient='index', columns=[f'BLUP_{trait}'])
            
            # Add Intercept
            intercept = result.params['Intercept']
            temp_df[f'Predicted_{trait}'] = temp_df[f'BLUP_{trait}'] + intercept
            temp_df['Analysis_Group'] = run_name
            
            results_container.append(temp_df)
            debug_log.append(f"--- {run_name} Summary ---\n{result.summary().as_text()}")

        except Exception as e:
            debug_log.append(f"ERROR in {run_name}: {str(e)}")
    
    progress_bar.progress(100, text="Finalizing Results...")
    time.sleep(0.5)
    progress_bar.empty()
    
    if results_container:
        final_df = pd.concat(results_container)
        return True, final_df, "\n".join(debug_log)
    else:
        return False, None, "\n".join(debug_log)

def run_parental_model(df, trait, male_col, female_col, row_col, col_col, expt_col):
    """
    Runs a dedicated GCA model: Yield ~ (1|Male) + (1|Female) + Spatial
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
        else:
            formula = f"{trait} ~ 1"
            
        # Variance Components
        vc = {
            "Male_GCA": f"0 + C({male_col})",
            "Female_GCA": f"0 + C({female_col})",
            "SpatialRow": f"0 + C(Unique_Row)",
            "SpatialCol": f"0 + C(Unique_Col)"
        }
        
        progress_bar.progress(50, text="Fitting Parental Model...")
        model = smf.mixedlm(formula, model_data, groups="Global_Group", vc_formula=vc)
        result = model.fit()
        
        progress_bar.progress(80, text="Extracting GCA Values...")
        
        re_dict = result.random_effects[1]
        
        male_gca = {}
        female_gca = {}
        
        for key, val in re_dict.items():
            if key.startswith("Male_GCA["):
                match = re.search(r"\[C\(" + re.escape(male_col) + r"\)\]\[(.*?)\]", key)
                name = match.group(1) if match else key
                male_gca[name] = val
            elif key.startswith("Female_GCA["):
                match = re.search(r"\[C\(" + re.escape(female_col) + r"\)\]\[(.*?)\]", key)
                name = match.group(1) if match else key
                female_gca[name] = val
                
        df_male = pd.DataFrame.from_dict(male_gca, orient='index', columns=[f'GCA_{trait}'])
        df_female = pd.DataFrame.from_dict(female_gca, orient='index', columns=[f'GCA_{trait}'])
        
        progress_bar.empty()
        return True, df_male, df_female, result.summary().as_text()
        
    except Exception as e:
        progress_bar.empty()
        return False, None, None, str(e)

# --- MAIN APP ---

def main():
    st.title("🧬 Plant Breeding Trial Analytics")
    st.write("Upload your trial CSV. Perform spatial correction and genetic analysis.")

    # --- SIDEBAR (Explicit calls to avoid Magic Print errors) ---
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
    if df is not None:
        st.sidebar.divider()
        st.sidebar.header("2. Map Columns")
        st.sidebar.info("Please identify the columns in your file.")
        
        all_cols = ["Select Column..."] + df.columns.tolist()
        
        # Use st.sidebar.selectbox explicitly
        col_map['expt'] = st.sidebar.selectbox("Experiment ID", all_cols)
        col_map['geno'] = st.sidebar.selectbox("Genotype/Hybrid", all_cols)
        col_map['row'] = st.sidebar.selectbox("Plot Row", all_cols)
        col_map['col'] = st.sidebar.selectbox("Plot Column", all_cols)
        col_map['male'] = st.sidebar.selectbox("Male Parent", all_cols)
        col_map['female'] = st.sidebar.selectbox("Female Parent", all_cols)
        
        # Check if mapping is complete
        if "Select Column..." in col_map.values():
            st.sidebar.warning("⚠️ You must map all columns above to proceed.")
            # Return safely without a context manager
            return 

        st.sidebar.divider()
        st.sidebar.header("3. Filter & Configure")
        unique_expts = df[col_map['expt']].unique().tolist()
        selected_expts = st.sidebar.multiselect("Select Experiments to Include", unique_expts, default=unique_expts)
        
        potential_traits = [c for c in df.columns if c not in col_map.values()]
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
    tab_qc, tab_spatial, tab_perf, tab_parents = st.tabs([
        "🛡️ QC & Cleaning", 
        "🗺️ Spatial Analysis", 
        "📊 Genotype Performance", 
        "👨‍👩‍👧 Parental GCA"
    ])

    # --- TAB 1: QC ---
    with tab_qc:
        st.header("Data Quality Control")
        if cleaning_log:
            for msg in cleaning_log:
                st.error(msg) if "CRITICAL" in msg else st.warning(msg)
        else:
            st.success("No statistical anomalies detected.")

        col1, col2 = st.columns(2)
        
        # Refactored to explicit calls to prevent DeltaGenerator print error
        col1.write(f"**Selected Experiments:** {len(selected_expts)}")
        col1.write(f"**Total Plots:** {len(df_clean)}")
        col1.write(f"**Unique Hybrids:** {df_clean[col_map['geno']].nunique()}")
        
        col2.write("**Missing Values:**")
        col2.dataframe(df_clean[selected_traits].isnull().sum())

    # --- TAB 2: SPATIAL ---
    with tab_spatial:
        st.header("Spatial Field Map")
        c1, c2 = st.columns([1, 3])
        # Refactored to explicit calls
        map_trait = c1.selectbox("View Trait", selected_traits)
        map_expt = c1.selectbox("View Experiment", selected_expts)
        
        map_data = df_clean[df_clean[col_map['expt']] == map_expt]
        if not map_data.empty:
            pivot = map_data.pivot_table(index=col_map['row'], columns=col_map['col'], values=map_trait)
            fig = px.imshow(pivot, color_continuous_scale='Viridis', title=f"{map_expt}: {map_trait}", aspect="auto")
            fig.update_yaxes(autorange="reversed")
            c2.plotly_chart(fig, use_container_width=True)
        else:
            c2.warning("No data.")

    # --- TAB 3: GENOTYPE PERFORMANCE ---
    with tab_perf:
        st.header("Hybrid Performance Analysis")
        
        col_opts, col_act = st.columns([2, 1])
        # Refactored to explicit calls
        perf_trait = col_opts.selectbox("Trait to Analyze", selected_traits, key='perf_trait')
        analysis_mode = col_opts.radio(
            "Analysis Strategy", 
            ["Analyze experiments separately", "Analyze all experiments as one group"],
            help="Separate: Runs a spatial model for each trial loop. Group: Runs one model with Experiment as a fixed effect."
        )
        separate_flag = True if analysis_mode == "Analyze experiments separately" else False

        col_act.write("") 
        col_act.write("") 
        run_btn = col_act.button("🚀 Run Hybrid Analysis", type="primary")

        if run_btn:
            success, res_df, debug = run_hybrid_model(
                df_clean, perf_trait, col_map['geno'], col_map['row'], col_map['col'], col_map['expt'], separate_flag
            )
            
            if success:
                st.subheader("Top Performing Hybrids")
                res_df = res_df.sort_values(by=f"Predicted_{perf_trait}", ascending=False)
                
                fig = px.bar(
                    res_df.head(20),
                    x=f"Predicted_{perf_trait}",
                    y=res_df.head(20).index,
                    color="Analysis_Group" if separate_flag else None,
                    orientation='h',
                    title=f"Top 20 Hybrids ({analysis_mode})"
                )
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(res_df)
                st.download_button("Download Results CSV", res_df.to_csv().encode('utf-8'), f"Results_{perf_trait}.csv")
                
                with st.expander("Statistical Output (Debug)"):
                    st.text(debug)
            else:
                st.error("Model Failed.")
                st.text(debug)

    # --- TAB 4: PARENTAL GCA ---
    with tab_parents:
        st.header("Parental GCA (General Combining Ability)")
        st.markdown("""
        **Statistical Rigor:** This module fits a Linear Mixed Model: `Trait ~ (1|Male) + (1|Female) + SpatialCorrection`.
        This isolates the true genetic breeding value (GCA) of the parents.
        """)
        
        gca_trait = st.selectbox("Trait for GCA", selected_traits, key='gca_trait')
        
        if st.button("Calculate GCA"):
            success, male_df, female_df, debug = run_parental_model(
                df_clean, gca_trait, col_map['male'], col_map['female'], col_map['row'], col_map['col'], col_map['expt']
            )
            
            if success:
                c_male, c_fem = st.columns(2)
                
                # Refactored to explicit calls
                c_male.subheader("Male GCA (Best to Worst)")
                male_df = male_df.sort_values(by=f"GCA_{gca_trait}", ascending=False)
                c_male.dataframe(male_df.style.background_gradient(cmap="Blues"))
                
                c_fem.subheader("Female GCA (Best to Worst)")
                female_df = female_df.sort_values(by=f"GCA_{gca_trait}", ascending=False)
                c_fem.dataframe(female_df.style.background_gradient(cmap="Reds"))
                    
                with st.expander("GCA Model Details"):
                    st.text(debug)
            else:
                st.error("GCA Model Failed")
                st.text(debug)

if __name__ == "__main__":
    main()
