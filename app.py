import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.formula.api as smf

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Trial Data Analytics",
    page_icon="🧬",
    layout="wide"
)

# --- HELPER FUNCTIONS ---

def validate_data(df):
    """Checks for critical issues in the dataset structure."""
    if df.empty:
        return False, ["The uploaded file is empty."]
    return True, []

def clean_curveballs(df, traits):
    """
    Identifies and cleans statistical anomalies.
    """
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

def run_stats_model(df, trait, gen_col, row_col, col_col, expt_col):
    """
    Runs a Linear Mixed Model (LMM) handling multiple trials.
    
    Logic:
    1. Creates Unique Spatial IDs (Expt_Row, Expt_Col) so spatial smoothing 
       doesn't bleed across different trials.
    2. If multiple experiments exist, adds Experiment as a Fixed Effect.
    """
    try:
        model_data = df.dropna(subset=[trait, gen_col, row_col, col_col, expt_col]).copy()
        
        # --- CRITICAL: Create Unique Spatial IDs ---
        # Row 1 in Expt A is NOT the same as Row 1 in Expt B.
        model_data["Unique_Row"] = model_data[expt_col].astype(str) + "_" + model_data[row_col].astype(str)
        model_data["Unique_Col"] = model_data[expt_col].astype(str) + "_" + model_data[col_col].astype(str)

        # Determine Formula
        # Check if we have multiple experiments selected
        n_expts = model_data[expt_col].nunique()
        
        if n_expts > 1:
            # Multi-Trial: Yield ~ Experiment + (1|Genotype) + (1|UniqueRow) + (1|UniqueCol)
            formula = f"{trait} ~ C({expt_col})"
        else:
            # Single Trial: Yield ~ 1 + (1|Genotype) + (1|Row) + (1|Col)
            formula = f"{trait} ~ 1"
            # Use a dummy column for grouping if statsmodels needs it, 
            # but we usually group by a dummy constant for single trial analysis
            model_data["Global_Group"] = 1

        # Define Random Effects (Variance Components)
        vc = {
            "Genotype": f"0 + C({gen_col})",
            "SpatialRow": f"0 + C(Unique_Row)",
            "SpatialCol": f"0 + C(Unique_Col)"
        }

        # Fit Model
        # We group by "Global_Group" (constant) so the random effects are treated globally across the dataset
        # This is required because Unique_Row handles the nesting implicitly
        model_data["Global_Group"] = 1
        
        model = smf.mixedlm(
            formula, 
            model_data, 
            groups="Global_Group", 
            vc_formula=vc
        )
        
        result = model.fit()
        
        # --- EXTRACT BLUPs ---
        re = result.random_effects[1] # Get the random effects dict
        
        geno_blups = {}
        for key, val in re.items():
            if key.startswith("Genotype["):
                # Clean string to get genotype name
                # Format is usually Genotype[C(Name)][GenotypeName]
                # We split by brackets to extract the inner name
                import re as regex
                match = regex.search(r"\[C\(" + regex.escape(gen_col) + r"\)\]\[(.*?)\]", key)
                if match:
                    name = match.group(1)
                    geno_blups[name] = val
                else:
                    # Fallback for simpler string patterns
                    geno_blups[key] = val
        
        blup_df = pd.DataFrame.from_dict(geno_blups, orient='index', columns=[f'BLUP_{trait}'])
        
        # Add Intercept (Grand Mean) to make values relatable
        # Note: In multi-trial, this adds the reference level intercept. 
        # It ranks correctly, but absolute values are relative to the reference experiment.
        intercept = result.params['Intercept']
        blup_df[f'Predicted_{trait}'] = blup_df[f'BLUP_{trait}'] + intercept
        
        return True, blup_df, result.summary()
        
    except Exception as e:
        return False, None, str(e)

# --- MAIN APP ---

def main():
    st.title("🧬 Plant Breeding Trial Analysis")
    st.write("Upload your trial CSV. Handles Multi-trial files and and spatial correction.")

    # 1. SIDEBAR - DATA UPLOAD
    with st.sidebar:
        st.header("1. Upload Data")
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success("File Uploaded")
            except Exception as e:
                st.error(f"Error reading file: {e}")
                return
        else:
            st.info("Awaiting CSV file upload.")
            return

    # 2. COLUMN MAPPING
    st.sidebar.header("2. Map Columns")
    all_cols = df.columns.tolist()
    
    def get_idx(options, query):
        matches = [i for i, x in enumerate(options) if query.lower() in x.lower()]
        return matches[0] if matches else 0

    col_expt = st.sidebar.selectbox("Experiment ID", all_cols, index=get_idx(all_cols, "expt"))
    col_genotype = st.sidebar.selectbox("Genotype/Hybrid", all_cols, index=get_idx(all_cols, "name"))
    col_row = st.sidebar.selectbox("Plot Row", all_cols, index=get_idx(all_cols, "row"))
    col_col = st.sidebar.selectbox("Plot Column", all_cols, index=get_idx(all_cols, "col"))
    col_male = st.sidebar.selectbox("Male Parent", all_cols, index=get_idx(all_cols, "male"))
    col_female = st.sidebar.selectbox("Female Parent", all_cols, index=get_idx(all_cols, "female"))
    
    # 3. FILTER TRIALS
    st.sidebar.header("3. Filter Data")
    unique_expts = df[col_expt].unique().tolist()
    selected_expts = st.sidebar.multiselect("Select Experiments to Analyze", unique_expts, default=unique_expts)
    
    if not selected_expts:
        st.warning("Select at least one experiment.")
        return
        
    # Filter DataFrame
    df_filtered = df[df[col_expt].isin(selected_expts)].copy()
    
    # 4. VARIATE SELECTION
    st.sidebar.header("4. Select Traits")
    potential_traits = [c for c in all_cols if c not in [col_row, col_col, col_expt, col_genotype]]
    selected_traits = st.sidebar.multiselect("Phenotypes", potential_traits, default=[x for x in potential_traits if "yield" in x.lower()])

    if not selected_traits:
        st.warning("Select a trait.")
        return

    # --- MAIN TABS ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "🛡️ QC & Cleaning", 
        "🗺️ Spatial Analysis", 
        "📊 Genetic Performance", 
        "👨‍👩‍👧 Combining Ability"
    ])

    df_clean, cleaning_log = clean_curveballs(df_filtered, selected_traits)

    # --- TAB 1: QC ---
    with tab1:
        st.header("Data Quality Control")
        if cleaning_log:
            for msg in cleaning_log:
                st.warning(msg) if "CRITICAL" not in msg else st.error(msg)
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"Analyzing **{len(selected_expts)}** Experiment(s)")
            st.write(f"**Total Plots:** {len(df_clean)}")
            st.write(f"**Genotypes:** {df_clean[col_genotype].nunique()}")
        with col2:
            st.write("**Missing Values:**")
            st.dataframe(df_clean[selected_traits].isnull().sum())

    # --- TAB 2: SPATIAL ---
    with tab2:
        st.header("Spatial Field Map")
        st.markdown("View the field layout. Since plot coordinates overlap between trials, **select one experiment** to view at a time.")
        
        col_map_1, col_map_2 = st.columns([1, 3])
        
        with col_map_1:
            trait_to_map = st.selectbox("Trait", selected_traits)
            expt_to_map = st.selectbox("Select Experiment", selected_expts)
        
        with col_map_2:
            # Filter for just this map
            map_df = df_clean[df_clean[col_expt] == expt_to_map]
            
            if not map_df.empty:
                pivot = map_df.pivot_table(index=col_row, columns=col_col, values=trait_to_map)
                
                fig = px.imshow(
                    pivot, 
                    color_continuous_scale='Viridis',
                    title=f"{expt_to_map}: {trait_to_map}",
                    aspect="auto"
                )
                fig.update_yaxes(autorange="reversed")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data for this selection.")

    # --- TAB 3: BLUPs ---
    with tab3:
        st.header("Genotype Performance (BLUPs)")
        st.markdown("This model accounts for Spatial Variation across **all selected experiments**.")
        
        trait_stats = st.selectbox("Select Trait", selected_traits, key='stats_select')
        
        if st.button(f"Run Analysis for {trait_stats}"):
            with st.spinner("Running Multi-Environment Linear Mixed Model..."):
                success, results_df, debug_info = run_stats_model(
                    df_clean, trait_stats, col_genotype, col_row, col_col, col_expt
                )
                
                if success:
                    st.success("Analysis Complete")
                    
                    results_df = results_df.sort_values(by=f'Predicted_{trait_stats}', ascending=False)
                    
                    # Graph
                    top_20 = results_df.head(20)
                    fig_bar = px.bar(
                        top_20, 
                        x=f'Predicted_{trait_stats}',
                        y=top_20.index,
                        orientation='h',
                        title=f"Top 20 Genotypes (Across {len(selected_expts)} Trials)",
                        color=f'Predicted_{trait_stats}',
                        color_continuous_scale='Greens'
                    )
                    fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_bar, use_container_width=True)
                    
                    # Table
                    st.dataframe(results_df)
                    csv = results_df.to_csv().encode('utf-8')
                    st.download_button("Download BLUPs CSV", csv, f"BLUPs_{trait_stats}.csv", "text/csv")
                    
                    with st.expander("Model Convergence Details"):
                        st.text(debug_info)
                else:
                    st.error("Analysis Failed")
                    st.text(debug_info)

    # --- TAB 4: PARENTS ---
    with tab4:
        st.header("Parental Analysis")
        analysis_trait = st.selectbox("Select Trait", selected_traits, key='parent_select')
        
        # Simple means for parents (aggregated across selected experiments)
        col_p1, col_p2 = st.columns(2)
        
        with col_p1:
            st.subheader("Male Performance")
            male_stats = df_clean.groupby(col_male)[analysis_trait].agg(['mean', 'count']).sort_values('mean', ascending=False)
            # Using standard Streamlit dataframe instead of style to avoid matplotlib issues just in case
            st.dataframe(male_stats) 

        with col_p2:
            st.subheader("Female Performance")
            fem_stats = df_clean.groupby(col_female)[analysis_trait].agg(['mean', 'count']).sort_values('mean', ascending=False)
            st.dataframe(fem_stats)

if __name__ == "__main__":
    main()
