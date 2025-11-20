import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.formula.api as smf
import re # Import regex module explicitly

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Canola Breeding Analytics",
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
    """
    try:
        model_data = df.dropna(subset=[trait, gen_col, row_col, col_col, expt_col]).copy()
        
        # --- CRITICAL: Create Unique Spatial IDs ---
        # Row 1 in Expt A is NOT the same as Row 1 in Expt B.
        model_data["Unique_Row"] = model_data[expt_col].astype(str) + "_" + model_data[row_col].astype(str)
        model_data["Unique_Col"] = model_data[expt_col].astype(str) + "_" + model_data[col_col].astype(str)

        # Determine Formula
        n_expts = model_data[expt_col].nunique()
        
        if n_expts > 1:
            # Multi-Trial: Yield ~ Experiment + (1|Genotype) + (1|UniqueRow) + (1|UniqueCol)
            formula = f"{trait} ~ C({expt_col})"
        else:
            # Single Trial: Yield ~ 1 + (1|Genotype) + (1|Row) + (1|Col)
            formula = f"{trait} ~ 1"

        # Define Random Effects (Variance Components)
        vc = {
            "Genotype": f"0 + C({gen_col})",
            "SpatialRow": f"0 + C(Unique_Row)",
            "SpatialCol": f"0 + C(Unique_Col)"
        }

        # Fit Model
        # Group by a constant so random effects are treated globally
        model_data["Global_Group"] = 1
        
        model = smf.mixedlm(
            formula, 
            model_data, 
            groups="Global_Group", 
            vc_formula=vc
        )
        
        result = model.fit()
        
        # --- EXTRACT BLUPs ---
        re_dict = result.random_effects[1] # Get the random effects dict
        
        geno_blups = {}
        for key, val in re_dict.items():
            # Parse Genotype BLUPs
            if key.startswith("Genotype["):
                # Format is usually Genotype[C(Name)][GenotypeName]
                # We split by brackets to extract the inner name
                match = re.search(r"\[C\(" + re.escape(gen_col) + r"\)\]\[(.*?)\]", key)
                if match:
                    name = match.group(1)
                    geno_blups[name] = val
                else:
                    # Fallback if regex fails
                    geno_blups[key] = val
        
        blup_df = pd.DataFrame.from_dict(geno_blups, orient='index', columns=[f'BLUP_{trait}'])
        
        # Add Intercept (Grand Mean) to make values relatable
        intercept = result.params['Intercept']
        blup_df[f'Predicted_{trait}'] = blup_df[f'BLUP_{trait}'] + intercept
        
        # Return summary as STRING to avoid Streamlit rendering the object
        return True, blup_df, str(result.summary())
        
    except Exception as e:
        return False, None, str(e)

# --- MAIN APP ---

def main():
    st.title("🧬 Gold Standard Plant Breeding Analytics")
    st.write("Upload your trial CSV. Handles Multi-Environment Trials (MET) and spatial correction.")

    # --- SIDEBAR CONFIGURATION ---
    # Consolidate all sidebar elements into one block to prevent 'magic' print errors
    with st.sidebar:
        st.header("1. Upload Data")
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        
        # Initialize df to None
        df = None
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success("File Uploaded")
            except Exception as e:
                st.error(f"Error reading file: {e}")
                return
        
        if df is not None:
            st.divider()
            st.header("2. Map Columns")
            all_cols = df.columns.tolist()
            
            # Helper to find index
            def get_idx(options, query):
                matches = [i for i, x in enumerate(options) if query.lower() in x.lower()]
                return matches[0] if matches else 0

            col_expt = st.selectbox("Experiment ID", all_cols, index=get_idx(all_cols, "expt"))
            col_genotype = st.selectbox("Genotype/Hybrid", all_cols, index=get_idx(all_cols, "name"))
            col_row = st.selectbox("Plot Row", all_cols, index=get_idx(all_cols, "row"))
            col_col = st.selectbox("Plot Column", all_cols, index=get_idx(all_cols, "col"))
            col_male = st.selectbox("Male Parent", all_cols, index=get_idx(all_cols, "male"))
            col_female = st.selectbox("Female Parent", all_cols, index=get_idx(all_cols, "female"))
            
            st.divider()
            st.header("3. Filter Data")
            unique_expts = df[col_expt].unique().tolist()
            selected_expts = st.multiselect("Select Experiments", unique_expts, default=unique_expts)
            
            st.divider()
            st.header("4. Select Traits")
            # Exclude mapping columns from trait list
            potential_traits = [c for c in all_cols if c not in [col_row, col_col, col_expt, col_genotype]]
            selected_traits = st.multiselect("Phenotypes", potential_traits, default=[x for x in potential_traits if "yield" in x.lower()])

    # --- MAIN PAGE LOGIC ---
    # Exit if data not ready
    if df is None:
        st.info("👈 Waiting for CSV upload in the sidebar.")
        return

    if not selected_expts:
        st.warning("Please select at least one experiment in the sidebar.")
        return

    if not selected_traits:
        st.warning("Please select at least one trait in the sidebar.")
        return

    # Filter DataFrame
    df_filtered = df[df[col_expt].isin(selected_expts)].copy()
    df_clean, cleaning_log = clean_curveballs(df_filtered, selected_traits)

    # --- TABS ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "🛡️ QC & Cleaning", 
        "🗺️ Spatial Analysis", 
        "📊 Genetic Performance", 
        "👨‍👩‍👧 Combining Ability"
    ])

    # --- TAB 1: QC ---
    with tab1:
        st.header("Data Quality Control")
        if cleaning_log:
            for msg in cleaning_log:
                if "CRITICAL" in msg:
                    st.error(msg)
                else:
                    st.warning(msg)
        
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
        st.markdown("Since plot coordinates overlap between trials, **select one experiment** to view at a time.")
        
        col_map_1, col_map_2 = st.columns([1, 3])
        
        with col_map_1:
            trait_to_map = st.selectbox("Trait", selected_traits)
            expt_to_map = st.selectbox("Select Experiment", selected_expts)
        
        with col_map_2:
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
                st.warning("No data found for this selection.")

    # --- TAB 3: BLUPs ---
    with tab3:
        st.header("Genotype Performance (BLUPs)")
        st.markdown("This model accounts for Spatial Variation across **all selected experiments**.")
        
        trait_stats = st.selectbox("Select Trait for Analysis", selected_traits, key='stats_select')
        
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
                    
                    # Table and Download
                    st.dataframe(results_df)
                    csv = results_df.to_csv().encode('utf-8')
                    st.download_button("Download BLUPs CSV", csv, f"BLUPs_{trait_stats}.csv", "text/csv")
                    
                    with st.expander("View Model Summary"):
                        st.text(debug_info) # Converted to string in function
                else:
                    st.error("Analysis Failed")
                    st.error(debug_info)

    # --- TAB 4: PARENTS ---
    with tab4:
        st.header("Parental Analysis")
        analysis_trait = st.selectbox("Select Trait for Parents", selected_traits, key='parent_select')
        
        col_p1, col_p2 = st.columns(2)
        
        with col_p1:
            st.subheader("Male Performance")
            male_stats = df_clean.groupby(col_male)[analysis_trait].agg(['mean', 'count']).sort_values('mean', ascending=False)
            st.dataframe(male_stats) 

        with col_p2:
            st.subheader("Female Performance")
            fem_stats = df_clean.groupby(col_female)[analysis_trait].agg(['mean', 'count']).sort_values('mean', ascending=False)
            st.dataframe(fem_stats)

if __name__ == "__main__":
    main()
