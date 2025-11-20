import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.formula.api as smf
import io

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Canola Breeding Analytics",
    page_icon="🧬",
    layout="wide"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .stAlert {
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---

def validate_data(df):
    """Checks for critical issues in the dataset structure."""
    issues = []
    
    # Check if empty
    if df.empty:
        return False, ["The uploaded file is empty."]
        
    return True, issues

def clean_curveballs(df, traits):
    """
    Identifies and cleans statistical anomalies (curveballs).
    1. Converts non-numeric traits to numeric (coercing errors).
    2. Detects negative values in yield data (biological impossibility).
    """
    log = []
    df_clean = df.copy()
    
    for trait in traits:
        # 1. Force Numeric
        if not pd.api.types.is_numeric_dtype(df_clean[trait]):
            df_clean[trait] = pd.to_numeric(df_clean[trait], errors='coerce')
            log.append(f"⚠️ Column '{trait}' contained non-numeric data. These values were converted to NA.")
        
        # 2. Negative Value Detection (The "Curveball")
        neg_mask = df_clean[trait] < 0
        neg_count = neg_mask.sum()
        if neg_count > 0:
            df_clean.loc[neg_mask, trait] = np.nan
            log.append(f"🚨 **CRITICAL:** Found {neg_count} negative values in '{trait}'. These are biologically impossible for yield/counts and have been set to NA to prevent model bias.")
            
    return df_clean, log

def run_stats_model(df, trait, gen_col, row_col, col_col):
    """
    Runs a Linear Mixed Model (LMM).
    Equation: Trait ~ 1 + (1|Genotype) + (1|Row) + (1|Col)
    Returns BLUPs (Best Linear Unbiased Predictions).
    """
    try:
        # Drop NAs for the specific run
        model_data = df.dropna(subset=[trait, gen_col, row_col, col_col])
        
        # Define Formula: Intercept + Random Effects
        # 'vc' structure implies variance components (random effects)
        model = smf.mixedlm(
            f"{trait} ~ 1", 
            model_data, 
            groups="Expt_Dummy",  # Dummy group if single experiment
            vc_formula={
                "Genotype": f"0 + C({gen_col})",
                "Row": f"0 + C({row_col})",
                "Col": f"0 + C({col_col})"
            }
        )
        
        # Add a dummy group column for statsmodels requirement if checking single field
        model_data["Expt_Dummy"] = 1
        
        result = model.fit()
        
        # Extract BLUPs (Random Effects)
        re = result.random_effects[1] # Get the random effects dict
        
        # Parse Genotype BLUPs
        geno_blups = {}
        for key, val in re.items():
            if key.startswith("Genotype["):
                # Clean string to get genotype name
                geno_name = key.replace(f"Genotype[C({gen_col})][", "").replace("]", "")
                geno_blups[geno_name] = val
        
        blup_df = pd.DataFrame.from_dict(geno_blups, orient='index', columns=[f'BLUP_{trait}'])
        
        # Add Intercept to get Predicted Value
        intercept = result.params['Intercept']
        blup_df[f'Predicted_{trait}'] = blup_df[f'BLUP_{trait}'] + intercept
        
        return True, blup_df, result.summary()
        
    except Exception as e:
        return False, None, str(e)

# --- MAIN APP ---

def main():
    st.title("🧬 Gold Standard Plant Breeding Analytics")
    st.markdown("### Hybrid Trial Analysis Pipeline")
    st.write("Upload your trial CSV. The app will validate structure, clean anomalies (like negative yields), and perform spatial and genetic analysis.")

    # 1. SIDEBAR - DATA UPLOAD
    with st.sidebar:
        st.header("1. Upload Data")
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success("File Uploaded Successfully")
            except Exception as e:
                st.error(f"Error reading file: {e}")
                return
        else:
            st.info("Awaiting CSV file upload.")
            return

    # 2. COLUMN MAPPING (CRITICAL FOR ROBUSTNESS)
    st.sidebar.header("2. Map Columns")
    st.sidebar.info("Map your file headers to the required analysis fields.")
    
    all_cols = df.columns.tolist()
    
    # Heuristic to guess columns
    def get_idx(options, query):
        matches = [i for i, x in enumerate(options) if query.lower() in x.lower()]
        return matches[0] if matches else 0

    col_genotype = st.sidebar.selectbox("Genotype/Hybrid Name", all_cols, index=get_idx(all_cols, "name"))
    col_row = st.sidebar.selectbox("Plot Row", all_cols, index=get_idx(all_cols, "row"))
    col_col = st.sidebar.selectbox("Plot Column", all_cols, index=get_idx(all_cols, "col"))
    col_male = st.sidebar.selectbox("Male Parent", all_cols, index=get_idx(all_cols, "male"))
    col_female = st.sidebar.selectbox("Female Parent", all_cols, index=get_idx(all_cols, "female"))
    
    # 3. VARIATE SELECTION
    st.sidebar.header("3. Select Traits")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    # Exclude spatial cols from numeric choices if selected
    potential_traits = [c for c in all_cols if c not in [col_row, col_col]]
    
    selected_traits = st.sidebar.multiselect(
        "Select Phenotypes to Analyze (e.g., Yield, Oil)", 
        potential_traits,
        default=[x for x in potential_traits if "yield" in x.lower()]
    )

    if not selected_traits:
        st.warning("Please select at least one trait in the sidebar to begin analysis.")
        return

    # --- MAIN ANALYSIS TABS ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "🛡️ QC & Cleaning", 
        "🗺️ Spatial Analysis", 
        "📊 Genetic Performance", 
        "👨‍👩‍👧 Combining Ability"
    ])

    # --- LOGIC: DATA CLEANING ---
    df_clean, cleaning_log = clean_curveballs(df, selected_traits)

    # --- TAB 1: QC ---
    with tab1:
        st.header("Data Quality Control")
        
        # Display Cleaning Log
        if cleaning_log:
            for msg in cleaning_log:
                if "CRITICAL" in msg:
                    st.error(msg)
                else:
                    st.warning(msg)
        else:
            st.success("✅ No data anomalies detected.")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Dataset Overview")
            st.write(f"**Total Plots:** {len(df)}")
            st.write(f"**Unique Genotypes:** {df[col_genotype].nunique()}")
            st.write(f"**Male Parents:** {df[col_male].nunique()}")
            st.write(f"**Female Parents:** {df[col_female].nunique()}")
        
        with col2:
            st.subheader("Missing Data Profile")
            missing = df_clean[selected_traits].isnull().sum()
            st.dataframe(missing, use_container_width=True)

        st.subheader("Raw Data Preview")
        st.dataframe(df_clean.head())

    # --- TAB 2: SPATIAL ---
    with tab2:
        st.header("Spatial Field Map")
        st.markdown("Visualizing the field layout helps identify environmental trends (e.g., a salty patch in the corner).")
        
        trait_to_map = st.selectbox("Select Trait for Heatmap", selected_traits)
        
        # Heatmap
        if df_clean[col_row].nunique() > 1 and df_clean[col_col].nunique() > 1:
            pivot = df_clean.pivot_table(index=col_row, columns=col_col, values=trait_to_map)
            
            fig = px.imshow(
                pivot, 
                color_continuous_scale='Viridis',
                title=f"Spatial Distribution: {trait_to_map}",
                labels=dict(x="Plot Column", y="Plot Row", color=trait_to_map),
                aspect="auto"
            )
            fig.update_yaxes(autorange="reversed") # Map orientation
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Insufficient Row/Column variation to generate a heatmap.")

    # --- TAB 3: GENETIC MODELING (BLUPs) ---
    with tab3:
        st.header("Genotype Performance (BLUPs)")
        st.markdown("""
        **Methodology:** We are fitting a Linear Mixed Model (LMM) where Genotypes, Rows, and Columns are treated as Random Effects. 
        This produces **BLUPs** (Best Linear Unbiased Predictions), which are 'shrunken' towards the mean to provide conservative, reliable rankings.
        """)
        
        trait_stats = st.selectbox("Select Trait for Ranking", selected_traits, key='stats_select')
        
        if st.button(f"Run Model for {trait_stats}"):
            with st.spinner("Fitting Linear Mixed Model... (This uses REML)"):
                success, results_df, debug_info = run_stats_model(df_clean, trait_stats, col_genotype, col_row, col_col)
                
                if success:
                    st.success("Model Converged Successfully!")
                    
                    # Sort and Display
                    results_df = results_df.sort_values(by=f'Predicted_{trait_stats}', ascending=False)
                    
                    # Top 10 Graph
                    top_10 = results_df.head(15)
                    fig_bar = px.bar(
                        top_10, 
                        y=top_10.index, 
                        x=f'Predicted_{trait_stats}',
                        orientation='h',
                        title=f"Top 15 Hybrids (BLUP Values) for {trait_stats}",
                        color=f'Predicted_{trait_stats}',
                        color_continuous_scale='Greens'
                    )
                    fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_bar, use_container_width=True)
                    
                    # Download Button
                    st.subheader("Full Results Table")
                    st.dataframe(results_df)
                    
                    csv = results_df.to_csv().encode('utf-8')
                    st.download_button(
                        label="📥 Download BLUPs as CSV",
                        data=csv,
                        file_name=f'BLUP_results_{trait_stats}.csv',
                        mime='text/csv',
                    )
                    
                    with st.expander("View Statistical Model Summary"):
                        st.text(debug_info)
                else:
                    st.error("Model Failed to Converge.")
                    st.text(debug_info)

    # --- TAB 4: PARENTAL ANALYSIS ---
    with tab4:
        st.header("Parental Analysis (GCA Proxy)")
        st.markdown("Analysis of Male and Female parent performance (General Combining Ability).")
        
        analysis_trait = st.selectbox("Select Trait for Parents", selected_traits, key='parent_select')
        
        col_p1, col_p2 = st.columns(2)
        
        # Male Analysis
        with col_p1:
            st.subheader("Top Male Parents")
            male_means = df_clean.groupby(col_male)[analysis_trait].mean().sort_values(ascending=False).reset_index()
            male_means.columns = ['Male Parent', f'Mean {analysis_trait}']
            st.dataframe(male_means.style.background_gradient(cmap='Blues'))
            
        # Female Analysis
        with col_p2:
            st.subheader("Top Female Parents")
            female_means = df_clean.groupby(col_female)[analysis_trait].mean().sort_values(ascending=False).reset_index()
            female_means.columns = ['Female Parent', f'Mean {analysis_trait}']
            st.dataframe(female_means.style.background_gradient(cmap='Reds'))

        # Interaction Heatmap (SCA visualization)
        st.subheader("Male x Female Interaction Grid")
        if len(male_means) < 50 and len(female_means) < 50: # Limit size for performance
            pivot_parents = df_clean.pivot_table(index=col_female, columns=col_male, values=analysis_trait)
            fig_parent = px.imshow(
                pivot_parents,
                title=f"Hybrid Performance Grid: {analysis_trait}",
                labels=dict(x="Male Parent", y="Female Parent", color=analysis_trait),
                color_continuous_scale='RdBu_r'
            )
            st.plotly_chart(fig_parent, use_container_width=True)
            
            csv_parents = pivot_parents.to_csv().encode('utf-8')
            st.download_button(
                label="📥 Download Interaction Matrix",
                data=csv_parents,
                file_name=f'Parent_Interaction_{analysis_trait}.csv',
                mime='text/csv',
            )
        else:
            st.info("Parental interaction grid is too large to visualize comfortably.")

if __name__ == "__main__":
    main()
