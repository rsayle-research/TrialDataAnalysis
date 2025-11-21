import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.formula.api as smf
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
    """
    if not key.startswith(f"{prefix}["):
        return key
    key_content = key[len(prefix)+1:]
    match = re.search(r'C\([^)]+\)\[([^\]]+)\]', key_content)
    if match:
        return match.group(1)
    if key_content.endswith(']'):
        key_content = key_content[:-1]
    inner_match = re.search(r'\[([^\]]+)\]$', key_content)
    if inner_match:
        return inner_match.group(1)
    return key_content

def clean_curveballs(df, traits):
    log = []
    df_clean = df.copy()
    for trait in traits:
        if not pd.api.types.is_numeric_dtype(df_clean[trait]):
            df_clean[trait] = pd.to_numeric(df_clean[trait], errors='coerce')
            log.append(f"⚠️ Column '{trait}' contained non-numeric data. Values converted to NA.")
        neg_mask = df_clean[trait] < 0
        neg_count = neg_mask.sum()
        if neg_count > 0:
            df_clean.loc[neg_mask, trait] = np.nan
            log.append(f"🚨 **CRITICAL:** Found {neg_count} negative values in '{trait}'. Set to NA.")
    return df_clean, log

def run_hybrid_model(df, trait, gen_col, row_col, col_col, expt_col, analyze_separate):
    progress_bar = st.progress(0, text="Initializing Model...")
    results_container = []
    summary_stats = []
    debug_log = []

    if analyze_separate:
        datasets = [(expt, df[df[expt_col] == expt].copy()) for expt in df[expt_col].unique()]
        total_runs = len(datasets)
    else:
        datasets = [("Combined Analysis", df.copy())]
        total_runs = 1

    for i, (run_name, model_data) in enumerate(datasets):
        step_val = int((i / total_runs) * 100)
        progress_bar.progress(step_val, text=f"Processing **{run_name}**: Preparing Matrix...")
        
        convergence_status = "❌ Failed"
        convergence_message = ""
        optimizer_used = "None"

        try:
            model_data = model_data.dropna(subset=[trait, gen_col, row_col, col_col]).copy()
            
            # --- CALC RAW STATS (Confidence) ---
            # Calculate N and SE before modeling to attach to results
            grp_stats = model_data.groupby(gen_col)[trait].agg(['count', 'sem'])
            grp_stats.columns = ['N_Plots', 'Raw_SE']

            if model_data[gen_col].nunique() < 2:
                 debug_log.append(f"ERROR in {run_name}: Not enough unique genotypes.")
                 convergence_message = f"Only {model_data[gen_col].nunique()} unique genotype(s)."
                 summary_stats.append({
                     'Experiment ID': run_name, 'Status': convergence_status, 'Message': convergence_message,
                     'Optimizer': optimizer_used, 'Mean': 'N/A', 'CV (%)': 'N/A', 'Genotype Var (Vg)': 'N/A',
                     'Residual Var (Ve)': 'N/A', 'Est. Heritability (H2, %)': 'N/A',
                     'Plots': len(model_data), 'Unique Genotypes': model_data[gen_col].nunique()
                 })
                 continue

            model_data["Unique_Row"] = model_data[expt_col].astype(str) + "_" + model_data[row_col].astype(str)
            model_data["Unique_Col"] = model_data[expt_col].astype(str) + "_" + model_data[col_col].astype(str)
            model_data["Global_Group"] = 1

            if not analyze_separate and model_data[expt_col].nunique() > 1:
                formula = f"{trait} ~ C({expt_col})"
            else:
                formula = f"{trait} ~ 1"

            vc = {
                "Genotype": f"0 + C({gen_col})",
                "SpatialRow": f"0 + C(Unique_Row)",
                "SpatialCol": f"0 + C(Unique_Col)"
            }

            progress_bar.progress(min(step_val + 10, 100), text=f"Processing **{run_name}**: Fitting Spatial Model (REML)...")

            model = smf.mixedlm(formula, model_data, groups="Global_Group", vc_formula=vc)
            result = None
            optimizers = ['powell', 'lbfgs', 'bfgs']

            for opt in optimizers:
                try:
                    result = model.fit(method=opt, reml=True)
                    optimizer_used = opt
                    if hasattr(result, 'converged'):
                        if result.converged:
                            convergence_status = "✓ Good"
                            convergence_message = "Model converged successfully"
                            break
                        else:
                            convergence_status = "⚠️ Caution"
                            convergence_message = "Weak convergence"
                    else:
                        convergence_status = "✓ Good"
                        convergence_message = "Model fitted successfully"
                        break
                except Exception as opt_error:
                    debug_log.append(f"Optimizer {opt} failed for {run_name}: {str(opt_error)}")
                    continue

            if result is None:
                raise Exception("Model fitting failed with all optimizers")

            progress_bar.progress(min(step_val + 20, 100), text=f"Processing **{run_name}**: Extracting BLUPs...")

            re_dict = result.random_effects[1]
            geno_blups = {}
            for key, val in re_dict.items():
                if key.startswith("Genotype["):
                    clean_name = extract_genotype_name(key, "Genotype")
                    geno_blups[clean_name] = val

            temp_df = pd.DataFrame.from_dict(geno_blups, orient='index', columns=[f'BLUP_{trait}'])
            temp_df.index.name = 'Genotype'
            intercept = result.params.get('Intercept', 0)
            temp_df[f'Predicted_{trait}'] = temp_df[f'BLUP_{trait}'] + intercept
            
            # JOIN STATS
            temp_df = temp_df.join(grp_stats, how='left')
            
            temp_df['Analysis_Group'] = run_name

            results_container.append(temp_df)
            debug_log.append(f"\n{'='*30}\nMODEL OUTPUT: {run_name}\n{'='*30}\n{result.summary().as_text()}")

            # Stats extraction
            v_g = result.vcomp[0] if (hasattr(result, "vcomp") and len(result.vcomp) > 0) else None
            v_e = result.scale
            h2 = 'N/A'
            h2_val = None

            if v_g is None or v_g <= 0:
                convergence_status = "⚠️ Caution"
                convergence_message = "No genetic variation detected (Vg ≤ 0)" if v_g is not None else "Genetic var failed"
            elif v_e is not None and model_data[gen_col].nunique() > 1:
                avg_plots_per_geno = model_data.groupby(gen_col).size().mean()
                h2_val = v_g / (v_g + (v_e / avg_plots_per_geno))
                h2 = round(h2_val * 100, 1)
                if convergence_status == "✓ Good":
                    if h2_val < 0.10:
                        convergence_status = "⚠️ Caution"
                        convergence_message = f"Low heritability ({h2}%)"
                    elif h2_val > 0.95:
                        convergence_status = "⚠️ Caution"
                        convergence_message = f"Unusually high heritability ({h2}%)"

            mean = model_data[trait].mean()
            std_dev = model_data[trait].std()
            cv = (std_dev / mean) * 100 if mean != 0 else 0

            summary_stats.append({
                'Experiment ID': run_name, 'Status': convergence_status, 'Message': convergence_message,
                'Optimizer': optimizer_used, 'Mean': round(mean, 3), 'CV (%)': round(cv, 1),
                'Genotype Var (Vg)': round(v_g, 3) if v_g is not None and v_g > 0 else 'N/A',
                'Residual Var (Ve)': round(v_e, 3) if v_e is not None else 'N/A',
                'Est. Heritability (H2, %)': h2,
                'Plots': len(model_data), 'Unique Genotypes': model_data[gen_col].nunique()
            })

        except Exception as e:
            error_msg = str(e)
            user_msg = "Not enough variation or singular matrix" if "singular" in error_msg.lower() else error_msg[:100]
            debug_log.append(f"\n{'='*30}\nERROR: {run_name}\n{'='*30}\n{error_msg}")
            summary_stats.append({
                'Experiment ID': run_name, 'Status': "❌ Failed", 'Message': user_msg,
                'Optimizer': optimizer_used, 'Mean': 'N/A', 'CV (%)': 'N/A', 'Genotype Var (Vg)': 'N/A',
                'Residual Var (Ve)': 'N/A', 'Est. Heritability (H2, %)': 'N/A',
                'Plots': 0, 'Unique Genotypes': 0
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
    progress_bar = st.progress(0, text="Calculating GCA (Parental BLUPs)...")
    male_results = []
    female_results = []
    debug_messages = []
    experiments = df[expt_col].dropna().unique().tolist()
    total = max(1, len(experiments))

    for i, expt in enumerate(experiments):
        progress_bar.progress(int((i / total) * 100), text=f"Running GCA for {expt}...")
        expt_data = df[df[expt_col] == expt].dropna(subset=[trait, male_col, female_col, row_col, col_col]).copy()

        if expt_data.empty:
            debug_messages.append(f"[{expt}] Skipped: no valid rows.")
            continue

        def get_stats(data, group_col, value_col):
            stats = data.groupby(group_col)[value_col].agg(['count', 'sem'])
            stats.columns = ['N_Progeny', 'Raw_SE']
            return stats

        male_stats = get_stats(expt_data, male_col, trait)
        female_stats = get_stats(expt_data, female_col, trait)

        expt_data["Unique_Row"] = expt_data[row_col].astype(str)
        expt_data["Unique_Col"] = expt_data[col_col].astype(str)
        expt_data["Global_Group"] = 1
        
        formula = f"{trait} ~ 1"
        vc = {
            "Male_GCA": f"0 + C({male_col})",
            "Female_GCA": f"0 + C({female_col})",
            "SpatialRow": "0 + C(Unique_Row)",
            "SpatialCol": "0 + C(Unique_Col)"
        }

        try:
            model = smf.mixedlm(formula, expt_data, groups="Global_Group", vc_formula=vc)
            result = model.fit(method='powell', reml=True)
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

            if male_gca:
                df_m = pd.DataFrame.from_dict(male_gca, orient='index', columns=[f'GCA_{trait}'])
                df_m.index.name = 'Male Parent'
                df_m = df_m.join(male_stats, how='left')
                df_m.reset_index(inplace=True)
                df_m['Group'] = expt
                male_results.append(df_m)

            if female_gca:
                df_f = pd.DataFrame.from_dict(female_gca, orient='index', columns=[f'GCA_{trait}'])
                df_f.index.name = 'Female Parent'
                df_f = df_f.join(female_stats, how='left')
                df_f.reset_index(inplace=True)
                df_f['Group'] = expt
                female_results.append(df_f)

            debug_messages.append(f"\n{'='*30}\nMODEL OUTPUT: {expt}\n{'='*30}\n{result.summary().as_text()}")

        except Exception as e:
            debug_messages.append(f"[{expt}] ERROR: {str(e)}")
            continue

    progress_bar.progress(100, text="Finalizing GCA Results...")
    time.sleep(0.2)
    progress_bar.empty()

    male_df = pd.concat(male_results, ignore_index=True) if male_results else None
    female_df = pd.concat(female_results, ignore_index=True) if female_results else None

    success_flag = True if (male_df is not None or female_df is not None) else False
    return success_flag, male_df, female_df, "\n".join(debug_messages)


# --- MAIN APP ---

def main():
    # Streamlit state initialization
    if 'results_df' not in st.session_state: st.session_state.results_df = None
    if 'stats_df' not in st.session_state: st.session_state.stats_df = None
    if 'debug_log' not in st.session_state: st.session_state.debug_log = None
    if 'analysis_mode' not in st.session_state: st.session_state.analysis_mode = "Analyze experiments separately"
    if 'trait_ran' not in st.session_state: st.session_state.trait_ran = None
    
    if 'gca_male_df' not in st.session_state: st.session_state.gca_male_df = None
    if 'gca_female_df' not in st.session_state: st.session_state.gca_female_df = None
    if 'gca_debug' not in st.session_state: st.session_state.gca_debug = None
    if 'gca_trait_ran' not in st.session_state: st.session_state.gca_trait_ran = None

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

    col_map = {}
    gca_enabled = False
    if df is not None:
        st.sidebar.divider()
        st.sidebar.header("2. Map Columns")
        all_cols = ["Select Column..."] + df.columns.tolist()

        col_map['expt'] = st.sidebar.selectbox("Experiment ID", all_cols)
        col_map['geno'] = st.sidebar.selectbox("Genotype", all_cols)
        col_map['row'] = st.sidebar.selectbox("Plot Row", all_cols)
        col_map['col'] = st.sidebar.selectbox("Plot Column", all_cols)

        with st.sidebar.expander("GCA Parental Lines (Optional)"):
            st.info("GCA analysis can be performed if parental lines are selected for hybrid crops.")
            col_map['male'] = st.selectbox("Male Parent (Required for GCA)", all_cols)
            col_map['female'] = st.selectbox("Female Parent (Required for GCA)", all_cols)

        gca_enabled = (col_map['male'] != "Select Column...") and (col_map['female'] != "Select Column...")

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

    if df is None:
        st.info("👈 Upload a CSV file to begin.")
        return

    if not selected_expts or not selected_traits:
        st.sidebar.warning("Please select experiments and traits.")
        return

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
                if "CRITICAL" in msg: st.error(msg)
                else: st.warning(msg)
        else:
            st.success("No statistical anomalies detected.")
        
        c_stats, c_missing = st.columns(2)
        with c_stats:
            st.markdown(f"**Total Plots:** `{len(df_clean)}`")
            st.markdown(f"**Unique Genotypes:** `{df_clean[col_map['geno']].nunique()}`")
        with c_missing:
            st.dataframe(df_clean[selected_traits].isnull().sum(), use_container_width=True)

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
            c2.warning(f"No data for {map_expt}")

    # --- TAB 3: GENOTYPE PERFORMANCE ---
    with tab_perf:
        st.header("Genotype Performance Analysis (BLUPs)")
        col_opts, col_act = st.columns([2, 1])
        perf_trait = col_opts.selectbox("Trait to Analyze", selected_traits, key='perf_trait')
        
        def update_analysis_mode():
            st.session_state.analysis_mode = st.session_state.temp_analysis_mode
            st.session_state.results_df = None

        analysis_mode = col_opts.radio(
            "Analysis Strategy", 
            ["Analyze experiments separately", "Analyze all experiments as one group"],
            key='temp_analysis_mode',
            on_change=update_analysis_mode
        )
        
        separate_flag = (st.session_state.analysis_mode == "Analyze experiments separately")
        col_act.write("") 
        col_act.write("") 
        if col_act.button("🚀 Run Genotype Analysis", type="primary"):
            success, res_df, stats_df, debug = run_hybrid_model(
                df_clean, perf_trait, col_map['geno'], col_map['row'], col_map['col'], col_map['expt'], separate_flag
            )
            st.session_state.results_df = res_df
            st.session_state.stats_df = stats_df
            st.session_state.debug_log = debug
            st.session_state.trait_ran = perf_trait
            st.session_state.analysis_mode_ran = st.session_state.analysis_mode 

        # --- RESULTS DISPLAY ---
        if st.session_state.results_df is not None and st.session_state.trait_ran is not None:
            res_df = st.session_state.results_df
            trait_ran = st.session_state.trait_ran
            
            st.subheader(f"Results for: {trait_ran}")
            
            # --- 1. EXPERIMENT SUMMARY TABLE ---
            if st.session_state.stats_df is not None:
                st.markdown("### 📋 Experiment Analysis Summary")
                
                # Function to color code status
                def color_status_rows(row):
                    color = ''
                    if 'Good' in row['Status']:
                        color = 'background-color: #d4edda; color: #155724' # Green
                    elif 'Caution' in row['Status']:
                        color = 'background-color: #fff3cd; color: #856404' # Yellow/Orange
                    elif 'Failed' in row['Status']:
                        color = 'background-color: #f8d7da; color: #721c24' # Red
                    return [color] * len(row)

                stats_styled = st.session_state.stats_df.style.apply(color_status_rows, axis=1)
                st.dataframe(stats_styled, use_container_width=True, hide_index=True)
                
                # Dedicated Download Button for Summary (Fixes Issue 4)
                st.download_button(
                    "Download Summary Table (CSV)",
                    st.session_state.stats_df.to_csv(index=False).encode('utf-8'),
                    "Experiment_Summary_Stats.csv"
                )
                st.divider()

            # --- 2. GENOTYPE RESULTS TABLE ---
            st.markdown("### 🏆 Genotype Rankings")
            
            # Sort by Predicted Value
            current_view = res_df.sort_values(by=f"Predicted_{trait_ran}", ascending=False)
            
            # Apply Gradient Coloring (Heatmap)
            # Target columns: BLUP_{trait} and Predicted_{trait}
            cols_to_color = [f'BLUP_{trait_ran}', f'Predicted_{trait_ran}']
            
            styled_results = current_view.style.background_gradient(
                subset=cols_to_color, 
                cmap="Blues"
            ).format({
                f'BLUP_{trait_ran}': "{:.3f}",
                f'Predicted_{trait_ran}': "{:.3f}",
                'Raw_SE': "{:.3f}"
            })
            
            st.dataframe(styled_results, use_container_width=True, hide_index=True)
            
            st.download_button("Download Genotype Results (CSV)", res_df.to_csv().encode('utf-8'), f"Genotype_Results_{trait_ran}.csv")

            # --- 3. MODEL OUTPUT DROPDOWN (Req 1) ---
            with st.expander("Model Outputs & Logs (Detailed Stats)"):
                st.text(st.session_state.debug_log)


    # --- TAB 4: PARENTAL GCA ---
    with tab_parents:
        if not gca_enabled:
            st.header("Parental GCA Disabled")
            st.warning("Select 'Male Parent' and 'Female Parent' in the sidebar to enable GCA.")
        else:
            st.header("Parental GCA (General Combining Ability)")
            st.markdown("""
            **Interpretation:**
            * **GCA:** The breeding value. Higher is better (usually).
            * **N_Progeny:** The number of times this parent was tested (Confidence metric).
            * **Raw_SE:** Standard error of the raw data. Lower means more consistent performance.
            """)
            
            gca_trait = st.selectbox("Trait for GCA", selected_traits, key='gca_trait')
            
            if st.button("Calculate GCA", key='run_gca_btn', type='primary'):
                success, male_df, female_df, debug = run_parental_model(
                    df_clean, gca_trait, col_map['male'], col_map['female'], col_map['row'], col_map['col'], col_map['expt']
                )
                
                if success:
                    st.session_state.gca_male_df = male_df
                    st.session_state.gca_female_df = female_df
                    st.session_state.gca_debug = debug
                    st.session_state.gca_trait_ran = gca_trait
                else:
                    st.error("GCA Model Failed")
                    st.text(debug)

            if st.session_state.gca_male_df is not None or st.session_state.gca_female_df is not None:
                st.caption(f"Showing results for: **{st.session_state.gca_trait_ran}**")
                col_m, col_f = st.columns(2)
                
                with col_m:
                    st.subheader("Male Parent Results")
                    if st.session_state.gca_male_df is not None:
                        m_df = st.session_state.gca_male_df.sort_values(by=f"GCA_{st.session_state.gca_trait_ran}", ascending=False)
                        m_df['Raw_SE'] = m_df['Raw_SE'].round(2)
                        st.dataframe(
                            m_df.style.background_gradient(subset=[f"GCA_{st.session_state.gca_trait_ran}"], cmap="Blues"), 
                            use_container_width=True, hide_index=True 
                        )
                        st.download_button(f"Download Male GCA (CSV)", m_df.to_csv(index=False).encode('utf-8'), f"Male_GCA_{st.session_state.gca_trait_ran}.csv", key='dl_male_gca')
                    else:
                        st.info("No Male GCA calculated.")

                with col_f:
                    st.subheader("Female Parent Results")
                    if st.session_state.gca_female_df is not None:
                        f_df = st.session_state.gca_female_df.sort_values(by=f"GCA_{st.session_state.gca_trait_ran}", ascending=False)
                        f_df['Raw_SE'] = f_df['Raw_SE'].round(2)
                        st.dataframe(
                            f_df.style.background_gradient(subset=[f"GCA_{st.session_state.gca_trait_ran}"], cmap="Reds"), 
                            use_container_width=True, hide_index=True
                        )
                        st.download_button(f"Download Female GCA (CSV)", f_df.to_csv(index=False).encode('utf-8'), f"Female_GCA_{st.session_state.gca_trait_ran}.csv", key='dl_female_gca')

                # --- MODEL OUTPUT DROPDOWN (Req 1) ---
                with st.expander("Model Outputs & Logs (Detailed Stats)"):
                    st.text(st.session_state.gca_debug)

if __name__ == "__main__":
    main()
