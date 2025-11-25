import streamlit as st
import pandas as pd
import numpy as np
import re

st.set_page_config(page_title="Pricing Strategy Engine Pro", layout="wide")

# --- UTILS & CALCS ---

def clean_accounting_number(val):
    """
    Robust cleaner that handles:
    - Currency: $1,000.00 -> 1000.0
    - Accounting Negatives: (500.00) -> -500.0
    - Empty/Dash: '-' -> 0
    """
    if pd.isna(val) or val == '':
        return 0.0
    
    val_str = str(val).strip()
    
    if val_str == '-' or val_str == 'Placeholder':
        return 0.0
        
    # Check for accounting format (123.45) -> -123.45
    if val_str.startswith('(') and val_str.endswith(')'):
        val_str = '-' + val_str[1:-1]
        
    # Remove common non-numeric chars
    val_str = val_str.replace('$', '').replace(',', '').replace(' ', '')
    
    try:
        return float(val_str)
    except:
        return 0.0

def load_data(uploaded_file):
    """Loads and cleans the main Data.csv file."""
    try:
        df = pd.read_csv(uploaded_file)
        
        # Map columns to standard names
        cols_map = {
            'Full Amount': 'Full_List_Price',            
            'Avg Laid-In': 'Base_Cost',                  
            'Avg Full Price (withDiscount)': 'Avg_Price',
            'Average Discount': 'Avg_Discount',
            'Units (L12mo)': 'Units', 
            'Container Type': 'Container_Type',          
            'Supplier Family': 'Family',                 
            'Supplier': 'Supplier',
            'Product': 'Product',
            'Segment': 'Segment',
            'Package': 'Package'
        }
        
        # Rename columns that exist
        df = df.rename(columns=cols_map)

        # Ensure critical columns exist, if not create them with 0
        expected_cols = ['Full_List_Price', 'Base_Cost', 'Avg_Price', 'Avg_Discount', 'Units']
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0.0

        # CLEAN DATA (Apply the new robust cleaner)
        for col in expected_cols:
            df[col] = df[col].apply(clean_accounting_number)

        # --- MAJOR CHANGE: DO NOT DROP ROWS ---
        # Instead of dropping rows with missing cost, we keep them so Units match.
        # We fill NaNs with 0 if any slipped through.
        df = df.fillna(0)
        
        return df
    except Exception as e:
        st.error(f"Error loading Data.csv: {e}")
        return None

def load_taxco(uploaded_file):
    """Loads the TAXCO.csv to get the tax per package."""
    try:
        df = pd.read_csv(uploaded_file)
        
        col_map = {}
        for col in df.columns:
            if 'Package' in col and 'ID' not in col:
                col_map[col] = 'Package'
            if 'Tax' in col:
                col_map[col] = 'Tax'
        
        df = df.rename(columns=col_map)
        
        if 'Package' not in df.columns or 'Tax' not in df.columns:
            st.warning("TAXCO file missing 'Package' or 'Tax' columns. Taxes set to 0.")
            return None
            
        df['Tax'] = df['Tax'].apply(clean_accounting_number)
        return df[['Package', 'Tax']]
    except Exception as e:
        st.error(f"Error loading TAXCO.csv: {e}")
        return None

def apply_strategy(df, tax_df, global_settings, pkg_overrides, fam_overrides, supp_overrides, prod_overrides):
    """
    The Master Calculation Engine.
    """
    res = df.copy()
    
    # --- 1. SETUP COST BASIS ---
    if tax_df is not None:
        tax_clean = tax_df.drop_duplicates(subset=['Package'])
        res = res.merge(tax_clean, on='Package', how='left')
        res['Tax'] = res['Tax'].fillna(0)
    else:
        res['Tax'] = 0

    # Real Cost 
    res['Current_Cost_Final'] = res['Base_Cost'] + res['Tax']
    
    # Real Current GP
    res['Current_Unit_GP'] = res['Full_List_Price'] - res['Current_Cost_Final']
    res['Current_Line_DGP'] = res['Current_Unit_GP'] * res['Units']
    
    # Metric: Discount Total
    res['Current_Discount_Total'] = (res['Avg_Price'] - res['Full_List_Price']) * res['Units']

    # --- 2. SANITIZE INPUTS ---
    # Apply robust cleaner to inputs too
    if not pkg_overrides.empty and 'Price Increase' in pkg_overrides.columns:
        pkg_overrides['Price Increase'] = pkg_overrides['Price Increase'].apply(clean_accounting_number)
    if not supp_overrides.empty and 'Price Increase' in supp_overrides.columns:
        supp_overrides['Price Increase'] = supp_overrides['Price Increase'].apply(clean_accounting_number)
    if not fam_overrides.empty:
        for col in ['Price Increase', 'Growth %', 'Pkg Split %', 'Keg Split %']:
            if col in fam_overrides.columns:
                fam_overrides[col] = fam_overrides[col].apply(clean_accounting_number)
    if prod_overrides is not None and not prod_overrides.empty:
        prod_overrides['Price Increase'] = prod_overrides['Price Increase'].apply(clean_accounting_number)

    # --- 3. APPLY LOGIC HIERARCHY ---
    res['Inc_Applied'] = float(global_settings['price_inc'])
    res['Growth_Applied'] = float(global_settings['growth'])
    res['Pkg_Split_Applied'] = float(global_settings['pkg_split'])
    res['Keg_Split_Applied'] = float(global_settings['keg_split'])
    res['Source_Rule'] = 'Global'

    def apply_map(source_df, key_col, val_col, target_col, rule_name):
        if not source_df.empty:
            # Only use valid overrides (non-zero or specifically set)
            # For this logic, we assume if it exists in the table, apply it.
            if val_col in source_df.columns:
                mapper = source_df.set_index(key_col)[val_col]
                mask = res[target_col].isin(mapper.index)
                # Map and fill existing
                res.loc[mask, 'Inc_Applied'] = res.loc[mask, target_col].map(mapper).fillna(res.loc[mask, 'Inc_Applied'])
                res.loc[mask, 'Source_Rule'] = rule_name

    # Package
    apply_map(pkg_overrides, 'Package', 'Price Increase', 'Package', 'Package Rule')
    
    # Family (Handles Inc, Growth, Splits)
    if not fam_overrides.empty:
        # Map Price
        if 'Price Increase' in fam_overrides.columns:
            mapper = fam_overrides.set_index('Supplier Family')['Price Increase']
            mask = res['Family'].isin(mapper.index)
            res.loc[mask, 'Inc_Applied'] = res.loc[mask, 'Family'].map(mapper).fillna(res.loc[mask, 'Inc_Applied'])
            res.loc[mask, 'Source_Rule'] = 'Family Rule'
            
        # Map Others
        for col, target in [('Growth %', 'Growth_Applied'), ('Pkg Split %', 'Pkg_Split_Applied'), ('Keg Split %', 'Keg_Split_Applied')]:
            if col in fam_overrides.columns:
                val_map = fam_overrides.set_index('Supplier Family')[col]
                # Only update where valid
                res.loc[mask, target] = res.loc[mask, 'Family'].map(val_map).fillna(res.loc[mask, target])

    # Supplier
    apply_map(supp_overrides, 'Supplier', 'Price Increase', 'Supplier', 'Supplier Rule')
    
    # Product
    if prod_overrides is not None:
        apply_map(prod_overrides, 'Product', 'Price Increase', 'Product', 'Product Exception')

    # --- 4. FINAL CALCULATIONS ---
    
    if 'Container_Type' not in res.columns:
         res['Container_Type'] = ''
         
    is_keg = res['Container_Type'].str.contains('Keg', case=False, na=False)
    
    # A. Cost Increase (Increase * Split)
    res['Applicable_Split'] = np.where(is_keg, res['Keg_Split_Applied'], res['Pkg_Split_Applied'])
    res['Cost_Increase'] = res['Inc_Applied'] * (res['Applicable_Split'] / 100.0)
    
    # B. New Unit GP
    res['New_Unit_GP'] = res['Current_Unit_GP'] + res['Inc_Applied'] - res['Cost_Increase']
    
    # C. New Units
    res['New_Units'] = res['Units'] * (1 + (res['Growth_Applied'] / 100.0))
    
    # D. Projected DGP
    res['New_DGP'] = res['New_Unit_GP'] * res['New_Units']
    
    # E. Export Columns
    res['New_Price_Full'] = res['Full_List_Price'] + res['Inc_Applied']
    res['New_Cost'] = res['Current_Cost_Final'] + res['Cost_Increase']
    
    # Handle division by zero for % columns
    res['Current_GP_Pct'] = np.where(res['Full_List_Price'] != 0, res['Current_Unit_GP'] / res['Full_List_Price'], 0)
    res['New_GP_Pct'] = np.where(res['New_Price_Full'] != 0, res['New_Unit_GP'] / res['New_Price_Full'], 0)
    
    res['GP_Change'] = res['New_Unit_GP'] - res['Current_Unit_GP']
    res['DGP_Change'] = res['New_DGP'] - res['Current_Line_DGP']

    return res

# --- APP UI ---

st.title("💰 Pricing Strategy Engine Pro")

with st.sidebar:
    st.header("1. Data")
    file_data = st.file_uploader("Upload 'Data.csv'", type=['csv'])
    file_tax = st.file_uploader("Upload 'TAXCO.csv' (Optional)", type=['csv'])
    
    st.divider()
    st.header("2. Global Defaults")
    glob_price = st.number_input("Global Price Increase ($)", 0.0, 20.0, 2.0)
    glob_growth = st.number_input("Global Volume Growth (%)", -20.0, 20.0, 0.0)
    st.caption("Supplier Cost Pass-through:")
    glob_pkg_split = st.slider("Pkg Supplier Share (%)", 0, 100, 70)
    glob_keg_split = st.slider("Keg Supplier Share (%)", 0, 100, 50)
    
    global_settings = {
        'price_inc': glob_price,
        'growth': glob_growth,
        'pkg_split': glob_pkg_split,
        'keg_split': glob_keg_split
    }

if file_data:
    df = load_data(file_data)
    tax_df = load_taxco(file_tax) if file_tax else None
    
    if df is not None:
        if tax_df is None:
            st.warning("Note: No Tax file uploaded. Costs exclude tax.")
            
        # Check for ZERO cost items to warn user
        zero_cost_count = len(df[df['Base_Cost'] == 0])
        if zero_cost_count > 0:
            st.warning(f"⚠️ Warning: {zero_cost_count} products have $0.00 Cost. Profit margins for these will be 100%. Check your Data source.")

        # --- SCENARIO BUILDER ---
        st.subheader("🛠️ Scenario Builder")
        t_pkg, t_fam, t_supp, t_prod = st.tabs(["📦 Package Rules", "🏭 Family Rules", "🚚 Supplier Rules", "💎 Product Exceptions"])
        
        with t_pkg:
            unique_pkgs = sorted(df['Package'].astype(str).unique())
            pkg_df = pd.DataFrame({'Package': unique_pkgs, 'Price Increase': [None]*len(unique_pkgs)})
            edited_pkg = st.data_editor(pkg_df, hide_index=True, use_container_width=True, height=250)
            
        with t_fam:
            unique_fams = sorted(df['Family'].astype(str).unique())
            fam_df = pd.DataFrame({
                'Supplier Family': unique_fams, 
                'Price Increase': [None]*len(unique_fams),
                'Growth %': [None]*len(unique_fams),
                'Pkg Split %': [None]*len(unique_fams),
                'Keg Split %': [None]*len(unique_fams)
            })
            edited_fam = st.data_editor(fam_df, hide_index=True, use_container_width=True, height=250)

        with t_supp:
            unique_supp = sorted(df['Supplier'].astype(str).unique())
            supp_df = pd.DataFrame({'Supplier': unique_supp, 'Price Increase': [None]*len(unique_supp)})
            edited_supp = st.data_editor(supp_df, hide_index=True, use_container_width=True, height=250)
            
        with t_prod:
            st.info("💡 Tip: Download the template below, open it in Excel, fill in your exceptions, and upload it back here.")
            
            # --- AUTO-GENERATE TEMPLATE ---
            sample_prods = df['Product'].head(5).tolist() if not df.empty else ["Example Product A"]
            template_df = pd.DataFrame({'Product': sample_prods, 'Price Increase': [0.50, 1.00, 5.00, 10.00, 2.00]})
            csv_template = template_df.to_csv(index=False).encode('utf-8')
            
            st.download_button("📥 Download Template CSV", csv_template, "product_exceptions_template.csv", "text/csv")
            
            st.markdown("---")
            prod_file = st.file_uploader("Upload Exceptions CSV", type=['csv'])
            prod_overrides = pd.read_csv(prod_file) if prod_file else None

        # --- RUN ---
        st.divider()
        if st.button("🚀 Run Scenario Logic", type="primary"):
            
            results = apply_strategy(df, tax_df, global_settings, edited_pkg, edited_fam, edited_supp, prod_overrides)
            
            # --- METRICS ---
            total_discount = results['Current_Discount_Total'].sum()
            total_current_dgp = results['Current_Line_DGP'].sum()
            total_proj_dgp = results['New_DGP'].sum()
            total_impact = total_proj_dgp - total_current_dgp
            
            st.markdown("### 📊 Key Metrics")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Discount (L12mo)", f"${total_discount:,.0f}")
            k2.metric("Current Total DGP", f"${total_current_dgp:,.0f}")
            k3.metric("Projected DGP", f"${total_proj_dgp:,.0f}")
            k4.metric("Total DGP Change Impact", f"${total_impact:,.0f}", delta_color="normal")
            
            st.divider()
            
            st.write("**Rules Breakdown:**")
            st.bar_chart(results['Source_Rule'].value_counts())
            
            with st.expander("🔎 View Detailed Data (Price Scenarios Format)", expanded=True):
                # --- PREPARE EXPORT DATAFRAME ---
                export_df = pd.DataFrame()
                export_df['Supplier Family'] = results['Family']
                export_df['Supplier'] = results['Supplier']
                export_df['Product'] = results['Product']
                export_df['Segment'] = results['Segment']
                export_df['Package'] = results['Package']
                export_df['Container Type'] = results['Container_Type']
                export_df['Full Amount'] = results['Full_List_Price']
                export_df['Avg Full Price (withDiscount)'] = results['Avg_Price']
                export_df['Average Discount'] = results['Avg_Discount']
                export_df['Price Increase'] = results['Inc_Applied']
                export_df['Avg Laid-In (w/Tax)'] = results['Current_Cost_Final']
                export_df['Laid-In Increase'] = results['Cost_Increase']
                export_df['New Laid-In'] = results['New_Cost']
                export_df['Current GP$'] = results['Current_Unit_GP']
                export_df['GP%'] = results['Current_GP_Pct']
                export_df['Best Net/Increase'] = results['New_Price_Full']
                export_df['New GP$'] = results['New_Unit_GP']
                export_df['New GP%'] = results['New_GP_Pct']
                export_df['GP Change'] = results['GP_Change']
                export_df['Units (L12mo)'] = results['Units']
                export_df['Projected Units 2026'] = results['New_Units']
                export_df['Total Discount (L12mo)'] = results['Current_Discount_Total']
                export_df['Current DGP'] = results['Current_Line_DGP']
                export_df['Projected DGP'] = results['New_DGP']
                export_df['DGP Change'] = results['DGP_Change']
                
                st.dataframe(export_df)
                
                csv_data = export_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="💾 Download 'Price Scenarios' Report (CSV)",
                    data=csv_data,
                    file_name="Price_Scenarios_Export.csv",
                    mime="text/csv"
                )
                
else:
    st.info("Please upload 'Data.csv' to begin.")
