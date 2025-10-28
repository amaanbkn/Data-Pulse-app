import streamlit as st
import pandas as pd
import plotly.express as px
import warnings
import pandas.api.types as ptypes
import numpy as np
import seaborn as sns  # NEW: For heatmaps
import matplotlib.pyplot as plt  # NEW: For displaying heatmap

from sklearn.linear_model import LinearRegression
from prophet import Prophet
from prophet.plot import plot_plotly

# Suppress warnings
warnings.filterwarnings('ignore')

# --- 1. Page Configuration ---
st.set_page_config(page_title="DataPulse MVP 2.5",
                   layout="wide",
                   initial_sidebar_state="expanded")

# --- 2. Title and Sidebar ---
st.title('🚀 DataPulse: Your Business Snapshot')
st.sidebar.header("Instructions")
st.sidebar.info(
    "1. Upload your CSV sales data.\n"
    "2. Confirm the auto-detected columns.\n"
    "3. Click 'Generate My Report'!\n"
    "4. Check all 3 tabs: Dashboard, Forecasts, & Insights."
)

# --- 3. Auto-Detection Function (No changes) ---
def auto_detect_columns(df):
    all_cols = df.columns.tolist()
    guesses = {'date': None, 'sales': None, 'product': None}
    
    for col in all_cols:
        col_lower = col.lower()
        if guesses['date'] is None:
            if 'date' in col_lower or 'time' in col_lower:
                try:
                    pd.to_datetime(df[col], errors='raise')
                    guesses['date'] = col
                except Exception:
                    pass
        if guesses['sales'] is None:
            if 'sales' in col_lower or 'amount' in col_lower or 'revenue' in col_lower or 'price' in col_lower:
                if ptypes.is_numeric_dtype(df[col]):
                    guesses['sales'] = col
        if guesses['product'] is None:
            if 'product' in col_lower or 'item' in col_lower or 'category' in col_lower or 'name' in col_lower:
                if ptypes.is_object_dtype(df[col]):
                    guesses['product'] = col
                    
    if guesses['sales'] is None:
        for col in all_cols:
            if ptypes.is_numeric_dtype(df[col]):
                guesses['sales'] = col
                break
                
    if guesses['product'] is None:
        for col in all_cols:
            if ptypes.is_object_dtype(df[col]):
                guesses['product'] = col
                break
                
    return guesses

# --- 4. File Uploader (No changes) ---
uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

# --- 5. Session State Management (No changes) ---
if 'current_file_name' not in st.session_state:
    st.session_state['current_file_name'] = None
    st.session_state['report_generated'] = False

if uploaded_file is not None and uploaded_file.name != st.session_state['current_file_name']:
    st.session_state['current_file_name'] = uploaded_file.name
    st.session_state['report_generated'] = False 
    st.session_state.pop('df_raw', None) 
    st.session_state.pop('df_processed', None) 
    st.session_state.pop('df_forecast', None) 
elif uploaded_file is None:
    st.session_state['current_file_name'] = None
    st.session_state['report_generated'] = False
    st.session_state.pop('df_raw', None)
    st.session_state.pop('df_processed', None)
    st.session_state.pop('df_forecast', None)

# --- 6. Main Application Logic (No changes) ---
if uploaded_file is not None:
    
    try:
        if 'df_raw' not in st.session_state:
            st.session_state['df_raw'] = pd.read_csv(uploaded_file)
        
        df_raw = st.session_state['df_raw'] 
        
        if not st.session_state['report_generated']:
            st.success("File Uploaded Successfully!")
            st.write("### 📊 Data Preview")
            st.dataframe(df_raw.head())
        
        # --- 7. Column Mapping (No changes) ---
        st.write("### ⚙️ Map Your Columns")
        st.info("We've made our best guess. Please confirm the columns are correct.")
        
        all_columns = df_raw.columns.tolist()
        column_guesses = auto_detect_columns(df_raw)
        
        date_col_index = all_columns.index(column_guesses['date']) if column_guesses['date'] else 0
        sales_col_index = all_columns.index(column_guesses['sales']) if column_guesses['sales'] else 0
        product_col_index = all_columns.index(column_guesses['product']) if column_guesses['product'] else 0

        col1, col2, col3 = st.columns(3)
        with col1:
            date_col = st.selectbox("Select your 'Date' column:", all_columns, index=date_col_index)
        with col2:
            sales_col = st.selectbox("Select your 'Sales Amount' column:", all_columns, index=sales_col_index)
        with col3:
            product_col = st.selectbox("Select your 'Product' column:", all_columns, index=product_col_index)
        
        # --- 8. Report Generation Button (No changes) ---
        if st.button("Generate My Report", type="primary"):
            st.session_state['report_generated'] = False 
            try:
                df = df_raw.copy()
                df[sales_col] = pd.to_numeric(df[sales_col], errors='coerce')
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                df.dropna(subset=[sales_col, date_col], inplace=True)
                st.session_state['df_processed'] = df 
                
                st.session_state['total_sales'] = df[sales_col].sum()
                st.session_state['avg_sale'] = df[sales_col].mean()
                
                st.session_state['top_products'] = df.groupby(product_col)[sales_col].sum().nlargest(10).reset_index()
                
                df_forecast = df.groupby(date_col)[sales_col].sum().reset_index()
                df_forecast['day_num'] = np.arange(len(df_forecast))
                X = df_forecast[['day_num']]
                y = df_forecast[sales_col]
                model = LinearRegression().fit(X, y)
                df_forecast['trend'] = model.predict(X)
                st.session_state['df_forecast'] = df_forecast 
                
                st.session_state['date_col'] = date_col
                st.session_state['sales_col'] = sales_col
                st.session_state['product_col'] = product_col
                
                st.session_state['report_generated'] = True
                st.rerun() 

            except Exception as e:
                st.error(f"Error during analysis: {e}")
                st.error("Please check your column selections. 'Date' and 'Sales' must be correct.")
                st.session_state['report_generated'] = False

    except Exception as e:
        st.error(f"An error occurred while reading the file: {e}")
        st.error("Please ensure you uploaded a valid CSV file.")
        st.session_state['report_generated'] = False

# --- 9. Report Display Block (No changes) ---
if st.session_state.get('report_generated', False):
    
    st.header(f"📈 Sales Report for {st.session_state['current_file_name']}")
    
    total_sales = st.session_state['total_sales']
    avg_sale = st.session_state['avg_sale']
    df = st.session_state['df_processed']
    df_forecast = st.session_state['df_forecast']
    top_products = st.session_state['top_products']
    date_col = st.session_state['date_col']
    sales_col = st.session_state['sales_col']
    product_col = st.session_state['product_col']

    # --- 10. MODIFIED: Draw 3 Tabs ---
    tab1, tab2, tab3 = st.tabs(["📊 Main Dashboard", "🔮 Forecasts", "💡 Insights"])
    
    # --- Main Dashboard Tab (No changes) ---
    with tab1:
        st.subheader("Top-Level Metrics")
        kpi1, kpi2 = st.columns(2)
        kpi1.metric("Total Sales", f"₹{total_sales:,.2f}")
        kpi2.metric("Average Sale Value", f"₹{avg_sale:,.2f}")
        st.subheader("Detailed Visuals")
        fig_time = px.line(df, x=date_col, y=sales_col, title="Sales Trend Over Time")
        st.plotly_chart(fig_time, use_container_width=True)
        fig_bar = px.bar(top_products, x=product_col, y=sales_col, title="Top 10 Selling Products")
        st.plotly_chart(fig_bar, use_container_width=True)

    # --- Forecasts Tab (No changes) ---
    with tab2:
        st.subheader("Future Forecasts")
        if len(df_forecast) < 10:
            st.warning("Not enough data to create a forecast.")
        else:
            st.markdown("### 1. Simple Linear Trend (scikit-learn)")
            st.write("This line shows the simple upward or downward trend in your sales.")
            fig_trend = px.scatter(df_forecast, x=date_col, y=sales_col, title="Sales Data with Trendline")
            fig_trend.add_scatter(x=df_forecast[date_col], y=df_forecast['trend'], mode='lines', name='Trend')
            st.plotly_chart(fig_trend, use_container_width=True)
            
            st.markdown("### 2. Advanced Forecast (Prophet)")
            st.write("This model predicts future sales and will not go below zero.")
            forecast_days = st.slider("Select number of days to forecast:", 30, 365, 90)

            try:
                df_prophet = df_forecast[[date_col, sales_col]].rename(columns={date_col: 'ds', sales_col: 'y'})
                m = Prophet(growth='linear').fit(df_prophet)
                future = m.make_future_dataframe(periods=forecast_days)
                forecast = m.predict(future)
                
                forecast['yhat'] = forecast['yhat'].apply(lambda x: max(0, x))
                forecast['yhat_lower'] = forecast['yhat_lower'].apply(lambda x: max(0, x))
                forecast['yhat_upper'] = forecast['yhat_upper'].apply(lambda x: max(0, x))
                
                st.markdown("#### 🔮 Your Forecast Summary")
                future_predictions = forecast.iloc[-forecast_days:]
                total_predicted_sales = future_predictions['yhat'].sum()
                avg_daily_predicted_sales = future_predictions['yhat'].mean()
                trend_direction = "up" if future_predictions['trend'].iloc[-1] > future_predictions['trend'].iloc[0] else "down"
                
                kpi1, kpi2 = st.columns(2)
                kpi1.metric(f"Total Predicted Sales (Next {forecast_days} Days)", f"₹{total_predicted_sales:,.2f}")
                kpi2.metric(f"Avg. Daily Predicted Sales", f"₹{avg_daily_predicted_sales:,.2f}")
                st.info(f"**Overall Trend:** Your sales are trending **{trend_direction}** over this period.")
                
                st.markdown("#### Sales Forecast Chart")
                st.write("The **black dots** are your actual sales, the **dark blue line** is our prediction, and the **light blue area** is the likely range.")
                fig_prophet = plot_plotly(m, forecast)
                fig_prophet.update_layout(title="Sales Forecast (Won't go below ₹0)", xaxis_title="Date", yaxis_title="Sales Amount")
                st.plotly_chart(fig_prophet, use_container_width=True)
                
                st.markdown("#### What Drives Your Sales?")
                st.write("The 'Trend' line is your overall growth. The 'Weekly' chart shows which days of the week are usually busiest.")
                fig_components = m.plot_components(forecast)
                st.pyplot(fig_components)
                
            except Exception as e:
                st.error(f"Error running Prophet forecast: {e}")
                st.info("Prophet works best with daily data for at least a few months.")
    
    # --- 11. NEW: Insights Tab ---
    with tab3:
        st.subheader("💡 Automated Business Insights")
        
        # --- Insight 1: Numeric Correlation ---
        st.markdown("### 1. Numeric Data Correlations")
        st.write("This heatmap shows if any numeric columns in your data are related. A '1' is a perfect match, a '0' is no match.")
        
        try:
            # We use the original df_raw, but only numeric columns
            numeric_df = df_raw.select_dtypes(include=np.number)
            
            if len(numeric_df.columns) < 2:
                st.info("No numeric columns found to correlate (besides Sales).")
            else:
                corr = numeric_df.corr()
                fig, ax = plt.subplots()
                sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Could not generate correlation heatmap: {e}")

        # --- Insight 2: Product Correlation (Market Basket Analysis) ---
        st.markdown("### 2. 'People Also Bought' Analysis")
        st.write("This shows which products are most often sold on the *same day*. A high score means they are frequently bought together.")

        try:
            # We need to 'pivot' the data. Rows = Dates, Columns = Products, Values = Count
            # This creates a "basket" for each day
            basket_df = df.pivot_table(index=date_col, 
                                       columns=product_col, 
                                       values=sales_col,  # We can use any column here, just need to count
                                       aggfunc='count').fillna(0)
            
            # Simple check: If a product was bought, set to 1, else 0.
            def encode_units(x):
                if x <= 0:
                    return 0
                if x >= 1:
                    return 1
            
            basket_df = basket_df.applymap(encode_units)

            if len(basket_df.columns) > 50:
                st.warning("More than 50 products found. Analysis will be slow and may be hard to read.")
            
            # Calculate the correlation between products
            product_corr = basket_df.corr()
            
            # Display the heatmap
            fig_prod, ax_prod = plt.subplots()
            sns.heatmap(product_corr, annot=True, fmt=".2f", cmap="viridis", ax=ax_prod)
            st.pyplot(fig_prod)
            
            # --- NEW: Plain-English Summary ---
            st.markdown("#### Top 5 Product Pairs")
            # Unstack the correlation matrix to get pairs
            corr_pairs = product_corr.unstack().sort_values(ascending=False)
            
            # Filter out self-correlations (Product A vs Product A)
            corr_pairs = corr_pairs[corr_pairs != 1.0]
            
            # Get the top 5 pairs (div by 2 because A-B is same as B-A)
            top_pairs = corr_pairs.head(10)[::2] 
            
            if top_pairs.empty:
                st.info("No strong product correlations found.")
            else:
                st.write("Customers who buy...")
                for (prod1, prod2), correlation in top_pairs.items():
                    if correlation > 0.1: # Only show pairs with some correlation
                        st.success(f"**{prod1}**... also tend to buy **{prod2}** (Score: {correlation:.2f})")
                    
        except Exception as e:
            st.error(f"Could not generate product analysis: {e}")
            st.info("This analysis works best when you have multiple products sold per day.")

# --- Final 'else' block ---
elif not uploaded_file:
    st.info("Awaiting CSV file upload...")