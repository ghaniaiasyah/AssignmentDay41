import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from plotly.subplots import make_subplots

# konfigurasi halaman
st.set_page_config(
    page_title = 'E-Commerce Business Transactions Dashboard',
    page_icon = '💵',
    layout = 'wide',
    initial_sidebar_state = 'expanded'
)

# fungsi load dataset
@st.cache_data
def load_data():
    df_sales = pd.read_csv('data finpro manipulated.csv')
    df_segment = pd.read_csv('data rfm finpro.csv')   # perbaiki read_csv
    return df_sales, df_segment

df_sales, df_segment = load_data()

# judul dashboard
st.title('E-Commerce Business Transactions Dashboard')
st.markdown('This dashboard provides an overview of e-commerce sales performance, showing monthly trends and the distribution of sales across countries, products, and customers')

st.sidebar.header('Navigations')

pilihan_halaman = st.sidebar.radio(
    'Page Selection:',
    ('Overview', 'Customer Segmentation')
)


# filter
# halaman 1
if pilihan_halaman == 'Overview':
    st.sidebar.markdown('### Filters')

    df_sales['Date'] = pd.to_datetime(df_sales['Date'])

    # tanggal
    min_date = df_sales['Date'].min().date()
    max_date = df_sales['Date'].max().date()

    date_range = st.sidebar.date_input(
        'Choose Date Range:',
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date_filter = pd.Timestamp(date_range[0])
        end_date_filter = pd.Timestamp(date_range[1])

        filtered_df = df_sales[
            (df_sales['Date'] >= start_date_filter) &
            (df_sales['Date'] <= end_date_filter)
        ]
    else:
        filtered_df = df_sales

    # wilayah
    selected_regions = st.sidebar.multiselect(
        'Choose Country:',
        options=sorted(df_sales['Country'].unique().tolist()),
        default=sorted(df_sales['Country'].unique().tolist())
    )
    filtered_df = filtered_df[filtered_df['Country'].isin(selected_regions)]

    # produk
    selected_product = st.sidebar.multiselect(
        'Choose Product:',
        options=sorted(df_sales['ProductName'].unique().tolist()),
        default=sorted(df_sales['ProductName'].unique().tolist())
    )
    filtered_df = filtered_df[filtered_df['ProductName'].isin(selected_product)]

    # ambil daftar customer yang muncul di filtered_df
    customers_filtered = filtered_df[['CustomerNo']].drop_duplicates()

    # join dengan df_segment untuk ambil Recency
    df_segment = df_segment.rename(columns={
    'Dataset_CustomerNo': 'CustomerNo',
    'RECENCY': 'Recency'
})
    
    recency_join = customers_filtered.merge(
    df_segment[['CustomerNo', 'Recency']],
    on='CustomerNo',
    how='left'
)

    avg_recency = recency_join['Recency'].mean()

# halaman 2
else:
    st.sidebar.markdown('### Filters')

    df_sales['Date'] = pd.to_datetime(df_sales['Date'])

    # tanggal
    min_date = df_sales['Date'].min().date()
    max_date = df_sales['Date'].max().date()

    date_range = st.sidebar.date_input(
        'Choose Date Range:',
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date_filter = pd.Timestamp(date_range[0])
        end_date_filter = pd.Timestamp(date_range[1])

        filtered_seg_sales = df_sales[
            (df_sales['Date'] >= start_date_filter) &
            (df_sales['Date'] <= end_date_filter)
        ]
    else:
        filtered_seg_sales = df_sales

    # wilayah
    selected_regions = st.sidebar.multiselect(
        'Choose Country:',
        options=sorted(df_sales['Country'].unique().tolist()),
        default=sorted(df_sales['Country'].unique().tolist())
    )
    filtered_seg_sales = filtered_seg_sales[filtered_seg_sales['Country'].isin(selected_regions)]

    # produk
    selected_product = st.sidebar.multiselect(
        'Choose Product:',
        options=sorted(df_sales['ProductName'].unique().tolist()),
        default=sorted(df_sales['ProductName'].unique().tolist())
    )
    filtered_seg_sales = filtered_seg_sales[filtered_seg_sales['ProductName'].isin(selected_product)]

    # segmen
    if 'Segment' in df_sales.columns:
        selected_segment = st.sidebar.multiselect(
            'Choose Customer Segment:',
            options=sorted(df_sales['Segment'].unique().tolist()),
            default=sorted(df_sales['Segment'].unique().tolist())
        )
        filtered_seg_sales = filtered_seg_sales[filtered_seg_sales['Segment'].isin(selected_segment)]

    # join dengan df_segment based on CustomerNo
    df_segment = df_segment.rename(columns={
    'Dataset_CustomerNo': 'CustomerNo',
    'RECENCY': 'Recency'
})
    customers_filtered = filtered_seg_sales[['CustomerNo']].drop_duplicates()

    seg_data = customers_filtered.merge(
        df_segment,      
        on='CustomerNo',
        how='inner'
    )

# konten halaman utama
if pilihan_halaman == 'Overview':
      # kpi
    st.subheader('Sales Performance Summary')

    seg_data = customers_filtered.merge(
        df_segment,      
        on='CustomerNo',
        how='inner'
    )

    total_customers = filtered_df['CustomerNo'].nunique()
    total_transactions = filtered_df['TransactionNo'].nunique()
    total_sales = filtered_df['TotalPrice'].sum()
    avg_monetary = filtered_df.groupby('CustomerNo')['TotalPrice'].sum().mean()
    avg_frequency = seg_data['FREQUENCY'].mean()

    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)

    with kpi1:
        st.metric("Total Customers", f"{total_customers:,}")
    with kpi2:
        st.metric("Total Transactions", f"{total_transactions:,}")
    with kpi3:
        st.metric("Total Sales", f"${total_sales:,.0f}")
    with kpi4:
        st.metric("AVG Monetary", f"${avg_monetary:,.0f}")
    with kpi5:
        st.metric("AVG Frequency", f"{avg_frequency:,.0f}")
    with kpi6:
        st.metric("AVG Recency (days)", f"{avg_recency:,.0f}")
else:
    
        # kpi
    st.subheader('Customer Segmentation Summary')

    total_customers = seg_data['CustomerNo'].nunique()
    total_transactions = filtered_seg_sales['TransactionNo'].nunique()
    total_sales = filtered_seg_sales['TotalPrice'].sum()
    avg_monetary = seg_data['MONETARY'].mean()
    avg_frequency = seg_data['FREQUENCY'].mean()
    avg_recency = seg_data['Recency'].mean()
    
    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)

    with kpi1:
        st.metric("Total Customers", f"{total_customers:,}")
    with kpi2:
        st.metric("Total Transactions", f"{total_transactions:,}")
    with kpi3:
        st.metric("Total Sales", f"${total_sales:,.0f}")
    with kpi4:
        st.metric("AVG Monetary", f"${avg_monetary:,.0f}")
    with kpi5:
        st.metric("AVG Frequency", f"{avg_frequency:,.0f}")
    with kpi6:
        st.metric("AVG Recency (days)", f"{avg_recency:,.0f}")

#visualisasi halaman 1
if pilihan_halaman == 'Overview':

    tab1, tab2, tab3, tab4 = st.tabs(['Trends', 'Products','Customers','Countries'])

    with tab1: #trends
        st.markdown("### Trends")

        # agregasi per bulan
        df_trend = (
            filtered_df
            .resample('M', on='Date')
            .agg(
                TotalCustomers=('CustomerNo', 'nunique'),
                TotalTransactions=('TransactionNo', 'nunique'),
                TotalSales=('TotalPrice', 'sum')
            )
            .reset_index()
        )

        df_trend['Month'] = df_trend['Date'].dt.strftime('%b %Y')

        fig_trend = make_subplots(specs=[[{"secondary_y": True}]])

        fig_trend.add_trace(
            go.Scatter(
                x=df_trend['Month'],
                y=df_trend['TotalCustomers'],
                name="Total Customers",
                mode='lines+markers'
            ),
            secondary_y=False
        )

        fig_trend.add_trace(
            go.Scatter(
                x=df_trend['Month'],
                y=df_trend['TotalTransactions'],
                name="Total Transactions",
                mode='lines+markers'
            ),
            secondary_y=False
        )

        fig_trend.add_trace(
            go.Scatter(
                x=df_trend['Month'],
                y=df_trend['TotalSales'] / 1_000_000,  # ke $M
                name="Total Sales (in $M)",
                mode='lines+markers'
            ),
            secondary_y=True
        )

        fig_trend.update_layout(
            title="Total Customer, Total Transactions and Total Sales Trends",
            xaxis_title="Month",
        )
        fig_trend.update_yaxes(title_text="Customers / Transactions", secondary_y=False)
        fig_trend.update_yaxes(title_text="Sales (in $M)", secondary_y=True)

        st.plotly_chart(fig_trend, use_container_width=True)



    with tab2: # products
        st.markdown("### Top Products")

        prod_agg = (
            filtered_df
            .groupby('ProductName')
            .agg(
                TotalSales=('TotalPrice', 'sum'),
                TotalTransactions=('TransactionNo', 'nunique')
            )
            .reset_index()
            .sort_values('TotalSales', ascending=False)
            .head(5)
        )

        fig_prod = make_subplots(specs=[[{"secondary_y": True}]])

        fig_prod.add_trace(
            go.Bar(
                x=prod_agg['ProductName'],
                y=prod_agg['TotalSales'],
                name="Total Sales",
                marker=dict(color="#1f77b4")  
            ),
            secondary_y=False
        )

        fig_prod.add_trace(
            go.Scatter(
                x=prod_agg['ProductName'],
                y=prod_agg['TotalTransactions'],
                name="Total Transactions",
                mode='lines+markers',
                line=dict(color="#ff7f0e", width=2),  
                marker=dict(color="#ff7f0e", size=6)   
            ),
            secondary_y=True
        )

        fig_prod.update_layout(
            title="Top 5 Products by Total Sales",
            xaxis_title="Product",
        )
        fig_prod.update_yaxes(title_text="Total Sales", secondary_y=False)
        fig_prod.update_yaxes(title_text="Total Transactions", secondary_y=True)

        st.plotly_chart(fig_prod, use_container_width=True)


    with tab3: #customers
        st.markdown("### Top Customers")

        cust_agg = (
            filtered_df
            .groupby('CustomerNo')
            .agg(
                TotalSales=('TotalPrice', 'sum'),
                TotalTransactions=('TransactionNo', 'nunique')
            )
            .reset_index()
            .sort_values('TotalSales', ascending=False)
            .head(5)
        )

        cust_agg['CustomerNo_str'] = cust_agg['CustomerNo'].astype(str)

        fig_cust = make_subplots(specs=[[{"secondary_y": True}]])

        fig_cust.add_trace(
            go.Bar(
                x=cust_agg['CustomerNo_str'],
                y=cust_agg['TotalSales'],
                name="Total Sales",
                marker=dict(color="#1f77b4")   
            ),
            secondary_y=False
        )

        fig_cust.add_trace(
            go.Scatter(
                x=cust_agg['CustomerNo_str'],
                y=cust_agg['TotalTransactions'],
                name="Total Transactions",
                mode='lines+markers',
                line=dict(color="#ff7f0e", width=2),
                marker=dict(color="#ff7f0e", size=6)
            ),
            secondary_y=True
        )

        fig_cust.update_layout(
            title="Top 5 Customers by Total Sales",
            xaxis_title="CustomerNo",
        )

        fig_cust.update_xaxes(type='category')
        fig_cust.update_yaxes(title_text="Total Sales", secondary_y=False)
        fig_cust.update_yaxes(title_text="Total Transactions", secondary_y=True)

        st.plotly_chart(fig_cust, use_container_width=True)


    with tab4: # countries
        st.markdown("### Countries Overview")

        col1, col2 = st.columns(2)

        # total sales by countries
        country_sales_raw = (
            filtered_df
            .groupby('Country')
            .agg(TotalSales=('TotalPrice', 'sum'))
            .reset_index()
        )

        # non UK -> others
        country_sales_raw['Country_grouped'] = country_sales_raw['Country'].where(
            country_sales_raw['Country'] == 'United Kingdom',
            'Others'
        )

        country_sales = (
            country_sales_raw
            .groupby('Country_grouped', as_index=False)['TotalSales']
            .sum()
            .rename(columns={'Country_grouped': 'Country'})
        )

        with col1:
            fig_country_sales = px.pie(
                country_sales,
                values='TotalSales',
                names='Country',
                title='Total Sales by Country',
                color = 'Country',
                color_discrete_map={'United Kingdom': '#1f77b4', 'Others': '#ff7f0e'           # oren
                })
        
            st.plotly_chart(fig_country_sales, use_container_width=True)


        # total cust by country
        country_cust_raw = (
            filtered_df
            .groupby('Country')
            .agg(TotalCustomers=('CustomerNo', 'nunique'))
            .reset_index()
        )

        country_cust_raw['Country_grouped'] = country_cust_raw['Country'].where(
            country_cust_raw['Country'] == 'United Kingdom',
            'Others'
        )

        country_customers = (
            country_cust_raw
            .groupby('Country_grouped', as_index=False)['TotalCustomers']
            .sum()
            .rename(columns={'Country_grouped': 'Country'})
        )

        with col2:
            fig_country_cust = px.pie(
                country_customers,
                values='TotalCustomers',
                names='Country',
                title='Total Customers by Country',
                color = 'Country',
                color_discrete_map={'United Kingdom': '#1f77b4', 'Others': '#ff7f0e'           # oren
                })
            st.plotly_chart(fig_country_cust, use_container_width=True)

else:  # Halaman 2: Customer Segmentation Dashboard
    df_segment = df_segment.rename(columns={
    'Dataset_CustomerNo': 'CustomerNo',
    'MONETARY': 'Monetary',
    'FREQUENCY': 'Frequency',
    'RECENCY': 'Recency',
    'F Score': 'F_Score',
    'M Score': 'M_Score',
    'R Score': 'R_Score',
    'Customer Segment': 'Segment'
})
    filtered_seg = filtered_seg_sales
    
    customers_filtered = (
        filtered_seg[['CustomerNo']]
        .drop_duplicates()
    )

    seg_data = customers_filtered.merge(
        df_segment,     
        on='CustomerNo',
        how='left'
    )
    
    tab1, tab2, tab3 = st.tabs(['Trend', 'Segments', 'Products'])

    with tab1: #trend
        st.markdown("### Total Sales Trend by Segment")

        filtered_seg['Date'] = pd.to_datetime(filtered_seg['Date'])

        # agregasi per bulan & segment
        df_trend_seg = (
            filtered_seg
            .groupby([pd.Grouper(key='Date', freq='M'), 'Segment'])
            .agg(TotalSales=('TotalPrice', 'sum'))
            .reset_index()
        )

        df_trend_seg['Month'] = df_trend_seg['Date'].dt.strftime('%b %Y')

        fig_trend_seg = px.line(
            df_trend_seg,
            x='Month',
            y='TotalSales',
            color='Segment',
            markers=True,
            title='Total Sales Trend by Segment'
        )
        fig_trend_seg.update_yaxes(title_text="Total Sales")
        fig_trend_seg.update_xaxes(title_text="Month")

        st.plotly_chart(fig_trend_seg, use_container_width=True)
    
    with tab2: # segment
        st.markdown("### Recency, Frequency, and Monetary KPIs per Segment")

        # KPI 
        seg_sales_agg = (
            filtered_seg
            .groupby('Segment')
            .agg(
                TotalCustomers=('CustomerNo', 'nunique'),
                TotalSales=('TotalPrice', 'sum'),
                TotalTransactions=('TransactionNo', 'nunique')
            )
            .reset_index()
        )

        # total untuk persen
        total_cust_all = seg_sales_agg['TotalCustomers'].sum()
        total_sales_all = seg_sales_agg['TotalSales'].sum()
        total_trans_all = seg_sales_agg['TotalTransactions'].sum()

        seg_sales_agg['PctCustomers'] = (seg_sales_agg['TotalCustomers'] / total_cust_all * 100).round(2)
        seg_sales_agg['PctSales'] = (seg_sales_agg['TotalSales'] / total_sales_all * 100).round(2)
        seg_sales_agg['PctTransactions'] = (seg_sales_agg['TotalTransactions'] / total_trans_all * 100).round(2)

        # avg recency
        recency_seg = (
            seg_data
            .groupby('Segment')
            .agg(AvgRecency=('Recency', 'mean'))
            .reset_index()
        )

        seg_kpi = seg_sales_agg.merge(recency_seg, on='Segment', how='left')

        # rapikan urutan & nama kolom
        seg_kpi = seg_kpi[[
            'Segment',
            'TotalCustomers', 'PctCustomers',
            'TotalSales', 'PctSales',
            'TotalTransactions', 'PctTransactions',
            'AvgRecency'
        ]]

        seg_kpi = seg_kpi.rename(columns={
            'Segment': 'Customer Segment',
            'TotalCustomers': 'Total Customers',
            'PctCustomers': '% Customers',
            'TotalSales': 'Total Sales',
            'PctSales': '% Total Sales',
            'TotalTransactions': 'Total Transactions',
            'PctTransactions': '% Total Transactions',
            'AvgRecency': 'AVG Days Since Last Transaction'
        })

        # tamilkan tabel
        st.dataframe(
            seg_kpi.style
            .format({
                'Total Customers': '{:,.0f}',
                '% Customers': '{:.2f} %',
                'Total Sales': '{:,.2f}',
                '% Total Sales': '{:.2f} %',
                'Total Transactions': '{:,.0f}',
                '% Total Transactions': '{:.2f} %',
                'AVG Days Since Last Transaction': '{:.2f}'
            })
            .background_gradient(
                subset=['Total Sales', 'Total Transactions', 'Total Customers'],
                cmap='Blues'
            )
        )

        # distribusi score rfm
        st.markdown("### Distribution of AVG R, F, M Score per Segment")

        rfm_scores = (
            seg_data
            .groupby('Segment')
            .agg(
                Avg_R=('R_Score', 'mean'),
                Avg_F=('F_Score', 'mean'),
                Avg_M=('M_Score', 'mean')
            )
            .reset_index()
        )

        rfm_melt = rfm_scores.melt(
            id_vars='Segment',
            value_vars=['Avg_R', 'Avg_F', 'Avg_M'],
            var_name='ScoreType',
            value_name='AvgScore'
        )

        rfm_melt['ScoreType'] = rfm_melt['ScoreType'].map({
            'Avg_R': 'R Score',
            'Avg_F': 'F Score',
            'Avg_M': 'M Score'
        })

        fig_rfm = px.bar(
            rfm_melt,
            x='Segment',
            y='AvgScore',
            color='ScoreType',
            barmode='group',
            title='Distribution of AVG R, F, M Score per Segment'
        )
        fig_rfm.update_xaxes(title_text="Customer Segment")
        fig_rfm.update_yaxes(title_text="Average Score")

        st.plotly_chart(fig_rfm, use_container_width=True)

    with tab3: # products
        st.markdown("### Number of Sales by Products and Segments")

        prod_seg = (
            filtered_seg
            .groupby(['ProductName', 'Segment'])
            .agg(
                TotalTransactions=('TransactionNo', 'nunique')
            )
            .reset_index()
        )

        # top 10 produk
        topN = 10
        top_products = (
            prod_seg
            .groupby('ProductName')['TotalTransactions']
            .sum()
            .nlargest(topN)
            .index
        )

        prod_seg_top = prod_seg[prod_seg['ProductName'].isin(top_products)]

        fig_prod_seg = px.bar(
            prod_seg_top,
            x='TotalTransactions',
            y='ProductName',
            color='Segment',
            orientation='h',
            barmode='stack',
            title=f'Number of Sales by Products and Segments (Top {topN} Products)'
        )
        fig_prod_seg.update_xaxes(title_text="Number of Sales / Transactions")
        fig_prod_seg.update_yaxes(title_text="Product")

        st.plotly_chart(fig_prod_seg, use_container_width=True)



