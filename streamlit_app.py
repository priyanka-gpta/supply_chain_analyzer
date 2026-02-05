import streamlit as st
from google import genai
from google.genai import types
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from dotenv import load_dotenv
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from io import BytesIO

# Load environment variables from .env file
load_dotenv()

# ===== PAGE CONFIGURATION =====
st.set_page_config(
    page_title="Supply Chain AI Analyzer",
    page_icon="📊",
    layout="wide"
)

# ===== SIDEBAR FOR API KEY =====
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("---")

# Try to get API key from environment first (for local development only)
env_api_key = os.getenv("GEMINI_API_KEY")

if env_api_key:
    # Developer mode - using environment variable
    api_key = env_api_key
    st.sidebar.success("✅ API Key loaded from environment")
else:
    # Public deployment - users enter their own key
    st.sidebar.markdown("""
    ### 🔑 Get Your Free API Key
    
    1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
    2. Click "Create API Key"
    3. Copy and paste it below
    
    **✨ It's 100% free!** No credit card required.
    """)
    
    api_key = st.sidebar.text_input(
        "Paste your Gemini API Key here:",
        type="password",
        placeholder="AIza...",
        help="Your key is used only for this session and never stored"
    )
    
    if api_key:
        st.sidebar.success("✅ API Key received!")
    else:
        st.sidebar.info("👆 Enter your API key to get started")

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📖 How to Use:
1. Upload your supply chain CSV file
2. Click 'Analyze Data'
3. View graphs and AI insights!

### 📋 CSV Format Required:
- Date
- Supplier
- Product
- Order_Quantity
- Delivery_Time_Days
- Inventory_Level
- Order_Value
- Status
""")

# ===== MAIN INTERFACE =====
st.title("📊 Supply Chain AI Anomaly Detector")
st.markdown("Upload your supply chain data and let AI identify risks, patterns, and anomalies.")
st.markdown("---")

# Initialize session state for storing analysis results
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'analysis_text' not in st.session_state:
    st.session_state.analysis_text = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None

# ===== HELPER FUNCTIONS =====
def generate_pdf(report_text):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    
    styles = getSampleStyleSheet()
    story = []
    
    # Split paragraphs
    for line in report_text.split("\n"):
        if line.strip() == "":
            story.append(Spacer(1, 12))
        else:
            story.append(Paragraph(line, styles["Normal"]))
            story.append(Spacer(1, 8))
    
    doc.build(story)
    
    buffer.seek(0)
    return buffer

# ===== FILE UPLOAD =====
uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=['csv'],
    help="Make sure your CSV has the required columns"
)

if uploaded_file is not None:
    # Load the data
    try:
        df = pd.read_csv(uploaded_file)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Display data preview
        st.success(f"✅ File loaded successfully! {len(df)} records found.")
        
        with st.expander("📋 View Data Preview (First 10 rows)"):
            st.dataframe(df.head(10))
        
        # ===== ANALYZE BUTTON =====
        if st.button("🔍 Analyze Data", type="primary"):
            
            if not api_key:
                st.error("⚠️ Please enter your Gemini API key or add it to .env file!")
            else:
                # Configure AI with new API
                client = genai.Client(api_key=api_key)
                
                # ===== GENERATE GRAPHS =====
                st.markdown("---")
                st.subheader("📈 Visual Analysis")
                
                with st.spinner("Generating graphs..."):
                    # Set style
                    sns.set_style("whitegrid")
                    
                    # Create tabs for different graphs
                    tab1, tab2, tab3, tab4 = st.tabs([
                        "🚚 Delivery Times", 
                        "📦 Inventory Levels", 
                        "📊 Order Quantities",
                        "💰 Order Values"
                    ])
                    
                    # Graph 1: Delivery Times
                    with tab1:
                        fig1, ax1 = plt.subplots(figsize=(10, 5))
                        for supplier in df['Supplier'].unique():
                            supplier_data = df[df['Supplier'] == supplier]
                            ax1.plot(supplier_data['Date'], supplier_data['Delivery_Time_Days'], 
                                   marker='o', label=supplier, alpha=0.7, linewidth=2)
                        ax1.set_title('Delivery Times by Supplier Over Time', fontsize=14, fontweight='bold')
                        ax1.set_xlabel('Date')
                        ax1.set_ylabel('Delivery Time (Days)')
                        ax1.legend()
                        ax1.grid(True, alpha=0.3)
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig1)
                        st.caption("👆 Track delivery performance trends across suppliers")
                    
                    # Graph 2: Inventory Levels
                    with tab2:
                        fig2, ax2 = plt.subplots(figsize=(10, 5))
                        ax2.plot(df['Date'], df['Inventory_Level'], color='green', linewidth=2)
                        ax2.axhline(y=800, color='red', linestyle='--', linewidth=2, label='Critical Level (800)')
                        ax2.fill_between(df['Date'], df['Inventory_Level'], alpha=0.3, color='green')
                        ax2.set_title('Inventory Levels Over Time', fontsize=14, fontweight='bold')
                        ax2.set_xlabel('Date')
                        ax2.set_ylabel('Inventory Level (Units)')
                        ax2.legend()
                        ax2.grid(True, alpha=0.3)
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig2)
                        st.caption("👆 Monitor stock levels and identify potential shortages")
                    
                    # Graph 3: Order Quantities
                    with tab3:
                        fig3, ax3 = plt.subplots(figsize=(10, 5))
                        for product in df['Product'].unique():
                            product_data = df[df['Product'] == product]
                            ax3.scatter(product_data['Date'], product_data['Order_Quantity'], 
                                      label=product, alpha=0.6, s=100)
                        ax3.set_title('Order Quantities by Product', fontsize=14, fontweight='bold')
                        ax3.set_xlabel('Date')
                        ax3.set_ylabel('Order Quantity')
                        ax3.legend()
                        ax3.grid(True, alpha=0.3)
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig3)
                        st.caption("👆 Spot demand spikes and ordering patterns")
                    
                    # Graph 4: Order Values
                    with tab4:
                        fig4, ax4 = plt.subplots(figsize=(10, 5))
                        scatter = ax4.scatter(df['Date'], df['Order_Value'], 
                                            c=df['Order_Quantity'], cmap='viridis', 
                                            alpha=0.6, s=100)
                        ax4.set_title('Order Values Over Time', fontsize=14, fontweight='bold')
                        ax4.set_xlabel('Date')
                        ax4.set_ylabel('Order Value ($)')
                        cbar = plt.colorbar(scatter, ax=ax4)
                        cbar.set_label('Order Quantity')
                        ax4.grid(True, alpha=0.3)
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig4)
                        st.caption("👆 Detect price volatility and cost trends")
                
                # ===== CALCULATE METRICS =====
                metrics = {
                    'total_records': len(df),
                    'date_range': f"{df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}",
                    'avg_delivery_time': round(df['Delivery_Time_Days'].mean(), 2),
                    'max_delivery_time': df['Delivery_Time_Days'].max(),
                    'min_inventory': df['Inventory_Level'].min(),
                    'avg_inventory': round(df['Inventory_Level'].mean(), 2),
                    'suppliers': df['Supplier'].unique().tolist(),
                    'products': df['Product'].unique().tolist(),
                    'delayed_orders': len(df[df['Status'] == 'Delayed']),
                    'total_order_value': round(df['Order_Value'].sum(), 2)
                }
                
                delivery_by_supplier = df.groupby('Supplier')['Delivery_Time_Days'].agg(['mean', 'max'])
                
                # Get anomaly examples
                anomaly_examples = pd.concat([
                    df[df['Delivery_Time_Days'] > 10].head(3),
                    df[df['Inventory_Level'] < 800].head(3),
                    df[df['Order_Quantity'] > 500].head(3)
                ]).drop_duplicates().to_string(index=False)
                
                # ===== AI ANALYSIS =====
                #st.markdown("---")
                #st.subheader("🤖 AI-Powered Insights")
                
                prompt = f"""
Analyze this supply chain data and provide a CONCISE, EXECUTIVE-LEVEL summary.

**KEY METRICS:**
- Date Range: {metrics['date_range']}
- Total Records: {metrics['total_records']}
- Average Delivery Time: {metrics['avg_delivery_time']} days
- Minimum Inventory Level: {metrics['min_inventory']} units
- Delayed Orders: {metrics['delayed_orders']}
- Suppliers: {', '.join(metrics['suppliers'])}
- Products: {', '.join(metrics['products'])}

**DELIVERY PERFORMANCE BY SUPPLIER:**
{delivery_by_supplier.to_string()}

**SAMPLE ANOMALOUS RECORDS:**
{anomaly_examples}

**INSTRUCTIONS:**
Provide a brief analysis in this EXACT format (keep each section to 2-3 bullet points MAX):

🚨 **TOP 3 CRITICAL ISSUES**
- [Most urgent problem]
- [Second priority]
- [Third priority]

📊 **KEY FINDINGS**
- [Major pattern 1]
- [Major pattern 2]

💡 **IMMEDIATE ACTIONS**
- [Action 1]
- [Action 2]

Keep your response under 200 words total. Be specific with dates, numbers, and supplier names.
"""
                
                with st.spinner("AI is analyzing your data..."):
                    try:
                        # Use new API format
                        response = client.models.generate_content(
                            model='gemini-2.5-flash',
                            contents=prompt
                        )
                        analysis = response.text
                        
                        # Store in session state
                        st.session_state.analysis_text = analysis
                        st.session_state.metrics = metrics
                        st.session_state.analysis_complete = True
                        
                        # Display AI analysis in a nice box
                        #st.markdown(analysis)
                        
                        st.success("✅ Analysis complete!")
                        
                    except Exception as e:
                        st.error(f"❌ AI Analysis Error: {str(e)}")
                        st.info("💡 Possible issues: Invalid API key, rate limit exceeded, or connection problem")
        
        # ===== DISPLAY CACHED RESULTS (persists after download clicks) =====
        if st.session_state.analysis_complete and st.session_state.analysis_text:
            st.markdown("---")
            st.subheader("🤖 AI-Powered Insights")
            st.markdown(st.session_state.analysis_text)
            
            # Display metrics
            if st.session_state.metrics:
                st.markdown("---")
                st.subheader("📋 Summary Metrics")
                
                metrics = st.session_state.metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Records", metrics['total_records'])
                    st.metric("Avg Delivery Time", f"{metrics['avg_delivery_time']} days")
                
                with col2:
                    st.metric("Max Delivery Time", f"{metrics['max_delivery_time']} days")
                    st.metric("Delayed Orders", metrics['delayed_orders'])
                
                with col3:
                    st.metric("Min Inventory", f"{metrics['min_inventory']} units")
                    st.metric("Avg Inventory", f"{metrics['avg_inventory']} units")
                
                with col4:
                    st.metric("Total Order Value", f"${metrics['total_order_value']:,.2f}")
                    st.metric("Suppliers", len(metrics['suppliers']))
                
                # Download button
                st.markdown("---")
                report_text = f"""SUPPLY CHAIN ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*70}

{st.session_state.analysis_text}

{'='*70}

KEY METRICS:
- Total Records: {metrics['total_records']}
- Date Range: {metrics['date_range']}
- Average Delivery Time: {metrics['avg_delivery_time']} days
- Maximum Delivery Time: {metrics['max_delivery_time']} days
- Minimum Inventory Level: {metrics['min_inventory']} units
- Average Inventory Level: {metrics['avg_inventory']} units
- Delayed Orders: {metrics['delayed_orders']}
- Total Order Value: ${metrics['total_order_value']:,.2f}
- Suppliers: {', '.join(metrics['suppliers'])}
- Products: {', '.join(metrics['products'])}
"""
                pdf_file = generate_pdf(report_text)
                
                st.download_button(
                    label="📥 Download Full Report",
                    data=pdf_file,
                    file_name=f"supply_chain_analysis_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    key="download_report_cached"
                )
    
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        st.info("💡 Make sure your CSV has the required columns: Date, Supplier, Product, Order_Quantity, Delivery_Time_Days, Inventory_Level, Order_Value, Status")

else:
    # Show placeholder when no file is uploaded
    st.info("👆 Upload a CSV file to get started!")
    
    # Show example of what the tool can do
    st.markdown("---")
    st.subheader("✨ What This Tool Does:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📊 Visual Analytics
        - Track delivery times by supplier
        - Monitor inventory levels
        - Identify demand spikes
        - Detect price volatility
        """)
    
    with col2:
        st.markdown("""
        ### 🤖 AI Insights
        - Detect anomalies automatically
        - Risk assessment
        - Pattern recognition
        - Actionable recommendations
        """)

# ===== FOOTER =====
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Built with Streamlit & Google Gemini AI | Priyanka Gupta</p>
</div>
""", unsafe_allow_html=True)