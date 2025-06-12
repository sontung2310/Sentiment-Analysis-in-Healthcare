import streamlit as st
import plotly.graph_objects as go
from pymongo import MongoClient
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()



# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="📊",
    layout="wide"
)

# MongoDB connection
@st.cache_resource
def init_connection():
    connection_string = os.getenv("MONGODB_URI")
    client = MongoClient(connection_string)
    return client

@st.cache_data(ttl=60)
def load_data():
    try:
        client = init_connection()
        db = client["healthcare_dashboard"]
        collection = db["sentiment"]
        
        # Fetch all documents
        cursor = collection.find({})
        data = list(cursor)
        
        if not data:
            st.warning("No data found in the database")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        return df
        
    except Exception as e:
        st.error(f"Error connecting to database: {str(e)}")
        return pd.DataFrame()

def create_donut_chart(df):
    if df.empty:
        return None
    
    # Count sentiment values
    sentiment_counts = df['result'].value_counts()
    
    # Define colors
    colors = {
        'Positive': '#2E8B57',    # Green
        'Negative': '#DC143C',    # Red
        'Neutral': '#FFB347'      # Orange
    }
    
    color_list = [colors.get(sentiment, '#808080') for sentiment in sentiment_counts.index]
    
    # Create donut chart
    fig = go.Figure(data=[go.Pie(
        labels=sentiment_counts.index,
        values=sentiment_counts.values,
        hole=.4,
        marker_colors=color_list,
        textinfo='label+percent',
        textposition='outside',
        textfont_size=16
    )])
    
    fig.update_layout(
        title={
            'text': 'Sentiment Distribution',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24}
        },
        font=dict(size=14),
        showlegend=True,
        height=500
    )
    
    return fig

# Main app
st.title("📊 Sentiment Analysis Dashboard")

# Load data
df = load_data()

if not df.empty:
    # Convert timestamp to datetime and sort by latest
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp', ascending=False)
    
    # Create two columns layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 Latest Reviews")
        
        # Display latest reviews table
        latest_reviews = df.head(10)  # Show latest 10 reviews
        
        for idx, row in latest_reviews.iterrows():
            # Create a container for each review
            with st.container():
                # Format timestamp
                if 'timestamp' in row:
                    timestamp_str = row['timestamp'].strftime("%Y-%m-%d %H:%M")
                else:
                    timestamp_str = "N/A"
                
                # Color code based on sentiment
                if row['result'] == 'Positive':
                    sentiment_color = "🟢"
                elif row['result'] == 'Negative':
                    sentiment_color = "🔴"
                else:
                    sentiment_color = "🟡"
                
                # Display review info
                st.markdown(f"**{sentiment_color} {row['result']}** - {timestamp_str}")
                
                # Handle long reviews with expandable text
                review_text = row['input_text']
                if len(review_text) > 100:
                    # Show truncated version by default
                    st.text(review_text[:100] + "...")
                    
                    # Add expander for full text
                    with st.expander("📖 Read full review"):
                        st.text(review_text)
                else:
                    # Show full text if it's short
                    st.text(review_text)
                
                st.markdown("---")
    
    with col2:
        st.subheader("📊 Distribution")
        
        # Create and display donut chart
        fig = create_donut_chart(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Show simple stats
        total = len(df)
        st.write(f"**Total Reviews:** {total}")
        
        sentiment_counts = df['result'].value_counts()
        for sentiment, count in sentiment_counts.items():
            percentage = (count / total * 100)
            st.write(f"**{sentiment}:** {count} ({percentage:.1f}%)")
            
else:
    st.error("No data available to display")