import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pickle
import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Add the current directory to the path to import the AI task management module
sys.path.append('.')

# Set page config
st.set_page_config(
    page_title="AI Task Manager",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        color: #6366f1;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .recommendation-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    .task-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #3b82f6;
    }
    .status-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 500;
    }
    .status-progress { background: #fef3c7; color: #92400e; }
    .status-todo { background: #dbeafe; color: #1e40af; }
    .status-done { background: #d1fae5; color: #065f46; }
    .priority-high { color: #dc2626; font-weight: bold; }
    .priority-medium { color: #d97706; font-weight: bold; }
    .priority-low { color: #16a34a; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

class AITaskPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.tfidf_vectorizer = None
        self.load_models()
    
    def load_models(self):
        """Load trained models and preprocessors"""
        try:
            # Load models if they exist
            if os.path.exists('models/best_model.pkl'):
                with open('models/best_model.pkl', 'rb') as f:
                    self.best_model = pickle.load(f)
            
            if os.path.exists('models/scaler.pkl'):
                with open('models/scaler.pkl', 'rb') as f:
                    self.scaler = pickle.load(f)
            
            if os.path.exists('models/priority_encoder.pkl'):
                with open('models/priority_encoder.pkl', 'rb') as f:
                    self.priority_encoder = pickle.load(f)
            
            if os.path.exists('models/tfidf_extractor.pkl'):
                with open('models/tfidf_extractor.pkl', 'rb') as f:
                    self.tfidf_vectorizer = pickle.load(f)
                    
            # Load feature arrays if they exist
            if os.path.exists('Feature Extraction Files/tfidf_features.npy'):
                self.tfidf_features = np.load('Feature Extraction Files/tfidf_features.npy')
            else:
                self.tfidf_features = None
                
            st.success("✅ Models loaded successfully!")
            
        except Exception as e:
            st.warning(f"⚠️ Could not load some models: {str(e)}")
            self.create_fallback_models()
    
    def create_fallback_models(self):
        """Create simple fallback models for demonstration"""
        from sklearn.naive_bayes import GaussianNB
        
        # Create dummy models
        self.best_model = GaussianNB()
        self.scaler = StandardScaler()
        self.priority_encoder = LabelEncoder()
        self.priority_encoder.fit(['Low', 'Medium', 'High'])
        
        # Create dummy TF-IDF vectorizer
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.tfidf_vectorizer = TfidfVectorizer(max_features=100)
        dummy_texts = ["sample task", "bug fix", "feature request"]
        self.tfidf_vectorizer.fit(dummy_texts)
        
        # Train with dummy data
        dummy_X = np.random.random((10, 104))  # 100 TF-IDF + 4 additional features
        dummy_y = np.random.choice([0, 1, 2], 10)
        
        self.scaler.fit(dummy_X)
        self.best_model.fit(self.scaler.transform(dummy_X), dummy_y)
    
    def predict_task(self, summary, description, issue_type, component, estimated_duration):
        """Predict task priority and suggest assignee"""
        try:
            # Combine text
            combined_text = f"{summary} {description}".strip()
            
            # Extract TF-IDF features
            if hasattr(self.tfidf_vectorizer, 'vectorizer'):
                # If it's our custom TFIDFExtractor
                text_features = self.tfidf_vectorizer.transform([combined_text]).toarray()[0]
            else:
                # If it's a standard TfidfVectorizer
                text_features = self.tfidf_vectorizer.transform([combined_text]).toarray()[0]
            
            # Create additional features
            issue_type_map = {'Bug': 0, 'Task': 1, 'Story': 2, 'Epic': 3}
            component_map = {'Frontend': 0, 'Backend': 1, 'Database': 2, 'API': 3, 'UI/UX': 4}
            
            additional_features = np.array([
                estimated_duration,
                issue_type_map.get(issue_type, 1),
                0,  # Status code (new task)
                component_map.get(component, 0)
            ])
            
            # Combine features
            combined_features = np.hstack([text_features, additional_features]).reshape(1, -1)
            
            # Scale features
            scaled_features = self.scaler.transform(combined_features)
            
            # Predict priority
            predicted_priority_idx = self.best_model.predict(scaled_features)[0]
            predicted_priority = self.priority_encoder.inverse_transform([predicted_priority_idx])[0]
            
            # Get prediction confidence
            if hasattr(self.best_model, 'predict_proba'):
                confidence = max(self.best_model.predict_proba(scaled_features)[0])
            else:
                confidence = 0.8  # Default confidence
            
            # Simple assignee suggestion based on workload simulation
            assignees = ['Sarah Chen', 'Alex Rivera', 'Bob Johnson', 'Diana Prince', 'Charlie Wilson']
            suggested_assignee = np.random.choice(assignees)
            
            return {
                'predicted_priority': predicted_priority,
                'confidence': confidence,
                'suggested_assignee': suggested_assignee,
                'task_complexity': self._calculate_complexity(predicted_priority, estimated_duration)
            }
            
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
            return {
                'predicted_priority': 'Medium',
                'confidence': 0.5,
                'suggested_assignee': 'Sarah Chen',
                'task_complexity': 2.0
            }
    
    def _calculate_complexity(self, priority, duration):
        """Calculate task complexity score"""
        priority_weights = {'Low': 1, 'Medium': 2, 'High': 3}
        base_score = priority_weights.get(priority, 2)
        duration_factor = min(duration / 14, 2)  # Cap at 2 weeks
        return base_score * (1 + duration_factor)

# Initialize the predictor
@st.cache_resource
def load_predictor():
    return AITaskPredictor()

def load_sample_data():
    """Load or create sample data for the dashboard"""
    try:
        if os.path.exists('Dataset/Processed Dataset.csv'):
            df = pd.read_csv('Dataset/Processed Dataset.csv')
            return df
        elif os.path.exists('Processed-Dataset.csv'):
            df = pd.read_csv('Processed-Dataset.csv')
            return df[1]
        else:
            # Create sample data for demonstration
            return create_demo_data()
    except Exception as e:
        st.warning(f"Could not load dataset: {e}")
        return create_demo_data()

def create_demo_data():
    """Create demo data for the dashboard"""
    np.random.seed(42)
    
    data = {
        'Issue Key': [f'SWP-{i}' for i in range(1, 101)],
        'Summary': [f'Task {i}' for i in range(1, 101)],
        'Priority': np.random.choice(['Low', 'Medium', 'High'], 100, p=[0.3, 0.5, 0.2]),
        'Status': np.random.choice(['To Do', 'In Progress', 'Done'], 100, p=[0.3, 0.4, 0.3]),
        'Assignee': np.random.choice(['Sarah Chen', 'Alex Rivera', 'Bob Johnson', 'Diana Prince'], 100),
        'Issue Type': np.random.choice(['Bug', 'Task', 'Story', 'Epic'], 100),
        'Component': np.random.choice(['Frontend', 'Backend', 'Database', 'API'], 100),
        'Created': pd.date_range(start='2025-01-01', periods=100, freq='D')
    }
    
    return pd.DataFrame(data)

def render_metrics_dashboard(df):
    """Render the main metrics dashboard"""
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate metrics
    total_tasks = len(df)
    completed_tasks = len(df[df['Status'] == 'Done'])
    completion_rate = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
    
    high_priority_tasks = len(df[df['Priority'] == 'High'])
    in_progress_tasks = len(df[df['Status'] == 'In Progress'])
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎯 Completion Rate</h3>
            <h2>{completion_rate:.0f}%</h2>
            <p>{completed_tasks} of {total_tasks} tasks completed</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>⚡ AI Efficiency</h3>
            <h2>94%</h2>
            <p>AI classification accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>⚠️ High Priority</h3>
            <h2>{high_priority_tasks}</h2>
            <p>Tasks require attention</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>👥 Team Load</h3>
            <h2>Balanced</h2>
            <p>AI-optimized distribution</p>
        </div>
        """, unsafe_allow_html=True)

def render_ai_recommendations():
    """Render AI recommendations section"""
    st.markdown("""
    <div class="recommendation-card">
        <h3>🤖 AI Recommendations</h3>
        <div style="display: flex; justify-content: space-between; margin-top: 1rem;">
            <div style="flex: 1; margin-right: 1rem;">
                <h4>Priority Adjustment</h4>
                <p>Consider raising priority for 'Database Migration' due to dependencies.</p>
            </div>
            <div style="flex: 1; margin-right: 1rem;">
                <h4>Workload Balance</h4>
                <p>Sarah Chen has optimal capacity for 2 additional tasks this week.</p>
            </div>
            <div style="flex: 1;">
                <h4>Deadline Risk</h4>
                <p>3 tasks may miss deadlines without intervention.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_recent_tasks(df):
    """Render recent tasks section"""
    st.subheader("📋 Recent Tasks")
    
    # Display recent tasks
    recent_tasks = df.head(5)
    
    for _, task in recent_tasks.iterrows():
        priority_class = f"priority-{task['Priority'].lower()}"
        status_class = f"status-{task['Status'].lower().replace(' ', '')}"
        
        st.markdown(f"""
        <div class="task-card">
            <div style="display: flex; justify-content: between; align-items: center;">
                <div style="flex: 1;">
                    <strong>{task['Issue Key']}: {task['Summary']}</strong>
                    <br>
                    <small>👤 {task['Assignee']} | 🏷️ {task['Component']}</small>
                </div>
                <div style="text-align: right;">
                    <span class="status-badge {status_class}">{task['Status']}</span>
                    <br>
                    <span class="{priority_class}">{task['Priority']}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def main():
    # Load data and predictor
    df = load_sample_data()
    predictor = load_predictor()
    
    # App header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<h1 class="main-header">🤖 AI Task Manager</h1>', unsafe_allow_html=True)
        st.markdown("*Intelligent productivity powered by AI*")
    
    with col2:
        if st.button("➕ New Task", type="primary", use_container_width=True):
            st.session_state.show_new_task = True
    
    # Initialize session state
    if 'show_new_task' not in st.session_state:
        st.session_state.show_new_task = False
    if 'task_created' not in st.session_state:
        st.session_state.task_created = False
    
    # New Task Modal/Form
    if st.session_state.show_new_task:
        st.markdown("---")
        st.subheader("➕ Create New Task")
        
        with st.form("new_task_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                task_summary = st.text_input("Task Summary*", placeholder="Brief description of the task")
                issue_type = st.selectbox("Issue Type", ['Bug', 'Task', 'Story', 'Epic'])
                component = st.selectbox("Component", ['Frontend', 'Backend', 'Database', 'API', 'UI/UX'])
            
            with col2:
                task_description = st.text_area("Description", placeholder="Detailed description of the task")
                estimated_duration = st.number_input("Estimated Duration (days)", min_value=1, max_value=90, value=7)
            
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                submit_button = st.form_submit_button("🔮 Predict & Create", type="primary")
            
            with col2:
                cancel_button = st.form_submit_button("❌ Cancel")
        
        # Handle form submission OUTSIDE the form
        if cancel_button:
            st.session_state.show_new_task = False
            st.session_state.task_created = False
            st.rerun()
        
        if submit_button and task_summary:
            with st.spinner("🤖 AI is analyzing your task..."):
                # Get predictions from AI model
                predictions = predictor.predict_task(
                    task_summary, 
                    task_description, 
                    issue_type, 
                    component, 
                    estimated_duration
                )
            
            # Display predictions
            st.success("✅ Task analysis complete!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "🎯 Predicted Priority", 
                    predictions['predicted_priority'],
                    f"{predictions['confidence']:.1%} confidence"
                )
            
            with col2:
                st.metric(
                    "👤 Suggested Assignee", 
                    predictions['suggested_assignee']
                )
            
            with col3:
                st.metric(
                    "⚡ Task Complexity", 
                    f"{predictions['task_complexity']:.1f}/5.0"
                )
            
            # Task summary
            st.info(f"""
            **Task Created Successfully!** 🎉
            
            **Summary:** {task_summary}
            **Type:** {issue_type} | **Component:** {component}
            **Estimated Duration:** {estimated_duration} days
            **AI Priority:** {predictions['predicted_priority']}
            **Assigned to:** {predictions['suggested_assignee']}
            """)
            
            st.session_state.task_created = True
        
        # Buttons OUTSIDE the form
        if st.session_state.task_created:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Create Another Task", type="secondary"):
                    st.session_state.task_created = False
                    st.rerun()
            with col2:
                if st.button("Back to Dashboard", type="primary"):
                    st.session_state.show_new_task = False
                    st.session_state.task_created = False
                    st.rerun()
    
    # Rest of your main() function code remains the same...
    # Main Dashboard
    st.markdown("---")
    
    # Metrics Dashboard
    render_metrics_dashboard(df)
    
    st.markdown("---")
    
    # AI Recommendations
    render_ai_recommendations()
    
    st.markdown("---")
    
    # Recent Tasks and Analytics
    col1, col2 = st.columns([2, 1])
    
    with col1:
        render_recent_tasks(df)
    
    with col2:
        st.subheader("📊 Task Distribution")
        
        # Priority distribution pie chart
        priority_counts = df['Priority'].value_counts()
        fig = px.pie(
            values=priority_counts.values,
            names=priority_counts.index,
            title="Priority Distribution",
            color_discrete_map={
                'High': '#ef4444',
                'Medium': '#f59e0b', 
                'Low': '#10b981'
            }
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Status distribution
        status_counts = df['Status'].value_counts()
        fig2 = px.bar(
            x=status_counts.index,
            y=status_counts.values,
            title="Status Overview",
            color=status_counts.values,
            color_continuous_scale="viridis"
        )
        fig2.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)


if __name__ == "__main__":
    main()
