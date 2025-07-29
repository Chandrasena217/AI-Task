# AI Task Manager - Enhanced with ML Integration

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
import pickle
import os
import sys

# Add the current directory to the path to import from ai_task_management_fixed.py
sys.path.append('.')

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AI Task Manager",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
    .task-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .priority-high {
        border-left-color: #d62728 !important;
    }
    .priority-medium {
        border-left-color: #ff7f0e !important;
    }
    .priority-low {
        border-left-color: #2ca02c !important;
    }
    .metric-card {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
    }
    .create-task-form {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'tasks' not in st.session_state:
    st.session_state.tasks = []
if 'show_create_form' not in st.session_state:
    st.session_state.show_create_form = False

# Load processed dataset
@st.cache_data
def load_processed_data():
    """Load the processed dataset"""
    try:
        df = pd.read_csv('Processed-Dataset.csv')[1]
        # Convert date columns
        date_columns = ['Created', 'Due Date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        return df
    except FileNotFoundError:
        st.error("Processed-Dataset.csv not found. Please ensure the file exists in the current directory.")
        return pd.DataFrame()

# Load AI models
@st.cache_resource
def load_ai_models():
    """Load the trained AI models and components"""
    models = {}
    try:
        # Load models if they exist
        if os.path.exists('models/best_model.pkl'):
            with open('models/best_model.pkl', 'rb') as f:
                models['priority_model'] = pickle.load(f)
        
        if os.path.exists('models/scaler.pkl'):
            with open('models/scaler.pkl', 'rb') as f:
                models['scaler'] = pickle.load(f)
        
        if os.path.exists('models/priority_encoder.pkl'):
            with open('models/priority_encoder.pkl', 'rb') as f:
                models['priority_encoder'] = pickle.load(f)
        
        if os.path.exists('models/tfidf_extractor.pkl'):
            with open('models/tfidf_extractor.pkl', 'rb') as f:
                models['tfidf_extractor'] = pickle.load(f)
        
        # Load feature extraction components
        if os.path.exists('Feature Extraction Files/tfidf_features.npy'):
            models['tfidf_features'] = np.load('Feature Extraction Files/tfidf_features.npy')
        
        return models
    except Exception as e:
        st.error(f"Error loading AI models: {str(e)}")
        return {}

# Enhanced AI predictor with ML integration
class EnhancedAIPredictor:
    def __init__(self, models, df):
        self.models = models
        self.df = df
        self.assignees = df['Assignee'].unique().tolist() if 'Assignee' in df.columns else ['Default User']
        
    def predict_priority(self, text, features=None):
        """Predict priority using trained ML model"""
        try:
            if 'priority_model' in self.models and 'tfidf_extractor' in self.models:
                # Extract TF-IDF features for the new text
                tfidf_features = self.models['tfidf_extractor'].transform([text]).toarray()
                
                # Create additional features (simplified)
                additional_features = np.array([[7, 0, 0, 0]])  # Default values
                
                # Combine features
                combined_features = np.hstack([tfidf_features, additional_features])
                
                # Scale features
                if 'scaler' in self.models:
                    scaled_features = self.models['scaler'].transform(combined_features)
                else:
                    scaled_features = combined_features
                
                # Predict
                prediction = self.models['priority_model'].predict(scaled_features)[0]
                confidence = np.max(self.models['priority_model'].predict_proba(scaled_features)[0])
                
                # Convert prediction back to label
                if 'priority_encoder' in self.models:
                    priority_label = self.models['priority_encoder'].inverse_transform([prediction])[0]
                else:
                    priority_label = ['Low', 'Medium', 'High'][prediction % 3]
                
                return priority_label, confidence
            else:
                # Fallback to rule-based prediction
                return self._rule_based_priority(text)
        except Exception as e:
            st.error(f"Error in ML prediction: {str(e)}")
            return self._rule_based_priority(text)
    
    def _rule_based_priority(self, text):
        """Fallback rule-based priority prediction"""
        text_lower = text.lower()
        high_priority_keywords = ['critical', 'urgent', 'bug', 'error', 'crash', 'security', 'production']
        medium_priority_keywords = ['improvement', 'feature', 'update', 'refactor']
        
        high_score = sum(1 for keyword in high_priority_keywords if keyword in text_lower)
        medium_score = sum(1 for keyword in medium_priority_keywords if keyword in text_lower)
        
        if high_score > 0:
            return 'High', 0.8 + (high_score * 0.05)
        elif medium_score > 0:
            return 'Medium', 0.7 + (medium_score * 0.05)
        else:
            return 'Low', 0.6
    
    def suggest_assignee(self, task_info):
        """Suggest assignee based on component expertise and workload"""
        try:
            component = task_info.get('component', 'General')
            
            # Component-based expertise mapping
            component_experts = {
                'Frontend': ['Emma Wilson', 'Alice Brown'],
                'Backend': ['Alex Rivera', 'Bob Johnson'],
                'Database': ['Mike Johnson', 'Charlie Wilson'],
                'API': ['Sarah Chen', 'Alex Rivera'],
                'UI/UX': ['Emma Wilson', 'Alice Brown'],
                'Authentication': ['Bob Johnson', 'Sarah Chen'],
                'Payment Processing': ['Alex Rivera', 'Mike Johnson'],
            }
            
            # Get potential assignees for the component
            potential_assignees = component_experts.get(component, self.assignees)
            
            # Calculate current workload for each assignee
            current_tasks = self.df.groupby('Assignee').size() if 'Assignee' in self.df.columns else pd.Series()
            
            # Find assignee with lowest workload from potential candidates
            best_assignee = None
            min_workload = float('inf')
            confidence = 0.7
            
            for assignee in potential_assignees:
                if assignee in self.assignees:
                    workload = current_tasks.get(assignee, 0)
                    if workload < min_workload:
                        min_workload = workload
                        best_assignee = assignee
                        confidence = max(0.6, 0.9 - (workload * 0.1))
            
            if best_assignee is None:
                best_assignee = self.assignees[0] if self.assignees else 'Default User'
                confidence = 0.5
            
            return best_assignee, confidence
            
        except Exception as e:
            st.error(f"Error in assignee suggestion: {str(e)}")
            return self.assignees[0] if self.assignees else 'Default User', 0.5

# Load data and models
df = load_processed_data()
models = load_ai_models()

# Initialize AI predictor
if not df.empty:
    ai_predictor = EnhancedAIPredictor(models, df)
else:
    ai_predictor = None

# Header
st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <h1 style="color: #1f77b4; margin-bottom: 0.5rem;">🤖 AI Task Manager</h1>
    <p style="color: #666; font-size: 1.2rem;">Intelligent productivity powered by AI</p>
</div>
""", unsafe_allow_html=True)

# Metrics row
if not df.empty:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_tasks = len(df)
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0;">{total_tasks}</h3>
            <p style="margin: 0;">Total Tasks</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        high_priority = len(df[df['Priority'] == 'High']) if 'Priority' in df.columns else 0
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0;">{high_priority}</h3>
            <p style="margin: 0;">High Priority</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        in_progress = len(df[df['Status'] == 'In Progress']) if 'Status' in df.columns else 0
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0;">{in_progress}</h3>
            <p style="margin: 0;">In Progress</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        completed = len(df[df['Status'] == 'Done']) if 'Status' in df.columns else 0
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0;">{completed}</h3>
            <p style="margin: 0;">Completed</p>
        </div>
        """, unsafe_allow_html=True)

# Create Task Section
st.markdown("---")
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("### 📝 Task Management")

with col2:
    if st.button("➕ Create New Task", type="primary"):
        st.session_state.show_create_form = not st.session_state.show_create_form

# Task Creation Form
if st.session_state.show_create_form:
    st.markdown('<div class="create-task-form">', unsafe_allow_html=True)
    st.markdown("#### Create New Task")
    
    with st.form("create_task_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            task_title = st.text_input("Task Title*", placeholder="Enter task title...")
            task_priority = st.selectbox("Priority", ["Low", "Medium", "High"])
            task_component = st.selectbox("Component", [
                "Frontend", "Backend", "Database", "API", "UI/UX", 
                "Authentication", "Payment Processing", "General"
            ])
        
        with col2:
            task_type = st.selectbox("Task Type", ["Task", "Bug", "Story", "Epic"])
            task_status = st.selectbox("Status", ["To Do", "In Progress", "Done"])
            due_date = st.date_input("Due Date", value=datetime.now() + timedelta(days=7))
        
        task_description = st.text_area("Description*", placeholder="Describe the task in detail...")
        
        submitted = st.form_submit_button("🤖 Create Task with AI Analysis", type="primary")
        
        if submitted and task_title and task_description and ai_predictor:
            # Combine title and description for AI analysis
            combined_text = f"{task_title} {task_description}"
            
            # Get AI predictions
            predicted_priority, priority_confidence = ai_predictor.predict_priority(combined_text)
            suggested_assignee, assignee_confidence = ai_predictor.suggest_assignee({
                'component': task_component,
                'priority': predicted_priority
            })
            
            # Create new task
            new_task = {
                'id': len(st.session_state.tasks) + len(df) + 1,
                'title': task_title,
                'description': task_description,
                'priority': task_priority,
                'predicted_priority': predicted_priority,
                'priority_confidence': priority_confidence,
                'issue_type': task_type,
                'status': task_status,
                'component': task_component,
                'suggested_assignee': suggested_assignee,
                'assignee_confidence': assignee_confidence,
                'created': datetime.now(),
                'due_date': due_date,
                'ai_analyzed': True
            }
            
            # Add to session state
            st.session_state.tasks.insert(0, new_task)
            
            # Show success message with AI analysis
            st.success("✅ Task created successfully with AI analysis!")
            
            # Display AI insights
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"🤖 **AI Priority Prediction:** {predicted_priority} (Confidence: {priority_confidence:.2f})")
            with col2:
                st.info(f"👤 **Suggested Assignee:** {suggested_assignee} (Confidence: {assignee_confidence:.2f})")
            
            # Hide form after creation
            st.session_state.show_create_form = False
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# Recent Tasks Section
st.markdown("### 📋 Recent Tasks")

# Combine session tasks with dataset tasks for display
all_tasks = []

# Add newly created tasks from session state
for task in st.session_state.tasks:
    all_tasks.append({
        'Title': task['title'],
        'Priority': task['priority'],
        'Status': task['status'],
        'Component': task['component'],
        'Assignee': task.get('suggested_assignee', 'Unassigned'),
        'Created': task['created'],
        'AI_Analyzed': task.get('ai_analyzed', False),
        'AI_Priority': task.get('predicted_priority', ''),
        'Priority_Confidence': task.get('priority_confidence', 0),
        'Assignee_Confidence': task.get('assignee_confidence', 0)
    })

# Add recent tasks from dataset
if not df.empty and len(df) > 0:
    recent_df = df.head(10).copy()
    for _, row in recent_df.iterrows():
        all_tasks.append({
            'Title': row.get('Summary', 'No Title'),
            'Priority': row.get('Priority', 'Medium'),
            'Status': row.get('Status', 'Open'),
            'Component': row.get('Component', 'General'),
            'Assignee': row.get('Assignee', 'Unassigned'),
            'Created': row.get('Created', datetime.now()),
            'AI_Analyzed': False,
            'AI_Priority': '',
            'Priority_Confidence': 0,
            'Assignee_Confidence': 0
        })

# Display tasks
if all_tasks:
    for i, task in enumerate(all_tasks[:15]):  # Show top 15 tasks
        priority_class = f"priority-{task['Priority'].lower()}"
        
        ai_badge = ""
        if task['AI_Analyzed']:
            ai_badge = f"""
            <div style="float: right;">
                <span style="background: #28a745; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8rem;">
                    🤖 AI Analyzed
                </span>
            </div>
            """
        
        confidence_info = ""
        if task['AI_Analyzed']:
            confidence_info = f"""
            <small style="color: #666;">
                AI Priority: {task['AI_Priority']} ({task['Priority_Confidence']:.2f}) | 
                Assignee Confidence: {task['Assignee_Confidence']:.2f}
            </small><br>
            """
        
        st.markdown(f"""
        <div class="task-card {priority_class}">
            {ai_badge}
            <h4 style="margin: 0 0 0.5rem 0; color: #333;">{task['Title']}</h4>
            {confidence_info}
            <div style="display: flex; gap: 1rem; align-items: center; flex-wrap: wrap;">
                <span style="background: #e9ecef; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem;">
                    📊 {task['Priority']}
                </span>
                <span style="background: #e9ecef; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem;">
                    📋 {task['Status']}
                </span>
                <span style="background: #e9ecef; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem;">
                    🏗️ {task['Component']}
                </span>
                <span style="background: #e9ecef; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem;">
                    👤 {task['Assignee']}
                </span>
                <span style="color: #666; font-size: 0.8rem;">
                    📅 {task['Created'].strftime('%Y-%m-%d') if hasattr(task['Created'], 'strftime') else str(task['Created'])[:10]}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("No tasks available. Create your first task using the button above!")

# Analytics Section
if not df.empty:
    st.markdown("---")
    st.markdown("### 📊 Task Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Priority distribution
        if 'Priority' in df.columns:
            priority_counts = df['Priority'].value_counts()
            fig_priority = px.pie(
                values=priority_counts.values,
                names=priority_counts.index,
                title="Task Priority Distribution",
                color_discrete_map={
                    'High': '#d62728',
                    'Medium': '#ff7f0e',
                    'Low': '#2ca02c'
                }
            )
            st.plotly_chart(fig_priority, use_container_width=True)
    
    with col2:
        # Status distribution
        if 'Status' in df.columns:
            status_counts = df['Status'].value_counts()
            fig_status = px.bar(
                x=status_counts.index,
                y=status_counts.values,
                title="Task Status Distribution",
                labels={'x': 'Status', 'y': 'Count'}
            )
            st.plotly_chart(fig_status, use_container_width=True)

# AI Model Status
st.markdown("---")
st.markdown("### 🤖 AI Model Status")

col1, col2, col3 = st.columns(3)

with col1:
    model_status = "✅ Loaded" if 'priority_model' in models else "❌ Not Available"
    st.metric("Priority Prediction Model", model_status)

with col2:
    feature_status = "✅ Loaded" if 'tfidf_extractor' in models else "❌ Not Available"
    st.metric("Feature Extraction", feature_status)

with col3:
    scaler_status = "✅ Loaded" if 'scaler' in models else "❌ Not Available"
    st.metric("Feature Scaling", scaler_status)

if not models:
    st.warning("⚠️ AI models not found. Please run the ai_task_management_fixed.py script first to train the models.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    Built with ❤️ using Streamlit and AI/ML models
</div>
""", unsafe_allow_html=True)
