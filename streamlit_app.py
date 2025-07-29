import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import required modules from your existing code
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

# Download NLTK data if not already present
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except:
        pass

# Initialize NLTK
download_nltk_data()

# Set page config
st.set_page_config(
    page_title="AI Task Management Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .priority-high {
        color: #ff4b4b;
        font-weight: bold;
    }
    .priority-medium {
        color: #ff8c00;
        font-weight: bold;
    }
    .priority-low {
        color: #00b4d8;
        font-weight: bold;
    }
    .task-card {
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

class AITaskManager:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = set(stopwords.words('english')) if stopwords else set()
        self.load_models()
        self.load_data()
    
    def load_models(self):
        """Load pre-trained models if they exist"""
        try:
            if os.path.exists('models/best_model.pkl'):
                with open('models/best_model.pkl', 'rb') as f:
                    self.model = pickle.load(f)
            else:
                # Fallback model
                self.model = RandomForestClassifier(n_estimators=50, random_state=42)
            
            if os.path.exists('models/scaler.pkl'):
                with open('models/scaler.pkl', 'rb') as f:
                    self.scaler = pickle.load(f)
            else:
                self.scaler = StandardScaler()
            
            if os.path.exists('models/priority_encoder.pkl'):
                with open('models/priority_encoder.pkl', 'rb') as f:
                    self.priority_encoder = pickle.load(f)
            else:
                self.priority_encoder = LabelEncoder()
                self.priority_encoder.fit(['Low', 'Medium', 'High'])
            
            if os.path.exists('models/tfidf_extractor.pkl'):
                with open('models/tfidf_extractor.pkl', 'rb') as f:
                    self.tfidf_extractor = pickle.load(f)
            else:
                self.tfidf_extractor = TfidfVectorizer(max_features=1000, stop_words='english')
        except Exception as e:
            st.error(f"Error loading models: {e}")
            # Initialize fallback models
            self.model = RandomForestClassifier(n_estimators=50, random_state=42)
            self.scaler = StandardScaler()
            self.priority_encoder = LabelEncoder()
            self.priority_encoder.fit(['Low', 'Medium', 'High'])
            self.tfidf_extractor = TfidfVectorizer(max_features=1000, stop_words='english')
    
    def load_data(self):
        """Load the processed dataset"""
        try:
            self.df = pd.read_csv('Processed-Dataset.csv')
        except:
            st.error("Could not load Processed-Dataset.csv. Please ensure the file exists.")
            # Create sample data for demonstration
            self.df = self.create_sample_data()
    
    def create_sample_data(self):
        """Create sample data for demonstration"""
        np.random.seed(42)
        priorities = ['Low', 'Medium', 'High']
        issue_types = ['Bug', 'Task', 'Story', 'Epic']
        statuses = ['To Do', 'In Progress', 'Done', 'Blocked']
        assignees = ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank']
        
        n_samples = 200
        data = {
            'Issue Key': [f"SWP-{i}" for i in range(1, n_samples + 1)],
            'Summary': [f"Task {i}: Sample task description" for i in range(1, n_samples + 1)],
            'Priority': np.random.choice(priorities, n_samples),
            'Issue Type': np.random.choice(issue_types, n_samples),
            'Status': np.random.choice(statuses, n_samples),
            'Assignee': np.random.choice(assignees, n_samples),
            'Created': pd.date_range(start='2024-01-01', end='2025-07-29', periods=n_samples),
            'Days to Complete': np.random.randint(1, 30, n_samples)
        }
        return pd.DataFrame(data)
    
    def preprocess_text(self, text):
        """Preprocess text for model input"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        try:
            tokens = word_tokenize(text)
            tokens = [token for token in tokens if token not in self.stopwords]
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
            return ' '.join(tokens)
        except:
            return text
    
    def predict_task_assignment(self, task_summary, task_description="", estimated_days=7):
        """Predict task assignment and priority"""
        try:
            # Preprocess text
            processed_summary = self.preprocess_text(task_summary)
            processed_description = self.preprocess_text(task_description)
            combined_text = f"{processed_summary} {processed_description}"
            
            # For demonstration, we'll use a simple assignment logic
            # In practice, this would use your trained models
            assignees = ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank']
            priorities = ['Low', 'Medium', 'High']
            
            # Simple heuristic-based assignment
            if any(word in combined_text.lower() for word in ['bug', 'error', 'fix', 'issue']):
                predicted_priority = 'High'
                confidence = 0.85
            elif any(word in combined_text.lower() for word in ['feature', 'new', 'implement']):
                predicted_priority = 'Medium'
                confidence = 0.75
            else:
                predicted_priority = 'Low'
                confidence = 0.65
            
            # Calculate workload for each assignee
            current_workload = self.df[self.df['Status'].isin(['To Do', 'In Progress'])].groupby('Assignee').size()
            
            # Assign to person with lowest workload
            if len(current_workload) > 0:
                suggested_assignee = current_workload.idxmin()
                workload_score = current_workload.min() / current_workload.max() if current_workload.max() > 0 else 0
            else:
                suggested_assignee = np.random.choice(assignees)
                workload_score = 0.5
            
            return {
                'suggested_assignee': suggested_assignee,
                'predicted_priority': predicted_priority,
                'confidence': confidence,
                'workload_score': workload_score,
                'estimated_days': estimated_days
            }
        except Exception as e:
            st.error(f"Error in prediction: {e}")
            return {
                'suggested_assignee': 'Alice',
                'predicted_priority': 'Medium',
                'confidence': 0.5,
                'workload_score': 0.5,
                'estimated_days': estimated_days
            }

# Initialize the AI Task Manager
@st.cache_resource
def load_task_manager():
    return AITaskManager()

def main():
    st.title("🤖 AI-Powered Task Management Dashboard")
    st.markdown("---")
    
    # Load the task manager
    task_manager = load_task_manager()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "📊 Dashboard Overview",
        "➕ Add New Task",
        "📋 Task Management",
        "👥 Team Analytics",
        "📈 Performance Insights"
    ])
    
    if page == "📊 Dashboard Overview":
        dashboard_overview(task_manager)
    elif page == "➕ Add New Task":
        add_new_task(task_manager)
    elif page == "📋 Task Management":
        task_management(task_manager)
    elif page == "👥 Team Analytics":
        team_analytics(task_manager)
    elif page == "📈 Performance Insights":
        performance_insights(task_manager)

def dashboard_overview(task_manager):
    st.header("Dashboard Overview")
    
    df = task_manager.df
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_tasks = len(df)
        st.metric("Total Tasks", total_tasks)
    
    with col2:
        completed_tasks = len(df[df['Status'] == 'Done'])
        completion_rate = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        st.metric("Completion Rate", f"{completion_rate:.1f}%")
    
    with col3:
        active_tasks = len(df[df['Status'].isin(['To Do', 'In Progress'])])
        st.metric("Active Tasks", active_tasks)
    
    with col4:
        avg_completion_time = df[df['Status'] == 'Done']['Days to Complete'].mean()
        st.metric("Avg Completion Time", f"{avg_completion_time:.1f} days" if not pd.isna(avg_completion_time) else "N/A")
    
    st.markdown("---")
    
    # Charts Row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Task Status Distribution")
        status_counts = df['Status'].value_counts()
        fig_status = px.pie(values=status_counts.values, names=status_counts.index,
                           title="Task Status Distribution")
        st.plotly_chart(fig_status, use_container_width=True)
    
    with col2:
        st.subheader("Priority Distribution")
        priority_counts = df['Priority'].value_counts()
        colors = {'High': '#ff4b4b', 'Medium': '#ff8c00', 'Low': '#00b4d8'}
        fig_priority = px.bar(x=priority_counts.index, y=priority_counts.values,
                             title="Priority Distribution",
                             color=priority_counts.index,
                             color_discrete_map=colors)
        st.plotly_chart(fig_priority, use_container_width=True)
    
    # Team Workload
    st.subheader("Team Workload Analysis")
    workload_data = df[df['Status'].isin(['To Do', 'In Progress'])].groupby('Assignee').size().reset_index()
    workload_data.columns = ['Assignee', 'Active Tasks']
    
    if not workload_data.empty:
        fig_workload = px.bar(workload_data, x='Assignee', y='Active Tasks',
                             title="Current Team Workload",
                             color='Active Tasks',
                             color_continuous_scale='Reds')
        st.plotly_chart(fig_workload, use_container_width=True)
    else:
        st.info("No active tasks found.")

def add_new_task(task_manager):
    st.header("Add New Task")
    st.markdown("Enter task details below and let AI suggest the optimal assignment.")
    
    with st.form("new_task_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            task_summary = st.text_input("Task Summary*", placeholder="Enter a brief task summary...")
            task_type = st.selectbox("Task Type", ["Task", "Bug", "Story", "Epic"])
            estimated_days = st.number_input("Estimated Days", min_value=1, max_value=365, value=7)
        
        with col2:
            task_description = st.text_area("Task Description", placeholder="Enter detailed description...")
            due_date = st.date_input("Due Date", value=datetime.now() + timedelta(days=7))
            reporter = st.selectbox("Reporter", ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"])
        
        submitted = st.form_submit_button("🤖 Get AI Recommendation", type="primary")
        
        if submitted and task_summary:
            with st.spinner("AI is analyzing the task..."):
                prediction = task_manager.predict_task_assignment(
                    task_summary, task_description, estimated_days
                )
            
            st.success("AI Analysis Complete!")
            
            # Display AI Recommendations
            st.subheader("🎯 AI Recommendations")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Suggested Assignee", prediction['suggested_assignee'])
                st.metric("Confidence Score", f"{prediction['confidence']:.0%}")
            
            with col2:
                priority_color = {
                    'High': '🔴', 'Medium': '🟡', 'Low': '🟢'
                }
                st.metric("Predicted Priority", 
                         f"{priority_color.get(prediction['predicted_priority'], '⚪')} {prediction['predicted_priority']}")
                st.metric("Workload Score", f"{prediction['workload_score']:.2f}")
            
            with col3:
                st.metric("Estimated Duration", f"{prediction['estimated_days']} days")
                completion_date = datetime.now() + timedelta(days=prediction['estimated_days'])
                st.metric("Expected Completion", completion_date.strftime("%Y-%m-%d"))
            
            # Confirmation section
            st.subheader("Confirm Task Creation")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Accept AI Recommendation", type="primary"):
                    # Here you would save the task to your database
                    st.success("Task created successfully with AI recommendations!")
                    st.balloons()
            
            with col2:
                if st.button("✏️ Modify Assignment"):
                    st.info("You can manually modify the assignment before saving.")

def task_management(task_manager):
    st.header("Task Management")
    
    df = task_manager.df
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status_filter = st.multiselect("Filter by Status", 
                                      options=df['Status'].unique(),
                                      default=df['Status'].unique())
    
    with col2:
        priority_filter = st.multiselect("Filter by Priority",
                                        options=df['Priority'].unique(),
                                        default=df['Priority'].unique())
    
    with col3:
        assignee_filter = st.multiselect("Filter by Assignee",
                                        options=df['Assignee'].unique(),
                                        default=df['Assignee'].unique())
    
    # Filter dataframe
    filtered_df = df[
        (df['Status'].isin(status_filter)) &
        (df['Priority'].isin(priority_filter)) &
        (df['Assignee'].isin(assignee_filter))
    ]
    
    st.markdown(f"**Showing {len(filtered_df)} of {len(df)} tasks**")
    
    # Recent Tasks
    st.subheader("Recent Tasks")
    
    # Display tasks in cards
    for index, task in filtered_df.head(10).iterrows():
        with st.container():
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            
            with col1:
                priority_emoji = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}
                st.markdown(f"**{task['Issue Key']}**: {task['Summary']}")
                st.caption(f"Assignee: {task['Assignee']} | Created: {task['Created']}")
            
            with col2:
                st.markdown(f"{priority_emoji.get(task['Priority'], '⚪')} {task['Priority']}")
            
            with col3:
                status_emoji = {'Done': '✅', 'In Progress': '🔄', 'To Do': '📋', 'Blocked': '🚫'}
                st.markdown(f"{status_emoji.get(task['Status'], '📋')} {task['Status']}")
            
            with col4:
                st.markdown(f"⏱️ {task['Days to Complete']} days")
            
            st.markdown("---")

def team_analytics(task_manager):
    st.header("Team Analytics")
    
    df = task_manager.df
    
    # Team Performance Overview
    st.subheader("Team Performance Overview")
    
    team_stats = df.groupby('Assignee').agg({
        'Issue Key': 'count',
        'Days to Complete': 'mean',
        'Status': lambda x: (x == 'Done').sum()
    }).round(2)
    
    team_stats.columns = ['Total Tasks', 'Avg Completion Time', 'Completed Tasks']
    team_stats['Completion Rate'] = (team_stats['Completed Tasks'] / team_stats['Total Tasks'] * 100).round(1)
    
    st.dataframe(team_stats, use_container_width=True)
    
    # Individual Performance Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Task Distribution by Team Member")
        fig_team = px.bar(team_stats.reset_index(), x='Assignee', y='Total Tasks',
                         title="Tasks per Team Member")
        st.plotly_chart(fig_team, use_container_width=True)
    
    with col2:
        st.subheader("Average Completion Time")
        fig_time = px.bar(team_stats.reset_index(), x='Assignee', y='Avg Completion Time',
                         title="Average Days to Complete")
        st.plotly_chart(fig_time, use_container_width=True)
    
    # AI Efficiency Score
    st.subheader("AI Efficiency Score")
    st.info("The AI Efficiency Score is calculated based on task completion rate, average completion time, and workload balance.")
    
    # Calculate efficiency scores
    efficiency_scores = {}
    for assignee in team_stats.index:
        completion_rate = team_stats.loc[assignee, 'Completion Rate']
        avg_time = team_stats.loc[assignee, 'Avg Completion Time']
        
        # Normalize scores (this is a simplified calculation)
        time_score = max(0, 100 - avg_time * 3)  # Lower time is better
        efficiency_score = (completion_rate * 0.6 + time_score * 0.4)
        efficiency_scores[assignee] = min(100, max(0, efficiency_score))
    
    efficiency_df = pd.DataFrame(list(efficiency_scores.items()), columns=['Assignee', 'Efficiency Score'])
    
    fig_efficiency = px.bar(efficiency_df, x='Assignee', y='Efficiency Score',
                           title="AI Efficiency Score by Team Member",
                           color='Efficiency Score',
                           color_continuous_scale='Greens',
                           range_y=[0, 100])
    st.plotly_chart(fig_efficiency, use_container_width=True)

def performance_insights(task_manager):
    st.header("Performance Insights")
    
    df = task_manager.df
    
    # Convert Created column to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['Created']):
        df['Created'] = pd.to_datetime(df['Created'])
    
    # Tasks created over time
    st.subheader("Task Creation Trends")
    
    df['Month'] = df['Created'].dt.to_period('M')
    monthly_tasks = df.groupby('Month').size().reset_index()
    monthly_tasks.columns = ['Month', 'Tasks Created']
    monthly_tasks['Month'] = monthly_tasks['Month'].astype(str)
    
    fig_trends = px.line(monthly_tasks, x='Month', y='Tasks Created',
                        title="Tasks Created Over Time")
    st.plotly_chart(fig_trends, use_container_width=True)
    
    # Priority vs Completion Time
    st.subheader("Priority vs Completion Time Analysis")
    
    completed_tasks = df[df['Status'] == 'Done']
    if not completed_tasks.empty:
        fig_priority_time = px.box(completed_tasks, x='Priority', y='Days to Complete',
                                  title="Completion Time by Priority")
        st.plotly_chart(fig_priority_time, use_container_width=True)
    else:
        st.info("No completed tasks found for analysis.")
    
    # Task Type Analysis
    st.subheader("Task Type Performance")
    
    if 'Issue Type' in df.columns:
        type_stats = df.groupby('Issue Type').agg({
            'Days to Complete': 'mean',
            'Issue Key': 'count'
        }).round(2)
        type_stats.columns = ['Avg Completion Time', 'Total Tasks']
        
        fig_type = px.scatter(type_stats.reset_index(), x='Total Tasks', y='Avg Completion Time',
                             size='Total Tasks', color='Issue Type',
                             title="Task Type Analysis")
        st.plotly_chart(fig_type, use_container_width=True)

if __name__ == "__main__":
    main()
