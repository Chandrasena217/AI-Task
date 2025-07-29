import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title='AI Task Manager',
    page_icon='🤖',
    layout='wide',
    initial_sidebar_state='collapsed'
)

# Custom CSS to match the screenshot design
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border: 1px solid #e1e5e9;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.5rem;
    }
    
    .metric-description {
        font-size: 0.8rem;
        color: #888;
        margin-top: 0.25rem;
    }
    
    .ai-recommendations {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin: 2rem 0;
    }
    
    .ai-recommendations h3 {
        color: white;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
    }
    
    .recommendation-item {
        background: rgba(255,255,255,0.1);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
    }
    
    .recommendation-title {
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .task-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ddd;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .task-card.high {
        border-left-color: #ff4757;
    }
    
    .task-card.medium {
        border-left-color: #ffa502;
    }
    
    .task-card.low {
        border-left-color: #2ed573;
    }
    
    .priority-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: bold;
        text-transform: uppercase;
    }
    
    .priority-high {
        background: #ffebee;
        color: #c62828;
    }
    
    .priority-medium {
        background: #fff3e0;
        color: #ef6c00;
    }
    
    .priority-low {
        background: #e8f5e8;
        color: #2e7d32;
    }
    
    .status-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.75rem;
        background: #e3f2fd;
        color: #1565c0;
    }
    
    .new-task-btn {
        background: #6c5ce7;
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: bold;
        cursor: pointer;
        text-decoration: none;
        display: inline-block;
    }
    
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 2rem;
    }
    
    .header-title {
        margin: 0;
    }
    
    .header-subtitle {
        color: #666;
        margin-top: 0.25rem;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.show_new_task = False

@st.cache_data
def load_data():
    """Load and cache the dataset"""
    try:
        df = pd.read_csv('Processed-Dataset.csv')
        return df
    except:
        try:
            df = pd.read_csv('Dataset/Processed Dataset.csv')
            return df
        except:
            # Create sample data if files not found
            return create_sample_data()

def create_sample_data():
    """Create sample data for demonstration"""
    data = {
        'Issue Key': ['SWP-1', 'SWP-2', 'SWP-3'],
        'Summary': [
            'Implement AI Task Classification',
            'User Interface Design Review', 
            'Database Migration Tool'
        ],
        'Priority': ['High', 'Medium', 'High'],
        'Status': ['In Progress', 'To Do', 'Done'],
        'Assignee': ['Sarah Chen', 'Alex Rivera', 'Bob Johnson'],
        'Issue Type': ['Development', 'Design', 'Task'],
        'Days to Complete': [7, 3, 15],
        'Created': pd.date_range('2025-07-20', periods=3),
        'Due Date': pd.date_range('2025-07-27', periods=3)
    }
    return pd.DataFrame(data)

def calculate_metrics(df):
    """Calculate dashboard metrics"""
    total_tasks = len(df)
    completed_tasks = len(df[df['Status'] == 'Done'])
    high_priority_tasks = len(df[df['Priority'] == 'High'])
    in_progress_tasks = len(df[df['Status'] == 'In Progress'])
    
    completion_rate = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
    ai_efficiency = 94  # Mock AI efficiency score
    
    # Calculate team load balance
    if 'Assignee' in df.columns:
        workload = df[df['Status'] != 'Done'].groupby('Assignee').size()
        team_balance = "Balanced" if workload.std() < 2 else "Unbalanced"
    else:
        team_balance = "Balanced"
    
    return {
        'completion_rate': completion_rate,
        'completed_tasks': completed_tasks,
        'total_tasks': total_tasks,
        'ai_efficiency': ai_efficiency,
        'high_priority_tasks': high_priority_tasks,
        'team_balance': team_balance
    }

def show_metric_card(title, value, description, color="blue"):
    """Display a metric card"""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color: {color};">{value}</div>
        <div class="metric-label">{title}</div>
        <div class="metric-description">{description}</div>
    </div>
    """, unsafe_allow_html=True)

def show_ai_recommendations():
    """Display AI recommendations section"""
    st.markdown("""
    <div class="ai-recommendations">
        <h3>⚡ AI Recommendations</h3>
        
        <div class="recommendation-item">
            <div class="recommendation-title">Priority Adjustment</div>
            <div>Consider raising priority for 'Database Migration' due to dependencies.</div>
        </div>
        
        <div class="recommendation-item">
            <div class="recommendation-title">Workload Balance</div>
            <div>Sarah Chen has optimal capacity for 2 additional tasks this week.</div>
        </div>
        
        <div class="recommendation-item">
            <div class="recommendation-title">Deadline Risk</div>
            <div>3 tasks may miss deadlines without intervention.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def show_recent_tasks(df):
    """Display recent tasks section"""
    st.markdown("### 📋 Recent Tasks")
    
    # Get recent tasks (limit to 10)
    recent_tasks = df.head(10)  
    
    for _, task in recent_tasks.iterrows():
        priority_class = task['Priority'].lower()
        priority_color = {
            'high': 'priority-high',
            'medium': 'priority-medium', 
            'low': 'priority-low'
        }.get(priority_class, 'priority-medium')
        
        # Mock AI score
        ai_score = np.random.randint(75, 99)
        
        st.markdown(f"""
        <div class="task-card {priority_class}">
            <div style="display: flex; justify-content: space-between; align-items: start;">
                <div style="flex: 1;">
                    <div style="font-weight: bold; margin-bottom: 0.5rem;">{task['Summary']}</div>
                    <div style="display: flex; gap: 0.5rem; align-items: center; margin-bottom: 0.5rem;">
                        <span class="priority-badge {priority_color}">{task['Priority']}</span>
                        <span class="status-badge">{task['Status']}</span>
                        <span style="color: #666; font-size: 0.8rem;">{task['Issue Type']}</span>
                    </div>
                    <div style="color: #666; font-size: 0.9rem;">
                        👤 {task['Assignee']} • 📅 {task.get('Created', datetime.now().strftime('%Y-%m-%d'))}
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="color: #666; font-size: 0.8rem;">AI Score</div>
                    <div style="font-weight: bold; color: #2ed573;">{ai_score}%</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def show_new_task_modal():
    """Show new task creation modal"""
    with st.container():
        st.markdown("### ✨ Create New Task")
        
        with st.form("new_task_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                summary = st.text_input("Task Summary*", placeholder="Enter task summary...")
                description = st.text_area("Description", placeholder="Describe the task...")
                priority = st.selectbox("Priority", ["Low", "Medium", "High"])
                
            with col2:
                issue_type = st.selectbox("Issue Type", ["Bug", "Task", "Story", "Epic"])
                assignee = st.selectbox("Assignee", ["Sarah Chen", "Alex Rivera", "Bob Johnson", "Alice Brown"])
                due_date = st.date_input("Due Date", min_value=datetime.now().date())
            
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                submitted = st.form_submit_button("Create Task", use_container_width=True)
                
            if submitted and summary:
                st.success(f"✅ Task '{summary}' created successfully!")
                st.session_state.show_new_task = False
                st.rerun()

def main():
    """Main application"""
    # Load data
    df = load_data()
    metrics = calculate_metrics(df)
    
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
        <div>
            <h1 class="header-title">🤖 AI Task Manager</h1>
            <p class="header-subtitle">Intelligent productivity powered by AI</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.button("➕ New Task", key="new_task_btn"):
            st.session_state.show_new_task = True
    
    # Show new task modal if requested
    if st.session_state.show_new_task:
        show_new_task_modal()
        if st.button("❌ Cancel", key="cancel_task"):
            st.session_state.show_new_task = False
            st.rerun()
        st.divider()
    
    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        show_metric_card(
            "Completion Rate", 
            f"{metrics['completion_rate']:.0f}%",
            f"{metrics['completed_tasks']} of {metrics['total_tasks']} tasks completed",
            "#6c5ce7"
        )
    
    with col2:
        show_metric_card(
            "AI Efficiency", 
            f"{metrics['ai_efficiency']}%",
            "AI classification accuracy",
            "#00b894"
        )
    
    with col3:
        show_metric_card(
            "High Priority", 
            str(metrics['high_priority_tasks']),
            "Tasks require attention",
            "#fdcb6e"
        )
    
    with col4:
        show_metric_card(
            "Team Load", 
            metrics['team_balance'],
            "AI-optimized distribution",
            "#fd79a8"
        )
    
    # AI Recommendations
    show_ai_recommendations()
    
    # Recent Tasks
    show_recent_tasks(df)

if __name__ == "__main__":
    main()
