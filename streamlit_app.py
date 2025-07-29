import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os
from datetime import datetime, timedelta

# Import custom modules
from ai_task_management_fixed import (
    TFIDFExtractor, Word2VecExtractor, WorkloadManager,
    preprocess_text, clean_text
)

st.set_page_config(page_title='AI Task Management System', layout='wide')

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

def load_models():
    try:
        # Load the saved models and encoders
        with open('models/best_model.pkl', 'rb') as f:
            best_model = pickle.load(f)
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('models/priority_encoder.pkl', 'rb') as f:
            priority_encoder = pickle.load(f)
        with open('models/tfidf_extractor.pkl', 'rb') as f:
            tfidf_extractor = pickle.load(f)
        
        return best_model, scaler, priority_encoder, tfidf_extractor
    except:
        st.error("Error loading models. Please ensure model files exist in the 'models' directory.")
        return None, None, None, None

def initialize_app():
    if not st.session_state.initialized:
        # Load the dataset
        try:
            df = pd.read_csv('Dataset/Processed Dataset.csv')
            st.session_state.df = df
            
            # Initialize WorkloadManager
            st.session_state.workload_manager = WorkloadManager(df)
            
            # Load models
            models = load_models()
            if all(model is not None for model in models):
                (st.session_state.best_model, 
                 st.session_state.scaler,
                 st.session_state.priority_encoder,
                 st.session_state.tfidf_extractor) = models
                
            st.session_state.initialized = True
            
        except Exception as e:
            st.error(f"Error initializing app: {str(e)}")
            return False
    return True

# Main App UI
def main():
    st.title("AI Task Management System")
    
    if not initialize_app():
        st.stop()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Task Creation", "Task Analysis", "Workload Dashboard"])
    
    if page == "Task Creation":
        show_task_creation()
    elif page == "Task Analysis":
        show_task_analysis()
    else:
        show_workload_dashboard()

def show_task_creation():
    st.header("Create New Task")
    
    # Task input form
    with st.form("task_form"):
        summary = st.text_input("Task Summary")
        description = st.text_area("Task Description")
        
        col1, col2 = st.columns(2)
        with col1:
            issue_type = st.selectbox("Issue Type", 
                                    ['Bug', 'Feature', 'Task', 'Story', 'Epic'])
            component = st.selectbox("Component",
                                   ['Frontend', 'Backend', 'Database', 'API', 'UI/UX'])
        
        with col2:
            status = st.selectbox("Status",
                                ['To Do', 'In Progress', 'Done'])
            due_date = st.date_input("Due Date", 
                                   min_value=datetime.now().date())
        
        submit = st.form_submit_button("Create Task")
        
        if submit:
            process_task_creation(summary, description, issue_type, 
                                component, status, due_date)

def process_task_creation(summary, description, issue_type, component, status, due_date):
    try:
        # Preprocess text
        combined_text = f"{summary} {description}"
        processed_text = clean_text(combined_text)
        
        # Extract features
        features = st.session_state.tfidf_extractor.transform([processed_text])
        
        # Create additional features
        additional_features = np.array([
            (due_date - datetime.now().date()).days,  # Duration
            0,  # Issue Type code (dummy)
            0,  # Status code (dummy)
            0   # Component code (dummy)
        ]).reshape(1, -1)
        
        # Combine features
        combined_features = np.hstack([features.toarray(), additional_features])
        
        # Scale features
        scaled_features = st.session_state.scaler.transform(combined_features)
        
        # Predict priority
        priority_pred = st.session_state.best_model.predict(scaled_features)[0]
        priority_label = st.session_state.priority_encoder.inverse_transform([priority_pred])[0]
        
        # Get assignee suggestion
        task_features = {
            'Priority': priority_label,
            'Duration': (due_date - datetime.now().date()).days
        }
        
        suggested_assignee, assignment_details = st.session_state.workload_manager.suggest_assignee(task_features)
        
        # Display results
        st.success("Task processed successfully!")
        
        st.subheader("Task Details:")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Predicted Priority:**", priority_label)
            st.write("**Suggested Assignee:**", suggested_assignee)
        with col2:
            st.write("**Task Complexity:**", f"{assignment_details['task_complexity']:.2f}")
            st.write("**Workload Score:**", f"{assignment_details['workload_score']:.2f}")
        
    except Exception as e:
        st.error(f"Error processing task: {str(e)}")

def show_task_analysis():
    st.header("Task Analysis")
    
    if 'df' not in st.session_state:
        st.warning("No data available for analysis.")
        return
    
    df = st.session_state.df
    
    # High-level metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Tasks", len(df))
    with col2:
        st.metric("Open Tasks", len(df[df['Status'] != 'Done']))
    with col3:
        st.metric("Completed Tasks", len(df[df['Status'] == 'Done']))
    with col4:
        st.metric("Average Duration", f"{df['Days to Complete'].mean():.1f} days")
    
    # Priority Distribution
    st.subheader("Priority Distribution")
    priority_dist = df['Priority'].value_counts()
    st.bar_chart(priority_dist)
    
    # Issue Type Analysis
    st.subheader("Issue Type Distribution")
    issue_dist = df['Issue Type'].value_counts()
    st.bar_chart(issue_dist)

def show_workload_dashboard():
    st.header("Workload Dashboard")
    
    if 'workload_manager' not in st.session_state:
        st.warning("Workload manager not initialized.")
        return
    
    workload = st.session_state.workload_manager.user_workload
    
    # Workload Overview
    st.subheader("Current Workload Distribution")
    if not workload.empty:
        st.bar_chart(workload)
        
        # Detailed workload table
        st.subheader("Detailed Workload")
        workload_df = pd.DataFrame({
            'Assignee': workload.index,
            'Active Tasks': workload.values
        })
        st.dataframe(workload_df)
    else:
        st.info("No workload data available.")

if __name__ == "__main__":
    main()