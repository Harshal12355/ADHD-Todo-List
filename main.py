import streamlit as st
import os
from ai_todo import get_subtasks, rank_tasks, set_model_config, update_chains, save_todo_list, load_todo_list, get_saved_lists
from datetime import datetime

st.set_page_config(page_title="AI ADHD-Friendly To-Do List", layout="wide")

# Model selection in sidebar
st.sidebar.title("Model Settings")
model_provider = st.sidebar.radio("Choose Model Provider:", 
                                 ["Local", "OpenAI"],
                                 index=0)

if model_provider == "Local":
    local_models = ["gemma3:4b", "gemma3:1b", "deepseek-r1:1.5b", "deepseek-r1:7b"]
    # Default to gemma3:4b if no model is selected
    selected_model = st.sidebar.selectbox("Select Local Model:", local_models, index=0)
    
    # Apply the model selection
    set_model_config("local", selected_model)
    st.sidebar.success(f"Using local model: {selected_model}")
    
else:  # OpenAI
    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY", "")
    openai_api_key = st.sidebar.text_input("OpenAI API Key:", value=api_key, type="password")
    
    if openai_api_key:
        # Save API key to environment variable
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # Model selection
        openai_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
        selected_model = st.sidebar.selectbox("Select OpenAI Model:", openai_models, index=0)
        
        # Apply the model selection
        set_model_config("openai", selected_model)
        st.sidebar.success(f"Using OpenAI model: {selected_model}")
    else:
        st.sidebar.warning("Please enter your OpenAI API key to use OpenAI models.")
        # Default to local model if no API key
        set_model_config("local", "gemma3:4b")
        st.sidebar.info("Using default local model: gemma3:4b")

# Update the chains with the selected model
update_chains()

st.title("🧠 ADHD-Friendly AI To-Do List")
st.write("Add your tasks, and I'll organize and break them down into actionable steps!")

# Initialize session state
if 'tasks' not in st.session_state:
    st.session_state.tasks = {}  # {task: {"subtasks": [...], "completed": False}}
if 'subtasks_status' not in st.session_state:
    st.session_state.subtasks_status = {}  # {task: {subtask: completed_status}}
if 'ranked_tasks' not in st.session_state:
    st.session_state.ranked_tasks = []  # List of tasks in ranked order

# Task input form
with st.form(key="task_form"):
    task_input = st.text_area("Enter one or more tasks (one per line)")
    submit_button = st.form_submit_button(label="Add & Organize Tasks")

# Handle new task submission
if submit_button and task_input.strip():
    tasks = [t.strip() for t in task_input.splitlines() if t.strip()]
    
    if tasks:
        with st.spinner("Organizing and breaking down your tasks..."):
            # First, rank the tasks
            ranked_tasks = rank_tasks(tasks)
            st.session_state.ranked_tasks = ranked_tasks
            
            # Then, break down each task
            for task in ranked_tasks:
                if task not in st.session_state.tasks:
                    subtasks_text = get_subtasks(task)
                    subtasks_list = [s.strip() for s in subtasks_text.split('\n') if s.strip()]
                    
                    # Initialize task entry
                    st.session_state.tasks[task] = {
                        "subtasks": subtasks_list,
                        "completed": False
                    }
                    
                    # Initialize subtasks status
                    st.session_state.subtasks_status[task] = {
                        subtask: False for subtask in subtasks_list
                    }
            
            st.success(f"Added and organized {len(tasks)} tasks!")

# Display all tasks and their subtasks
if st.session_state.tasks:
    st.subheader("📋 Your Prioritized Tasks")
    
    # Get the tasks in ranked order, or if not available, use the keys
    display_tasks = st.session_state.ranked_tasks or list(st.session_state.tasks.keys())
    
    for i, task in enumerate(display_tasks):
        if task in st.session_state.tasks:
            # Task checkbox for main task
            task_completed = st.checkbox(
                f"Task {i+1}: {task}", 
                value=st.session_state.tasks[task]["completed"],
                key=f"task_{i}"
            )
            
            # Update task completion status
            if task_completed != st.session_state.tasks[task]["completed"]:
                st.session_state.tasks[task]["completed"] = task_completed
                # If main task is checked, mark all subtasks as completed
                if task_completed:
                    for subtask in st.session_state.subtasks_status[task]:
                        st.session_state.subtasks_status[task][subtask] = True
            
            # Create an expander for each task
            with st.expander("View subtasks", expanded=not task_completed):
                subtasks = st.session_state.tasks[task]["subtasks"]
                
                for j, subtask in enumerate(subtasks):
                    # Subtask checkbox
                    subtask_completed = st.checkbox(
                        subtask,
                        value=st.session_state.subtasks_status[task].get(subtask, False),
                        key=f"subtask_{i}_{j}"
                    )
                    
                    # Update subtask completion status
                    st.session_state.subtasks_status[task][subtask] = subtask_completed
                
                # Check if all subtasks are completed and update main task accordingly
                all_subtasks_completed = all(st.session_state.subtasks_status[task].values())
                if all_subtasks_completed and not st.session_state.tasks[task]["completed"]:
                    st.session_state.tasks[task]["completed"] = True
                    st.rerun()  # Refresh to update the main task checkbox

    # Add options to manage tasks
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Completed Tasks"):
            # Remove tasks that are marked as completed
            completed_tasks = [task for task in st.session_state.tasks if st.session_state.tasks[task]["completed"]]
            for task in completed_tasks:
                del st.session_state.tasks[task]
                if task in st.session_state.subtasks_status:
                    del st.session_state.subtasks_status[task]
                if task in st.session_state.ranked_tasks:
                    st.session_state.ranked_tasks.remove(task)
            st.rerun()
            
    with col2:
        if st.button("Clear All Tasks"):
            # Clear all tasks
            st.session_state.tasks = {}
            st.session_state.subtasks_status = {}
            st.session_state.ranked_tasks = []
            st.rerun()
else:
    st.info("Add some tasks to get started!")

# To-do list management
st.sidebar.title("📅 Daily To-Do Lists")

# Save current list
with st.sidebar.expander("Save Current List"):
    today = datetime.now().strftime("%Y-%m-%d")
    default_name = f"todo_list_{today}"
    save_name = st.text_input("List Name:", value=default_name)
    
    if st.button("Save List"):
        if st.session_state.tasks:  # Only save if there are tasks
            file_path = save_todo_list(
                st.session_state.tasks,
                st.session_state.subtasks_status,
                st.session_state.ranked_tasks,
                save_name
            )
            st.sidebar.success(f"Saved to {os.path.basename(file_path)}")
        else:
            st.sidebar.warning("No tasks to save!")

# Load saved list
with st.sidebar.expander("Load Saved List"):
    saved_lists = get_saved_lists()
    
    if saved_lists:
        selected_list = st.selectbox(
            "Select a saved list:",
            options=saved_lists,
            format_func=lambda x: x.replace("todo_list_", "").replace(".json", "")
        )
        
        if st.button("Load List"):
            result = load_todo_list(selected_list)
            
            if result:
                tasks, subtasks_status, ranked_tasks = result
                # Update session state
                st.session_state.tasks = tasks
                st.session_state.subtasks_status = subtasks_status
                st.session_state.ranked_tasks = ranked_tasks
                st.sidebar.success(f"Loaded list: {selected_list}")
                st.rerun()
            else:
                st.sidebar.error("Failed to load list!")
    else:
        st.sidebar.info("No saved lists found.")

# Create new list
if st.sidebar.button("Start New List"):
    # Confirm before clearing
    if st.session_state.tasks and not st.sidebar.checkbox("Confirm clear current list"):
        st.sidebar.warning("Check to confirm clearing your current list")
    else:
        # Clear the session state
        st.session_state.tasks = {}
        st.session_state.subtasks_status = {}
        st.session_state.ranked_tasks = []
        st.sidebar.success("Started a new list!")
        st.rerun()

# Add a horizontal line to separate model settings from list management
st.sidebar.markdown("---")