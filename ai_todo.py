import json
import os
from datetime import datetime
from langchain.chat_models import ChatOllama
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from dotenv import load_dotenv

# Model selection function
def get_llm(provider="local", model_name=None, temperature=0.2):
    """
    Returns the appropriate language model based on provider.
    
    Args:
        provider (str): Either "local" or "openai"
        model_name (str): Model name for the selected provider
        temperature (float): Temperature setting for the model
        
    Returns:
        LLM instance
    """
    if provider.lower() == "openai":
        # Check if API key is set
        load_dotenv()  # Load environment variables from .env file
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
            
        # Default to gpt-3.5-turbo if no model specified
        openai_model = model_name or "gpt-3.5-turbo"
        return ChatOpenAI(model=openai_model, temperature=temperature)
    else:
        # Default to gemma3:4b if no model specified
        local_model = model_name or "gemma3:4b"
        return ChatOllama(model=local_model, temperature=temperature)

# Global variable to store the current model provider and name
# Default to local model
CURRENT_PROVIDER = "local"
CURRENT_MODEL = "gemma3:4b"

# Function to set the model configuration
def set_model_config(provider="local", model_name=None):
    """
    Sets the model configuration to use.
    
    Args:
        provider (str): Either "local" or "openai"
        model_name (str): Model name for the selected provider
    """
    global CURRENT_PROVIDER, CURRENT_MODEL
    CURRENT_PROVIDER = provider.lower()
    
    if CURRENT_PROVIDER == "openai" and not model_name:
        CURRENT_MODEL = "gpt-3.5-turbo"
    elif CURRENT_PROVIDER == "local" and not model_name:
        CURRENT_MODEL = "gemma3:4b"
    else:
        CURRENT_MODEL = model_name

    return get_llm(CURRENT_PROVIDER, CURRENT_MODEL)

# Initialize the LLM with default settings
llm = get_llm(CURRENT_PROVIDER, CURRENT_MODEL)

# Define the prompts with clearer instructions
subtask_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a productivity assistant specialized in helping people with ADHD. 
    Break down the given task into 3-5 clear, actionable subtasks that are easy to follow.
    Make each step concrete, specific, and achievable.
    Use simple language and include visual cues or time estimates when helpful.
    Number each step clearly (1., 2., etc.).
    IMPORTANT: Return ONLY the numbered list without ANY introduction, thinking process, or conclusion."""),
    ("human", "{task}")
])

ranking_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a productivity assistant specialized in helping people with ADHD prioritize tasks.
    Your job is to analyze and rank the following tasks in order of importance, considering urgency, impact, and difficulty.
    IMPORTANT: Return ONLY the ranked tasks as a numbered list (1., 2., etc.), without ANY explanation, thinking process, or commentary.
    Each task should appear on a new line in priority order (most important task first)."""),
    ("human", "Please rank these tasks by importance:\n{tasks}")
])

# Define chains - these will be updated when the model is changed
subtask_chain = subtask_prompt | llm | StrOutputParser()
ranking_chain = ranking_prompt | llm | StrOutputParser()

# Function to update chains when model changes
def update_chains():
    """Updates the chains to use the current model configuration."""
    global subtask_chain, ranking_chain, llm
    
    # Get the current LLM based on global settings
    llm = get_llm(CURRENT_PROVIDER, CURRENT_MODEL)
    
    # Redefine the chains with the new LLM
    subtask_chain = subtask_prompt | llm | StrOutputParser()
    ranking_chain = ranking_prompt | llm | StrOutputParser()

def get_subtasks(task):
    """Break down a task into ADHD-friendly actionable subtasks."""
    response = subtask_chain.invoke({"task": task})
    
    # Clean up the response to remove any thinking text
    lines = [line.strip() for line in response.split('\n')]
    # Keep only lines that start with numbers or have bullet points
    clean_lines = [line for line in lines if (line and (
        line[0].isdigit() or 
        line.startswith('•') or 
        line.startswith('-') or
        line.startswith('*')
    ))]
    
    # If no structured lines found, return the original response
    if not clean_lines:
        return response
    
    # Limit to 5 subtasks maximum to avoid overwhelming the user
    clean_lines = clean_lines[:5]
    
    return '\n'.join(clean_lines)

def rank_tasks(task_list):
    """
    Rank a list of tasks by importance using the current LLM.
    
    Args:
        task_list (list): List of task strings to rank
    
    Returns:
        list: The same tasks in ranked order (most important first)
    """
    if not task_list:
        return []
    
    if len(task_list) <= 1:
        return task_list
    
    try:
        # Format the tasks for the prompt
        tasks_text = "\n".join([f"{i+1}. {task}" for i, task in enumerate(task_list)])
        
        # Use a more direct prompt to ensure proper ranking
        prompt = f"""Rank these tasks in order of priority (most important first):
        
{tasks_text}

Return ONLY the task numbers in priority order, separated by commas. Example: 3,1,2,4"""
        
        # Get the ranked order directly
        response = llm.invoke(prompt).content.strip()
        
        # Extract numbers from the response
        import re
        numbers = [int(num) for num in re.findall(r'\d+', response)]
        
        # Map numbers back to tasks (adjusting for 0-based indexing)
        ranked_tasks = []
        for num in numbers:
            if 1 <= num <= len(task_list):  # Ensure index is valid
                task = task_list[num-1]
                if task not in ranked_tasks:  # Avoid duplicates
                    ranked_tasks.append(task)
        
        # Add any missing tasks at the end
        for task in task_list:
            if task not in ranked_tasks:
                ranked_tasks.append(task)
        
        return ranked_tasks
        
    except Exception as e:
        print(f"Error ranking tasks: {e}")
        # If anything goes wrong, return the original list
        return task_list

def save_todo_list(tasks, subtasks_status, ranked_tasks, filename=None):
    """
    Save the current to-do list to a file.
    
    Args:
        tasks (dict): The tasks dictionary
        subtasks_status (dict): The subtasks status dictionary
        ranked_tasks (list): The ranked task list
        filename (str, optional): Custom filename. If None, uses today's date.
    
    Returns:
        str: The path to the saved file
    """
    # Create a data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_lists")
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate filename based on date if not provided
    if not filename:
        today = datetime.now().strftime("%Y-%m-%d")
        filename = f"todo_list_{today}.json"
    elif not filename.endswith('.json'):
        filename = f"{filename}.json"
    
    file_path = os.path.join(data_dir, filename)
    
    # Prepare data to save
    data = {
        "tasks": tasks,
        "subtasks_status": subtasks_status,
        "ranked_tasks": ranked_tasks,
        "saved_at": datetime.now().isoformat()
    }
    
    # Save to file
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    return file_path

def load_todo_list(filename):
    """
    Load a to-do list from a file.
    
    Args:
        filename (str): The filename to load
    
    Returns:
        tuple: (tasks, subtasks_status, ranked_tasks) or None if file not found
    """
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_lists")
    file_path = os.path.join(data_dir, filename)
    
    if not os.path.exists(file_path):
        return None
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return data.get("tasks", {}), data.get("subtasks_status", {}), data.get("ranked_tasks", [])
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def get_saved_lists():
    """
    Get a list of all saved to-do lists.
    
    Returns:
        list: List of filenames sorted by date (newest first)
    """
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_lists")
    os.makedirs(data_dir, exist_ok=True)
    
    files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    # Sort by modification time (newest first)
    files.sort(key=lambda x: os.path.getmtime(os.path.join(data_dir, x)), reverse=True)
    return files

def get_task_suggestions(completed_tasks=None, incomplete_tasks=None):
    """
    Generate task suggestions based on task history
    
    Args:
        completed_tasks (list): List of recently completed tasks
        incomplete_tasks (list): List of incomplete tasks
    
    Returns:
        list: Suggested tasks to add
    """
    if not completed_tasks:
        completed_tasks = []
    if not incomplete_tasks:
        incomplete_tasks = []
    
    # Prepare the context for the LLM
    context = ""
    
    if completed_tasks:
        context += "Recently completed tasks:\n"
        context += "\n".join([f"- {task}" for task in completed_tasks[:10]])
        context += "\n\n"
    
    if incomplete_tasks:
        context += "Tasks still in progress:\n"
        context += "\n".join([f"- {task}" for task in incomplete_tasks[:10]])
        context += "\n\n"
    
    # If we don't have any context, provide some defaults
    if not context:
        context = "No previous task history available."
    
    # Create the prompt
    prompt = f"""Based on the following task history, suggest 3-5 new tasks that would be helpful to add to today's to-do list.
Focus on suggesting practical, actionable tasks. Include a mix of:
- Important tasks that might have been overlooked
- Next logical steps if there are incomplete tasks
- Self-care or personal development tasks
- Any routine tasks that might be missing

{context}

Return ONLY a list of suggested tasks, one per line. No explanations or additional text."""

    try:
        # Get suggestions from LLM
        response = llm.invoke(prompt).content.strip()
        
        # Extract task suggestions (one per line)
        suggestions = [line.strip() for line in response.split('\n') if line.strip()]
        
        # Remove any bullet points or numbering
        suggestions = [
            s[2:].strip() if s.startswith('- ') else 
            s[2:].strip() if s.startswith('* ') else
            s[2:].strip() if (s[0].isdigit() and s[1] in ['.', ')']) else s 
            for s in suggestions
        ]
        
        # Return up to 5 suggestions
        return suggestions[:5]
    
    except Exception as e:
        print(f"Error generating task suggestions: {e}")
        # Fallback suggestions if the LLM fails
        fallbacks = [
            "Review and plan weekly goals",
            "Schedule time for self-care",
            "Clean up and organize workspace",
            "Follow up on pending emails or messages",
            "Take a short walk for focus and mental clarity"
        ]
        import random
        return random.sample(fallbacks, min(3, len(fallbacks)))