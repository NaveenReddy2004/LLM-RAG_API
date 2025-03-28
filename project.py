import os
import psutil
import subprocess
import faiss
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

import logging

# Configure logging
logging.basicConfig(
    filename="execution_logs.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def open_application(app_name):
    """Open an application (e.g., Chrome, Notepad, Calculator)."""
    apps = {
        "chrome": "chrome" if os.name == "posix" else "C:/Program Files/Google/Chrome/Application/chrome.exe",
        "notepad": "notepad" if os.name == "nt" else "gedit",
        "calculator": "calc" if os.name == "nt" else "gnome-calculator",
    }
    if app_name in apps:
        subprocess.Popen(apps[app_name], shell=True)
        return f"{app_name} opened successfully!"
    return f"Application '{app_name}' not found."

def get_system_usage():
    """Retrieve CPU and RAM usage."""
    return {"cpu_usage": psutil.cpu_percent(interval=1), "ram_usage": psutil.virtual_memory().percent}

def run_command(command):
    """Execute a shell command."""
    try:
        result = subprocess.run(command, shell=True, text=True, capture_output=True)
        return result.stdout if result.returncode == 0 else result.stderr
    except Exception as e:
        return str(e)

# Function metadata for retrieval
FUNCTION_METADATA = {
    "open_application": "Opens an application like Chrome, Notepad, or Calculator.",
    "get_system_usage": "Gets CPU and RAM usage statistics.",
    "run_command": "Executes a shell command and returns the output.",
}


# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert function descriptions to embeddings
function_names = list(FUNCTION_METADATA.keys())
descriptions = list(FUNCTION_METADATA.values())
description_embeddings = np.array(embedding_model.encode(descriptions)).astype('float32')

# Store embeddings in FAISS index
dimension = description_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(description_embeddings)
function_mapping = {i: name for i, name in enumerate(function_names)}

def retrieve_function(user_query):
    """Retrieve the best-matching function using FAISS."""
    query_embedding = np.array(embedding_model.encode([user_query])).astype('float32')
    _, indices = index.search(query_embedding, 1)
    best_match = function_mapping[indices[0][0]]
    return best_match


import time

def generate_and_execute_code(function_name, **kwargs):
    """Dynamically generate and execute function calls with logging."""
    functions = {
        "open_application": open_application,
        "get_system_usage": get_system_usage,
        "run_command": run_command,
    }

    if function_name in functions:
        start_time = time.time()
        try:
            result = functions[function_name](**kwargs)
            execution_time = round(time.time() - start_time, 4)

            # Log execution details
            logging.info(f"Function: {function_name}, Parameters: {kwargs}, Result: {result}, Time: {execution_time}s")

            return {"function": function_name, "output": result, "execution_time": execution_time}
        except Exception as e:
            logging.error(f"Error executing function {function_name}: {str(e)}")
            return {"error": str(e)}
    else:
        logging.warning(f"Function '{function_name}' not found.")
        return {"error": f"Function '{function_name}' not found."}


import json


def load_custom_functions():
    """Load custom user-defined functions from a JSON file."""
    try:
        with open("custom_functions.json", "r") as file:
            custom_funcs = json.load(file)
            print("Loaded Custom Functions:", custom_funcs)  # Debugging Line
            return custom_funcs
    except FileNotFoundError:
        print("Error: custom_functions.json not found.")
        return {}



def execute_custom_function(function_name, **kwargs):
    """Dynamically execute a user-defined function."""
    custom_functions = load_custom_functions()

    if function_name in custom_functions:
        function_code = custom_functions[function_name]["code"]
        exec_globals = {}
        exec(function_code, exec_globals)  # Execute the function code in a safe environment

        if function_name in exec_globals:
            function_ref = exec_globals[function_name]
            return function_ref(**kwargs)
        else:
            return {"error": f"Function '{function_name}' is not defined properly."}
    else:
        return {"error": f"Custom function '{function_name}' not found."}


def retrieve_function(user_query):
    """Retrieve function name based on user query, including custom functions."""
    FUNCTION_METADATA = {
        "open notepad": "open_application",
        "open calculator": "open_application",
        "check cpu usage": "get_system_usage",
        "run shell command": "run_command"
    }

    # Check predefined functions first
    for key, value in FUNCTION_METADATA.items():
        if key in user_query.lower():
            return value

    # Check custom functions
    custom_functions = load_custom_functions()
    for function_name in custom_functions.keys():
        if function_name.lower() in user_query.lower():
            print(f"Matched Custom Function: {function_name}")  # Debugging Line
            return function_name

    return None  # No matching function found



app = FastAPI()

class FunctionRequest(BaseModel):
    user_query: str
    parameters: dict = {}

@app.post("/execute_function/")
async def execute_function(request: FunctionRequest):
    function_name = retrieve_function(request.user_query)

    if function_name in FUNCTION_METADATA:
        result = generate_and_execute_code(function_name, **request.parameters)
    elif function_name:
        result = execute_custom_function(function_name, **request.parameters)
    else:
        result = {"error": "No matching function found."}

    return {"query": request.user_query, "matched_function": function_name, "result": result}




# Run API Server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
