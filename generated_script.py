import os
import subprocess

def open_application(app_name):
    apps = {
        "chrome": "chrome" if os.name == "posix" else "C:/Program Files/Google/Chrome/Application/chrome.exe",
        "notepad": "notepad" if os.name == "nt" else "gedit",
        "calculator": "calc" if os.name == "nt" else "gnome-calculator",
    }
    if app_name in apps:
        subprocess.Popen(apps[app_name], shell=True)
        print(f"{app_name} opened successfully!")
    else:
        print(f"Application '{app_name}' not found.")

# Execute the function
open_application("chrome")
