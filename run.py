import os
import sys
import subprocess
import time
import webbrowser

def install_dependencies():
    print("Checking and installing dependencies...")
    try:
        # Run pip install in the current python environment
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependencies verified and installed successfully!")
    except Exception as e:
        print(f"Error installing dependencies: {e}")
        print("Please run: pip install -r requirements.txt manually.")

def main():
    # Make sure dependencies are installed
    install_dependencies()
    
    print("\nStarting the Predictive Analytics Backend Server...")
    print("Serving frontend at http://127.0.0.1:8000")
    
    # Open browser after a brief delay
    def open_browser():
        time.sleep(2)
        webbrowser.open("http://127.0.0.1:8000")
        
    import threading
    threading.Thread(target=open_browser, daemon=True).start()

    # Start FastAPI server via uvicorn
    try:
        import uvicorn
        uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
    except KeyboardInterrupt:
        print("\nStopping server.")
    except Exception as e:
        print(f"Failed to start uvicorn server: {e}")
        print("Make sure you are in the workspace root directory and run: uvicorn backend.main:app --reload")

if __name__ == "__main__":
    main()
