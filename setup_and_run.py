#!/usr/bin/env python3
"""
Setup and run script for Bangla dataset evaluation
This script handles environment setup and runs the evaluation
"""

import subprocess
import sys
import os
import platform

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_system():
    """Check system information"""
    print("🖥️ System Information:")
    print(f"  OS: {platform.system()} {platform.release()}")
    print(f"  Architecture: {platform.machine()}")
    print(f"  Python: {platform.python_version()}")

def find_dataset_files():
    """Find dataset files in current directory and parent directories"""
    dataset_files = []
    
    # First check for the bangla dataset folder structure
    bangla_dataset_path = r"d:\bangla dataset"
    if os.path.exists(bangla_dataset_path):
        data_path = os.path.join(bangla_dataset_path, "data", "data.json")
        data_v2_path = os.path.join(bangla_dataset_path, "data_v2", "data_v2.json")
        
        if os.path.exists(data_path):
            dataset_files.append(("Bangla Dataset Folder", bangla_dataset_path))
        
        return dataset_files
    
    # Fallback to checking current directory
    for file in os.listdir('.'):
        if file.endswith('.csv') or file.endswith('.json'):
            if 'dataset' in file.lower() or 'data' in file.lower() or 'bangla' in file.lower():
                dataset_files.append(file)
    return dataset_files

def main():
    print("🚀 Bangla Dataset Evaluation Setup")
    print("=" * 50)
    
    # Check system
    check_system()
    print()
    
    # Check Python version
    if not check_python_version():
        return
    
    # Install requirements
    if not install_requirements():
        return
    
    # Check for dataset files
    print("\n📁 Looking for dataset files...")
    dataset_files = find_dataset_files()
    
    if dataset_files:
        print("Found dataset files/folders:")
        for i, item in enumerate(dataset_files, 1):
            if isinstance(item, tuple):
                name, path = item
                print(f"  {i}. {name}: {path}")
                # Show folder contents
                data_path = os.path.join(path, "data", "data.json")
                data_v2_path = os.path.join(path, "data_v2", "data_v2.json")
                if os.path.exists(data_path):
                    size_gb = os.path.getsize(data_path) / (1024**3)
                    print(f"     - data.json: {size_gb:.1f} GB")
                if os.path.exists(data_v2_path):
                    size_gb = os.path.getsize(data_v2_path) / (1024**3)
                    print(f"     - data_v2.json: {size_gb:.1f} GB")
            else:
                print(f"  {i}. {item}")
        
        # Auto-select dataset
        selected_dataset = dataset_files[0]
        if isinstance(selected_dataset, tuple):
            print(f"Using dataset folder: {selected_dataset[1]}")
            cmd = [sys.executable, "main_evaluation.py", "--dataset", "FOLDER:" + selected_dataset[1]]
        else:
            print(f"Using dataset: {selected_dataset}")
            cmd = [sys.executable, "main_evaluation.py", "--dataset", selected_dataset]
        
        # Run evaluation
        print(f"\n🔥 Starting evaluation...")
        print("This may take a while depending on your GPU and dataset size...")
        print("The system will load 20,000 samples from your dataset.")
        
        subprocess.run(cmd)
        
    else:
        print("❌ No dataset files found!")
        print("\nPlease ensure your dataset is available at:")
        print("  - d:\\bangla dataset\\data\\data.json")  
        print("  - d:\\bangla dataset\\data_v2\\data_v2.json")
        print("\nOr add dataset files to this directory.")

if __name__ == "__main__":
    main()