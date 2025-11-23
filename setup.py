"""
Setup script to initialize the project structure.
"""

import os
from pathlib import Path

def setup_project():
    """Create necessary directories and files."""
    directories = ['data', 'models', 'logs']
    
    print("Setting up project structure...")
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created/verified directory: {directory}/")
    
    # Create .gitkeep files to ensure directories are tracked
    for directory in directories:
        gitkeep = Path(directory) / '.gitkeep'
        if not gitkeep.exists():
            gitkeep.touch()
    
    print("\n✅ Project setup complete!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. (Optional) Migrate legacy data: python migrate_legacy_data.py")
    print("3. Run full pipeline: python main.py --mode full")
    print("4. Launch dashboard: streamlit run dashboard.py")


if __name__ == "__main__":
    setup_project()

