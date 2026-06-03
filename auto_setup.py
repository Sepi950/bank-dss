import os, subprocess, sys

def run_setup_if_needed():
    """Jalankan setup_models.py jika model belum ada."""
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'lr_model.pkl')
    if not os.path.exists(model_path):
        print("Model belum ada, menjalankan setup...")
        setup_path = os.path.join(os.path.dirname(__file__), 'setup_models.py')
        subprocess.run([sys.executable, setup_path], check=True)
        print("Setup selesai!")
