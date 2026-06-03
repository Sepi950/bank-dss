import os, subprocess, sys, traceback

def run_setup_if_needed():
    BASE = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE, 'models', 'lr_model.pkl')

    if not os.path.exists(model_path):
        import streamlit as st
        with st.spinner("Mempersiapkan model pertama kali... (1-2 menit)"):
            os.makedirs(os.path.join(BASE, 'data'), exist_ok=True)
            os.makedirs(os.path.join(BASE, 'models'), exist_ok=True)
            setup_path = os.path.join(BASE, 'setup_models.py')
            result = subprocess.run(
                [sys.executable, setup_path],
                capture_output=True, text=True, check=False
            )
            if result.returncode != 0:
                st.error("Setup gagal. Detail error:")
                st.code(result.stderr[-3000:])
                st.stop()
            else:
                st.rerun()
