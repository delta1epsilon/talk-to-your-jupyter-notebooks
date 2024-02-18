run-notebooks-rag:
	@echo "Starting ollama serve..."
	@ollama serve >/dev/null 2>&1 &
	@echo "Starting Streamlit app..."
	@streamlit run talk_to_notebooks_app.py
