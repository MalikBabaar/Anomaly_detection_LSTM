# AIOps LSTM Autoencoder Project

## Setup Instructions

1. **Activate virtual environment**  

   ```bash
   # Windows
   venv\Scripts\activate

   # macOS / Linux
   source venv/bin/activate

2. Install project dependencies

  pip install -r requirements.txt

3. Install Dashboard dependencies

  cd Dashboard

  pip install -r requirements.txt

4. Install Trainer dependencies

  cd malik/malik/trainer

  pip install -r requirements.txt


1. Run APIs

Terminal 1 – Main app API:

uvicorn app:app --reload --host 0.0.0.0 --port 5000

Terminal 2 – Trainer API:

uvicorn malik.malik.trainer.api:app --reload --host 0.0.0.0 --port 9000

2. Run Dashboard

Terminal 3 – Streamlit Dashboard:

cd Dashboard

streamlit run app.py

3. Run MLflow UI

Terminal 4 – MLflow dashboard:

mlflow ui --backend-store-uri "file:///C:/aiops_project_LSTM_Autoencoder/mlruns" --port 5001

Running Training from CLI:

You can train the LSTM model directly using the command line:

python -m malik.malik.trainer.train_lstm --inputs "malik/malik/new_data.csv" --outdir foldername --epochs <num_epochs>


