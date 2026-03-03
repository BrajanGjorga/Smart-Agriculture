# How to Run the Smart Agriculture App

Follow these steps to run the irrigation/fan prediction web app locally.

---

## 1. Requirements

- **Python 3** installed
- Required packages: `flask`, `joblib`, `pandas`

Install dependencies if needed:

```bash
pip install flask joblib pandas
```

- The trained model and config must exist in the repo:
  - `notebooks/models/model.pkl`
  - `notebooks/models/config.json`

---

## 2. Open a terminal and go to the app folder

```bash
cd path\to\Smart-Agriculture\app
```

**Example (Windows):**

```powershell
cd "c:\Users\User\OneDrive\Documents\Machine Learning\Smart-Agriculture\app"
```

---

## 3. Start the app

```bash
python app.py
```

You should see something like:

- `Model found : True`
- `[OK] Pipeline loaded successfully.`
- `[OK] Config loaded. ... threshold: 0.8`
- `Running on http://127.0.0.1:5000`

---

## 4. Open the app in your browser

Go to:

**http://127.0.0.1:5000**

Use the web interface for predictions and schedule.

---

## 5. Stop the app

In the same terminal, press **Ctrl+C**.

---

## Quick reference

| Step | Action |
|------|--------|
| 1 | Install: `pip install flask joblib pandas` |
| 2 | `cd Smart-Agriculture\app` |
| 3 | `python app.py` |
| 4 | Open http://127.0.0.1:5000 |
| 5 | Stop with **Ctrl+C** |
