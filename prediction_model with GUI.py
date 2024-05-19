import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

def load_and_process_data(file_path):
    # Load the data
    df = pd.read_excel(file_path)
    
    # Rename the column to match the expected column name
    if 'Branch_code' in df.columns:
        df.rename(columns={'Branch_code': 'Branch_\ncode'}, inplace=True)
    
    # Select relevant columns
    df = df[['Inst Code', 'Institution Name', 'Branch_\ncode', 'Rank']]
    
    # Drop rows with missing values
    df.dropna(inplace=True)
    
    return df

def train_model(df):
    # Encode categorical features
    le_inst_code = LabelEncoder()
    le_inst_name = LabelEncoder()
    le_branch_code = LabelEncoder()
    
    df['Inst Code'] = le_inst_code.fit_transform(df['Inst Code'])
    df['Institution Name'] = le_inst_name.fit_transform(df['Institution Name'])
    df['Branch_\ncode'] = le_branch_code.fit_transform(df['Branch_\ncode'])
    
    # Prepare features and target
    X = df[['Rank']]
    y = df[['Inst Code', 'Institution Name', 'Branch_\ncode']]
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save the encoders and model
    joblib.dump(le_inst_code, 'le_inst_code.pkl')
    joblib.dump(le_inst_name, 'le_inst_name.pkl')
    joblib.dump(le_branch_code, 'le_branch_code.pkl')
    joblib.dump(model, 'rank_predictor_model.pkl')
    
    return model, le_inst_code, le_inst_name, le_branch_code

def predict_institution_branch(rank, model, le_inst_code, le_inst_name, le_branch_code):
    # Make a prediction
    prediction = model.predict([[rank]])
    
    # Decode the prediction
    inst_code = le_inst_code.inverse_transform([prediction[0][0]])[0]
    inst_name = le_inst_name.inverse_transform([prediction[0][1]])[0]
    branch_code = le_branch_code.inverse_transform([prediction[0][2]])[0]
    
    return inst_code, inst_name, branch_code

def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
    entry_file_path.delete(0, tk.END)
    entry_file_path.insert(0, file_path)

def predict():
    input_file = entry_file_path.get()
    ranks = [int(rank_entry.get())]
    
    df = load_and_process_data(input_file)
    model, le_inst_code, le_inst_name, le_branch_code = train_model(df)
    
    inst_code, inst_name, branch_code = predict_institution_branch(ranks[0], model, le_inst_code, le_inst_name, le_branch_code)
    messagebox.showinfo("Prediction Result", f"Institution Code: {inst_code}\nInstitution Name: {inst_name}\nBranch Code: {branch_code}")

# Create the main window
root = tk.Tk()
root.title("Rank Predictor")

# File Path Entry
label_file_path = tk.Label(root, text="Select Input File:")
label_file_path.grid(row=0, column=0, padx=5, pady=5, sticky="w")
entry_file_path = tk.Entry(root, width=50)
entry_file_path.grid(row=0, column=1, columnspan=2, padx=5, pady=5, sticky="we")
button_browse = tk.Button(root, text="Browse", command=browse_file)
button_browse.grid(row=0, column=3, padx=5, pady=5)

# Rank Entry
label_rank = tk.Label(root, text="Enter Rank:")
label_rank.grid(row=1, column=0, padx=5, pady=5, sticky="w")
rank_entry = tk.Entry(root)
rank_entry.grid(row=1, column=1, columnspan=2, padx=5, pady=5, sticky="we")

# Predict Button
button_predict = tk.Button(root, text="Predict", command=predict)
button_predict.grid(row=2, column=1, columnspan=2, padx=5, pady=5, sticky="we")

# Start the GUI main loop
root.mainloop()
