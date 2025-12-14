import streamlit as st
import os
import pandas as pd
import pickle
from sklearn.preprocessing import  OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
#models
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor

#Preprocessing
def create_preprocessor(df, target_col):
    # ... (Feature identification code) ...
    features=[col for col in df.columns if col != target_col]
    numerical_features=df[features].select_dtypes(include=['int64','float64']).columns
    categorical_features=df[features].select_dtypes(include=['object','category']).columns
    numerical_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ],
        remainder='passthrough'
    )
    return preprocessor

#Models Dictionary
classificationModels={
    'Logistic Regression' : LogisticRegression(max_iter=1000,random_state=42),
    'K-Nearest Neighbous' : KNeighborsClassifier(),
    'Random Forest' : RandomForestClassifier(random_state=42),
    'Support Vector Machine' : SVC(random_state=42)
}

regressionModels={
    'Linear Regression' : LinearRegression(),
    'Decision Tree' : DecisionTreeRegressor(random_state=42),
    'Gradient Booster' : GradientBoostingRegressor(random_state=42)
}

#Choosing model

def run_automl(df,target_col,task_type):
    X=df.drop(columns=[target_col])
    y=df[target_col]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    preprocessor=create_preprocessor(df,target_col)

    result=[]
    if task_type=='Classification':
        modelRun=classificationModels
        metrics='Accuracy'
    else:
        modelRun=regressionModels
        metrics='R2 Score'
    for name,model in modelRun.items():
        fullPipeline=Pipeline([
            ('preprocessor',preprocessor),
            ('model',model)
        ])

        fullPipeline.fit(X_train,y_train)
        y_pred=fullPipeline.predict(X_test)

        if task_type=='Classification':
            score=accuracy_score(y_test,y_pred)
        else:
            score=r2_score(y_test,y_pred)
        result.append({'Model':name,metrics:score})
    board=pd.DataFrame(result).sort_values(by=metrics,ascending=False)
    bestModelName=board.iloc[0]['Model']
    bestModel=Pipeline([('preprocessor',preprocessor),('model',modelRun[bestModelName])])
    bestModel.fit(X,y)
    return bestModel,board

#Creating streamlit UI

@st.cache_data
def load_data(file):
    # Try the default UTF-8 first
    try:
        data = pd.read_csv(file)
        return data
    except UnicodeDecodeError:
        # If UTF-8 fails, try common encodings
        try:
            # Most common alternative for non-English datasets
            data = pd.read_csv(file, encoding='Windows-1252')
            return data
        except UnicodeDecodeError:
            # Another common alternative
            data = pd.read_csv(file, encoding='latin1')
            return data
        except Exception as e:
            # If all else fails, raise a helpful error message
            st.error(f"Failed to load file with common encodings. Please check the file's encoding. Error: {e}")
            return pd.DataFrame() # Return empty data frame to prevent crash

def main():
    st.set_page_config(
        page_title='PurnaAutoML',
        page_icon='assets\logo.jpg',
        layout='centered'
    )
    st.title("AutoML Engine")
    st.write("Custom pipeline for automated training and comparison of fundamental ML algorithms.")

    with st.sidebar:
        st.header("1. Upload Data")
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
        
        st.header("2. Select Task")
        task_type = st.radio("Is this Classification or Regression?", ["Classification", "Regression"])

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
        st.subheader("Data Preview")
        st.dataframe(df.head())

        target_col = st.selectbox("Select the Target Column", df.columns)

        if st.button("Train All Models"):
            with st.spinner(f"Running custom {task_type} pipeline..."):
                
                # --- CALL THE NEW SCALING/SKLEARN FUNCTION ---
                best_model_pipeline, leaderboard = run_automl(df, target_col, task_type)
                
                # Display Results
                st.success("Training Complete!")
                
                st.subheader("Model Performance Leaderboard")
                
                # Get the name of the main metric
                metric_name = 'Accuracy' if task_type == 'Classification' else 'R2 Score'
                
                # Display the leaderboard table
                st.dataframe(leaderboard.style.highlight_max(subset=[metric_name], axis=0))

                # Display the best model
                st.subheader("Best Model Found:")
                st.write(f"The best model is **{leaderboard.iloc[0]['Model']}** with a {metric_name} of **{leaderboard.iloc[0][metric_name]:.4f}**.")

               

                st.markdown("---")
                st.subheader("Download Best Model")

                # Create the filename
                metric_name = 'Accuracy' if task_type == 'Classification' else 'R2 Score'
                best_model_name = leaderboard.iloc[0]['Model']
                
                # We need a function to create the download file content
                def get_model_download_link(model_pipeline, model_name, metric_score):
                    # Serialize the entire pipeline object into a byte stream
                    serialized_model = pickle.dumps(model_pipeline)
                    
                    # Construct a descriptive filename
                    filename = f"{model_name}.pkl"
                    
                    return serialized_model, filename

                current_metric_score = leaderboard.iloc[0][metric_name]
                download_data, download_filename = get_model_download_link(
                    best_model_pipeline, 
                    best_model_name, 
                    current_metric_score
                )
                
                # Creating the Download Button
                st.download_button(
                    label="Download Model(.pkl)",
                    data=download_data,
                    file_name=download_filename,
                    mime="application/octet-stream"
                )
                
                st.caption("The downloaded file contains the entire Scikit-learn Pipeline (preprocessing + model).")
                st.caption("You can load it using: `pickle.loads(file_content)`")



if __name__ == "__main__":
    main()