from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pandas as pd
from pydantic import BaseModel
import random
import plotly.express as px

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# Charger le dataset
df = pd.read_csv("updated_cancerSEER.csv")

# Préparer la colonne "Age" : Nettoyage et regroupement
def process_age_column(df):
    # Convertir "Age" en numérique, en remplaçant "90+ years" par 90
    df['Age'] = df['Age'].str.replace(" years", "").replace("90+", "90").astype(int)
    # Ajouter une colonne "Age Group"
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    labels = ["0–10", "11–20", "21–30", "31–40", "41–50", "51–60", "61–70", "71–80", "81–90", "90+"]
    df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    return df

df = process_age_column(df)

# Configuration des templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    # Données pour le graphique de répartition par groupe d'âge
    age_group_data = df.groupby(["Age Group", "Sex"]).size().reset_index(name="Count")

    # Données pour le graphique bar chart (répartition par sexe et année)
    bar_data = df.groupby(["Yeardiag", "Type", "Sex"]).size().reset_index(name="Count")


    # Données pour le line chart (évolution des cas par cancer avec split homme/femme)
    line_chart_data = df.groupby(["Yeardiag", "Type", "Sex"]).size().reset_index(name="Count")

    # Données pour le graphique par type de cancer et âge
    cancer_age_data = df.groupby(["Type", "Age Group"]).size().reset_index(name="Count")

    ethnicity_totals = df.groupby("Race")["Type"].count()

    # Calculate percentage for each combination of Race and Type
    ethnicity_cancer_data = (
        df.groupby(["Race", "Type"])
        .size()
        .reset_index(name="Count")
    )

    # Normalize by total cases per ethnicity
    ethnicity_cancer_data["Percentage"] = (ethnicity_cancer_data["Count"] / ethnicity_cancer_data["Race"].map(ethnicity_totals)) * 100

    # Ensure Age is a string before applying .str operations
    df["Age"] = df["Age"].astype(str)

    # Replace " years" and handle "90+" appropriately, then convert to numeric
    df["Age"] = df["Age"].str.replace(" years", "", regex=False).replace("90+", "90")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")  # Convert to numeric and handle invalid values
    # Filter out rows where the cause of death is "Alive"
    box_plot_data = df[df["COD"] != "Alive"][["Age", "COD"]]
    # Group by Type and COD to calculate counts
    vital_status_data = df.groupby(["Type", "COD"]).size().reset_index(name="Count")

    # Optionally, calculate percentages for visualization (if needed)
    type_totals = vital_status_data.groupby("Type")["Count"].sum()
    vital_status_data["Percentage"] = (vital_status_data["Count"] / vital_status_data["Type"].map(type_totals)) * 100

    # Group by Race and COD (vital status) to calculate counts
    vital_status_ethnicity_data = df.groupby(["Race", "COD"]).size().reset_index(name="Count")

    # Optionally, calculate percentages for each ethnicity
    ethnicity_totals = vital_status_ethnicity_data.groupby("Race")["Count"].sum()
    vital_status_ethnicity_data["Percentage"] = (vital_status_ethnicity_data["Count"] /vital_status_ethnicity_data["Race"].map(ethnicity_totals)) * 100

    # Group by year of diagnosis and cancer type, then calculate average age
    line_graph_data = df.groupby(["Yeardiag", "Type"])["Age"].mean().reset_index()

    # Rename columns for clarity (optional)
    line_graph_data.columns = ["Year", "CancerType", "AverageAge"]

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "age_group_data": age_group_data.to_dict(orient="records"),
            "line_chart_data": line_chart_data.to_dict(orient="records"),
            "cancer_age_data": cancer_age_data.to_dict(orient="records"),
            "ethnicity_cancer_data": ethnicity_cancer_data.to_dict(orient="records"),
            "box_plot_data": box_plot_data.to_dict(orient="records"),
            "vital_status_data": vital_status_data.to_dict(orient="records"),
            "vital_status_ethnicity_data": vital_status_ethnicity_data.to_dict(orient="records"),
            "line_graph_data": line_graph_data.to_dict(orient="records"),
        },
    )

@app.get("/context", response_class=HTMLResponse)
def context(request: Request):
    return templates.TemplateResponse("context.html", {"request": request})

# Define the expected input format
class PatientData(BaseModel):
    Age: int
    Sex: str  # "Male" or "Female"
    CancerType: str
    Race: str

# Dummy prediction endpoint (Replace with actual model later)
@app.post("/predict/")
def predict_lifestatus(patient: PatientData):
    # Placeholder prediction (Random probabilities)
    prob_alive = round(random.uniform(50, 95), 2)  # Random between 50% and 95%
    prob_dead = round(100 - prob_alive, 2)  # Complementary probability

    return {
        "Probability_Alive": prob_alive,
        "Probability_Dead": prob_dead
    }

# ML Page route
@app.get("/ml-model")
def ml_page(request: Request):
    # Dummy dropdown values (Replace with actual categories when model is ready)
    cancer_types = ["Lung Cancer", "Breast Cancer", "Prostate Cancer", "Colon Cancer", "Skin Cancer"]
    races = ["White", "Black", "Asian", "Hispanic", "Native American", "Other"]

    return templates.TemplateResponse("ml_model.html", {
        "request": request,
        "cancer_types": cancer_types,
        "races": races
    })