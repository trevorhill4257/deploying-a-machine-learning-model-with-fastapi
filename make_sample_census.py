"""
make_sample_census.py

Helper to generate a SYNTHETIC census.csv with the same schema as the UCI
Census Income (Adult) dataset so the pipeline can run end-to-end in this
workspace. Replace data/census.csv with the real dataset from the project
starter repo before final submission.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 4000

workclass = ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
             "Local-gov", "State-gov", "Without-pay", "Never-worked"]
education = ["Bachelors", "HS-grad", "11th", "Masters", "9th", "Some-college",
            "Assoc-acdm", "Assoc-voc", "7th-8th", "Doctorate", "Prof-school",
            "5th-6th", "10th", "1st-4th", "Preschool", "12th"]
marital = ["Married-civ-spouse", "Divorced", "Never-married", "Separated",
           "Widowed", "Married-spouse-absent", "Married-AF-spouse"]
occupation = ["Tech-support", "Craft-repair", "Other-service", "Sales",
              "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
              "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
              "Transport-moving", "Priv-house-serv", "Protective-serv",
              "Armed-Forces"]
relationship = ["Wife", "Own-child", "Husband", "Not-in-family",
                "Other-relative", "Unmarried"]
race = ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"]
sex = ["Female", "Male"]
country = ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada",
           "Germany", "India", "Japan", "Mexico", "Philippines", "Cuba"]


def edu_num(e):
    return education.index(e) + 1


rows = []
for _ in range(N):
    e = rng.choice(education)
    rows.append({
        "age": int(rng.integers(17, 90)),
        "workclass": rng.choice(workclass),
        "fnlgt": int(rng.integers(12000, 1500000)),
        "education": e,
        "education-num": edu_num(e),
        "marital-status": rng.choice(marital),
        "occupation": rng.choice(occupation),
        "relationship": rng.choice(relationship),
        "race": rng.choice(race),
        "sex": rng.choice(sex),
        "capital-gain": int(rng.choice([0, 0, 0, rng.integers(0, 99999)])),
        "capital-loss": int(rng.choice([0, 0, 0, rng.integers(0, 4356)])),
        "hours-per-week": int(rng.integers(1, 99)),
        "native-country": rng.choice(country),
    })

df = pd.DataFrame(rows)

# Create a learnable signal so metrics are meaningful but not trivial.
score = (
    (df["age"] / 90)
    + (df["education-num"] / 16)
    + (df["hours-per-week"] / 99)
    + (df["capital-gain"] > 0).astype(float)
    + (df["occupation"].isin(["Exec-managerial", "Prof-specialty"])).astype(float)
)
noise = rng.normal(0, 0.4, size=N)
df["salary"] = np.where(score + noise > 2.2, ">50K", "<=50K")

df.to_csv("data/census.csv", index=False)
print(f"Wrote data/census.csv with {len(df)} rows.")
print(df["salary"].value_counts())
