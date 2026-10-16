import numpy as np, pandas as pd

def engineer(df):
    d = df.copy()
    d["IncomePerExpYear"] = d.MonthlyIncome / (d.TotalWorkingYears + 1)
    d["PromotionDelayRatio"] = d.YearsSinceLastPromotion / (d.YearsAtCompany + 1)
    d["LongNoPromotion"] = (d.YearsSinceLastPromotion >= 4).astype(int)
    d["EngagementScore"] = d[["JobSatisfaction","EnvironmentSatisfaction","RelationshipSatisfaction",
                              "JobInvolvement","WorkLifeBalance"]].mean(axis=1)
    d["WorkloadStress"] = ((d.OverTime == "Yes") & ((d.BusinessTravel == "Travel_Frequently") |
                           (d.WorkLifeBalance <= 2))).astype(int)
    d["CompaniesPerYear"] = d.NumCompaniesWorked / (d.TotalWorkingYears + 1)
    return d

CAT = ["BusinessTravel","Department","EducationField","Gender","JobRole","MaritalStatus","OverTime"]
