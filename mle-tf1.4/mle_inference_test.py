from googleapiclient import discovery
from oauth2client.client import GoogleCredentials

PROJECT = 'ksalama-gcp-playground'  # change to your project name
MODEL_NAME = 'census_estimator'  # change to your model name
VERSION = 'v1'  # change to your model version


def estimate(project, model_name, version, instances):
    credentials = GoogleCredentials.get_application_default()
    api = discovery.build('ml', 'v1', credentials=credentials,
                          discoveryServiceUrl='https://storage.googleapis.com/cloud-ml/discovery/ml_v1_discovery.json')

    request_data = {'instances': instances}

    model_url = 'projects/{}/models/{}/versions/{}'.format(project, model_name, version)
    response = api.projects().predict(body=request_data, name=model_url).execute()

    return response


##############################################
# WRITE YOUR PREDICTION TEST HERE
##############################################
# list dictionary items of serving with json,
# or list of csv string(s) if serving with csv

instances = [
    # {"CRIM": 0.02729,
    #  "ZN": 0.0,
    #  "INDUS": 7.07,
    #  "CHAS": 0,
    #  "NOX": 0.469,
    #  "RM": 7.185,
    #  "AGE": 61.1,
    #  "DIS": 4.9671,
    #  "RAD": 2,
    #  "TAX": 242,
    #  "PTRATIO": 17.8,
    #  "B": 392.83,
    #  "LSTAT": 4.03},
    #
    # {"CRIM": 0.03729,
    #  "ZN": 0.0,
    #  "INDUS": 7.07,
    #  "CHAS": 1,
    #  "NOX": 0.269,
    #  "RM": 5.185,
    #  "AGE": 81.1,
    #  "DIS": 4.9671,
    #  "RAD": 4,
    #  "TAX": 142,
    #  "PTRATIO": 17.8,
    #  "B": 592.83,
    #  "LSTAT": 3.03}
    {"age": 39, "workclass": "State-gov", "education": "Bachelors", "education_num": 13,
     "marital_status": "Never-married", "occupation": "Adm-clerical", "relationship": "Not-in-family", "race": "White",
     "gender": "Male", "capital_gain": 2174, "capital_loss": 0, "hours_per_week": 40, "native_country": "United-States"}
]

predictions = estimate(instances=instances
                       , project=PROJECT
                       , model_name=MODEL_NAME
                       , version=VERSION)

print(predictions)

#############################################
