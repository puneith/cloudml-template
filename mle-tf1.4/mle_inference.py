from googleapiclient import discovery
from oauth2client.client import GoogleCredentials

PROJECT = 'your-project-name'  # change to your project name
MODEL_NAME = 'your_model_name'  # change to your model name
VERSION = 'your.model.version'  # change to your model version


def estimate(project, model_name, version, instances):

    credentials = GoogleCredentials.get_application_default()
    api = discovery.build('ml', 'v1', credentials=credentials,
                          discoveryServiceUrl='https://storage.googleapis.com/cloud-ml/discovery/ml_v1_discovery.json')

    request_data = {'instances': instances}

    model_url = 'projects/{}/models/{}/versions/{}'.format(project, model_name, version)
    response = api.projects().predict(body=request_data, name=model_url).execute()

    return response['predictions']


##############################################
# WRITE YOUR MODEL PREDICTION TEST HERE
##############################################

# list dictionary items of serving with json,
# or list of csv string(s) if serving with csv
instances = []

predictions = estimate(instances=instances
                       , project=PROJECT
                       , model_name=MODEL_NAME
                       , version=VERSION)

print(predictions)

#############################################
