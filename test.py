import requests

url = "https://asia-southeast2-apiecocycle.cloudfunctions.net/img_classifier_model"

data = {
    "url": "https://i.pinimg.com/564x/49/a9/3a/49a93a52cc95159317d433be9261b664.jpg"
}

result = requests.post(url, json=data).json()
print(result)
