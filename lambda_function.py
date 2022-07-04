import json
import pickle
import boto3 
import numpy

s3 = boto3.resource('s3')
s3.Bucket('spookydata').download_file('model/simple_QL_saved_water.pkl', '/tmp/simple_QL_saved_water.pkl') 
s3.Bucket('spookydata').download_file('model/simple_QL_saved_light.pkl', '/tmp/simple_QL_saved_light.pkl') 

def read_Q(name):
    with (open(f"/tmp/simple_QL_saved_{name}.pkl", "rb")) as openfile:
        Q_v = pickle.load(openfile)
    return Q_v

def predict(value, type):
    Q_v = read_Q(type)
    print(value)
    b_a = min(Q_v.keys(), key=lambda x:abs(float(x)-value))
    print(Q_v[b_a])
    return max(Q_v[b_a], key=Q_v[b_a].get)
    #return max(Q_v[b_a])
     
def lambda_handler(event, context):
    payload = {
        "water" : "off",
        "light" : "off"
    }
    action_w = predict(event["humidity_soil"], "water")
    action_l = predict(event["luminosity"]/100, "light")
    
    print(action_w, action_l)
    
    if action_w == "20" :
        payload["water"] = "on"
    else :
        payload["water"] = "off"
        
    if action_l == "20" :
        payload["light"] = "on"
    else :
        payload["light"] = "off"

    iot = boto3.client('iot-data')
    response = iot.publish(
        topic='spooky/sub',
        qos=1,
        payload=json.dumps(payload)
    )
    print(response)

    return {
        'statusCode': 200,
        'body': response
    }
