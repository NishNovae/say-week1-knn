# src/say_week1_knn/main.py

import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
import pickle

data_list = []
kn = KNeighborsClassifier()
#kn.fit([30, 600], [1])

def yes_or_no(prompt):
    while True:
        yn = input(prompt + " (y/n)")
        if yn == "y" or yn == "Y":
            return True
        elif yn == "n" or yn == "N":
            return False
        else:
            print("Wrong format. Please input either 'y' or 'n'. Retrying...")
            continue

def grab_data_path(data="testdata.csv"):
    this_path = os.path.abspath(__file__)
    data_path = os.path.dirname(this_path) + "/data/" + data
    return data_path

def get_model_path(model="model.pkl"):
    this_path = os.path.abspath(__file__)
    model_path = os.path.dirname(this_path) + "/data/" + model
    return model_path

def parse_data(data_path):
    global input_val, output_val 
    with open(data_path, "r") as data:
        for line in data:
            data_list.append(line.strip().split(","))    
    print(f"[INFO] data collected from: {data_path}")
    return

def bulk_train():
    for length, width, answer in data_list:
        prediction = predict(length, width)
        if prediction == answer:
            update()
            print("Correct!")
        else:
            update_wrong()
            print("Wrong!")

def prompt():
    global input_val, output_val
    print("User prompt activated! Please input the length and width.")
    while True:
        try:
            length = input("length: ")
            length = float(length)

            width = input("weight: ")
            width = float(width)
        except ValueError:
            print("[ERROR] length/width should be convertable to floating point values. Please try again.\n")
            continue
        else:
            prediction = predict(length, width)
            print(f"The prediction for given length: {length}, width: {width} is: {prediction}")

            correct = yes_or_no("Was this prediction correct?")
            if correct:
                update_correct() # updates model
            else:
                update_wrong()
            
            print("[INFO] Model updated!")
            finish = yes_or_no("Finish training?")
            if finish:
                return
            else:
                continue

def predict(length, width, initial_pred = "Bream"):
    # training data is data_list
    pred = initial_pred
#    if kn.predict([[width, length]])[0] == 1:
#        return "Smelt"
#    else:
#        return "Bream"
#    return pred
    return pred

def update_correct():
    pass

def update_wrong():
    pass

# debug
manual = yes_or_no("Welcome! Do you wish to train the model manually?")
if manual:
    print("Proceeding to manual training.\n")
    prompt()
else:
    print("Proceeding to training from prefab data.\n")
    parse_data(grab_data_path())
    bulk_train()
