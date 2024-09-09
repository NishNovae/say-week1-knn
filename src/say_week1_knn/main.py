# src/say_week1_knn/main.py

import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pickle
import random

data_list = []
kn = KNeighborsClassifier(n_neighbors=5)
NOT_PREPARED = True
x_data, y_data = [], []

def create_dummy():
    global x_data, y_data
    x_data, y_data = [], []
    for i in range (10):
        x_data.append( [round(random.uniform(10, 41), 2), round(random.uniform(10, 1000), 2) ])
        y_data.append( random.choice( ["Bream", "Smelt"] ) )

def yes_or_no(prompt):
    while True:
        yn = input(prompt + " [y/n] ")
        if yn == "y" or yn == "Y":
            return 1
        elif yn == "n" or yn == "N":
            return 0
        elif yn == "exit":
            return -1
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
    global x_data, y_data

    with open(data_path, "r") as data:
        next(data)      # skips line "Length,Width,Label"
        for line in data:
            length, width, answer = line.strip().split(",")
            x_data.append([float(length), float(width)])
            y_data.append(answer)

    print(f"[INFO] data collected from: {data_path}")
    return

def bulk_train(x_list=[], y_list=[]):
    global NOT_PREPARED, x_data, y_data
    NOT_PREPARED = False

    for i in range (len(x_list)):
        x_data.append(x_list[i])
        y_data.append(y_list[i])
        print(x_data[-2:])
        print(y_data[-2:])

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
    kn.fit(x_train, y_train)

    # debug
    print("data len: ", len(x_train))
    bulk_test(x_test, y_test)

def bulk_test(x_test, y_test):
    prediction = kn.predict(x_test)
    print("y_test length: ", len(y_test))
    tmp = 0
    for i in range(len(y_test)):
        if prediction[i] == y_test[i]:
            tmp += 1
    
    acc = tmp/len(y_test) * 100
    print(f"RESULT: {acc} % accuracy\n")


def prompt():
    global NOT_PREPARED
    if NOT_PREPARED:
        print("[INFO] Initializing dummy data of size 10...\n")
        create_dummy()
        bulk_train()

    print("[INFO] Proceeding to manual prompt.\n")
    print("[INFO] User prompt activated! Please input the length and width.")
    global input_val, output_val
    while True:
        try:
            print()
            line = input("length, width [whitespace-separated]: ")
            length, width = line.strip().split()
            length = float(length)
            width = float(width)

        except ValueError:
            print("[ERROR] length/width should be convertable to floating point values. Please try again.\n")
            continue

        else:
            val, prediction = predict(length, width)
            print(f"The prediction for given length: {length}, width: {width} is: {prediction}")
            correct = yes_or_no("Was this prediction correct?")

            if correct == 1:
                update(length, width, val) # updates model
            elif correct == 0:
                update(length, width, 1-val)
            else:
                print("Exiting...")
                return
            
            # print("[INFO] Model updated!")

def predict(length, width):
    # training data is data_list

    if kn.predict([[length, width]])[0] == 1: # 1 == Smelt
        return 1, "Smelt"
    else:
        return 0, "Bream"

def update(length, width, val):

    print(f"length:{length}, width:{width}, val:{val}")

    if val == 1:
        answer = "Smelt"
    elif val == 0:
        answer = "Bream"
    else:
        return
    bulk_train([[length, width]], [answer])

def save_model():
    model_path = get_model_path()
    with open(model_path, "wb") as model:
        pickle.dump(kn, model)

# debug
print("[INFO] Beginning fish prediction program! type 'exit' to finish whenever a (y/n) prompt comes up.\n")
bulk = yes_or_no("Welcome! Do you wish to bulk-train the model before entering prompt?")

if bulk == 1:
    print("[INFO] Proceeding to bulk-train from prefab data...\n")
    parse_data(grab_data_path())
#    print(data_list)
    bulk_train()
    print("[INFO] Bulk-train complete!\n")
    prompt()

elif bulk == 0:
    prompt()

else:
    print("[INFO] Finishing program...")
save_model()

