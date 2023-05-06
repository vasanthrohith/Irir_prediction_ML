# import tkinter as tk
from tkinter import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tkinter import messagebox
import pyttsx3


class base():
    def __init__(self):
        print('base started')

    def base(self):
        ob = home_page()
        ob.home()


class home_page(base):

    def home(self):
        def predict_fn():
            global result_l  # global -> to access this label outside this fn(to refresh)
            print("predicting")
            global sep_len, sep_wid, pet_len, pet_wid
            sep_len = sep_len_e.get()
            sep_wid = sep_wid_e.get()
            pet_len = pet_len_e.get()
            pet_wid = pet_wid_e.get()
            # print(f" sepal len: {sep_len}  sepal wid : {sep_wid}  petal len : {pet_len}  petal wid : {pet_wid}")
            obj = nave_bayes()
            family = obj.nave_bayes_fn(sep_len, sep_wid, pet_len, pet_wid)[0]
            # print("--------------->",obj.nave_bayes_fn(sep_len,sep_wid,pet_len,pet_wid)[0])

            # result_l.config(text=o,font=('Helvetica','20','bold'),bg='green',fg='yellow')

            result_l = Label(frame1, text=family, font=('Helvetica', '20', 'bold'), bg='green', fg='yellow')
            result_l.place(x=400, y=440)
            print(family)

        def refresh_fn():
            result_l.destroy()
            sep_len_e.delete(0, 'end')
            sep_wid_e.delete(0, 'end')
            pet_len_e.delete(0, 'end')
            pet_wid_e.delete(0, 'end')

        def audio_fn():
            obj = nave_bayes()  # Here also we have to call the nave_bayes fn to retrieve the family name
            # print("--------------->",obj.nave_bayes_fn(sep_len,sep_wid,pet_len,pet_wid)[0])

            # result_l.config(text=obj.nave_bayes_fn(sep_len,sep_wid,pet_len,pet_wid)[0],font=('Helvetica','20','bold'),bg='green',fg='yellow')

            mic = pyttsx3.init()
            voices = mic.getProperty('voices')
            mic.setProperty('voice', voices[1].id)
            mic.say(text=obj.nave_bayes_fn(sep_len, sep_wid, pet_len, pet_wid)[0])
            mic.runAndWait()

            # print(f" sepal len: {sep_len}  sepal wid : {sep_wid}  petal len : {pet_len}  petal wid : {pet_wid}")

        # Main window---------------
        win = Tk()
        win.title("Iris prediction")
        win.geometry("700x700")

        # Creating frame
        frame1 = Frame(win, bg='green', highlightbackground="brown", highlightthickness=1)
        frame1.pack(padx=3, pady=3, ipadx=50, ipady=50, expand=True, fill='both')

        # widgets----
        # Labels
        header_l = Label(frame1, text='Iris Prediction', font=('Helvetica', '28', 'bold'), bg='green')
        header_l.pack(pady=10)

        sep_len_l = Label(frame1, text='Sepal length', font=('Helvetica', '20', 'bold'), bg='green')
        sep_len_l.place(x=100, y=160)

        sep_wid_l = Label(frame1, text='Sepal width', font=('Helvetica', '20', 'bold'), bg='green')
        sep_wid_l.place(x=100, y=230)

        pet_len_l = Label(frame1, text='Petal length', font=('Helvetica', '20', 'bold'), bg='green')
        pet_len_l.place(x=100, y=300)

        pet_wid_l = Label(frame1, text='Petal width', font=('Helvetica', '20', 'bold'), bg='green')
        pet_wid_l.place(x=100, y=370)

        family_l = Label(frame1, text='Family', font=('Helvetica', '20', 'bold'), bg='green')
        family_l.place(x=100, y=440)

        # result_l=Label(frame1,text='',bg='green')
        # result_l.place(x=400,y=440)

        # Entries
        sep_len_en = StringVar()
        sep_len_e = Entry(frame1, width=10, textvariable=sep_len_en, font=('Helvetica', '16', 'bold'))
        sep_len_e.place(x=400, y=160)

        sep_wid_en = StringVar()
        sep_wid_e = Entry(frame1, width=10, textvariable=sep_wid_en, font=('Helvetica', '16', 'bold'))
        sep_wid_e.place(x=400, y=230)

        pet_len_en = StringVar()
        pet_len_e = Entry(frame1, width=10, textvariable=pet_len_en, font=('Helvetica', '16', 'bold'))
        pet_len_e.place(x=400, y=300)

        pet_wid_en = StringVar()
        pet_wid_e = Entry(frame1, width=10, textvariable=pet_wid_en, font=('Helvetica', '16', 'bold'))
        pet_wid_e.place(x=400, y=370)

        # Buttons
        check_btn = Button(frame1, text='Check', font=('Helvetica', '16', 'bold'), relief=RAISED, borderwidth=4,
                           command=predict_fn)
        check_btn.place(x=230, y=540)

        refresh_btn = Button(frame1, text='refresh', font=('Helvetica', '16', 'bold'), relief=RAISED, borderwidth=4,
                             command=refresh_fn)
        refresh_btn.place(x=330, y=540)

        mic_btn = Button(frame1, text='🔊', font=('Helvetica', '16'), relief=RAISED, borderwidth=4, command=audio_fn)
        mic_btn.place(x=440, y=540)

        win.mainloop()


class nave_bayes():
    def nave_bayes_fn(self, sep_len, sep_wid, pet_len, pet_wid):
        print("algo")
        # print(f" sepal len: {sep_len}  sepal wid : {sep_wid}  petal len : {pet_len}  petal wid : {pet_wid}")

        # Nave bayes algorithm Algorith

        dataset = pd.read_csv(
            "C:\\Users\\vasanth rohith\\OneDrive - Kau Yan College\\Documents\\MachineLearning\\Datasets/Iris_new.csv")
        x = dataset.iloc[:, 0:-1].values
        y = dataset.iloc[:, -1].values

        # Encoding Y
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y1 = le.fit_transform(y)

        # Splitting
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=0)

        # training model
        from sklearn.naive_bayes import GaussianNB
        classifier = GaussianNB()
        classifier.fit(x_train, y_train)

        y_pred = classifier.predict(x_test)

        try:

            sep_len = float(sep_len)
            sep_wid = float(sep_wid)
            pet_len = float(pet_len)
            pet_wid = float(pet_wid)
            # print(type(sep_len))
            # if type(sep_len)=='float' and type(sep_wid)=='float' and type(pet_len)=='float' and type(sep_len)=='float':
            #     print("Good")
            input_pred = classifier.predict([[sep_len, sep_wid, pet_len, pet_wid]])

            # print("--------------->",input_pred)

            return input_pred  # returning the family name


        except Exception as e:
            print(e)
            messagebox.showerror("Error", "Pease enter the values correctly")

        # accuracy
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        acc = (sum(np.diag(cm)) / len(y_test))

        from sklearn import metrics
        metrics.accuracy_score(y_test, y_pred)


obb = base()
obb.base()

