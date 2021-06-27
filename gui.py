from numpy import mod
from tkinter import *
#from tkinter import ttk
#from tkinter.ttk import *
import os
import sys
import matplotlib.pyplot as plt

path_to_Main1 = r'C:\Users\Admin\pracExc\Project_Bitcoin'
sys.path.append(path_to_Main1)

import Main1  

path_assets = 'C:\\Users\\Admin\\pracExc\\Project_Bitcoin\\Assets'

class Start_Page:
    def __init__(self, master,path_to_asset ='C:\\Users\\Admin\\pracExc\\Project_Bitcoin\\Assets' ):
    


        # master is the whole page that holds the elements
        master.title('Liam Bitcoin Project')
        master.configure(background = '#1a82dd')
        
        
        self.frame = Frame(master, background= '#1a82dd')
        self.frame.pack(side="top", expand=True, fill="both")

        def On_Train():
            for widgets in self.frame.winfo_children():
                widgets.destroy()
            self.frame.destroy()
            mod.train()
            Second_Page(master,path_assets)

        def On_Pretrain():
            for widgets in self.frame.winfo_children():
                widgets.destroy()
            self.frame.destroy()
            mod.retrieve_model(Main1.model_path)
            Second_Page(master, path_assets)
        
        self.header = Label(self.frame, text = "Bitcoin Analysis",font = ('Arial', 18, 'bold') , justify= CENTER, pady= 20 , background= '#1a82dd' )
        self.header.pack()
        self.desc_main = Label(self.frame, text = "Please choose one of the following options:\n" , font = ('Arial', 11,'bold'), background= '#1a82dd' )
        self.desc_main.pack()
        self.desc_train = Label(self.frame, text = "1. Train Model in order to train a new model and observe its training live.\n" , font = ('Arial', 11, 'bold'), background= '#1a82dd', pady = 20 )
        self.desc_train.pack()
        train_pic = PhotoImage(file= os.path.join(path_to_asset,'Train.gif' ))
        train_but = Button(self.frame,text= "Train Model",height= 175, width= 500, image= train_pic,font= ('Courier', 18, 'bold'), foreground = 'blue', compound= 'left', command= On_Train  )
        train_but.img = train_pic
        train_but.pack()
        self.desc_pretrained = Label(self.frame, text = "2. Pretrained Model to skip the training process.\n" , font = ('Arial', 11,'bold'), background= '#1a82dd', pady = 20  )
        self.desc_pretrained.pack()
        pretrained_pic = PhotoImage(file= os.path.join(path_to_asset,'Pretrained.gif' ))
        pretrained_but = Button(self.frame,text= "Pretrained Model",height= 175, width= 500, image= pretrained_pic,font= ('Courier', 18, 'bold'), foreground = 'blue', compound= 'left', command= On_Pretrain )
        pretrained_but.img = pretrained_pic
        pretrained_but.pack()
        spc_lbl1 = Label(self.frame, pady= 15, background= '#1a82dd')
        spc_lbl1.pack()

    



class Second_Page:
    def __init__(self, master,path_to_asset ='C:\\Users\\Admin\\pracExc\\Project_Bitcoin\\Assets' ):
    


        def eval_window():
            mod.df.plot( y='Weighted_Price')
            plt.title("Weighted price")
            plt.show()
            mod.predict_plot_test()

        def forecast_window():
            fore_wind = mid_pred(master)
            

            

        # master is the whole page that holds the elements
        master.title('Liam Bitcoin Project')
        master.configure(background = '#1a82dd')
        
        
        self.frame = Frame(master, background= '#1a82dd')
        self.frame.pack(side="top", expand=True, fill="both")

        def To_First_Page():
            for widgets in self.frame.winfo_children():
                widgets.destroy()
            self.frame.destroy()
            Start_Page(master,path_assets)

        
        self.header = Label(self.frame, text = "Bitcoin Analysis",font = ('Arial', 18, 'bold') , justify= CENTER, pady= 20 , background= '#1a82dd' )
        self.header.pack()
        self.desc_main = Label(self.frame, text = "Please choose one of the following options:\n" , font = ('Arial', 11,'bold'), background= '#1a82dd' )
        self.desc_main.pack()
        self.desc_eval = Label(self.frame, text = "1. Evaluate the model according to real data.\n" , font = ('Arial', 11, 'bold'), background= '#1a82dd', pady = 20 )
        self.desc_eval.pack()
        eval_pic = PhotoImage(file= os.path.join(path_to_asset,'Stats.gif' ))
        eval_but = Button(self.frame,text= "Evaluate Model",height= 175, width= 500, image= eval_pic,font= ('Courier', 18, 'bold'), foreground = 'blue', compound= 'left', command= eval_window )
        eval_but.img = eval_pic
        eval_but.pack()
        self.desc_forecasting = Label(self.frame, text = "2. Predict future results .\n" , font = ('Arial', 11,'bold'), background= '#1a82dd', pady = 20  )
        self.desc_forecasting.pack()
        forecasting_pic = PhotoImage(file= os.path.join(path_to_asset,'forecasting.gif' ))
        forecasting_but = Button(self.frame,text= "Forecast",height= 175, width= 500, image= forecasting_pic,font= ('Courier', 18, 'bold'), foreground = 'blue', compound= 'left', command= forecast_window )
        forecasting_but.img = forecasting_pic
        forecasting_but.pack()
        spc_lbl1 = Label(self.frame, pady= 7, background= '#1a82dd')
        spc_lbl1.pack()
        back_but = Button(self.frame, text = "Back",height= 1, width= 10, command= To_First_Page )
        back_but.pack()
        spc_lbl2 = Label(self.frame, pady= 4, background= '#1a82dd')
        spc_lbl2.pack()

class mid_pred:


    def __init__(self,master):

        self.win = Toplevel(master)
        self.win.title("future hours selection")
        self.instruct = Label(self.win, text = "Please choose how many future hours would you like to predict (up to 10):\n")
        self.instruct.pack()
        self.preds = StringVar()
        self.opt = Spinbox(self.win, from_ = 1, to = 10, textvariable= self.preds)
        self.opt.pack()

        spc_lbl3 = Label(self.win, pady= 4)
        spc_lbl3.pack()

        def to_preds ():
            mod.predict_plot_future(int(self.preds.get()))

        self.to_plot_but = Button(self.win, text='continue', height= 1, width= 10, command= to_preds)
        self.to_plot_but.pack()

    
def main():            

    root = Tk()
    global mod
    mod = Main1.Model()
    FP = Start_Page(root, path_assets)
    root.mainloop()
    
if __name__ == "__main__": main()