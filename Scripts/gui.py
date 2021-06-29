from numpy import mod
from tkinter import *
import os
import sys

# Path of folder containing Main1.py file

path_to_Main1 = r'C:\Users\Admin\pracExc\Project_Bitcoin'
sys.path.append(path_to_Main1)

import Main1  

# Path to Assets folder
path_assets = 'C:\\Users\\Admin\\pracExc\\Project_Bitcoin\\Assets'

class Start_Page:
    """
    A class used to represent the first gui page

    ...

    Attributes
    ----------

     frame : Frame 
        element dedicated to contain all other elements in window (master)

    """



    def __init__(self, master,path_to_asset ='C:\\Users\\Admin\\pracExc\\Project_Bitcoin\\Assets' ):
        
        """

        Parameters
        ----------
        master : Tk
            the base toplevel window
        path_to_asset : str
            the path to Assets folder in local machine

        """


        # master is the whole page that holds the elements
        master.title('Liam Bitcoin Project')
        master.configure(background = '#1a82dd')
        
        
        self.frame = Frame(master, background= '#1a82dd')
        self.frame.pack(side="top", expand=True, fill="both")


        # train the model upon clicking the 'Train Model' button and move to second page 

        def On_Train():

            for widgets in self.frame.winfo_children():
                widgets.destroy()
            self.frame.destroy()
            mod.train()
            Second_Page(master,path_assets)


        # load the trained model upon clicking the 'trained model' and move to second page 

        def On_Pretrain():

            for widgets in self.frame.winfo_children():
                widgets.destroy()
            self.frame.destroy()
            mod.retrieve_model(Main1.model_path)
            Second_Page(master, path_assets)
        
        header = Label(self.frame, text = "Bitcoin Analysis",font = ('Arial', 18, 'bold') , justify= CENTER, pady= 20 , background= '#1a82dd' )
        header.pack()
        desc_main = Label(self.frame, text = "Please choose one of the following options:\n" , font = ('Arial', 11,'bold'), background= '#1a82dd' )
        desc_main.pack()
        desc_train = Label(self.frame, text = "1. Train Model in order to train a new model and observe its training live.\n" , font = ('Arial', 11, 'bold'), background= '#1a82dd', pady = 20 )
        desc_train.pack()
        train_pic = PhotoImage(file= os.path.join(path_to_asset,'Train.gif' ))
        train_but = Button(self.frame,text= "Train Model",height= 175, width= 500, image= train_pic,font= ('Courier', 18, 'bold'), foreground = 'blue', compound= 'left', command= On_Train  )
        train_but.img = train_pic
        train_but.pack()
        desc_pretrained = Label(self.frame, text = "2. Pretrained Model to skip the training process.\n" , font = ('Arial', 11,'bold'), background= '#1a82dd', pady = 20  )
        desc_pretrained.pack()
        pretrained_pic = PhotoImage(file= os.path.join(path_to_asset,'Pretrained.gif' ))
        pretrained_but = Button(self.frame,text= "Pretrained Model",height= 175, width= 500, image= pretrained_pic,font= ('Courier', 18, 'bold'), foreground = 'blue', compound= 'left', command= On_Pretrain )
        pretrained_but.img = pretrained_pic
        pretrained_but.pack()
        spc_lbl1 = Label(self.frame, pady= 15, background= '#1a82dd')
        spc_lbl1.pack()

    



class Second_Page:
    """
    A class used to represent the first gui page

    ...

    Attributes
    ----------

     frame : Frame 
        element dedicated to contain all other elements in window (master)

    """



    def __init__(self, master,path_to_asset ='C:\\Users\\Admin\\pracExc\\Project_Bitcoin\\Assets' ):
    
        
        """

        Parameters
        ----------
        master : Tk
            the base toplevel window
        path_to_asset : str
            the path to Assets folder in local machine

        """


        # plot and show original graph of data, then show predicted results vs real results of test phase

        def eval_window():
            mod.plot_orgdata()
            mod.predict_plot_test()
            mod.plot_loss()
            mod.plot_mae()

        # open mid_pred window 

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

        
        header = Label(self.frame, text = "Bitcoin Analysis",font = ('Arial', 18, 'bold') , justify= CENTER, pady= 20 , background= '#1a82dd' )
        header.pack()
        desc_main = Label(self.frame, text = "Please choose one of the following options:\n" , font = ('Arial', 11,'bold'), background= '#1a82dd' )
        desc_main.pack()
        desc_eval = Label(self.frame, text = "1. Evaluate the model according to real data.\n" , font = ('Arial', 11, 'bold'), background= '#1a82dd', pady = 20 )
        desc_eval.pack()
        eval_pic = PhotoImage(file= os.path.join(path_to_asset,'Stats.gif' ))
        eval_but = Button(self.frame,text= "Evaluate Model",height= 175, width= 500, image= eval_pic,font= ('Courier', 18, 'bold'), foreground = 'blue', compound= 'left', command= eval_window )
        eval_but.img = eval_pic
        eval_but.pack()
        desc_forecasting = Label(self.frame, text = "2. Predict future results .\n" , font = ('Arial', 11,'bold'), background= '#1a82dd', pady = 20  )
        desc_forecasting.pack()
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

    """
    A class used to represent the first gui page

    ...

    Attributes
    ----------

     win : Toplevel 
        pop up window dedicated to hold all other elements 
     preds : StringVar
        holds the value of spinbox in order to pass it to predict_plot_future function
        
    """

    def __init__(self,master):

        """

        Parameters
        ----------
        master : Tk
            the base toplevel window

        """

        self.win = Toplevel(master)
        self.win.title("future hours selection")
        instruct = Label(self.win, text = "Please choose how many future hours would you like to predict (up to 10):\n")
        instruct.pack()
        self.preds = StringVar()
        opt = Spinbox(self.win, from_ = 1, to = 10, textvariable= self.preds)
        opt.pack()

        spc_lbl3 = Label(self.win, pady= 4)
        spc_lbl3.pack()


        # call predict_plot_future upon clicking on continue button 

        def to_preds ():
            mod.predict_plot_future(int(self.preds.get()))

        to_plot_but = Button(self.win, text='continue', height= 1, width= 10, command= to_preds)
        to_plot_but.pack()

    
def main():            

    root = Tk()
    global mod
    mod = Main1.Model()
    Start_Page(root, path_assets)
    root.mainloop()
    
if __name__ == "__main__": main()