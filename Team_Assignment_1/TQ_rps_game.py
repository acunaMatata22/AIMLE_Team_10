"""
Python implementation of Rock-Paper-Scissors!

Rules:
Each player chooses a move (simultaneously) from the choices: rock, paper or scissors. 
If both players choose the same move, the round ends in a tie. 
Otherwise:
    Rock beats Scissors
    Scissors beats Paper
    Paper beats Rock.

To modify the GUI, please refer to:
Tkinter Documentation: https://docs.python.org/3/library/tk.html
                       https://www.geeksforgeeks.org/tkinter-optionmenu-widget/
"""

## Import required libraries
import random
import tkinter as tk
from tkinter import *
import numpy as np
from TQ_bayes_net import v_predict_move
from TQ_inv_bayes_net import inv_predict_move

# load training data, declared global in get_human_move
training_data = np.load("training_data.npy", allow_pickle=True) ## Load historical data
training_data = np.concatenate((training_data[:-1, :], training_data[1:,1].reshape(-1,1)), axis=1)

# record round moves
def save_data(hm, cm):
    '''
    This function collects data 
    '''
    data.append([hm,cm])

# organize new data into the right format
def save_real_time_data(data, computer_move):
    newSamples = np.concatenate((np.array(data)[-1, :], np.array(computer_move).reshape(1))) # both 1d
    return newSamples

# add newly collected real-time data to the loaded training_data
def feedback(newSamples, training_data):
    training_data = np.concatenate((training_data, newSamples.reshape(1, -1)), axis=0) # both 2d
    return training_data

def update_scores(winner):
    """
    returns updated total scores
    """
    global total_computer_score
    global total_human_score
    if (winner == 'human'):
        total_human_score += 1
        total_computer_score += 0
    if (winner == 'computer'):
        total_human_score += 0
        total_computer_score += 1
    if (winner == 'tie'):
        total_human_score += 0
        total_computer_score += 0
        
    return total_human_score, total_computer_score

def select_winner(computer_move, human_move):
    """
    return: winner of the round
    """
    if computer_move == human_move:
        return 'tie'
    if computer_move == 'rock':
        if human_move == 'paper':
            return 'human'
        return 'computer'
    elif computer_move == 'paper':
        if human_move == 'rock':
            return 'computer'
        return 'human'
    elif computer_move == 'scissors':
        if human_move == 'paper':
            return 'computer'
        return 'human'

def get_computer_move():
    """
    Using randint()m which is an inbuilt function of the random module in Python3, generate a number between (1,3)
    where, 1 - Rock, 2 - Paper, 3 - Scissors
    returns string representing what ai move (rock | paper | scissors)
    """
    move = random.randint(1, 3)
    if move == 1:
        return 'rock'
    elif move == 2:
        return 'paper'
    else:
        return 'scissors'

# called in get_human_move()
def get_ai_move(data, user_strat):
    '''
    Implement the win-stay, lose-shift or the win-shift, lose-shift strategy
    '''
    
    # check if there is any data available from previous rounds
    if data:
        last_human_move = np.array(data)[-1,0]
        last_ai_move = np.array(data)[-1,1]
        
        # get the winner of the previous round
        winner = select_winner(last_ai_move,last_human_move)
        
        if user_strat == 'win-stay_lose-shift':
            # implement win-stay, lose-shift strategy
            if winner == "computer":
                # if the AI won the last round, choose the same move again
                return last_ai_move
            elif winner == "human":
                # if the AI lost the last round, shift to the next move
                if last_human_move == 'rock':
                    return 'paper'
                elif last_human_move == 'paper':
                    return 'scissors'
                elif last_human_move == 'scissors':
                    return 'rock'
            elif winner == 'tie':
                # if the last round was a tie, choose a random move
                return get_computer_move()
        elif user_strat == 'win-shift_lose-shift':
            # implement win-shift, lose-shift strategy
            if winner == "computer":
                # if the AI won the last round, choose the opponent's previous move
                return last_human_move
            elif winner == "human":
                # if the AI lost the last round, shift to the next move
                if last_human_move == 'rock':
                    return 'paper'
                elif last_human_move == 'paper':
                    return 'scissors'
                elif last_human_move == 'scissors':
                    return 'rock'
            elif winner == 'tie':
                # if the last round was a tie, choose a random move
                return get_computer_move()
        elif user_strat == 'random':
            return get_computer_move()
    else:
        # if there is no previous data, choose a random move
        return get_computer_move()
    
def get_real_time_bayes_net_human_move(human_move, computer_move, training_data):
    '''
    Implement a Bayesian Network that takes as input the previous round's moves and predicts the next move you should play
        The choice of Bayes network to use (V-DAG (Prediction|Human Move and Computer Move) or `]Naive Bayes (Inverted V-DAG) (Human Move|Prediction)x(Computer Move|Prediction))
    '''
    if bayes == 'V-DAG':
        predicted_move = v_predict_move(human_move, computer_move, training_data)
    elif bayes == 'Inv(V-DAG)':
        predicted_move = inv_predict_move(human_move, computer_move, training_data)
    
    if predicted_move == 'rock':
        print('Recommended next move: paper')
    elif predicted_move == 'paper':
        print('Recommended next move: scissors')
    elif predicted_move == 'scissors':
        print('Recommended next move: rock')    

# called in display_module()
def get_human_move(human_move, tt):
    """
    returns a valid move from the human (rock, paper, or scissors) and updates the scores and returns a winner
    """

    global continue_playing_button
    global training_data

    # Game Mode <<------------------------------------
    computer_move = get_ai_move(data,user_strat) 
    
    # Select the winner
    winner = select_winner(computer_move, human_move)

    # Update the scores
    update_scores(winner)
    
    if enable_feedback == 'Yes':
        if data: # 1st round has no saved game data
            # [prev_hm, prev_cm, cur_cm]
            newSamples = save_real_time_data(data, computer_move)

            # add newly collected real-time data to loaded training_data
            training_data = feedback(newSamples, training_data)
    
    # sanity check
    print(training_data.shape)
    
    ## Predict opponent's most likely next move
    # get_bayes_net_human_move(human_move, computer_move)
    get_real_time_bayes_net_human_move(human_move, computer_move, training_data)
    
    # original save_data
    save_data(human_move, computer_move)

    # Print round summary
    HM_label=Label(Window, foreground='black',background='white', text='Human move was {}'.format(human_move))
    HM_label.place(x = 240,y = 300) 

    CM_label=Label(Window, foreground='black',background='white', text='Computer move was {}'.format(computer_move))
    CM_label.place(x = 240,y = 340) 

    W_label=Label(Window, foreground='black',background='white', text='Winner is {}'.format(winner))
    W_label.place(x = 240,y = 380) 

    CSH_label=Label(Window, foreground='black',background='white', text='Current score for human: {}'.format(total_human_score))
    CSH_label.place(x = 240,y = 420) 

    CSC_label=Label(Window, foreground='black',background='white', text='Current score for computer: {}'.format(total_computer_score))
    CSC_label.place(x = 240,y = 460) 

    continue_playing_button=Button(Window, foreground='black',background='white',text='Continue Playing',command=lambda xx= tt: reset(xx))
    continue_playing_button.place(x = 100, y = 500)

    labels.extend([HM_label,CM_label,W_label,CSH_label,CSC_label,continue_playing_button])


def reset(tt):
    '''
    This function is used to reset a round after it has been played as well as to terminate the game and display the game summary 
    once the input number of rounds have been played
    '''
    for label in labels:
        label.destroy()
    global count
    count += 1
    display_module(tt)

    if count == int(tt.get()):
        for label in labels:
            label.destroy()

        for label in labels2:
            label.destroy()

        GS_label=Label(Window, foreground='black',background='white', text='Game Summary')
        GS_label.place(x = 240,y = 280) 

        # Print round summary
        TSH_label=Label(Window, foreground='black',background='white', text='Total score for Human: {}'.format(total_human_score))
        TSH_label.place(x = 240,y = 320) 

        TSC_label=Label(Window, foreground='black',background='white', text='Total score for Computer: {}'.format(total_computer_score))
        TSC_label.place(x = 240,y = 360) 

        labels2.extend([GS_label,TSH_label,TSC_label])


        if total_computer_score > total_human_score:
            computer_label=Label(Window, foreground='black',background='white', text='Computer Wins!')
            computer_label.place(x = 240,y = 400) 
            labels2.append(computer_label)

        elif total_computer_score < total_human_score:
            human_label=Label(Window, foreground='black',background='white', text='Human Wins!')
            human_label.place(x = 240,y = 400) 
            labels2.append(human_label)

        else:
            tie_label=Label(Window, foreground='black',background='white', text='Series ended in a tie')
            tie_label.place(x = 240,y = 400) 
            labels2.append(tie_label)

        ## Clicking "Reset Game" terminates the game and dumps all data collected in this game
        reset_button=Button(Window, foreground='black',background='white',text='Reset Game',command= lambda xx= "reset": reset_game(xx))
        reset_button.pack()
        reset_button.place(x = 240, y = 500)


def display_module(tt): # take user's move
    '''
    This function to asks you to select a move

    '''
    round_label=Label(Window, foreground='black',background='white', text='Round {}'.format(count+1))
    round_label.place(x = 40, y = 280)

    select_label=Label(Window, foreground='black',background='white', text='Enter your move (rock|paper|scissors):')
    select_label.place(x = 40, y = 240)

    rock_button=Button(Window, foreground='black',background='white',text='Rock',command=lambda t= "rock": get_human_move(t,tt))
    rock_button.place(x = 330, y = 240)

    paper_button=Button(Window, foreground='black',background='white',text='Paper',command=lambda t= "paper": get_human_move(t,tt))
    paper_button.place(x = 430, y = 240)

    scissors_button=Button(Window, foreground='black',background='white',text='Scissors',command= lambda t= "scissors": get_human_move(t,tt))
    scissors_button.place(x = 530, y = 240)

    labels2.extend([round_label,select_label,round_label,paper_button,scissors_button,rock_button])

def reset_game(xx):
    '''
    This function is used to reset a game after all rounds have been played as well as saves previously collected moves
    Note that clicking on reset dumps all previously collected data onto a numpy file and resets the game
    You might want to modify this function such that you keep collecting data and the numpy dumping happens only when you click "Exit Program"
    '''
    if xx == "reset":
        for label in labels2:
            label.destroy()
    np.array(data).dump(open('data.npy', 'wb'))
    welcome()

def welcome(): # take in user_strat, game rounds
    '''
    This welcome function asks you to enter the number of rounds you would like to play and enables you to start playing the game
    '''
    games_label=Label(Window, foreground='black',background='white', text='Enter the names of rounds you want to play:') 
    games_label.place(x = 40,y = 100)

    user_entry = Entry(Window, width = 5)
    user_entry.pack()
    user_entry.place(x = 330, y = 97) 
    
    # Create the list of options
    options_list = ["random", "win-stay_lose-shift", "win-shift_lose-shift"]
    options_list_2 = ['Yes', 'No']
    options_list_3 = ['V-DAG', 'Inv(V-DAG)']
    
    # Variable to store the option user
    # selected in OptionMenu
    value_inside = tk.StringVar(Window)
    value_inside_2 = tk.StringVar(Window)
    value_inside_3 = tk.StringVar(Window)
    
    # Set the default value of the variable
    value_inside.set("Select an Option")
    value_inside_2.set("Select an Option")
    value_inside_3.set("Select an Option")
    
    ##
    strategy_label=Label(Window, foreground='black',background='white', text='You wish to play against:') 
    strategy_label.place(x = 40,y = 80)
    
    feedback_label=Label(Window, foreground='black',background='white', text='Do u want real-time model update?') 
    feedback_label.place(x = 40,y = 60)      
    
    bayes_label=Label(Window, foreground='black',background='white', text='Choose ur Bayes net:') 
    bayes_label.place(x = 40,y = 40)  
    
    # Create the optionmenu widget and passing 
    # the options_list and value_inside to it.
    question_menu = tk.OptionMenu(Window, value_inside, *options_list)
    question_menu.pack()
    question_menu.place(x = 330, y = 77) 
    
    question_menu_2 = tk.OptionMenu(Window, value_inside_2, *options_list_2)
    question_menu_2.pack()
    question_menu_2.place(x = 330, y = 57)
    
    question_menu_3 = tk.OptionMenu(Window, value_inside_3, *options_list_3)
    question_menu_3.pack()
    question_menu_3.place(x = 330, y = 37)
    
    # start_game_button=Button(Window,text='Start Playing!',command= lambda t= user_entry: playgame(t))
    # t is a tuple
    start_game_button = Button(Window, text='Start Playing!', command=lambda t=(value_inside, user_entry, value_inside_2, value_inside_3): playgame(t))
    start_game_button.pack()
    start_game_button.place(x = 330, y = 130)
    

def playgame(t):
    '''
    This function controls the round logic, based on how many rounds you would like to play
    '''
    nums_label=Label(Window, foreground='black',background='white', text='You will play {} games against a {} strategy'.format(t[1].get(), t[0].get())) # t[1] is a widget, t.get() is a value
    nums_label.place(x = 40,y = 180) 
    end_label=Label(Window, foreground='black',background='white', text='----------------------------------------------')
    end_label.place(x = 40,y = 200) 
    
    global total_computer_score
    global total_human_score
    global labels
    global labels2
    global data
    global count
    global user_strat
    global enable_feedback
    global bayes
    
    user_strat = t[0].get()
    enable_feedback = t[2].get()
    bayes = t[3].get()
    
    count = 0
    data = []
    total_human_score = 0
    total_computer_score = 0
    labels = []
    labels2 = []
    labels2.extend([nums_label, end_label])

    ## Call display function to select a move
    display_module(t[1])


if __name__ == '__main__':
    
    ## Initialize a Tkinter GUI window
    Window = Tk()
    ## Set GUI window dimensions
    Window.geometry("650x550")
    ## Set GUI background color
    Window.configure(background='white')
    ## Sets a title for the GUI
    Window.title("Rock - Paper - Scissors")
    ## Welcome statement
    welcome_label=Label(Window, foreground='black',background='white', text='Welcome to Rock, Paper, Scissors! --Presented to you by 24787 TA/CA')
    welcome_label.place(x = 40,y = 60) 
    welcome_label.pack()
    ## Call the welcome function
    welcome()
    ## Clicking "Exit Program" terminates the game and exits the program
    exitButton=Button(Window,text='Exit program',command=Window.destroy).place(x = 352, y = 500)
    # reset_button=Button(Window, foreground='black',background='white',text='Reset Game',command= lambda xx= "reset": reset_game(xx))
    Window.mainloop()
    
