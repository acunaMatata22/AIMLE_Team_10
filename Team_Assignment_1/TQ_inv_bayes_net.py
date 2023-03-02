"""
Official Documentation: https://pomegranate.readthedocs.io/en/latest/index.html
                        https://pomegranate.readthedocs.io/en/latest/NaiveBayes.html
Github: https://github.com/jmschrei/pomegranate
"""

# Import required libraries
import numpy as np
from pomegranate import *

# ****************Change .npy Name when necessary*********************
# training_data = np.load("training_data.npy", allow_pickle=True) ## Load historical data
# training_data = np.concatenate((training_data[:-1, :], training_data[1:,1].reshape(-1,1)), axis=1) ## Re-arrange the array such that column 1 contains previous human moves, column 2 contains previous computer moves and column 3 contains the next computer moves

def inv_predict_move(human_move, computer_move, training_data): # previous moves
    
    # Initialize Y
    labels = DiscreteDistribution({'rock': 1./3, 'paper': 1./3, 'scissors': 1./3})

    # Initial CPT of P(A|Y)
    A_givenY = ConditionalProbabilityTable(
        [[ 'rock', 'rock', 1./3 ],
         [ 'rock', 'paper', 1./3 ],
         [ 'rock', 'scissors', 1./3 ],
         [ 'paper', 'rock', 1./3 ],
         [ 'paper', 'paper', 1./3 ],
         [ 'paper', 'scissors', 1./3 ],
         [ 'scissors', 'rock', 1./3 ],
         [ 'scissors', 'paper', 1./3 ],
         [ 'scissors', 'scissors', 1./3]], [labels])
    
    # Initial CPT of P(B|Y)
    B_givenY = ConditionalProbabilityTable(
        [[ 'rock', 'rock', 1./3 ],
         [ 'rock', 'paper', 1./3 ],
         [ 'rock', 'scissors', 1./3 ],
         [ 'paper', 'rock', 1./3 ],
         [ 'paper', 'paper', 1./3 ],
         [ 'paper', 'scissors', 1./3 ],
         [ 'scissors', 'rock', 1./3 ],
         [ 'scissors', 'paper', 1./3 ],
         [ 'scissors', 'scissors', 1./3]], [labels])
    
    # Create nodes
    s1 = State(A_givenY, name="human")
    s2 = State(B_givenY, name="computer")
    s3 = State(labels, name="Y")
    
    # # This can represent Inv(V-DAG) if edges are defined right
    model = BayesianNetwork("Rock Paper Scissors")
    
    model.add_states(s1, s2, s3)
    
    # Below is Inv(V-DAG)
    model.add_edge(s3, s1)
    model.add_edge(s3, s2)

    # # Finalize the Bayesian Network
    model.bake()

    model.fit(training_data) # will update Y and CPTs
    
    # The following line returns the action that maximizes P(prediction|human_move,computer_move)
    prediction = model.predict([[human_move, computer_move, None]]) # <<------------------ takes in previous hm, cm
    
    print ("Argmax_Prediction:{}".format(prediction[-1][-1]))
    
    output = prediction[-1][-1]
    
    # To generate predictions probabilities for each of the possible actions, provide as evidence "Human": "Rock" and "Computer":     # "Paper" to your model
    predictions = model.predict_proba({"human": human_move, "computer": computer_move}) # <<------------------ takes in previous hm, cm
        
        # Print prediction probabilities for each node
    for node, prediction in zip(model.states, predictions):
        if isinstance(prediction, str):
            print(f"{node.name}: {prediction}") ## Prints the current state (previous moves) of the Human and Computer Nodes in your Bayesian Network
        else:
            print(f"{node.name}")
            for value, probability in prediction.parameters[0].items():
                print(f"    {value}: {probability:.4f}")  ## Prints the probability for each possible action given the current state of the parents in your Bayesian Network
    
    return output

