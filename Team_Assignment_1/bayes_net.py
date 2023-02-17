''' 
This python program implements a naive Bayesian Network using the Pomegranate library
Official Documentation: https://pomegranate.readthedocs.io/en/latest/index.html
Github: https://github.com/jmschrei/pomegranate

A previous move is assumed to be sampled from a human node as well a computer node independently.
The naive Bayesian Network (specifically, the V-DAG (next move conditioned on the moves from the previous rounds)) is initialized
such that the a categorical prior distribution is assumed for each action for both the human and computer node such that each action is equally likely.
Based on the priors, the prediction (next move) node is updated using the appropriate Conditional Probability Table (CPT)
'''

# Import required libraries
import numpy as np
from pomegranate import *

'''
Uncomment this section if you would like to fit your Bayesian Network (V-DAG) using previously collected data
'''
# data = np.load("data.npy", allow_pickle=True) ## Load historical data
# data = np.concatenate((data[:-1, :], data[1:,1].reshape(-1,1)), axis=1) ## Re-arrange the array such that column 1 contains previous human moves, column 2 contains previous computer moves and column 3 contains the next computer moves
'''
Uncomment this section if you would like to fit your Bayesian Network (V-DAG) known priors or any other pmf values you see fit, else comment
it if you plan to fit it using previously collected data. Please note that either this section or the one above needs to be active during anytime!
'''
# Assume human samples from a categorical distribution comprising of 3 outcomes, each of which are equally likely 
human = DiscreteDistribution({'rock': 1./3, 'paper': 1./3, 'scissors': 1./3})

# Assume computer samples from a categorical distribution comprising of 3 outcomes, each of which are equally likely 
computer = DiscreteDistribution({'rock': 1./3, 'paper': 1./3, 'scissors': 1./3})

# Prediction is dependent on both the human and computer moves. 
prediction = ConditionalProbabilityTable(
        [[ 'rock', 'rock', 'rock', 1./3 ],
         [ 'rock', 'rock', 'paper', 1./3 ],
         [ 'rock', 'rock', 'scissors', 1./3 ],
         [ 'rock', 'paper', 'rock', 1./3 ],
         [ 'rock', 'paper', 'paper', 1./3 ],
         [ 'rock', 'paper', 'scissors', 1./3 ],
         [ 'rock', 'scissors', 'rock', 1./3 ],
         [ 'rock', 'scissors', 'paper', 1./3 ],
         [ 'rock', 'scissors', 'scissors', 1./3 ],
         [ 'paper', 'rock', 'rock', 1./3 ],
         [ 'paper', 'rock', 'paper', 1./3 ],
         [ 'paper', 'rock', 'scissors', 1./3 ],
         [ 'paper', 'paper', 'rock', 1./3 ],
         [ 'paper', 'paper', 'paper', 1./3 ],
         [ 'paper', 'paper', 'scissors', 1./3 ],
         [ 'paper', 'scissors', 'rock', 1./3 ],
         [ 'paper', 'scissors', 'paper', 1./3 ],
         [ 'paper', 'scissors', 'scissors', 1./3 ],
         [ 'scissors', 'rock', 'rock', 1./3 ],
         [ 'scissors', 'rock', 'paper', 1./3 ],
         [ 'scissors', 'rock', 'scissors', 1./3 ],
         [ 'scissors', 'paper', 'rock', 1./3 ],
         [ 'scissors', 'paper', 'paper', 1./3 ],
         [ 'scissors', 'paper', 'scissors', 1./3 ],
         [ 'scissors', 'scissors', 'rock', 1./3 ],
         [ 'scissors', 'scissors', 'paper', 1./3 ],
         [ 'scissors', 'scissors', 'scissors', 1./3 ]], [human, computer]) 

# State objects hold both the distribution and the asscoiated node/state name. Both state and node mean the same in regard to the Bayesian Network
s1 = State(human, name="human")
s2 = State(computer, name="computer")
s3 = State(prediction, name="prediction")


# Create the Bayesian network object using a suitable name
model = BayesianNetwork("Rock Paper Scissors")

# Add the three states to the network 
model.add_states(s1, s2, s3)

# Add edges which represent conditional dependencies, where the prediction node is 
# conditionally dependent on its parent nodes (Prediction is dependent on both human and computer moves)
model.add_edge(s1, s3)
model.add_edge(s2, s3)

# Finalize the Bayesian Network
model.bake()

# Uncomment only if you want to directly fit your Bayesian Network using data. 
# model.fit(data)

# Prints the model summary (all marginal and conditional probability distributions)
print ("Bayesian Network Summary: {}".format(model))

# Uncomment this line if you would like to predict, in this case the joint probability of (rock, paper) being the previous round moves and the next being scissors
print (model.probability([['rock', 'paper', 'rock']]))

# The following line returns the action that maximizes P(prediction|human_move,computer_move)
prediction = model.predict([['rock', 'paper', None]])
print ("Argmax_Prediction:{}".format(prediction[-1][-1]))

# To generate predictions probabilities for each of the possible actions, provide as evidence "Human": "Rock" and "Computer": "Paper" to your model
predictions = model.predict_proba({"human": "rock", "computer": "paper"})

# Print prediction probabilities for each node
for node, prediction in zip(model.states, predictions):
    if isinstance(prediction, str):
        print(f"{node.name}: {prediction}") ## Prints the current state (previous moves) of the Human and Computer Nodes in your Bayesian Network
    else:
        print(f"{node.name}")
        for value, probability in prediction.parameters[0].items():
            print(f"    {value}: {probability:.4f}")  ## Prints the probability for each possible action given the current state of the parents in your Bayesian Network