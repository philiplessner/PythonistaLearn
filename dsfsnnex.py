import numpy as np
import matplotlib.pyplot as plt
import NNplay as nn


raw_digits = [
          """11111
             1...1
             1...1
             1...1
             11111""",

          """..1..
             ..1..
             ..1..
             ..1..
             ..1..""",

          """11111
             ....1
             11111
             1....
             11111""",

          """11111
             ....1
             11111
             ....1
             11111""",

          """1...1
             1...1
             11111
             ....1
             ....1""",

          """11111
             1....
             11111
             ....1
             11111""",

          """11111
             1....
             11111
             1...1
             11111""",

          """11111
             ....1
             ....1
             ....1
             ....1""",

          """11111
             1...1
             11111
             1...1
             11111""",

          """11111
             1...1
             11111
             ....1
             11111"""]

def make_digit(raw_digit):
    return [1 if c == '1' else 0
            for row in raw_digit.split("\n")
            for c in row.strip()]

X = np.array(list(map(make_digit, raw_digits)))

y = np.array([[1 if i == j else 0 for i in range(10)]
           for j in range(10)])
print(X)
print(y)

# Train the Network
hyperparam = {'nhidden': 5,
              'eta': 1.5,
              'epochs': 10000,
              'minibatches': 1}      
#train_params = nn.fit(X, y, hyperparam)
train_paramf, cost = nn.fit(X, y, hyperparam)

# Print out the results
#train_paramf = train_params[-1]
prediction = nn.predict_proba(X, train_paramf)
#predictions = [nn.predict_proba(X, tp) for tp in train_params]
#errors = [nn.error(y, p) for p in predictions]
classes = nn.predict(X, train_paramf)
print('\nsyn0\n', train_paramf[0])
print('\nb0\n', train_paramf[1])
print('\nsyn1\n', train_paramf[2])
print('\nb1\n', train_paramf[3])
print('\nClass Probabilities\n', prediction)
print('\nClass\n', classes)
#print('\nError\n', errors[-1])
print('\nError\n', cost[-1])

fig, ax = plt.subplots(nrows=1, ncols=1)
#ax.semilogy(range(0, len(errors)), errors)
ax.semilogy(range(0, len(cost)), cost)
plt.show()
plt.clf()

#Styled 3
Z = ([0,1,1,1,0, 
0,0,0,1,1, 
0,0,1,1,0,           
0,0,0,1,1, 
0,1,1,1,0])
print('\nStyled 3\n', nn.predict_proba(Z, train_paramf), 
                      nn.predict(Z, train_paramf))
# Styled 8
Q = [0,1,1,1,0,  
1,0,0,1,1,  
0,1,1,1,0,  
1,0,0,1,1, 
0,1,1,1,0]
print('\nStyled 8\n', nn.predict_proba(Q, train_paramf), 
                      nn.predict(Q, train_paramf))

