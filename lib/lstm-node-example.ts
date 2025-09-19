
/*
An LSTM (Long Short-Term Memory) node is a fundamental building block of LSTM networks,
a type of recurrent neural network (RNN) architecture. It is designed to remember
information for long periods, making it particularly effective for sequence
prediction problems.

Here's a breakdown of what's inside a typical LSTM node:

1. Cell State (C_t): This is the core of the LSTM node. It acts as a conveyor belt,
   running straight down the entire chain of LSTM nodes, with only minor linear
   interactions. Information can be easily added to or removed from the cell state,
   allowing it to be preserved over many time steps.

2. Gates: An LSTM node has three "gates" that control the flow of information.
   These gates are composed of a sigmoid neural network layer and a pointwise
   multiplication operation. The sigmoid layer outputs numbers between 0 and 1,
   describing how much of each component of information should be let through.
   A value of 0 means "let nothing through," while a value of 1 means "let everything
   through."

   a. Forget Gate (f_t): This gate decides what information to throw away from the
      cell state. It looks at the previous hidden state (h_{t-1}) and the current
      input (x_t) and outputs a number between 0 and 1 for each number in the
      previous cell state (C_{t-1}).

   b. Input Gate (i_t): This gate decides which new information to store in the
      cell state. It has two parts:
      - A sigmoid layer (the "input gate layer") decides which values we'll update.
      - A tanh layer creates a vector of new candidate values, C̃_t, that could be
        added to the state.

   c. Output Gate (o_t): This gate decides what the next hidden state should be.
      The hidden state is a filtered version of the cell state. The output gate
      runs a sigmoid layer to decide which parts of the cell state to output. The
      cell state is then passed through a tanh function (to push the values to be
      between -1 and 1) and multiplied by the output of the sigmoid gate.

Here's a simplified diagram of an LSTM node:

               +--------------------------------------------------+
               |                                                  |
               |       +-------+     +-------+     +-------+      |
h_{t-1}, x_t ->--|------>| sigmoid|------>| sigmoid|------>| sigmoid|-----> h_t
               |       +-------+     +-------+     +-------+      |
               |          |             |             |           |
               |          | (Forget Gate) | (Input Gate)  | (Output Gate) |
               |          |             |             |           |
               |          |             |             |           |
               |          v             v             v           |
C_{t-1} ----------------->(X)----------->(+)----------->(tanh)------> C_t
               |          ^             ^             ^           |
               |          |             |             |           |
               |          |       +-------+           |           |
               |          +-------| tanh  |<----------+           |
               |                  +-------+                       |
               |                                                  |
               +--------------------------------------------------+

In this diagram:
- (X) represents pointwise multiplication.
- (+) represents pointwise addition.
- h_{t-1} is the hidden state from the previous time step.
- x_t is the input at the current time step.
- C_{t-1} is the cell state from the previous time step.
- h_t is the hidden state at the current time step.
- C_t is the cell state at the current time step.

This structure allows LSTMs to effectively learn and remember long-term dependencies
in sequential data, which is a major advantage over traditional RNNs.
*/
