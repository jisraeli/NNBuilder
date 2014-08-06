NNBuilder
=========
Building Neural Network Models Node by Node.

AdaScale: This NNBuilder version attempts adaptive scaling/preprocessing

In a nutshell:
	
	Before optimizing a new node the pseoudo-response gets preprocessed with a new scaler

	Predictions are made recrusively by:

		add prediction from the most recent node
			apply most recent inverse transform
				add prediction from previous node
					apply previous inverse transform
						etc.

	Expected benefits:

		Each node can recreate limited behavior.
		While the original pre-processing puts the data in a form
		that simplifies this task, we have no guarantee that any of
		the pseudo-responses after the 1st node are still in this "easy to handle" form.
		To get around this problem, we adaptively scale/preprocess pseudo-response
		to simplify learning for each node.
		Also, by maintaining mean 0 and variance 1 gaussian with b=1 
		may learn more effectively than other nodes.
	

