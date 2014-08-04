NNBuilder
=========
Building Neural Network Models Node by Node.


GaussianNodes: This version builds layers with both sigmoid and gaussian nodes.


RoadMap:

  Try training successive transforms on residuals after each node.
  
  For prediction apply nodes and inverse transforms alternatively in successccion (backwards).
  
  Consider nonconvex optimization methods for gaussian with varying width (decaying momentum, etc.)
  
  Why do some nodes get an early stop at starting point?
  
  Also consider:
  
    Fitting one type, 2nd type, then attempt 1st type again
    
    Alternating between nodes types
    
    Fitting both and then choosing one each time
    
    If gaussian nodes show potential, maybe consider elliptical nodes (with an actual covariance matrix)
    
    
