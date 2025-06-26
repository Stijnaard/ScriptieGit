# An Adaptive Sampling Method to Improve Physics-Informed Neural Networks

#### Abstract
Physics-Informed Neural Networks (PINNs) offer a mesh-free approach to solving partial differential equations (PDEs), yet often struggle with localized features and require full retraining for new problem settings. This thesis introduces a novel adaptive sampling method that reallocates collocation points during training based on a virtual spring model inspired by classical mesh refinement. By concentrating training points in regions with high PDE residuals, the method improves convergence efficiency without increasing the number of training points, particularly in problems with localized features. Experiments on 1D synthetic benchmarks demonstrate consistent performance gains in scenarios with steep gradients and non-smooth solutions, reducing both training time and solution error compared to static or random resampling approaches. While limitations exist for smoother problems, the method’s lightweight nature and potential for extension to higher dimensions suggest broad applicability. This work contributes an interpretable and effective tool for enhancing the training of PINNs under computational constraints.

#### Method
Our adaptive sampling method is based on a physical analogy. All the (points) nodes in our domain are connected by springs. The stiffness of each spring is proportional to the average error of the nodes it connects. Where the residual is large, the spring is “stiffer” and pulls its endpoints closer together; where the residual is small, the spring is soft and allows the points to spread apart.  In effect, the network of springs searches for an equilibrium configuration in which the spacing of nodes is inversely proportional to the residual error. This method can be found in the `PINN_adaptive.ipynb` file under 'adaptive node moving function'. See `node_moving.png` in this directory to see an illustration of how this method works.

![Adaptive Node Moving](node_moving.jpg?raw=true "Node Moving")



