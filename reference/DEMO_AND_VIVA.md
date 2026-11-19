# Professor demonstration guide

## 60-second project introduction

“Our project is Neuro-Genetic LeNet-5, a soft-computing hybrid for handwritten digit recognition. A Convolutional Neural Network learns to classify the pixels, while a Genetic Algorithm searches for a good CNN architecture and training configuration. We represent each candidate as a chromosome with nine genes. Its fitness rewards validation accuracy and penalises unnecessary trainable parameters. We compare the evolved model against the LeNet-style baseline and random search using the same candidate-evaluation budget. The React dashboard lets us test live handwriting and inspect the experiment evidence.”

## Suggested live sequence

1. Open Live Studio and click a built-in MNIST image. Explain that the ground-truth label is known because it comes from the held-out test split.
2. Point to the normalized 28 × 28 image. Explain grayscale conversion, contrast normalization, foreground cropping and centering.
3. Point to the prediction and top-three scores. Explain that softmax gives class scores; a close top-two gap means more ambiguity.
4. Draw the same digit manually and predict again. Then try a visually similar digit.
5. Show inference time, CPU/GPU route, trainable parameters and estimated MACs. Do not call MACs electrical power or claim that the app measures watts.
6. Open Experiment Lab. Explain the genetic convergence plot and compare the held-out test performance with the baseline and equal-budget random search.
7. Open Algorithm Guide. Explain chromosome genes, fitness, selection, crossover, mutation and elitism.
8. Open Research Notes. Explain the research-paper foundation, project contributions, the training/validation/test split and why the test split is not used by the Genetic Algorithm.

## Viva questions

**1. Which soft-computing techniques are combined?**

A Convolutional Neural Network and a Genetic Algorithm. The CNN learns the digit classifier; the GA searches the CNN architecture and hyper-parameters.

**2. What does a chromosome represent?**

A complete candidate design with nine genes: `c1`, `c2`, `kernel`, `fc1`, `fc2`, `activation`, `pooling`, `dropout` and `lr`.

**3. What is the fitness function?**

`validation_accuracy - 0.002 × (parameter_count / 100000)`. It balances classification performance and compactness.

**4. Why include a size penalty?**

When two candidate networks have similar validation accuracy, a smaller network is preferred because it uses fewer trainable parameters and generally requires less computation and memory.

**5. What is tournament selection?**

Randomly sample a small group of candidates and choose the one with the highest fitness as a parent.

**6. What is uniform crossover?**

For each gene, independently copy the value from parent A or parent B with equal probability.

**7. What is mutation?**

With the configured per-gene probability, replace a gene with another permitted value. This introduces diversity and helps avoid premature convergence.

**8. Why use elitism?**

Copy the best candidate unchanged into the next generation so the best-so-far candidate is not lost.

**9. Why compare against random search?**

To check whether the evolutionary operators help find a good candidate compared with blind sampling when both methods receive the same number of unique candidate evaluations.

**10. Why not let the GA use the test set?**

Using test accuracy during search leaks information from the final evaluation into model selection and can produce an overly optimistic estimate. The GA uses validation accuracy; the test split is reserved for final comparison.

**11. How does the CNN learn?**

It makes a prediction, computes cross-entropy loss, back-propagation calculates gradients, and Adam updates the weights to reduce the loss.

**12. Does the GA train the CNN weights?**

No. The GA chooses architecture and hyper-parameters. Back-propagation with Adam learns the CNN weights.

**13. How many possible candidate designs are in the search space?**

161,280 combinations from the nine genes.

**14. What does the model output?**

Ten softmax scores, one for each digit from 0 through 9. The highest score is the predicted class.

**15. Does the model always give the right digit?**

No. It is a learned classifier. Handwriting style, image quality and similarity between digits can cause errors. The dashboard shows top-three scores and the gap between the top two classes to help discuss ambiguity.

**16. What is different from the 1998 paper?**

The project uses a LeNet-style baseline and MNIST foundation, but it is a modern re-implementation rather than a bit-for-bit reproduction. It uses fully connected convolutional feature maps, average or max pooling, a softmax output with cross-entropy and Adam. The new project contribution is GA-based architecture/hyper-parameter search, size-aware fitness, an equal-budget random-search control and multi-seed final testing.

**17. What does the project measure as computational cost?**

Trainable parameter count, an estimated number of multiply-accumulate operations per image and measured inference latency. It reports the compute device detected by TensorFlow, but it does not measure electrical power in watts.

**18. What are the limitations?**

The layer pattern is fixed, the search space is limited to nine genes, the demo profile uses a smaller search budget, and MNIST contains isolated digits rather than full words or documents. Results can vary by random seed and hardware.

**19. What could be improved next?**

Evolve the number of layers, use multi-objective algorithms such as NSGA-II, add a fuzzy controller for learning rate, test on more difficult handwriting datasets, or add a carefully validated explanation method such as Grad-CAM.

## Report integrity reminder

Use actual values generated in `backend/artifacts/`. The `demo` profile is for interactive demonstration. Run the `full` profile before presenting the three-seed result table specified in the report template. Never report an expected accuracy as if it were a measured result.
