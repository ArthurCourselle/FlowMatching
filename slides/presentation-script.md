# Flow Matching Presentation Script (15-20 Minutes)

> **Note to Speaker**: This script is designed for a ~18 minute presentation. Pause frequently, engage with the slides, and point to the equations. The text in *italics* are stage directions.

---

## Slide 1: Title Slide (0:00 - 1:00)
**Speaker:**
"Good morning everyone. Today we are presenting our implementation and analysis of **Flow Matching**."
"This project was an opportunity for us to dive into the state-of-the-art of Generative Modeling. Our goal was not just to use written libraries, but to rebuild the method from scratch on 2D toy datasets to grasp the fundamental mathematics."
"We will first explain why this method exists, then the rigorous 'hard' maths behind it, and finally our implementation and results."

---

## Slide 2: Generative Modeling Landscape (1:00 - 3:00)
**Speaker:**
"Let's start with the context. We want to model a data distribution $p_{data}$—think of images or simply points on a 2D plane."
"Historically, we had GANs (Adversarial) which are unstable, and VAEs which often produce blurry results."
"Recently, **Diffusion Models** took over. They destroy data with noise and learn to reverse the process. They are amazing but have a flaw: the sampling is a stochastic process that mimics a Stochastic Differential Equation (SDE). It requires many steps, making it slow."
"**Flow Matching** is the answer to this. It keeps the quality of diffusion but uses **Deterministic** ODEs with straight trajectories. This means we can sample much faster."

---

## Slide 3: Intuition - The Vector Field (3:00 - 4:30)
**Speaker:**
"Before the maths, let's build an intuition."
"Imagine the data distribution as a crowd of people. We want to move them from a random formation (Noise) to a target shape (Data)."
"Instead of them walking randomly (Diffusion), imagine we blow a wind—a **vector field** $v_t$—that pushes everyone exactly where they need to go."
"If we know this wind velocity at every point and time, we just place a particle in the noise, and it flows deterministically to the data."
"Flow Matching is simply: **Training a Neural Network to predict this wind velocity.**"

---

## Slide 4: The Continuity Equation (4:30 - 6:30)
**Speaker:**
"Now, let's get rigorous. This is the part our teacher wants us to be precise about."
"We define a **Probability Path** $p_t$. This is a sequence of distributions starting at $p_0$ (Noise) and ending at $p_1$ (Data)."
"Any time-dependent vector field $v_t$ generates a flow $\phi_t$. If we move points along this flow, their density changes."
"The relationship between the changing density and the vector field is governed by the **Continuity Equation**:
$$ \frac{\partial p_t}{\partial t} + \nabla \cdot (p_t v_t) = 0 $$
"This equation simply says: Mass is conserved. If density decreases here, it must have flowed somewhere else."

---

## Slide 5: The Objective (6:30 - 8:00)
**Speaker:**
"So, we want to find $v_\theta$ (our model) that matches the 'true' field $u_t$."
"Ideally, we would minimize the regression loss:
$$ \mathcal{L} = || v_\theta - u_t ||^2 $$
"But here is the **fundamental problem**: We don't know the true field $u_t$! We don't even know the intermediate densities $p_t$. We only have samples from the end (Data) and the beginning (Noise)."
"We cannot train on something we don't know. This is where the magic happens."

---

## Slide 6: Conditional Flow Matching (CFM) (8:00 - 10:00)
**Speaker:**
"The solution proposed by Lipman et al. (2023) is **Conditional Flow Matching**."
"The idea is: 'The global crowd movement is complex, but the path of a *single* person is simple'."
"We assume the total probability path is an average of many simple paths, each conditioned on a specific data point $x_1$."
"The **CFM Theorem** proves a powerful result:
> If we match the vector field of these simple conditional paths, we essentially match the correct global field."
"This changes everything. Now we have a target we can compute!"

---

## Slide 7: Defining the Path (Optimal Transport) (10:00 - 12:00)
**Speaker:**
"Now we need to pick that 'simple conditional path'. We chose the **Optimal Transport** path."
"Mathematically, it's just a linear interpolation between a noise sample $x_0$ and a data sample $x_1$:
$$ x_t = (1-t)x_0 + t x_1 $$
"Why is this brilliant?"
"1. The trajectory is a straight line."
"2. The velocity (derivative) is just $x_1 - x_0$. It is **constant in time** for that pair."
"This provides an incredibly stable target for our neural network. It doesn't have to learn a curving, chaotic path—just the average of many straight lines."

---

## Slide 8: Implementation - Training Algorithm (12:00 - 13:30)
**Speaker:**
"Let's look at how we implemented this."
"The training loop is surprisingly simple:"
"1. We sample a batch of data $x_1$ and noise $x_0$."
"2. We sample a time $t$ between 0 and 1."
"3. We interpolate to get $x_t$."
"4. We calculate the target direction $u_t = x_1 - x_0$."
"5. We train the network to predict this direction from $(x_t, t)$."
"It is a standard regression problem. No adversarial training, no ELBO."

---

## Slide 9: Implementation - Architecture (13:30 - 14:30)
**Speaker:**
"For the network, we used a Multi-Layer Perceptron (MLP)."
"A crucial detail is how we feed the time $t$."
"We use **Sinusoidal Time Embeddings**, similar to Transformers."
"This is vital because the vector field changes over time. At $t=0$, the flow is about moving away from noise. At $t=1$, it's about forming fine details."
"The embeddings allow the simple MLP to be 'time-aware' and highly expressive."

---

## Slide 10: Implementation - Euler Solving (14:30 - 15:30)
**Speaker:**
"Once trained, generating data is just solving the ODE."
"We start with random noise."
"We use the **Euler Method** to step forward in time."
"Because we forced the paths to be straight during training (Optimal Transport), the learned field is very smooth."
"This allows us to take large steps. We can generate high-quality samples in just 20 to 100 steps."

---

## Slide 11: Quantitative & Qualitative Results (15:30 - 17:00)
**Speaker:**
*(Point to the reconstruction images)*
"Here you can see the results on the 'Checkerboard' and 'Circles' datasets."
"On the left is the noise. On the right is the generated data."
"Notice how the points don't just 'appear'—they flow into position."
"We visualized the vector field in the background. You can clearly see it pointing towards the nearest mode, as expected."

---

## Slide 12: Comparison with RealNVP (17:00 - 18:00)
**Speaker:**
"We compared our approach to RealNVP, a classic Normalizing Flow."
"RealNVP uses invertible layers. It's fast (1 step) but constrained in expressivity."
"Flow Matching is a 'Free-form' ODE. It can model any topology."
"While RealNVP struggles with disconnected modes (like the checkerboard) without complex engineering, Flow Matching separates them naturally."
"The trade-off is inference speed: RealNVP is instantaneous, Flow Matching requires the Euler loop."

---

## Slide 13: Conclusion (18:00 - 19:00)
**Speaker:**
"In conclusion, Flow Matching represents a paradigm shift."
"- It simplifies Generative Modeling to a **Conditional Regression** problem."
"- It unifies Diffusion and Flows."
"- The use of **Optimal Transport** paths makes it efficient."
"We successfully implemented this from scratch and verified these properties on 2D data."
"Thank you. We are ready for your questions."
