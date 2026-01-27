# Flow Matching Presentation Script (15-20 Minutes)

> **Note to Speaker**: This script is designed for a ~18 minute presentation. Pause frequently, engage with the slides, and point to the equations. The text in *italics* are stage directions.

---

## Outline

1. **Introduction & Context**
   - Generative Modeling: The Landscape
   - The Diffusion Paradigm vs. Flow Matching
   - Intuition: The Vector Field
2. **Mathematical Formalism**
   - From Static to Dynamic: The Flow
   - The Objective: Matching the Vector Field
   - Conditional Flow Matching (CFM)
   - Defining the Conditional Path
3. **Implementation**
   - Sampling: Solving the ODE
4. **Results**
   - Qualitative Results (2D)
   - Qualitative Results: CIFAR-10
5. **Comparison & Conclusion**
   - Flow Matching vs RealNVP
   - Thank You

---

## Slide 1: Title Slide (0:00 - 0:30)
**Speaker:**
"Good morning everyone. Today we are presenting our implementation and analysis of **Flow Matching**."
"Our goal was not just to use existing libraries, but to rebuild the method from scratch to grasp the fundamental mathematics."

---

## Slide 2: Outline (0:30 - 1:00)
**Speaker:**
"Here is our agenda. We'll start with the context and intuition, then dive into the mathematical formalism, show our implementation, present results, and finally compare with RealNVP."

---

## Slide 3: Generative Modeling: The Landscape (1:00 - 2:30)
**Speaker:**
"Let's start with the context. Our goal in generative modeling is to learn the data distribution $p_{data}$ from samples."
"Historically, we had **GANs** which are adversarial and often unstable, and **VAEs** which produce approximate but often blurry results."
"**Diffusion Models** took over recently. They achieve high quality but have a flaw: the sampling process requires many steps, making it slow. They also rely on a 'destruction' process—adding noise to data."

---

## Slide 4: The Diffusion Paradigm vs. Flow Matching (2:30 - 4:00)
**Speaker:**
*(Point to the comparison image)*
"This slide illustrates the key difference."
"Both approaches map noise to data, but they do it differently."
"Diffusion models use a 'destructive' process—they learn to reverse noise addition."
"Flow Matching uses a 'constructive' process—it directly learns the velocity field that transports noise to data."
"This leads to straighter trajectories and faster sampling."

---

## Slide 5: Intuition - The Vector Field (4:00 - 5:30)
**Speaker:**
"Before the maths, let's build intuition."
"Imagine moving points from a simple distribution (Noise) to a complex one (Data)."
"We define a **velocity field** $v_t(x)$ that tells each point where to go at each time step."
"By following these arrows, we morph noise into data."
"Flow Matching is simply: **Training a Neural Network to predict this velocity.**"

---

## Slide 6: From Static to Dynamic: The Flow (5:30 - 7:00)
**Speaker:**
"Now let's formalize this."
"Instead of predicting noise like Diffusion models, we predict **velocity**."
"We consider a probability density $p_t(x)$ moving over time."
"At $t=0$, we have Noise $\mathcal{N}(0, I)$. At $t=1$, we have Data $p_{data}$."
"This movement defines a **Flow** $\phi_t(x)$—the path of a particle."
"The ODE governing this is: $\frac{d}{dt}\phi_t(x) = v_t(\phi_t(x))$."

---

## Slide 7: The Objective: Matching the Vector Field (7:00 - 8:30)
**Speaker:**
"If we knew the ground truth vector field $u_t$, we could just regress it with a simple MSE loss."
"But here is the **fundamental problem**: We don't know $u_t$! We don't even know the intermediate densities $p_t$."
"We only have samples from the end (Data) and can sample the beginning (Noise)."
"We cannot train on something we don't know. This is where the magic happens."

---

## Slide 8: Conditional Flow Matching (CFM) (8:30 - 10:30)
**Speaker:**
"The solution proposed by Lipman et al. (2023) is **Conditional Flow Matching**."
"The key insight is: 'The global flow is complex, but the path conditioned on a *single* data point is simple'."
"We assume the total probability path is a mixture of simple paths, each conditioned on a data point $x_1$."
"The **CFM Theorem** proves that minimizing the loss w.r.t. the conditional field is equivalent to minimizing the global FM loss."
"This changes everything—now we have a target we can compute!"

---

## Slide 9: Defining the Conditional Path (10:30 - 12:00)
**Speaker:**
"Now we need to define that simple conditional path. We choose the **Optimal Transport** path—a straight line."
"Mathematically: $x_t = (1-t)x_0 + t x_1$, where $x_0$ is noise and $x_1$ is data."
"The corresponding vector field is simply: $u_t(x|x_1) = x_1 - x_0$."
"Why is this brilliant?"
"1. **Simplicity**: Constant velocity for each pair."
"2. **Efficiency**: Straight paths mean we can take larger steps during sampling."
"3. **Stability**: The regression target is bounded and well-behaved."

---

## Slide 10: Sampling - Solving the ODE (12:00 - 14:00)
**Speaker:**
*(Point to the code)*
"Once trained, generating data is just solving an ODE."
"We sample noise $x_0$, then use **Euler integration** from $t=0$ to $t=1$."
"The update is simple: $x_{t+dt} = x_t + v_\theta(x_t, t) \cdot dt$."
"Because we trained with straight OT paths, the learned field is smooth."
"This allows us to generate high-quality samples in just 20 to 100 steps."

---

## Slide 11: Qualitative Results - 2D (14:00 - 15:00)
**Speaker:**
*(Point to the reconstruction plot)*
"Here you can see results on 2D datasets."
"The model successfully learned the data distribution."
"Notice the trajectories are straight lines—this is the Optimal Transport at work."

---

## Slide 12: Qualitative Results - CIFAR-10 (15:00 - 16:00)
**Speaker:**
*(Point to the CIFAR-10 images)*
"We also scaled to high-dimensional data using a **U-Net** backbone."
"The model generates sharp samples by following the learned velocity field."
"This demonstrates Flow Matching works beyond toy 2D datasets."

---

## Slide 13: Flow Matching vs RealNVP (16:00 - 17:30)
**Speaker:**
"Let's compare with RealNVP, a classic Normalizing Flow."
"**RealNVP** uses constrained invertible architectures. It's fast—single pass sampling—but has limited expressivity."
"**Flow Matching** uses free-form architectures. It trains with simple MSE regression. Sampling requires ODE solving (iterative), but it's a universal approximator."
"The conclusion: Flow Matching offers higher quality and simpler training, at the cost of slower inference."

---

## Slide 14: Thank You (17:30 - 18:00)
**Speaker:**
"Thank you for your attention."
"Our code is available on GitHub. We are ready for your questions."
