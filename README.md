# Visual-stimulus-decoding

## Predicting stimulus change events from calcium imaging recordings in mouse visual cortex
**Goal:** to investigate whether neural population activity in the Visual Cortex can predict when a visual stimulus change occurs during a behavioral task. Specifically, I aim to build a computational model that decodes stimulus changes from neuronal population activity recorded in mice performing a visual change-detection task.

**Core question:** Can the activity of neuronal populations be used to predict whether a visual stimulus has changed during a behavioral task?

Previous research in systems neuroscience has demonstrated that sensory information is encoded by distributed populations of neurons rather than single neurons. Population decoding approaches have been widely used to study how neural activity represents stimuli, decisions, and behavioral states. Advances in optical physiology techniques such as Two-photon calcium imaging allow simultaneous recording of activity from hundreds of neurons. These datasets enable computational modeling of neural population dynamics and decoding of sensory information. 

## Dataset Details
1. Method: Two-photon calcium imaging (ΔF/F)
2. Subjects: Multiple mice — change-detection task
3. Neurons: Hundreds per session (Exc + Inh)
4. Signal: ΔF/F fluorescence per neuron
5. Timepoints: Stimulus events ~250 ms apart
6. Access: AllenSDK (Python)

## Variables & Hypothesis
**Dependent Variable Y**

Binary: did stimulus change? (0 / 1)

1 = new image presented;

0 = same image repeated

One value per stimulus event

**Independent Variables X**

ΔF/F per neuron in ±500 ms window

Population activity vector [n₁…nₖ]

Cell type label (Exc / Inh)

Temporal features before stimulus

**Mathematical Hypothesis**

Model estimates:  P(Y = 1 | X)   ·   H₀: ROC-AUC ≈ 0.5 (population activity carries no information about stimulus change)   ·   H₁: ROC-AUC > 0.75

**Confounding Variables**
1) Task engagement / attention state
2) Between-animal variability
3) Session-to-session drift
4) Licking / motor artifacts


