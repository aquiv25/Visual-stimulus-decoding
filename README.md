# Visual-stimulus-decoding

## Predicting stimulus change events from calcium imaging recordings in mouse visual cortex
**Goal:** to investigate whether neural population activity in the Visual Cortex can predict when a visual stimulus change occurs during an active behavioral task. To address this, I build a computational model that decodes stimulus changes from neuronal population activity recorded in mice performing a visual change-detection task.

**Core question:** Can the activity of neuronal populations be used to predict whether a visual stimulus has changed during a behavioral task?

Previous research in systems neuroscience has demonstrated that sensory information is encoded by distributed populations of neurons rather than single neurons. Population decoding approaches have been widely used to study how neural activity represents stimuli, decisions, and behavioral states. Advances in optical physiology techniques such as two-photon calcium imaging allow simultaneous recording of activity from hundreds of neurons. These datasets enable computational modeling of neural population dynamics and decoding of sensory information, including signals related to stimulus change detection — which are thought to reflect surprise responses and decision-related activity in visual cortex.

## Dataset Details
1) Method: Two-photon calcium imaging (ΔF/F)
2) Subjects: Multiple mice — active change-detection task
3) Neurons: 666 per session used
4) Signal: ΔF/F fluorescence per neuron
5) Timepoints: Image flashes every ~750 ms
6) Access: AllenSDK (Python) — Allen Visual Behavior Ophys Dataset

## Variables & Hypothesis
**Dependent Variable Y**

Binary: did the stimulus change? (0 / 1)

1 = new image presented (change trial)

0 = same image repeated (no-change trial)

One value per stimulus flash

**Independent Variables X**

ΔF/F per neuron in [onset, onset + 500 ms] window

Population activity vector [n₁…nₖ]

**Mathematical Hypothesis**

Model estimates:  P(Y = 1 | X)   ·   H₀: ROC-AUC ≈ 0.5 ·   H₁: ROC-AUC > 0.75

**Confounding Variables**
1) Between-animal variability
2) Task engagement / attention state
3) Session-to-session drift
4) Licking / motor artifacts


