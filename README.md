# Visual-stimulus-decoding

## Predicting stimulus change events from calcium imaging recordings in mouse visual cortex
**Goal:** to investigate whether neural population activity in the Visual Cortex can predict the novelty status of a visual stimulus — specifically, whether a presented image is being seen for the first time (novel) or has been seen before (familiar). To address this, I build a computational model that decodes novelty from neuronal population activity recorded in mice during passive visual stimulation.

**Core question:** Can the activity of neuronal populations be used to predict whether a visual stimulus is novel or familiar during passive viewing?

Previous research in systems neuroscience has demonstrated that sensory information is encoded by distributed populations of neurons rather than single neurons. Population decoding approaches have been widely used to study how neural activity represents stimuli, decisions, and behavioral states. Advances in optical physiology techniques such as two-photon calcium imaging allow simultaneous recording of activity from hundreds of neurons. These datasets enable computational modeling of neural population dynamics and decoding of sensory information, including signals related to stimulus familiarity and novelty — which are thought to reflect adaptation and predictive coding mechanisms in visual cortex.

## Dataset Details
1. Method: Two-photon calcium imaging (ΔF/F)
2. Subjects: Multiple mice — passive visual stimulation
3. Neurons: 85 per session used
4. Signal: ΔF/F fluorescence per neuron
5. Timepoints: Stimulus events ~250 ms apart
6. Access: AllenSDK (Python)

## Variables & Hypothesis
**Dependent Variable Y**

Binary: is this image novel?  (0 / 1)

1 = first presentation of this image (novel)

0 = image has been seen before (familiar)

One value per stimulus event

**Independent Variables X**

ΔF/F per neuron in ±500 ms window

Population activity vector [n₁…nₖ]

**Mathematical Hypothesis**

Model estimates:  P(Y = 1 | X)   ·   H₀: ROC-AUC ≈ 0.5 ·   H₁: ROC-AUC > 0.75

**Confounding Variables**
1) Between-animal variability
2) Session-to-session drift


