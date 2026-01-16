# Marty's role as a robot

- [x] Embodiment
- [ ] 
- [ ] Pose demonstration
- [ ] Moving the body part to communicate corrections (non verbal)
  - [ ] Arms (jokes on elbow)
  - [ ] Legs (careful to keep balance)
  - [ ] Back (bend forward/backward)
  - [ ] Ankles (twist left/right)


# Marty's role as a coach
- [ ] Bonus: Personalized user (tired, improving fast, remembering user's name)

# Global architecture
```mermaid
graph TD
    A[Camera] --> B[Mediapipe landmarks]
    B --> C{Poses analysis algorithm}
    C -- Every frame --> G[Video feedback]
    C -- Incorrect for 5 sec --> D[LLM Corrective Feedback]
    C -- Correct for 30 sec --> H[Pose feedback]
    H --> E
    D --> E[TTS]
    E --> F[Speaker]
```

## Using Mediapipe
- [x] Getting landmarks from video feed.
- [ ] Bigger screen

## Poses analysis algorithm
Video feedback
- [x] Defining target poses with thresholds
- [x] Analyzing incoming landmarks against target poses
- [x] Coloring landmarks and joints
- [ ] Déclencher corrective feedback when incorrect for more than 5 seconds
- [ ] Déclencher long feedback when correct for more than 30 seconds
- [ ] Bonus: Using 3D landmarks

## Corrective LLM Feedback
- [ ] Use non verbal marty communication
- [ ] Bonus: Different prompt regarding our prior situation analysis (e.g. very high error)
- [ ] Prerecord voice and marty poses for common corrections (e.g. arms too low, back not straight)

## LLM Feedback
- [ ] Prompt engineering
- [ ] Bonus: {user name} {number of correction done} {time spent}

## TTS
- [x] TTS macos instant
- [ ] Bonus: Emotional TTS like CosyVoice

# Optimizations

Using 25 sec to verify a pose and using the last 5 to start generating a thoughtful feedback (thinking mode).

While he's speaking, we can capture the 5 last seconds to see if we need to say something about it or not.

Bonus: Générer une séance de yoga personalisée et adaptées.

Joint: L-Elbow, Angle: 156
Joint: R-Elbow, Angle: 170
Joint: L-Knee, Angle: 148
Joint: R-Knee, Angle: 132
Joint: L-Hip, Angle: 168
Joint: R-Hip, Angle: 139
Joint: L-Elbow, Angle: 157
Joint: R-Elbow, Angle: 168
Joint: L-Knee, Angle: 144
Joint: R-Knee, Angle: 128
Joint: L-Hip, Angle: 167
Joint: R-Hip, Angle: 139
