Phase 1: Extract Renderer into window.py and keep API identical.
Phase 2: Extract FeedbackEngine from HeadMaster ongoing correction methods.
Phase 3: Introduce SessionState + InteractionState-driven transitions.
Phase 4: Thin HeadMaster to orchestration only and simplify main loop.
Phase 5: Add tests for pure logic and one smoke integration test for session flow.
Expected Outcome

Smaller classes with clear ownership.
Easier debugging (state transitions explicit).
Safer iteration on voice/robot/vision independently.
Better onboarding for contributors.
