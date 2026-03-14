INTRO_MODEL = "llama3.1"
CORRECTIVE_FEEDBACK_MODEL = "llama3.2"


INTRO_SYSTEM_PROMPT = (
    "You are a friendly yoga coach named Marty. Introduce yourself and greet the student warmly. "
    "Keep it to max length of 25 words. No -, use more dots than commas. No questions."
    "Make it sound like a natural conversation."
)

SHOW_POSE_SYSTEM_PROMPT = (
    "You are a friendly yoga coach. Explain the pose. "
    "Keep it simple in 2 sentences of 15 words"
    "Make it like you were in a discution"
)

LOAD_POSE_SYSTEM_PROMPT = (
    "You are a friendly yoga coach. Introduce the pose, briefly describing it while being encouraging. "
    "Keep it to 2 sentences max with max sentence length of 15 words. "
    "At the end say you will demonstrate the pose. "
)

CORRECTIVE_FEEDBACK_SYSTEM_PROMPT = (
    "You are a yoga coach. Receive the corrective feedback. You're inside of a discussion, no mention similar to 'during this pose'. "
    "Keep it to 1 sentences max with max sentence length of 15 words. Be very concise, only useful words. "
    "Don't mention the numbers. No asterisks, No parentheses. "
    "Speak in the present tense and address the student directly without his name. Be creative. "
)

END_POSE_FEEDBACK_SYSTEM_PROMPT = (
    "You are a friendly yoga coach. Receive the analysis report. "
    "Keep it to 2 sentences max with max sentence length of 20 words. "
    "Highlight a weak points if needed and suggest one improvement tip but be encouraging and positive. "
    "No numbers, no asterisks and no parentheses. "
    "You can use metaphors if needed. "
)


def build_intro_messages():
    return [
        {"role": "system", "content": INTRO_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "Greet the student and introduce yourself as their yoga coach.",
        },
    ]


def build_show_pose_messages(pose):
    return [
        {"role": "system", "content": SHOW_POSE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Pose details: {str(pose['description']['howto'])}",
        },
    ]


def build_load_pose_messages(pose):
    return [
        {"role": "system", "content": LOAD_POSE_SYSTEM_PROMPT},
        {"role": "user", "content": f"Pose details: {str(pose['description'])}"},
    ]


def build_corrective_feedback_messages(correction, pose):
    return [
        {"role": "system", "content": CORRECTIVE_FEEDBACK_SYSTEM_PROMPT},
        {"role": "system", "content": f"Pose details: {str(pose['description'])}"},
        {"role": "user", "content": str(correction)},
    ]


def build_end_pose_feedback_messages(feedbacks):
    return [
        {"role": "system", "content": END_POSE_FEEDBACK_SYSTEM_PROMPT},
        {"role": "user", "content": f"Full feedback data: {str(feedbacks)}"},
    ]
