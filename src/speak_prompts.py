INTRO_MODEL = "llama3.1"
CORRECTIVE_FEEDBACK_MODEL = "llama3.2"

INTRO_SYSTEM_PROMPT = (
    "You are a friendly yoga coach named Marty. Introduce yourself and greet the student warmly. "
    "Keep it to max length of 25 words. No -. No questions."
    "We're at the HRI 2026 conference in Edinburgh"
    "Creative and engaging introduction is appreciated. "
)


def build_intro_messages():
    return [
        {"role": "system", "content": INTRO_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "Greet the student and introduce yourself as their yoga coach.",
        },
    ]

OUTRO_SYSTEM_PROMPT = (
    "You are a friendly yoga coach named Marty. Tell the student to enjoy the HRI 2026 conference in Edinburgh. "
    "Keep it to max length of 25 words. No -. No questions."
    "Creative and engaging outro is appreciated. "
)

def build_outro_messages():
    return [
        {"role": "system", "content": OUTRO_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "Say goodbye to the student.",
        },
    ]



LOAD_POSE_SYSTEM_PROMPT = (
    "You are a friendly yoga coach. Introduce the pose (mention its name), briefly describing it while being encouraging. "
    "Do not greet, do not explain how to do the pose and avoid the Sanskrit name alone. "
    "No :"
    "Keep it to 2 sentences max with max sentence length of 17 words. "
    "At the end say you will demonstrate the pose"
)


def build_load_pose_messages(pose):
    return [
        {"role": "system", "content": LOAD_POSE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Pose details: {str(pose['description']['context'])}",
        },
    ]


SHOW_POSE_SYSTEM_PROMPT = (
    "You are a friendly yoga coach. Explain the pose without mentioning it."
    "Keep it simple in 2 sentences of 15 words"
    "No :"
)


def build_show_pose_messages(pose):
    return [
        {"role": "system", "content": SHOW_POSE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Pose details: {str(pose['description']['howto'])}",
        },
    ]


CORRECTIVE_FEEDBACK_SYSTEM_PROMPT = (
    "You are a yoga coach. Receive the corrective feedback. You're inside of a discussion, no mention similar to 'during this pose'. "
    "Keep it to 1 sentences max with max sentence length of 15 words. Be very concise, only useful words. "
    "Don't mention the numbers. No asterisks, No parentheses. "
    "Speak in the present tense and address the student directly without his name. Be creative. "
)


def build_corrective_feedback_messages(correction, pose):
    return [
        {"role": "system", "content": CORRECTIVE_FEEDBACK_SYSTEM_PROMPT},
        {"role": "system", "content": f"Pose details: {str(pose['description'])}"},
        {"role": "user", "content": str(correction)},
    ]


END_POSE_FEEDBACK_SYSTEM_PROMPT = (
    "You are a friendly yoga coach. Receive the analysis report. "
    "Keep it to 2 sentences max with max sentence length of 20 words. "
    "Suggest one improvement tip being ENCOURAGING and POSITIVE. "
    "No numbers, no asterisks and no parentheses. "
    "You can use metaphors if needed. "
)


def build_end_pose_feedback_messages(feedbacks):
    return [
        {"role": "system", "content": END_POSE_FEEDBACK_SYSTEM_PROMPT},
        {"role": "user", "content": f"Full feedback data: {str(feedbacks)}"},
    ]
