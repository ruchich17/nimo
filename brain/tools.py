"""Tool schemas exposed to Gemma. Same list drives sim and real backends."""

TOOLS = [
    {"type": "function", "function": {
        "name": "roll",
        "description": "Roll the dough with one stroke of the roller through its centre. "
                       "The dough spreads out along the stroke direction.",
        "parameters": {"type": "object", "properties": {
            "direction_deg": {"type": "number",
                              "description": "Stroke axis as seen in the image, 0 to 180. "
                                             "0 = left-right, 90 = up-down, "
                                             "45 = bottom-left to top-right, 135 = top-left to bottom-right. "
                                             "The arm can only roll 60 to 120 (roughly up-down)."},
            "pressure": {"type": "number", "description": "0.1 (light) to 0.8 (firm)"},
            "length_cm": {"type": "number", "description": "Stroke length in cm, 3 to 15"},
        }, "required": ["direction_deg", "pressure", "length_cm"]}}},
    {"type": "function", "function": {
        "name": "rotate_dough",
        "description": "Ask a person to turn the dough on the board, for strokes the arm cannot reach. "
                       "Positive = counter-clockwise.",
        "parameters": {"type": "object", "properties": {
            "deg": {"type": "number", "description": "Rotation in degrees, -90 to 90"},
        }, "required": ["deg"]}}},
    {"type": "function", "function": {
        "name": "dust_flour",
        "description": "Dust flour on the dough when it looks sticky or is tearing.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "inspect",
        "description": "Re-capture the camera image and look again, e.g. if the view is unclear.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "done",
        "description": "Finish: the dough is round and evenly thin.",
        "parameters": {"type": "object", "properties": {
            "reason": {"type": "string"}}, "required": ["reason"]}}},
    {"type": "function", "function": {
        "name": "stop",
        "description": "Emergency stop: something is wrong (no dough visible, hand in view, dough torn).",
        "parameters": {"type": "object", "properties": {
            "reason": {"type": "string"}}, "required": ["reason"]}}},
]

TOOL_NAMES = {t["function"]["name"] for t in TOOLS}
