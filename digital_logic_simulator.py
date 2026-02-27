import cv2
import numpy as np
import time
import sys

# MediaPipe may not be installed or might lack the 'solutions' submodule in some
# environments.  Wrap the import so the rest of the program can at least start.
try:
    import mediapipe as mp
    if not hasattr(mp, 'solutions'):
        # some pip installations (especially on Windows) expose mediapipe without
        # the solutions subpackage, which we require for hand tracking.
        raise ImportError("mediapipe module has no attribute 'solutions'")
except Exception as e:
    print(f"Warning: unable to import MediaPipe ({e}), hand detection will be disabled.")
    mp = None

# ================= CONFIGURATION =================
CAM_W, CAM_H = 900, 600
SIDEBAR_W = 400
HAND_CONFIDENCE = 0.7

# ================= GLOBAL STATE =================
class AppState:
    def __init__(self):
        self.mode = "LOGIC"
        self.op_idx = 0
        self.A = 0
        self.B = 0
        self.Cin = 0
        self.shift_reg = [0, 0, 0, 0]
        self.shift_buffer = []             # pending typed bits
        self.shift_gesture_active = False  # debounce
        # landmarks toggle
        self.show_landmarks = False
        self.sr_q = 0
        self.jk_q = 0
        self.d_q = 0
        self.t_q = 0

state = AppState()

# ================= LOGIC GATES =================
def AND(a, b):
    return a & b

def OR(a, b):
    return a | b

def NOT(a):
    return 1 - a

def NAND(a, b):
    return 1 - (a & b)

def NOR(a, b):
    return 1 - (a | b)

def XOR(a, b):
    return a ^ b

def XNOR(a, b):
    return 1 - (a ^ b)

# ================= ARITHMETIC CIRCUITS =================
def half_adder(a, b):
    s = a ^ b
    c = a & b
    return s, c

def full_adder(a, b, cin):
    s = a ^ b ^ cin
    cout = (a & b) | (cin & (a ^ b))
    return s, cout

def half_subtractor(a, b):
    d = a ^ b
    borrow = (1 - a) & b
    return d, borrow

def full_subtractor(a, b, bin_in):
    d = a ^ b ^ bin_in
    bout = ((1 - a) & b) | (bin_in & ((a ^ b) == 0))
    return d, bout

# ================= FLIP-FLOPS =================
def sr_flipflop(s, r, q_current):
    if s == 1 and r == 1:
        return q_current
    elif s == 1:
        return 1
    elif r == 1:
        return 0
    else:
        return q_current

def jk_flipflop(j, k, q_current):
    if j == 0 and k == 0:
        return q_current
    elif j == 1 and k == 0:
        return 1
    elif j == 0 and k == 1:
        return 0
    else:
        return 1 - q_current

def d_flipflop(d, q_current):
    return d

def t_flipflop(t, q_current):
    if t == 1:
        return 1 - q_current
    else:
        return q_current

# ================= TITLE TABLES =================
LOGIC_GATES = {
    0: {
        "name": "AND Gate",
        "table": ["AND GATE", "A B | OUT", "0 0 |  0", "0 1 |  0", "1 0 |  0", "1 1 |  1"],
        "func": AND
    },
    1: {
        "name": "OR Gate",
        "table": ["OR GATE", "A B | OUT", "0 0 |  0", "0 1 |  1", "1 0 |  1", "1 1 |  1"],
        "func": OR
    },
    2: {
        "name": "NOT Gate",
        "table": ["NOT GATE", "A | OUT", "0 |  1", "1 |  0"],
        "func": lambda a: NOT(a)
    },
    3: {
        "name": "NAND Gate",
        "table": ["NAND GATE", "A B | OUT", "0 0 |  1", "0 1 |  1", "1 0 |  1", "1 1 |  0"],
        "func": NAND
    },
    4: {
        "name": "NOR Gate",
        "table": ["NOR GATE", "A B | OUT", "0 0 |  1", "0 1 |  0", "1 0 |  0", "1 1 |  0"],
        "func": NOR
    },
    5: {
        "name": "XOR Gate",
        "table": ["XOR GATE", "A B | OUT", "0 0 |  0", "0 1 |  1", "1 0 |  1", "1 1 |  0"],
        "func": XOR
    },
    6: {
        "name": "XNOR Gate",
        "table": ["XNOR GATE", "A B | OUT", "0 0 |  1", "0 1 |  0", "1 0 |  0", "1 1 |  1"],
        "func": XNOR
    }
}

ARITHMETIC = {
    0: {
        "name": "Half Adder",
        "table": ["HALF ADDER", "A B | S C", "0 0 | 0 0", "0 1 | 1 0", "1 0 | 1 0", "1 1 | 0 1"],
        "func": half_adder
    },
    1: {
        "name": "Full Adder",
        "table": ["FULL ADDER", "A B Cin | S Cout", "0 0 0  | 0 0", "0 0 1  | 1 0", "0 1 0  | 1 0", "0 1 1  | 0 1", "1 0 0  | 1 0", "1 0 1  | 0 1", "1 1 0  | 0 1", "1 1 1  | 1 1"],
        "func": full_adder
    },
    2: {
        "name": "Half Subtractor",
        "table": ["HALF SUBTRACTOR", "A B | D B", "0 0 | 0 0", "0 1 | 1 1", "1 0 | 1 0", "1 1 | 0 0"],
        "func": half_subtractor
    },
    3: {
        "name": "Full Subtractor",
        "table": ["FULL SUBTRACTOR", "A B Bin | D Bout", "0 0 0  | 0 0", "0 0 1  | 1 1", "0 1 0  | 1 1", "0 1 1  | 0 1", "1 0 0  | 1 0", "1 0 1  | 0 0", "1 1 0  | 0 0", "1 1 1  | 1 1"],
        "func": full_subtractor
    }
}

FLIPFLOPS = {
    0: {
        "name": "SR Flip-Flop",
        "table": ["SR FLIP-FLOP", "S R | Q", "0 0 | Qn", "0 1 | 0", "1 0 | 1", "1 1 | X"],
        "func": sr_flipflop
    },
    1: {
        "name": "JK Flip-Flop",
        "table": ["JK FLIP-FLOP", "J K | Q", "0 0 | Qn", "0 1 | 0", "1 0 | 1", "1 1 | Q'"],
        "func": jk_flipflop
    },
    2: {
        "name": "D Flip-Flop",
        "table": ["D FLIP-FLOP", "D | Q", "0 | 0", "1 | 1"],
        "func": d_flipflop
    },
    3: {
        "name": "T Flip-Flop",
        "table": ["T FLIP-FLOP", "T | Q", "0 | Qn", "1 | Q'"],
        "func": t_flipflop
    }
}

# SHIFTREG dictionary removed; shift-register logic is handled manually by new helpers

# ================= HAND DETECTION =================
def detect_hand(hands, frame):
    """Process hand detection and return input values."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    A, B, Cin = 0, 0, 0
    # input_bit no longer used
    # initialize thumb and hand info
    # new outputs for shift logic
    thumb_up = False
    hand_side = None

    fist = False
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label
            
            # Extract key landmarks
            thumb_tip = hand_landmarks.landmark[4]
            thumb_ip = hand_landmarks.landmark[3]
            index_tip = hand_landmarks.landmark[8]
            index_pip = hand_landmarks.landmark[6]
            middle_tip = hand_landmarks.landmark[12]
            middle_pip = hand_landmarks.landmark[10]
            ring_tip = hand_landmarks.landmark[16]
            ring_pip = hand_landmarks.landmark[14]
            pinky_tip = hand_landmarks.landmark[20]
            pinky_pip = hand_landmarks.landmark[18]
            
            is_index_up = index_tip.y < index_pip.y
            is_middle_up = middle_tip.y < middle_pip.y
            
            # Assign inputs based on hand
            if label == "Right":
                A = 1 if is_index_up else 0
                Cin = 1 if is_middle_up else 0
            if label == "Left":
                B = 1 if is_index_up else 0
            
            # detect fist (all four fingers down)
            if (index_tip.y > index_pip.y and middle_tip.y > middle_pip.y
                and ring_tip.y > ring_pip.y and pinky_tip.y > pinky_pip.y):
                fist = True
            
            # new shift register gesture detection
            if state.mode == "SHIFTREG":
                thumb_up = thumb_tip.y < thumb_ip.y
                xs = [lm.x for lm in hand_landmarks.landmark]
                hand_side = "RIGHT" if np.mean(xs) > 0.5 else "LEFT"
    
    return A, B, Cin, thumb_up, hand_side, fist

# ================= PROCESS LOGIC GATES =================
def process_logic_gates():
    """Calculate logic gate output."""
    gate = LOGIC_GATES[state.op_idx]
    
    if state.op_idx == 2:  # NOT gate
        return gate['func'](state.A)
    else:
        return gate['func'](state.A, state.B)

# ================= PROCESS ARITHMETIC =================
def process_arithmetic():
    """Calculate arithmetic circuit output."""
    arith = ARITHMETIC[state.op_idx]
    
    if state.op_idx in [0, 2]:  # Half operations
        return arith['func'](state.A, state.B)
    else:  # Full operations
        return arith['func'](state.A, state.B, state.Cin)

# ================= PROCESS FLIP-FLOPS =================
def process_flipflop():
    """Update flip-flop state."""
    if state.op_idx == 0:
        state.sr_q = sr_flipflop(state.A, state.B, state.sr_q)
        return state.sr_q
    elif state.op_idx == 1:
        state.jk_q = jk_flipflop(state.A, state.B, state.jk_q)
        return state.jk_q
    elif state.op_idx == 2:
        state.d_q = d_flipflop(state.A, state.d_q)
        return state.d_q
    elif state.op_idx == 3:
        state.t_q = t_flipflop(state.A, state.t_q)
        return state.t_q

# ================= PROCESS SHIFT REGISTER =================
def process_shift_register(thumb_up, hand_side):
    """Perform a single shift when thumb is up on the given side.
    thumb_up: bool, hand_side: 'LEFT' or 'RIGHT' or None
    Uses state.shift_gesture_active for debounce."""
    if state.mode != "SHIFTREG":
        return
    if thumb_up and hand_side == "RIGHT" and not state.shift_gesture_active:
        state.shift_reg = [0] + state.shift_reg[:-1]
        state.shift_gesture_active = True
    elif thumb_up and hand_side == "LEFT" and not state.shift_gesture_active:
        state.shift_reg = state.shift_reg[1:] + [0]
        state.shift_gesture_active = True
    elif not thumb_up:
        state.shift_gesture_active = False
# ================= DRAW UI =================
def draw_ui(canvas, mp_draw, mp_hands, frame, results):
    """Draw sidebar UI with all information."""
    h, w, _ = canvas.shape
    x_off = CAM_W + 20
    
    # Background
    cv2.rectangle(canvas, (CAM_W, 0), (w, h), (20, 15, 35), -1)
    
    # Header
    cv2.putText(canvas, "DIGITAL LOGIC LAB", (x_off, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 100, 255), 2)
    cv2.putText(canvas, "="*25, (x_off, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 50, 200), 1)
    # current mode display
    cv2.putText(canvas, f"MODE : {state.mode}", (x_off, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    
    y_pos = 90
    
    # Draw hand landmarks on frame if toggle enabled
    if state.show_landmarks and results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # Mode-specific UI
    if state.mode == "LOGIC":
        draw_logic_ui(canvas, x_off, y_pos)
    elif state.mode == "ARITHMETIC":
        draw_arithmetic_ui(canvas, x_off, y_pos)
    elif state.mode == "FLIPFLOPS":
        draw_flipflop_ui(canvas, x_off, y_pos)
    elif state.mode == "SHIFTREG":
        draw_shiftreg_ui(canvas, x_off, y_pos)
    
    # Input display
    input_y = CAM_H - 100
    cv2.rectangle(canvas, (x_off-5, input_y-5), (w-10, CAM_H-10), (40, 35, 70), -1)
    cv2.putText(canvas, f"Input A: {state.A}  Input B: {state.B}", (x_off, input_y+15), 0, 0.6, (150, 200, 255), 1)
    
    if state.mode == "ARITHMETIC" and state.op_idx >= 1:
        cv2.putText(canvas, f"Carry-in: {state.Cin}", (x_off, input_y+40), 0, 0.6, (150, 200, 255), 1)
    
    cv2.putText(canvas, "L=Logic  A=Arith  F=FlipFlop  S=Shift", (x_off, input_y+65), 0, 0.5, (200, 150, 100), 1)

# ================= DRAW LOGIC UI =================
def draw_logic_ui(canvas, x_off, y_pos):
    """Draw logic gate UI."""
    gate = LOGIC_GATES[state.op_idx]
    
    cv2.putText(canvas, f"MODE: LOGIC GATES", (x_off, y_pos), 0, 0.65, (100, 255, 150), 2)
    y_pos += 40
    
    cv2.putText(canvas, f"Gate: {gate['name']}", (x_off, y_pos), 0, 0.6, (200, 200, 200), 1)
    y_pos += 25
    cv2.putText(canvas, f"Keys: 1-7 to select", (x_off, y_pos), 0, 0.5, (150, 150, 150), 1)
    y_pos += 35
    
    for line in gate['table']:
        cv2.putText(canvas, line, (x_off, y_pos), 0, 0.55, (180, 180, 220), 1)
        y_pos += 22
    
    y_pos += 20
    result = process_logic_gates()
    cv2.putText(canvas, f"Result: {result}", (x_off, y_pos), 0, 0.7, (0, 255, 100), 2)

# ================= DRAW ARITHMETIC UI =================
def draw_arithmetic_ui(canvas, x_off, y_pos):
    """Draw arithmetic circuit UI."""
    arith = ARITHMETIC[state.op_idx]
    
    cv2.putText(canvas, f"MODE: ARITHMETIC", (x_off, y_pos), 0, 0.65, (100, 200, 255), 2)
    y_pos += 40
    
    cv2.putText(canvas, f"Op: {arith['name']}", (x_off, y_pos), 0, 0.6, (200, 200, 200), 1)
    y_pos += 25
    cv2.putText(canvas, f"Keys: 1-4 to select", (x_off, y_pos), 0, 0.5, (150, 150, 150), 1)
    y_pos += 35
    
    for line in arith['table']:
        cv2.putText(canvas, line, (x_off, y_pos), 0, 0.55, (180, 180, 220), 1)
        y_pos += 22
    
    y_pos += 20
    result = process_arithmetic()
    cv2.putText(canvas, f"Result: {result}", (x_off, y_pos), 0, 0.7, (0, 255, 100), 2)

# ================= DRAW FLIP-FLOP UI =================
def draw_flipflop_ui(canvas, x_off, y_pos):
    """Draw flip-flop UI."""
    ff = FLIPFLOPS[state.op_idx]
    
    cv2.putText(canvas, f"MODE: FLIP-FLOPS", (x_off, y_pos), 0, 0.65, (100, 255, 200), 2)
    y_pos += 40
    
    cv2.putText(canvas, f"FF: {ff['name']}", (x_off, y_pos), 0, 0.6, (200, 200, 200), 1)
    y_pos += 25
    cv2.putText(canvas, f"Keys: 1-4 to select", (x_off, y_pos), 0, 0.5, (150, 150, 150), 1)
    y_pos += 35
    
    for line in ff['table']:
        cv2.putText(canvas, line, (x_off, y_pos), 0, 0.55, (180, 180, 220), 1)
        y_pos += 22
    
    y_pos += 20
    q = process_flipflop()
    cv2.putText(canvas, f"Q: {q}", (x_off, y_pos), 0, 0.7, (0, 255, 100), 2)

# ================= DRAW SHIFT REGISTER UI =================
def draw_shiftreg_ui(canvas, x_off, y_pos):
    """Draw shift register UI."""
    cv2.putText(canvas, f"MODE: SHIFT REGISTER", (x_off, y_pos), 0, 0.65, (255, 150, 100), 2)
    y_pos += 40
    
    cv2.putText(canvas, "TYPE 4 BITS (0/1) AND PRESS ENTER", (x_off, y_pos), 0, 0.5, (150, 150, 150), 1)
    y_pos += 40
    
    # Draw Register
    cv2.putText(canvas, "Register (4-bit):", (x_off, y_pos), 0, 0.6, (200, 200, 200), 1)
    y_pos += 30
    
    for i, bit in enumerate(state.shift_reg):
        color = (0, 255, 0) if bit else (100, 100, 100)
        cv2.rectangle(canvas, (x_off + i*65, y_pos), (x_off + i*65 + 55, y_pos + 50), color, -1)
        cv2.putText(canvas, str(bit), (x_off + i*65 + 18, y_pos + 35), 0, 0.8, (255, 255, 255), 2)

    y_pos += 70
    # show buffer being typed
    cv2.putText(canvas, f"Buffer: {''.join(map(str,state.shift_buffer))}", (x_off, y_pos), 0, 0.6, (200,200,100), 1)
    y_pos += 30
    # no more parallel input/bit display
    




# ================= HANDLE KEYBOARD INPUT =================
def handle_keyboard(key):
    """Process keyboard input."""
    if key == ord('q'):
        return False
    elif key == ord('l'):
        state.mode = "LOGIC"
        state.op_idx = 0
    elif key == ord('a'):
        state.mode = "ARITHMETIC"
        state.op_idx = 0
    elif key == ord('f'):
        state.mode = "FLIPFLOPS"
        state.op_idx = 0

    elif key == ord('d'):
        state.show_landmarks = not state.show_landmarks

    elif key == ord('s'):
        state.mode = "SHIFTREG"
        state.op_idx = 0
        state.shift_reg = [0, 0, 0, 0]
        state.shift_buffer = []
        state.shift_gesture_active = False
    # handle buffer typing in shift mode
    elif state.mode == "SHIFTREG":
        if key in (ord('0'), ord('1')):
            if len(state.shift_buffer) < 4:
                state.shift_buffer.append(int(chr(key)))
        elif key == 13:  # ENTER
            if state.shift_buffer:
                state.shift_reg = state.shift_buffer.copy()
                state.shift_buffer.clear()
    elif state.mode == "LOGIC" and ord('1') <= key <= ord('7'):
        state.op_idx = key - ord('1')
    elif state.mode == "ARITHMETIC" and ord('1') <= key <= ord('4'):
        state.op_idx = key - ord('1')
    elif state.mode == "FLIPFLOPS" and ord('1') <= key <= ord('4'):
        state.op_idx = key - ord('1')
    
    return True

# ================= MAIN EXECUTION =================
def main():
    """Main application loop."""
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(3, CAM_W)
    cap.set(4, CAM_H)
    
    # Initialize MediaPipe (if available)
    if mp is None:
        print("MediaPipe unavailable, cannot run gesture detection. Exiting.")
        return
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=HAND_CONFIDENCE,
        min_tracking_confidence=HAND_CONFIDENCE
    )
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Preprocess frame
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (CAM_W, CAM_H))
            
            # Detect hands
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            
            # Extract hand inputs (now includes thumb gesture info and fist)
            state.A, state.B, state.Cin, thumb_up, hand_side, fist = detect_hand(hands, frame)
            
            # Process mode-specific logic
            if state.mode == "SHIFTREG":
                process_shift_register(thumb_up, hand_side)
            elif state.mode == "FLIPFLOPS":
                process_flipflop()
            
            # Prepare canvas
            canvas = np.zeros((CAM_H, CAM_W + SIDEBAR_W, 3), dtype=np.uint8)
            canvas[:, :CAM_W] = frame
            
            # Draw UI
            draw_ui(canvas, mp_draw, mp_hands, frame, results)
            
            # Display
            cv2.imshow("Digital Logic Simulator", canvas)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if not handle_keyboard(key):
                break
    
    except KeyboardInterrupt:
        print("\nApplication interrupted by user.")
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Application closed successfully.")

if __name__ == "__main__":
    main()
