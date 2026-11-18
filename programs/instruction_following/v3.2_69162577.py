# G2 / GenieSim instruction board, skill library v3.2 (2026-10-25, Opus 5.5).
#
# v3.2 = v3.1 + two grasp-option fixes from the phase-1 check (evidence/2026-10-25_v31_phase1):
#   * Pen s48 fell back to closing along the 13 cm pen, which no jaw opening encloses: options
#     whose closing direction spans more than JAW_SPAN_M of the object are not offered.
#   * Steep escalations (35/50 deg, only after both hovers and the integrated planner fail) held
#     1/5 in v3 + v3.1; with a tilted closing axis the lower open fingertip sits 5.2 cm x (its
#     vertical component) below the TCP, and the one held grasp had it 8 mm above the table, the
#     misses <= 3 mm (number block s35: 19 mm into the table). The escalation now also offers
#     yaws perpendicular to the reach (both fingertips level), tries the table-clearing ones
#     first, and raises a steep grasp until that fingertip is 6 mm above the table.
#   * Not changed: 20-degree and vertical grasps. A fingertip nominally at or below the table
#     does not predict failure there (195/200 held in v3 + v3.1, 15/15 even > 10 mm "below").
#     Pen s46 (a 20-degree miss) stays unexplained; one of 41 v3.1 20-degree grasps failed vs
#     3 of 159 in v3.
# Everything else, including the aim, is v3.1.
#
# v3.1 = v3 with the grasp alignment rebuilt on what the 10-24 pad-measurement study measured
# (exec/pad_measurement_2026-10-24.md, kb/06):
#   * The "finger pads" v2.1/v3 measured were SAM 3's whole-gripper mask: PAD_PROMPTS sat on
#     the proximal knuckles and "nearest median to the camera" picked the gripper mask, so both
#     pad centres were one fixed point of the gripper body, (-0.5, +-2.0, -7.0) cm in the TCP
#     frame (94/94 measurements). Moving that point over the object put the TCP (= the
#     fingertips = GenieSim's Follow point) 7.0 cm x sin(tilt) along the tilt plus 2 cm across
#     the jaw away from it: the "2.4 cm camera bias" is 7.0 x sin(20 deg); v3 closed a median
#     1.9 cm from the true object, and 3 cm cubes fail beyond 2.9 cm across the jaw (13/13).
#   * There is no camera bias to cancel: camera poses are the simulator's own and FK matches
#     them to 0.02 mm; the head camera's footprint centre is a median 3 mm from the true centre.
#   * So the TCP itself is aimed (FK) at the object's footprint-rectangle centre ("grasp_xy",
#     2.3x closer to the true centre than the point-cloud median, whose visible side faces bias
#     it towards the camera). The head camera gives the aim; the wrist camera at hover gives
#     the grasp height and overrides the aim only on a gross (> 3 cm), size-consistent
#     disagreement; the pre-close wrist view is only logged (it leans ~2 cm to the camera).
#   * The correct differential (both distal pads segmented at their FK pixels, residual vs FK)
#     is measured once per attempt and logged; FINGER_BIAS_APPLY stays off in simulation.
#   * Retry on air: re-aim at the wrist's view of the object if it moved, 5 mm lower.
#
# One program for all ten instruction-board tasks. It consolidates the Fable 5.1 programs
# (pick_specific_object_v0 = grasp core v2.1, pick_instruction_v1 = instruction resolution,
# straighten_v1 = upright draft) into one structure and fixes what the run evidence of
# 10-22..10-24 showed to be broken (kb/06, "skill library v3"):
#   * quat_from_basis: v2.1's trace-only conversion returned (0, 1, 0, 0) for EVERY vertical
#     approach, so the yaw of a top-down grasp was silently ignored. Shepperd's method now;
#     the 20-degree tilted candidates (what v2.1 actually used in 94% of its grasps) are
#     still ranked first, the vertical ones are now real fallbacks with the intended yaw.
#   * Retry on air: v2.1 read the gripper right after close_gripper, but that observation
#     predates the close (GenieSim answers before the jaw moves), so the retry never fired in
#     any run. The opening is now read after the lift, where it is current.
#   * Lift: v2.1 lifted only with the nominal orientation and lost grasped objects to
#     "no IK solution" (logic_or s32/s41). Falls back to the measured tool orientation and
#     shorter lifts (PickUpOnGripper needs 2 cm).
#   * Wrist re-localization tracks the chosen object by position (nearest plausible SAM 3
#     instance to the finger pads) instead of by text identity, and rejects junk masks
#     (v2.1 took a 48 cm "object" as the grasp height in block_size, overnight 10-22).
#   * Approach escalation after both v2.1 hover heights fail: cuRobo integrated pose planning
#     (scene-aware IK) and steeper tilts along the reach direction.
#   * Every perception call is guarded; the program always returns its report.
#
# Per episode: resolve the sentence (Qwen3.6 via g2.resolve_instruction, rule parser as
# fallback) -> choose the target with the task's chooser on the head camera -> grasp:
# hover with the TCP above the object's footprint centre, wrist check (grasp height, finger
# differential, gross-move override), joint-interpolated descend, pre-close view (logged),
# close, lift, verify, retry once on air, hold through a GenieSim evaluation tick -> straighten_object additionally
# rotates the bottle upright (best current approach; the 90-degree roll is usually
# unreachable, see report["upright"]). result = report.
#
# Sandbox: only `import math`, no classes, no names starting with an underscore, only the
# whitelisted builtins (no getattr/isinstance/hasattr); tools are injected by the harness.
import math

# ----------------------------------------------------------------------------- constants
# G2 has no PyRoki model: IK and planning run on the cuRobo service. Full planner speed:
# each scoring stage allows ~200 rows (6.7 s at 30 Hz), kb/06.
STRATEGY = MotionStrategy(ik_solver="curobo", trajectory_planner="curobo", time_dilation_factor=1.0)
# Scene-aware IK + trajectory in one call (cuRobo plan_pose); used only when the plain IK
# candidates all end in scene collision.
STRATEGY_POSE = MotionStrategy(ik_solver="curobo", trajectory_planner="curobo", pose_planner="curobo-integrated", time_dilation_factor=1.0)
# Final centimetres (descend, correct, lift, rotate): joint interpolation without the depth
# voxels, which would otherwise report the fingers colliding with the table near the object.
STRATEGY_FINAL = MotionStrategy(ik_solver="curobo", trajectory_planner="interpolation")

ABOVE_M = 0.12              # hover height above the grasp height
ABOVE_HIGH_M = 0.05         # extra height for the second hover attempt
LIFT_M = 0.12
LIFT_FALLBACK_M = (0.12, 0.08, 0.05)
GRASP_BELOW_TOP_M = 0.035   # fingertips 3.5 cm below the top of the object cloud
FLOOR_CLEARANCE_M = 0.006   # but never lower than this above the cloud's lowest points
RESIDUAL_OK_RAD = 0.25      # accept a move whose end is within this of its last waypoint
SERVO_TOL_RAD = 0.04        # servo (up to three rounds) until the residual is below this
HOVER_SHIFT_MAX_M = 0.12   # largest aim change a wrist view may cause
WRIST_OVERRIDE_M = 0.03     # head vs wrist footprint centres disagree by more -> follow the wrist
                            # (10-24: |wrist - head| <= 2.1 cm in all 32 v3 hovers; head 3-6 mm from truth)
SIZE_MATCH = 0.35           # a wrist instance is the chosen object if each footprint side agrees within this
RETRY_LOWER_M = 0.005       # the retry on air goes this much lower (never under the floor clearance)
RETRY_MOVED_M = 0.02        # after an air close, re-aim at the wrist's view only if the object moved
                            # more than this (the wrist view itself leans ~1.3 cm towards the camera)
AIR_OPENING = 0.06          # normalized opening after a lift below which nothing is held
TRACK_RADIUS_M = 0.08       # a wrist instance must lie this close (xy) to the expected spot
MIN_SCORE = 0.25            # SAM 3 text scores: real matches 0.36-0.98, junk masks < 0.2
TRACK_MIN_SCORE = 0.12      # wrist tracking: proximity and plausibility already reject junk
MIN_POINTS = 30
TILT_RAD = math.radians(20.0)
ESCALATION_TILTS_RAD = (math.radians(35.0), math.radians(50.0))
FINGERTIP_HALF_SPAN_M = 0.052  # open-jaw fingertips at +-5.2 cm along the closing axis, TCP level (PADS_TOOL)
STEEP_TILT_RAD = math.radians(27.5)  # escalation tilts (35/50 deg) count as steep, the standard 20 deg does not
TIP_CLEARANCE_M = 0.006        # steep grasps: lower fingertip at least this far above the table (object bottom)
JAW_SPAN_M = 0.09              # an object wider than this along the closing axis does not fit the jaw
GRIP_BELOW_TOP_M = 0.002       # the TCP must still reach this far below the object's top
                               # (v3 held a 1.9 cm pen with the TCP 2 mm under its top)
SHOULDER_X = 0.13           # base_link x of the shoulder joints (FK, waist at the task pose)
SHOULDER_Y = 0.244

# Gripper geometry in the TCP frame (gripper_*_center_link: x = closing axis, z = approach,
# +y towards the wrist camera on the right arm, -y on the left), jaw fully open (1.0, as on
# every approach). Distal pads = centroids of SAM 3's pad mask on recorded hover frames,
# 10-24; the knuckles and the gripper body (median of SAM 3's whole-gripper mask, the point
# v2.1/v3 used as "pad midpoint") only serve to reject finger instances in the wrist view.
PADS_TOOL = {"left": ((-0.052, -0.009, -0.027), (0.052, -0.009, -0.027)),
             "right": ((0.052, 0.009, -0.027), (-0.052, 0.009, -0.027))}
GRIPPER_PARTS_TOOL = {"left": ((-0.044, -0.013, -0.067), (0.044, -0.013, -0.067), (-0.004, -0.020, -0.070)),
                      "right": ((0.044, 0.013, -0.067), (-0.044, 0.013, -0.067), (-0.004, 0.020, -0.070))}
FINGER_RADIUS_M = 0.03       # a wrist instance centred this close to a finger point is the gripper
PAD_MASK_POINTS = (1500, 20000)  # a distal pad mask (whole finger 17-37k, gripper 65-250k points)
PAD_GATE_M = 0.03            # measured pad must lie this close to its FK position
PAD_AGREE_M = 0.012          # both pads must agree (off the closing axis) for a bias estimate
# Camera-vs-FK bias from the two pads. In GenieSim it is 0 by construction (camera poses are
# the simulator's; FK matches them to 0.02 mm), so it is only measured and logged here; on a
# real robot set FINGER_BIAS_APPLY after checking the logged values.
FINGER_BIAS_APPLY = False
FINGER_BIAS_RANGE_M = (0.012, 0.03)

# Alternative phrasings for objects SAM 3 does not recognise by the benchmark's name.
SYNONYMS = {
    "dishwashing sponge": ["sponge", "red sponge", "red rectangular block", "red block"],
    "sprite": ["green can", "green soda can", "soda can"],
    "paper cup": ["cup", "white cup"],
    "mouse": ["computer mouse", "grey mouse"],
    "apple": ["red apple", "fruit"],
}
# Fallback queries when the resolver gives nothing usable (category / common-sense words).
# SAM 3 finds bottles with "bottle" and cans with "can" (0.89-0.91 on GenieSim head frames,
# 10-24); "drink bottle", "beverage", "drink" and "juice bottle" score <= 0.27.
CATEGORY_QUERIES = {
    "drink": ["bottle", "can", "soda can"],
    "beverage": ["bottle", "can", "soda can"],
    "fruit": ["fruit", "apple", "orange"],
    "stationery": ["pen", "pencil", "marker"],
    "toy": ["toy", "toy car"],
    "cleaning": ["sponge", "brush", "cleaning tool"],
    "snack": ["snack", "chips"],
    "red block": ["red block"],
}
# Common-sense cue -> (category, SAM 3 queries). Every "thirsty" scene of the board holds a
# beverage (the target) and a fruit; Qwen3.6 answered "the red apple" for it (v2.1 rerun s32/s33),
# so a resolver object outside the cue's category is demoted behind the category queries.
COMMON_SENSE_CUES = (
    (("thirst", "juice", "drink", "carbonated", "soda", "cola"), "beverage", ["bottle", "can", "soda can"]),
    (("vitamin", "fruit", "eat", "hungry"), "fruit", ["fruit", "apple", "orange", "peach"]),
    (("write", "note"), "stationery", ["pen", "pencil"]),
)
BEVERAGE_WORDS = ("bottle", "can", "drink", "drinks", "juice", "soda", "cola", "coke", "sprite", "fanta", "pepsi", "water", "tea", "milk", "coffee", "beverage")
FRUIT_WORDS = ("fruit", "apple", "orange", "peach", "lemon", "banana", "pear", "grape", "kiwi", "mango", "strawberry", "watermelon", "pineapple")
STATIONERY_WORDS = ("pen", "pencil", "marker", "eraser", "ruler", "stationery")
COLOR_WORDS = ("red", "green", "blue", "yellow", "purple", "orange", "pink", "black", "white", "cyan", "teal", "brown", "grey", "gray")
SHAPE_WORDS = ("cube", "cubic", "square", "rectangular", "cuboid", "cylindrical", "cylinder", "triangular", "hexagonal", "prism", "l-shaped", "arch", "semicircular")
TASK_KINDS = ("pick_specific_object", "pick_block_color", "pick_object_type", "pick_follow_logic_or", "pick_block_size",
              "pick_billiards_color", "pick_block_number", "pick_common_sense", "pick_block_shape", "straighten_object")

# ----------------------------------------------------------------------------- report
context = get_task_context()
frame = get_robot_state().base_frame
report = {
    "program": "instruction_board_v3_2",
    "task": context.task_name,
    "instruction": context.language,
    "stages": [],
    "moves": [],
    "attempts": [],
    "rows": 0,
    "terminated": False,
    "error": None,
}
last_command = {}


def stage(name, **fields):
    entry = {"stage": name, "rows_before": report["rows"]}
    entry.update(fields)
    report["stages"].append(entry)
    return entry


def r3(values):
    return [round(float(v), 4) for v in values]


def count_rows(result):
    try:
        report["rows"] += int(result.steps_executed)
    except Exception:
        pass
    if result.terminated:
        report["terminated"] = True


def error_text(result):
    if result.error is None:
        return "?"
    return str(result.error.message)[:90]


# ----------------------------------------------------------------------------- geometry
def quantile(values, fraction):
    ordered = sorted(values)
    return ordered[int((len(ordered) - 1) * fraction)]


def sampled(points, limit):
    count = len(points)
    if count <= limit:
        return points
    return points[::int(math.ceil(count / float(limit)))]


def footprint(xs, ys):
    """Minimum-area rectangle of the xy points (robust 2-98% extents; 5-degree search, then
    1-degree around the best): (long-axis yaw, length, width, (centre x, centre y)). Unlike the
    covariance yaw it is face-aligned for cubes; the 1-degree pass matters for Upright's
    5-degree tolerance. The rectangle centre is the grasp aim: the side faces a camera sees
    pull the point median towards it, the extents do not (10-24, head frames vs simulator
    truth, 63 objects: median 3.1 mm from the true centre vs 7.2 mm for the point median)."""
    def extents(angle):
        c, s = math.cos(angle), math.sin(angle)
        us = [x * c + y * s for x, y in zip(xs, ys)]
        vs = [y * c - x * s for x, y in zip(xs, ys)]
        u0, u1 = quantile(us, 0.02), quantile(us, 0.98)
        v0, v1 = quantile(vs, 0.02), quantile(vs, 0.98)
        um, vm = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
        return ((u1 - u0) * (v1 - v0), angle, u1 - u0, v1 - v0, (um * c - vm * s, um * s + vm * c))

    best = min([extents(step * math.pi / 36.0) for step in range(18)], key=lambda item: item[0])
    fine = [extents(best[1] + math.radians(delta)) for delta in (-4, -3, -2, -1, 1, 2, 3, 4)]
    best = min(fine + [best], key=lambda item: item[0])
    if best[2] >= best[3]:
        return best[1], best[2], best[3], best[4]
    return best[1] + 0.5 * math.pi, best[3], best[2], best[4]


def fill_ratio(xs, ys, yaw, length, width):
    """Share of 5 mm cells inside the footprint rectangle that hold points (cube top ~1)."""
    if length < 0.01 or width < 0.01:
        return 0.0
    c, s = math.cos(yaw), math.sin(yaw)
    us = [x * c + y * s for x, y in zip(xs, ys)]
    vs = [y * c - x * s for x, y in zip(xs, ys)]
    u0, v0 = quantile(us, 0.02), quantile(vs, 0.02)
    cells = set()
    for u, v in zip(us, vs):
        i, j = int((u - u0) / 0.005), int((v - v0) / 0.005)
        if 0 <= i <= int(length / 0.005) and 0 <= j <= int(width / 0.005):
            cells.add(i * 1000 + j)
    total = (int(length / 0.005) + 1) * (int(width / 0.005) + 1)
    return min(1.0, len(cells) / float(total))


def summarize(cloud):
    """Geometry of one object cloud (base frame) used by every chooser and grasp step."""
    if cloud is None:
        return None
    points = sampled(cloud.points, 3000)
    if len(points) < MIN_POINTS:
        return None
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    zs = [float(p[2]) for p in points]
    flat = sampled(points, 1200)
    fx = [float(p[0]) for p in flat]
    fy = [float(p[1]) for p in flat]
    yaw, length, width, middle = footprint(fx, fy)
    top, bottom = quantile(zs, 0.95), quantile(zs, 0.02)
    return {
        "cloud": cloud,
        "center": (quantile(xs, 0.5), quantile(ys, 0.5), quantile(zs, 0.5)),
        "grasp_xy": (middle[0], middle[1]),
        "top": top,
        "bottom": bottom,
        "height": top - bottom,
        "yaw": yaw,
        "length": length,
        "width": width,
        "points": len(cloud.points),
        "fill": fill_ratio(fx, fy, yaw, length, width),
    }


def describe(summary):
    if summary is None:
        return None
    return {"center": r3(summary["center"]), "grasp_xy": r3(summary["grasp_xy"]), "top": round(summary["top"], 4), "bottom": round(summary["bottom"], 4),
            "length": round(summary["length"], 4), "width": round(summary["width"], 4), "height": round(summary["height"], 4),
            "yaw_deg": round(math.degrees(summary["yaw"]), 1), "points": summary["points"], "fill": round(summary["fill"], 3)}


def plausible(summary, max_extent=0.3):
    """An object on the table in front of the robot, not a table patch or a far wall."""
    if summary is None or summary["points"] < MIN_POINTS:
        return False
    x, y, z = summary["center"]
    return (summary["length"] <= max_extent and summary["height"] <= max_extent
            and 0.2 <= x <= 1.05 and abs(y) <= 0.65 and 0.45 <= z <= 1.25)


def grasp_height(summary, reference=None):
    """Fingertips GRASP_BELOW_TOP_M under the top, never below the cloud bottom + clearance.
    A wrist cloud straight from above may hold only the top face; the object cannot be
    shorter than the head camera measured it (`reference`), so its bottom is bounded by that."""
    bottom = summary["bottom"]
    if reference is not None and summary["height"] < 0.5 * reference["height"]:
        bottom = summary["top"] - reference["height"]
    return max(summary["top"] - GRASP_BELOW_TOP_M, bottom + FLOOR_CLEARANCE_M)


def volume(summary):
    return summary["length"] * summary["width"] * (summary["height"] + 0.01)


def xy_distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


# ----------------------------------------------------------------------------- rotations
def quat_normalize(q):
    n = math.sqrt(sum(v * v for v in q)) or 1.0
    q = [v / n for v in q]
    if q[0] < 0.0:
        q = [-v for v in q]
    return (q[0], q[1], q[2], q[3])


def quat_from_basis(x, y, z):
    """wxyz quaternion of the rotation with columns x, y, z (Shepperd's method)."""
    m00, m01, m02 = x[0], y[0], z[0]
    m10, m11, m12 = x[1], y[1], z[1]
    m20, m21, m22 = x[2], y[2], z[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = (0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s)
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(max(1e-12, 1.0 + m00 - m11 - m22)) * 2.0
        q = ((m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s)
    elif m11 > m22:
        s = math.sqrt(max(1e-12, 1.0 + m11 - m00 - m22)) * 2.0
        q = ((m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s)
    else:
        s = math.sqrt(max(1e-12, 1.0 + m22 - m00 - m11)) * 2.0
        q = ((m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s)
    return quat_normalize(q)


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


def quat_axis_angle(axis, angle):
    n = math.sqrt(sum(v * v for v in axis)) or 1.0
    s = math.sin(angle / 2.0) / n
    return (math.cos(angle / 2.0), axis[0] * s, axis[1] * s, axis[2] * s)


def quat_column(q, index):
    """Column `index` of the rotation matrix of q: the tool axis x (0), y (1) or z (2) in base."""
    w, x, y, z = q
    if index == 0:
        return (1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y))
    if index == 1:
        return (2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x))
    return (2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y))


def approach_quat(yaw, tilt=0.0, tilt_dir=0.0):
    """Tool +z (approach) points down, tilted by `tilt` toward azimuth `tilt_dir`; tool x
    (the closing axis) follows `yaw` in the horizontal plane."""
    z = (math.sin(tilt) * math.cos(tilt_dir), math.sin(tilt) * math.sin(tilt_dir), -math.cos(tilt))
    ref = (math.cos(yaw), math.sin(yaw), 0.0)
    d = ref[0] * z[0] + ref[1] * z[1] + ref[2] * z[2]
    x = (ref[0] - d * z[0], ref[1] - d * z[1], ref[2] - d * z[2])
    n = math.sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]) or 1.0
    x = (x[0] / n, x[1] / n, x[2] / n)
    y = (z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0])
    return quat_from_basis(x, y, z)


def option(name, quat, rank):
    return {"name": name, "quat": quat_normalize(quat), "rank": rank}


def approach_tilt(quat):
    """Angle between the approach axis (tool z) and straight down."""
    return math.acos(max(-1.0, min(1.0, -quat_column(quat, 2)[2])))


def tip_drop(quat):
    """How far the lower open fingertip sits below the TCP: the closing axis (tool x) tilts out
    of the horizontal whenever the approach tilt has a component along it."""
    return FINGERTIP_HALF_SPAN_M * abs(quat_column(quat, 0)[2])


def span_across(summary, yaw):
    """The object's footprint extent along a closing direction `yaw`."""
    delta = yaw - summary["yaw"]
    return summary["length"] * abs(math.cos(delta)) + summary["width"] * abs(math.sin(delta))


def clears_table(summary, quat):
    """True if the TCP can reach GRIP_BELOW_TOP_M under the object's top with the lower
    fingertip TIP_CLEARANCE_M above the table."""
    return summary["bottom"] + TIP_CLEARANCE_M + tip_drop(quat) <= summary["top"] - GRIP_BELOW_TOP_M


def grasp_options(summary, across_only=False):
    """Orientation candidates. Rank = 10 * grasp class + tilt rank: class 0 closes across the
    footprint's short axis (yaw and yaw + 180 are the same grasp), class 1 along it; within a
    class the 20-degree forward tilt (what v2.1 actually executed) comes before vertical.
    v3.2: a closing direction wider than the jaw is not offered (pen s48 closed along its 13 cm).
    The 20-degree and vertical options keep v3.1's ranking and height: their fingertip touching
    the table does not predict failure (v3 + v3.1: 195/200 held, 100% even 10 mm "below")."""
    base = summary["yaw"]
    yaws = [(base + 0.5 * math.pi, 0), (base - 0.5 * math.pi, 0)]
    if not across_only:
        yaws = yaws + [(base, 1), (base + math.pi, 1)]
    options = []
    for yaw, grasp_class in yaws:
        if grasp_class and span_across(summary, yaw) > JAW_SPAN_M:
            continue
        label = "yaw%+.0f" % math.degrees(math.atan2(math.sin(yaw), math.cos(yaw)))
        options.append(option(label + "_tilt20", approach_quat(yaw, TILT_RAD, 0.0), 10 * grasp_class))
        options.append(option(label + "_vertical", approach_quat(yaw, 0.0, 0.0), 10 * grasp_class + 1))
    return options


def escalation_options(summary, center, arm):
    """Steeper tilts along the shoulder -> object direction, for hovers no plain candidate reaches.
    v3.2: also closing perpendicular to the reach (both fingertips level, no table contact) when
    the object fits the jaw that way; options that clear the table first, then lower tilt, then
    the smaller fingertip drop (number block s35: both across-yaws lay in the 35-degree tilt
    plane, fingertip 28 mm under the TCP)."""
    side = 1.0 if arm == "left" else -1.0
    reach_dir = math.atan2(center[1] - side * SHOULDER_Y, center[0] - SHOULDER_X)
    yaws = [summary["yaw"] + 0.5 * math.pi, summary["yaw"] - 0.5 * math.pi]
    for yaw in (reach_dir + 0.5 * math.pi, reach_dir - 0.5 * math.pi):
        # skip a yaw that repeats an across-yaw (closing axes within 10 degrees, either sign)
        repeats = any(abs(math.sin(yaw - other)) < math.sin(math.radians(10.0)) for other in yaws)
        if not repeats and span_across(summary, yaw) <= JAW_SPAN_M:
            yaws.append(yaw)
    ranked = []
    for tilt in ESCALATION_TILTS_RAD:
        for yaw in yaws:
            quat = approach_quat(yaw, tilt, reach_dir)
            name = "reach%+.0f_tilt%.0f_yaw%+.0f" % (math.degrees(reach_dir), math.degrees(tilt), math.degrees(math.atan2(math.sin(yaw), math.cos(yaw))))
            ranked.append((0 if clears_table(summary, quat) else 1, tilt, tip_drop(quat), option(name, quat, 0)))
    ranked.sort(key=lambda entry: (entry[0], entry[1], entry[2]))
    return [entry[3] for entry in ranked]


# ----------------------------------------------------------------------------- motion
def joint_distance(a, b):
    return sum(abs(float(x) - float(y)) for x, y in zip(a, b))


def current_joints(arm):
    return [float(v) for v in get_robot_state().joint_positions[arm]]


def ik_ranked(position, options, arm, samples=1, keep=5):
    """cuRobo IK for each orientation option (`samples` calls each: the solver's seeds are
    random), ordered by option rank, then by joint distance from the current configuration."""
    current = current_joints(arm)
    ranked = []
    for item in options:
        for sample in range(samples):
            try:
                ik = solve_ik(Pose(position=[float(v) for v in position], quaternion_wxyz=list(item["quat"]), frame=frame), current, arm=arm, backend="curobo")
            except Exception:
                continue
            if ik.ok and ik.joint_positions is not None:
                joints = [float(v) for v in ik.joint_positions]
                ranked.append((item["rank"], joint_distance(joints, current), item, joints))
    ranked.sort(key=lambda entry: (entry[0], entry[1]))
    return ranked[:keep]


def with_gripper(trajectory, gripper):
    """Copy of a planned trajectory that commands one normalized gripper value on every row."""
    return Trajectory(
        joint_positions=trajectory.joint_positions,
        dt_s=trajectory.dt_s,
        joint_names=trajectory.joint_names,
        planner=trajectory.planner,
        collision_aware=trajectory.collision_aware,
        expected_start=trajectory.expected_start,
        arm=trajectory.arm,
        embodiment=trajectory.embodiment,
        gripper_positions=[gripper for waypoint in trajectory.joint_positions],
    )


def execute(trajectory, arm, entry, gripper=None):
    """Execute, then servo up to three rounds onto the last waypoint (the simulated arm lags
    cuRobo's time-optimal end by ~0.1 rad; a residual of 0.09 rad once closed on air)."""
    if gripper is not None:
        trajectory = with_gripper(trajectory, gripper)
    done = execute_trajectory(trajectory)
    count_rows(done)
    worst = max([float(v) for v in done.final_errors.values()] or [0.0])
    entry.update({"exec_ok": done.ok, "rows": done.steps_executed, "residual_rad": round(worst, 4)})
    if done.terminated:
        entry["terminated"] = True
        return False
    if done.steps_executed == 0:
        entry["exec_error"] = error_text(done)
        return False
    final = trajectory.joint_positions[-1]
    last_command[arm] = [float(v) for v in final]
    rounds = 0
    while worst > SERVO_TOL_RAD and rounds < 3:
        servo = move_to_joints(final, tolerance=0.03, max_steps=20, arm=arm)
        count_rows(servo)
        worst = max([float(v) for v in servo.final_errors.values()] or [worst])
        rounds += 1
        entry.update({"servo_rounds": rounds, "residual_after_servo_rad": round(worst, 4)})
        if servo.terminated:
            entry["terminated"] = True
            return False
    return worst <= RESIDUAL_OK_RAD


def move_ik(position, options, arm, label, gripper=None, strategy=None, exclude=None, samples=1, keep=5):
    """IK candidates (ranked) -> plan each until one is accepted -> execute and converge.
    Returns the option used, or None."""
    entry = {"label": label, "target": r3(position)}
    report["moves"].append(entry)
    ranked = ik_ranked(position, options, arm, samples=samples, keep=keep)
    entry["ik_solutions"] = len(ranked)
    if not ranked:
        entry["fail"] = "no IK solution"
        return None
    for index, item in enumerate(ranked):
        try:
            plan = plan_motion(item[3], arm=arm, strategy=strategy or STRATEGY, target_point_cloud=exclude)
        except Exception as exc:
            entry.setdefault("plan_errors", []).append("exception: " + str(exc)[:80])
            continue
        if plan.ok and plan.trajectory is not None:
            entry.update({"option": item[2]["name"], "joint_distance": round(item[1], 3), "plan_attempt": index + 1,
                          "waypoints": len(plan.trajectory.joint_positions)})
            return item[2] if execute(plan.trajectory, arm, entry, gripper) else None
        entry.setdefault("plan_errors", []).append(error_text(plan))
    entry["fail"] = "planning refused every IK candidate"
    return None


def move_pose(position, options, arm, label, gripper=None, limit=4):
    """cuRobo integrated pose planning (scene-aware IK + trajectory) over the options in order."""
    entry = {"label": label, "target": r3(position)}
    report["moves"].append(entry)
    for index, item in enumerate(options[:limit]):
        try:
            plan = plan_motion(Pose(position=[float(v) for v in position], quaternion_wxyz=list(item["quat"]), frame=frame), arm=arm, strategy=STRATEGY_POSE)
        except Exception as exc:
            entry.setdefault("plan_errors", []).append("exception: " + str(exc)[:80])
            continue
        if plan.ok and plan.trajectory is not None:
            entry.update({"option": item["name"], "plan_attempt": index + 1, "waypoints": len(plan.trajectory.joint_positions)})
            return item if execute(plan.trajectory, arm, entry, gripper) else None
        entry.setdefault("plan_errors", []).append(error_text(plan))
    entry["fail"] = "integrated planner refused every option"
    return None


def hold_still(arm, label, gripper):
    """One padded chunk (30 rows) at the last commanded joints: GenieSim evaluates its
    checkers every 30 steps, so this guarantees one evaluation in the current pose."""
    target = last_command.get(arm) or current_joints(arm)
    entry = {"label": label}
    report["moves"].append(entry)
    try:
        plan = plan_motion(target, arm=arm, strategy=STRATEGY_FINAL)
    except Exception as exc:
        entry["fail"] = "exception: " + str(exc)[:80]
        return False
    if not plan.ok or plan.trajectory is None:
        entry["fail"] = error_text(plan)
        return False
    return execute(plan.trajectory, arm, entry, gripper)


def tool_pose(arm):
    pose = get_robot_state().end_effector_poses[arm]
    return [float(v) for v in pose.position], quat_normalize([float(v) for v in pose.quaternion_wxyz])


def gripper_opening(arm):
    return float(get_robot_state().gripper_positions.get(arm, 1.0))


# ----------------------------------------------------------------------------- perception
def wrist_camera(arm):
    return "hand_left" if arm == "left" else "hand_right"


def cloud_of(mask, camera):
    try:
        cloud = mask_to_point_cloud(mask, camera, target_frame=frame)
    except Exception:
        return None
    if cloud is None or len(cloud.points) < MIN_POINTS:
        return None
    return cloud


def text_instances(camera, text, min_score=MIN_SCORE, max_extent=0.3):
    """Plausible SAM 3 instances of `text` (score-sorted): [(score, summary, box)]."""
    try:
        segs = segment_text(camera, text)
    except Exception:
        return []
    found = []
    if not segs.ok:
        return found
    for seg in segs.segmentations:
        if float(seg.score) < min_score:
            continue
        summary = summarize(cloud_of(seg.mask, camera))
        if plausible(summary, max_extent):
            box = None if seg.box_xyxy is None else [float(v) for v in seg.box_xyxy]
            found.append((float(seg.score), summary, box))
    return found


def point_instances(camera, pixel, max_extent=0.3):
    """Plausible SAM 3 masks for one point prompt (score-sorted, up to three)."""
    try:
        segs = segment_points(camera, [(float(pixel[0]), float(pixel[1]))])
    except Exception:
        return []
    found = []
    if not segs.ok:
        return found
    for seg in segs.segmentations[:3]:
        summary = summarize(cloud_of(seg.mask, camera))
        if plausible(summary, max_extent):
            found.append((float(seg.score), summary, None))
    return found


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def query_variants(queries):
    """Each query, its short form, and its known synonyms, without repeats."""
    variants = []
    for query in queries:
        if not query:
            continue
        text = " ".join(str(query).lower().split())
        for candidate in [text, text.split(",")[0].split(" with ")[0].strip()] + SYNONYMS.get(text, []) + SYNONYMS.get(strip_article(text), []):
            if candidate and candidate not in variants:
                variants.append(candidate)
    return variants


def strip_article(text):
    for article in ("the ", "a ", "an "):
        if text.startswith(article):
            return text[len(article):]
    return text


def tool_frame(arm):
    """FK TCP position, quaternion and its axes (x = closing axis, y, z = approach) in base."""
    position, quat = tool_pose(arm)
    return position, quat, [quat_column(quat, 0), quat_column(quat, 1), quat_column(quat, 2)]


def to_base(position, axes, local):
    return [position[i] + axes[0][i] * local[0] + axes[1][i] * local[1] + axes[2][i] * local[2] for i in range(3)]


def to_local(axes, vector):
    return [sum(axes[k][i] * vector[i] for i in range(3)) for k in range(3)]


def wrist_view(arm):
    """Pose and intrinsics of the wrist camera (it rides on the gripper), or None."""
    cam = wrist_camera(arm)
    try:
        view = get_observation([cam]).cameras[cam]
        quat = quat_normalize([float(v) for v in view.camera_pose.quaternion_wxyz])
        k = view.intrinsics
        return {"position": [float(v) for v in view.camera_pose.position],
                "axes": [quat_column(quat, 0), quat_column(quat, 1), quat_column(quat, 2)],
                "fx": float(k[0][0]), "fy": float(k[1][1]), "cx": float(k[0][2]), "cy": float(k[1][2])}
    except Exception:
        return None


def project(view, point):
    """Pixel of a base-frame point in the wrist image (OpenCV optical frame), or None."""
    local = to_local(view["axes"], [point[i] - view["position"][i] for i in range(3)])
    if local[2] < 0.02:
        return None
    return (view["fx"] * local[0] / local[2] + view["cx"], view["fy"] * local[1] / local[2] + view["cy"])


def finger_points(arm):
    """Base-frame points of both distal pads, both knuckles and the gripper body (FK)."""
    position, quat, axes = tool_frame(arm)
    return [to_base(position, axes, local) for local in PADS_TOOL[arm] + GRIPPER_PARTS_TOOL[arm]]


def finger_check(arm):
    """The differential the v2.1/v3 pad prompts meant to measure, done on the actual fingers:
    each distal pad is segmented with a point prompt at its FK pixel (only pad-sized masks
    near the FK position count) and compared with FK. The common-mode residual is the
    camera-vs-FK bias; the difference along the closing axis is only the jaw opening, so
    agreement is checked off that axis. Logged; applied only if FINGER_BIAS_APPLY."""
    cam = wrist_camera(arm)
    check = {"pads": [], "bias": None, "consistent": False, "applied": False}
    view = wrist_view(arm)
    if view is None:
        check["error"] = "no wrist view"
        return check
    position, quat, axes = tool_frame(arm)
    residuals = []
    for local in PADS_TOOL[arm]:
        expected = to_base(position, axes, local)
        pixel = project(view, expected)
        entry = {"pixel": None if pixel is None else [round(pixel[0]), round(pixel[1])]}
        check["pads"].append(entry)
        if pixel is None or not (0.0 <= pixel[0] < 2.0 * view["cx"] and 0.0 <= pixel[1] < 2.0 * view["cy"]):
            continue
        try:
            segs = segment_points(cam, [(float(pixel[0]), float(pixel[1]))])
        except Exception:
            continue
        if not segs.ok:
            continue
        best = None
        for seg in segs.segmentations[:3]:
            cloud = cloud_of(seg.mask, cam)
            if cloud is None or not (PAD_MASK_POINTS[0] <= len(cloud.points) <= PAD_MASK_POINTS[1]):
                continue
            pts = sampled(cloud.points, 2000)
            middle = [quantile([float(q[i]) for q in pts], 0.5) for i in range(3)]
            gap = math.sqrt(sum((middle[i] - expected[i]) ** 2 for i in range(3)))
            if gap <= PAD_GATE_M and (best is None or gap < best[0]):
                best = (gap, middle, len(cloud.points))
        if best is not None:
            entry["residual_tool"] = r3(to_local(axes, [best[1][i] - expected[i] for i in range(3)]))
            entry["points"] = best[2]
            residuals.append([best[1][i] - expected[i] for i in range(3)])
    if len(residuals) < 2:
        report["finger_misses"] = report.get("finger_misses", 0) + 1
        return check
    bias = [0.5 * (residuals[0][i] + residuals[1][i]) for i in range(3)]
    spread = to_local(axes, [residuals[0][i] - residuals[1][i] for i in range(3)])
    size = math.sqrt(bias[0] ** 2 + bias[1] ** 2)
    check["bias"] = r3(bias)
    check["bias_xy_m"] = round(size, 4)
    check["consistent"] = math.sqrt(spread[1] ** 2 + spread[2] ** 2) < PAD_AGREE_M
    check["applied"] = FINGER_BIAS_APPLY and check["consistent"] and FINGER_BIAS_RANGE_M[0] <= size <= FINGER_BIAS_RANGE_M[1]
    return check


def is_finger(summary, fingers, top_limit):
    """The fingers are dark objects in every wrist frame: an instance centred near a pad, a
    knuckle or the gripper body (FK), or above the object's measured top, is the gripper."""
    if summary["center"][2] > top_limit:
        return True
    for point in fingers:
        if math.sqrt(sum((summary["center"][i] - point[i]) ** 2 for i in range(3))) < FINGER_RADIUS_M:
            return True
    return False


def size_matches(seen, reference):
    """Same object by footprint: each side within SIZE_MATCH (relative) or 1 cm."""
    for key in ("length", "width"):
        a, b = seen[key], reference[key]
        if abs(a - b) > max(0.01, SIZE_MATCH * max(a, b)):
            return False
    return True


def wrist_measure(arm, target, aim):
    """The tracked object in the wrist camera and where the TCP is relative to it.
    The object is the plausible SAM 3 instance whose footprint centre is nearest (xy) to the
    current aim, within TRACK_RADIUS_M, fingers excluded. Tracking by position keeps the
    identity the head-camera chooser decided (two blocks of one colour score alike)."""
    cam = wrist_camera(arm)
    fingers = []
    try:
        fingers = finger_points(arm)
    except Exception:
        pass
    top_limit = target["summary"]["top"] + 0.05  # fingers hover >= 9 cm above the top
    seen = None
    used = None
    rejected = 0
    for query in target["track_queries"]:
        near = []
        for score, summary, box in text_instances(cam, query, min_score=TRACK_MIN_SCORE):
            if is_finger(summary, fingers, top_limit):
                rejected += 1
                continue
            gap = xy_distance(summary["grasp_xy"], aim)
            if gap <= TRACK_RADIUS_M:
                near.append((gap, -score, summary))
        if near:
            near.sort(key=lambda item: (item[0], item[1]))
            seen = near[0][2]
            used = query
            break
    measure = {"aim": r3(aim[:2]), "query": used, "object": describe(seen), "fingers_rejected": rejected}
    try:
        position, quat, axes = tool_frame(arm)
        measure["tool"] = r3(position)
        measure["tool_quat"] = r3(quat)
        measure["tcp_to_aim"] = r3([aim[0] - position[0], aim[1] - position[1]])
        if seen is not None:
            offset = [seen["grasp_xy"][0] - position[0], seen["grasp_xy"][1] - position[1], 0.0]
            measure["tcp_to_object"] = r3(offset[:2])
            # along (x) / across (y) the closing axis: across is what small objects cannot tolerate
            measure["tcp_to_object_tool"] = r3(to_local(axes, offset)[:2])
    except Exception:
        pass
    if seen is not None:
        measure["object_minus_aim"] = r3([seen["grasp_xy"][0] - aim[0], seen["grasp_xy"][1] - aim[1]])
        measure["same_size"] = size_matches(seen, target["summary"])
    return measure, seen


# ----------------------------------------------------------------------------- task understanding
def parse_sentence(text):
    """Rule parser: arm, object phrase (first clause), and the attribute words the choosers use."""
    lowered = " ".join(text.lower().strip().rstrip(".").split())
    arm = None
    for side in ("left", "right"):
        if (side + " arm") in lowered or (side + " hand") in lowered:
            arm = side
            break
    clause = lowered.split(",")[0].strip()
    for side in ("left", "right"):
        for lead in (side + " arm ", side + " hand ", "the " + side + " arm ", "the " + side + " hand "):
            if clause.startswith(lead):
                clause = clause[len(lead):]
    for verb in ("picks up the ", "picks up ", "pick up the ", "pick up ", "picks the ", "pick the ", "grabs the ", "grab the ", "takes the ", "take the "):
        if clause.startswith(verb):
            clause = clause[len(verb):]
            break
    for location in (" on the table", " from the table", " on the desk", " from the desk"):
        if clause.endswith(location):
            clause = clause[: -len(location)]
    phrase = " ".join(clause.split())
    words = phrase.replace("-", " - ").split()
    parsed = {"arm": arm, "phrase": phrase, "size_rule": None, "digit": None, "shape": None, "color": None, "alternatives": []}
    for word in ("largest", "biggest", "smallest", "tiniest"):
        if word in lowered:
            parsed["size_rule"] = "largest" if word in ("largest", "biggest") else "smallest"
    if "number" in lowered:
        for token in lowered.replace(",", " ").split():
            if token.isdigit():
                parsed["digit"] = token
    for word in SHAPE_WORDS:
        if word in lowered:
            parsed["shape"] = word
            break
    for word in COLOR_WORDS:
        if word in words:
            parsed["color"] = word
            break
    if " or " in phrase:
        parsed["alternatives"] = [strip_article(part.strip()) for part in phrase.split(" or ") if part.strip()]
    return parsed


def kind_from_sentence(text, parsed):
    lowered = text.lower()
    if "straighten" in lowered or "original position" in lowered:
        return "straighten_object"
    if parsed["digit"] is not None or "shaped like number" in lowered:
        return "pick_block_number"
    if "billiard" in lowered:
        return "pick_billiards_color"
    if parsed["size_rule"] is not None:
        return "pick_block_size"
    if parsed["alternatives"]:
        return "pick_follow_logic_or"
    if parsed["arm"] is None:
        return "pick_common_sense"
    if "block" in lowered and parsed["shape"] is not None:
        return "pick_block_shape"
    if "block" in lowered and parsed["color"] is not None:
        return "pick_block_color"
    for word in CATEGORY_QUERIES:
        if word in lowered:
            return "pick_object_type"
    return "pick_specific_object"


def resolve(text):
    """g2.resolve_instruction (never raises; Qwen3.6 on the head image) plus its summary."""
    try:
        resolved = g2.resolve_instruction(text)
    except Exception as exc:
        resolved = {"ok": False, "error": str(exc)[:120]}
    if not resolved:
        resolved = {"ok": False}
    keys = ("ok", "arm", "query", "alternatives", "box_xyxy", "selection_rule", "confidence", "visible", "latency_s", "error")
    report["resolver"] = {k: resolved.get(k) for k in keys}
    return resolved


def in_category(query, category):
    words = str(query).lower().replace("-", " ").replace(",", " ").split()
    drink = any(word in BEVERAGE_WORDS for word in words)
    if category == "beverage":
        return drink
    if category == "fruit":
        return (not drink) and any(word in FRUIT_WORDS for word in words)
    return any(word in STATIONERY_WORDS for word in words)


def text_queries(kind, parsed, resolved):
    """Ordered SAM 3 queries for the text choosers: the resolver's concrete object first,
    then the rule parser's phrase and the category / common-sense fallbacks."""
    queries = []
    if resolved.get("ok") and resolved.get("query"):
        queries.append(resolved["query"])
        queries.extend(resolved.get("alternatives") or [])
    lowered = context.language.lower()
    if kind == "pick_common_sense":
        for words, category, fallback in COMMON_SENSE_CUES:
            if any(word in lowered for word in words):
                agrees = bool(queries) and in_category(queries[0], category)
                report["common_sense"] = {"category": category, "resolver_query": queries[0] if queries else None, "resolver_in_category": agrees}
                queries = (queries + fallback) if agrees else (fallback + queries)
                break
    elif kind in ("pick_object_type", "pick_follow_logic_or"):
        parts = parsed["alternatives"] or [parsed["phrase"]]
        for part in parts:
            queries.append(part)
            for word, fallback in CATEGORY_QUERIES.items():
                if word in part:
                    queries.extend(fallback)
    else:
        queries.append(parsed["phrase"])
    return query_variants(queries)


# ----------------------------------------------------------------------------- choosers
def target_of(summary, how, track_queries, **extra):
    target = {"summary": summary, "how": how, "track_queries": track_queries}
    target.update(extra)
    return target


def choose_by_text(queries, limit=8):
    """The best-scoring plausible head instance of the first query that finds one."""
    tried = []
    for query in queries[:limit]:
        tried.append(query)
        found = text_instances("head", query)
        if found:
            score, summary, box = found[0]
            return target_of(summary, "text", [query] + [q for q in queries if q != query][:2], query=query, score=round(score, 3), tried=tried)
    report["text_queries_tried"] = tried
    return None


def choose_by_box(box, noun, max_extent=0.15):
    """Fine-grained choices (printed number, shape) are made by the VLM on the image: take the
    mask under the centre of its box (plausible masks only), else the SAM 3 instance of the
    noun that overlaps its box most."""
    if not box:
        return None
    centre = ((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0)
    found = point_instances("head", centre, max_extent)
    instances = text_instances("head", noun, max_extent=max_extent)
    overlaps = sorted([(box_iou(box, item[2]), item) for item in instances if item[2] is not None], key=lambda pair: -pair[0])
    diag = {"box": [round(float(v), 1) for v in box], "point_masks": len(found), "noun_instances": len(instances),
            "best_iou": round(overlaps[0][0], 3) if overlaps else None}
    report["box_choice"] = diag
    if found:
        summary = found[0][1]
        if overlaps and overlaps[0][0] > 0.1:
            diag["point_vs_iou_xy_m"] = round(xy_distance(summary["center"], overlaps[0][1][1]["center"]), 4)
        return target_of(summary, "box_point", [noun], score=round(found[0][0], 3))
    if overlaps and overlaps[0][0] >= 0.2:
        return target_of(overlaps[0][1][1], "box_iou", [noun], iou=round(overlaps[0][0], 3))
    return None


def choose_by_size(rule, noun):
    """largest / smallest: every plausible instance of the noun class (not the VLM's colour
    guess, sweep 10-22) compared by footprint volume; implausibly large masks are dropped."""
    seen = []
    for query in (noun, "cube", "building block"):
        for score, summary, box in text_instances("head", query, max_extent=0.12):
            if all(xy_distance(summary["center"], other["center"]) > 0.02 for other in seen):
                seen.append(summary)
        if len(seen) >= 4:
            break
    report["size_candidates"] = [{"center": r3(s["center"]), "volume_cm3": round(1e6 * volume(s), 1)} for s in seen]
    if not seen:
        return None
    seen.sort(key=volume, reverse=(rule == "largest"))
    return target_of(seen[0], "size_" + rule, [noun], candidates=len(seen))


def shape_diagnostics(noun):
    """Footprint features of every head instance of the noun (data for the next shape chooser)."""
    rows = []
    for score, summary, box in text_instances("head", noun, max_extent=0.15):
        row = describe(summary)
        row["score"] = round(score, 3)
        row["box"] = None if box is None else [round(v) for v in box]
        rows.append(row)
    report["shape_features"] = rows


def choose_target(kind, parsed, resolved):
    box = resolved.get("box_xyxy") if resolved.get("ok") else None
    noun = "billiard ball" if kind == "pick_billiards_color" else "block"
    queries = text_queries(kind, parsed, resolved)
    target = None
    if kind == "pick_block_size":
        target = choose_by_size(parsed["size_rule"] or str(resolved.get("selection_rule")), noun)
        if target is None:
            target = choose_by_box(box, noun)
    elif kind in ("pick_block_number", "pick_block_shape"):
        if kind == "pick_block_shape":
            shape_diagnostics(noun)
        target = choose_by_box(box, noun)
        if target is None:
            target = choose_by_text(queries)
    else:
        target = choose_by_text(queries)
        if target is None:
            target = choose_by_box(box, noun, max_extent=0.3)
    if target is not None and kind in ("pick_block_number", "pick_block_shape", "pick_block_size", "pick_block_color"):
        # Track blocks in the wrist camera by the generic noun as well (SAM 3 does not read
        # "shaped like number 1"; the nearest block to the pads is the chosen one).
        if noun not in target["track_queries"]:
            target["track_queries"] = target["track_queries"] + [noun]
    return target


# ----------------------------------------------------------------------------- grasp
def approach(center, grasp_z, options, arm, summary, attempt):
    """Hover ABOVE_M over the grasp point: v2.1's ranked IK at two heights, then the
    integrated (scene-aware) planner, then steeper reach-direction tilts."""
    hover = (center[0], center[1], grasp_z + ABOVE_M)
    high = (center[0], center[1], grasp_z + ABOVE_M + ABOVE_HIGH_M)
    for label, position in (("above", hover), ("above_high", high)):
        chosen = move_ik(position, options, arm, label, gripper=1.0)
        if chosen is not None:
            attempt["approach"] = label
            return chosen
        if report["terminated"]:
            return None
    chosen = move_pose(hover, sorted(options, key=lambda item: item["rank"]), arm, "above_integrated", gripper=1.0)
    if chosen is None and not report["terminated"]:
        chosen = move_pose(hover, escalation_options(summary, center, arm), arm, "above_tilted", gripper=1.0, limit=4)
    if chosen is not None:
        attempt["approach"] = report["moves"][-1]["label"]
    return chosen


def lift(arm, center, grasp_z, chosen, attempt):
    """Lift with the grasp orientation; if its IK fails, with the measured tool orientation and
    shorter lifts (PickUpOnGripper needs 2 cm)."""
    if move_ik((center[0], center[1], grasp_z + LIFT_M), [chosen], arm, "lift", gripper=0.0, strategy=STRATEGY_FINAL, samples=2) is not None:
        attempt["lift"] = LIFT_M
        return True
    if report["terminated"]:
        return False
    position, quat = tool_pose(arm)
    for height in LIFT_FALLBACK_M:
        target = (position[0], position[1], position[2] + height)
        if move_ik(target, [option("measured", quat, 0)], arm, "lift_measured", gripper=0.0, strategy=STRATEGY_FINAL, samples=2) is not None:
            attempt["lift"] = height
            return True
        if report["terminated"]:
            return False
    attempt["lift"] = 0.0
    return False


def descend_and_close(arm, aim, grasp_z, chosen, cloud, attempt, label):
    """Joint-interpolated descend with the TCP to the aim at the grasp height, close, lift, and
    read the jaw opening (current only after the lift). The pre-close wrist view is logged only
    (TCP vs object along / across the jaw): from the descend pose the camera sees mostly the
    near side face, so its centre estimate sits ~2 cm towards the camera, p90 3.8 cm from the
    truth; the only v3 cases where it disagreed with the head by > 4 cm (straighten s31/s32)
    were 4.3-5.1 cm off while the head was 1.5-1.6 cm off (10-24 vs simulator truth)."""
    if move_ik((aim[0], aim[1], grasp_z), [chosen], arm, label, gripper=1.0, strategy=STRATEGY_FINAL, exclude=cloud, samples=3) is None:
        attempt["fail"] = label
        return False, aim
    if report["terminated"]:
        return False, aim
    measure, seen = wrist_measure(arm, attempt["target"], aim)
    attempt[label + "_measure"] = measure
    closed = close_gripper(arm=arm)
    count_rows(closed)
    if closed.terminated:
        return False, aim
    lifted = lift(arm, aim, grasp_z, chosen, attempt)
    opening = gripper_opening(arm)
    attempt.setdefault("openings", []).append(round(opening, 4))
    attempt["lifted"] = lifted
    return closed.ok and lifted and opening >= AIR_OPENING, aim


def pick(target, arm, options):
    """Approach with the TCP above the object's footprint centre (head camera), check at hover
    (grasp height and gross moves from the wrist view, finger differential logged), descend,
    close, lift, verify; one retry when the jaws closed on air."""
    summary = target["summary"]
    aim = (summary["grasp_xy"][0], summary["grasp_xy"][1], summary["center"][2])
    grasp_z = grasp_height(summary)
    cloud = summary["cloud"]
    attempt = {"arm": arm, "target": target, "center": r3(summary["center"]), "aim": r3(aim[:2]), "top": round(summary["top"], 4),
               "bottom": round(summary["bottom"], 4), "grasp_z": round(grasp_z, 4), "options": [item["name"] for item in options],
               "rows_start": report["rows"], "aim_source": "head"}
    report["attempts"].append(attempt)
    chosen = approach(aim, grasp_z, options, arm, summary, attempt)
    if chosen is None:
        attempt["fail"] = "approach"
        return False
    attempt["option"] = chosen["name"]
    # At hover the head view is occluded: the wrist camera gives the grasp height, the finger
    # differential (logged) and a guard against an object that is not where the head saw it.
    fingers = finger_check(arm)
    attempt["finger_check"] = fingers
    measure, seen = wrist_measure(arm, target, aim)
    attempt["hover_measure"] = measure
    if seen is not None:
        grasp_z = grasp_height(seen, summary)
        cloud = seen["cloud"]
        target["hover_seen"] = seen
        gap = xy_distance(seen["grasp_xy"], aim)
        if WRIST_OVERRIDE_M < gap <= HOVER_SHIFT_MAX_M and measure.get("same_size"):
            aim = (seen["grasp_xy"][0], seen["grasp_xy"][1], aim[2])
            attempt["aim_source"] = "wrist"
            if fingers.get("applied"):
                # a wrist-camera aim carries the camera's own bias; the pads measured it
                aim = (aim[0] - fingers["bias"][0], aim[1] - fingers["bias"][1], aim[2])
                attempt["aim_source"] = "wrist-finger_bias"
    attempt["aim_used"] = r3(aim[:2])
    # v3.2: on a steep escalation keep the lower fingertip TIP_CLEARANCE_M above the table (the
    # head camera's object bottom; a wrist cloud from above may hold only the top face). The
    # 35/50-degree grasps of v3 + v3.1 held 1/5, the one held with that fingertip 8 mm up, the
    # misses at <= 3 mm. 20-degree and vertical grasps keep v3.1's floor.
    steep = approach_tilt(chosen["quat"]) > STEEP_TILT_RAD
    tip = tip_drop(chosen["quat"])
    table_floor = summary["bottom"] + (TIP_CLEARANCE_M + tip if steep else FLOOR_CLEARANCE_M)
    attempt["tip_drop"] = round(tip, 4)
    if steep and grasp_z < table_floor:
        attempt["grasp_z_raised_from"] = round(grasp_z, 4)
        grasp_z = table_floor
    attempt["grasp_z_used"] = round(grasp_z, 4)
    held, aim = descend_and_close(arm, aim, grasp_z, chosen, cloud, attempt, "descend")
    if not held and not report["terminated"] and attempt.get("lifted") and attempt["openings"][-1] < AIR_OPENING:
        # Closed on air at the lifted pose (~LIFT_M above the grasp): re-open, look again from
        # here like a hover (the close may have pushed the object), descend 5 mm lower, close once more.
        retry = {"first_opening": attempt["openings"][-1], "rows_before": report["rows"], "aim_source": "previous"}
        attempt["retry"] = retry
        opened = open_gripper(arm=arm)
        count_rows(opened)
        # The finger check and the tracking assume an open jaw; give it three ticks to finish opening.
        settle = move_to_joints(last_command.get(arm) or current_joints(arm), tolerance=0.001, max_steps=3, arm=arm)
        count_rows(settle)
        measure, seen = wrist_measure(arm, target, aim)
        retry["measure"] = measure
        if seen is not None:
            grasp_z = grasp_height(seen, summary)
            cloud = seen["cloud"]
            if measure.get("same_size") and RETRY_MOVED_M < xy_distance(seen["grasp_xy"], aim) <= TRACK_RADIUS_M:
                aim = (seen["grasp_xy"][0], seen["grasp_xy"][1], aim[2])
                retry["aim_source"] = "wrist"
        # lower, but never below the floor (v3.1: never raise it; a steep escalation: raise it to
        # the fingertip floor if the wrist height is under it)
        if steep:
            grasp_z = max(grasp_z - RETRY_LOWER_M, table_floor)
        else:
            grasp_z = max(grasp_z - RETRY_LOWER_M, min(grasp_z, table_floor))
        retry["aim_used"] = r3(aim[:2])
        retry["grasp_z_used"] = round(grasp_z, 4)
        if not report["terminated"]:
            held, aim = descend_and_close(arm, aim, grasp_z, chosen, cloud, attempt, "retry_descend")
    attempt["held"] = held
    attempt["gripper_after"] = attempt.get("openings", [None])[-1]
    attempt["rows_end"] = report["rows"]
    if held and not report["terminated"]:
        hold_still(arm, "hold_lifted", 0.0)
    return held


# ----------------------------------------------------------------------------- straighten
def bottle_profile(summary):
    """Axis (footprint long axis), length, and which end is the cap: the narrower cross
    section near an end (straighten_v1)."""
    pts = sampled(summary["cloud"].points, 3000)
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    yaw = summary["yaw"]
    d = (math.cos(yaw), math.sin(yaw))
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    along = [(x - mx) * d[0] + (y - my) * d[1] for x, y in zip(xs, ys)]
    across = [(y - my) * d[0] - (x - mx) * d[1] for x, y in zip(xs, ys)]
    lo, hi = quantile(along, 0.02), quantile(along, 0.98)
    length = hi - lo

    def width_near(t0, t1):
        sel = [a for a, s in zip(across, along) if t0 <= s <= t1]
        return (quantile(sel, 0.95) - quantile(sel, 0.05)) if len(sel) > 10 else 0.0

    w_pos = width_near(lo + 0.7 * length, hi)
    w_neg = width_near(lo, lo + 0.3 * length)
    cap_sign = 1.0 if w_pos < w_neg else -1.0
    return {"yaw": yaw, "length": length, "cap_dir": (cap_sign * d[0], cap_sign * d[1], 0.0),
            "w_pos": round(w_pos, 4), "w_neg": round(w_neg, 4), "bottom": summary["bottom"]}


def upright(arm, profile):
    """Best current approach (straighten_v1, with the orientation math corrected): rise, then
    rotate the held bottle 90 degrees about the horizontal axis perpendicular to it so the cap
    end points up (minimal rotation from the MEASURED tool orientation), offering the IK a fan
    of yaws about world z (all keep the bottle vertical) at small position offsets; hold one
    evaluation tick upright, set down, release. Upright() only checks the tilt (<= 5 deg)."""
    info = {"cap_dir": r3(profile["cap_dir"]), "head_cap_dir": r3(profile.get("head_cap_dir") or profile["cap_dir"]),
            "w_pos": profile["w_pos"], "w_neg": profile["w_neg"], "length": round(profile["length"], 4)}
    report["upright"] = info
    position, quat = tool_pose(arm)
    hold_z = max(position[2] + 0.04, profile["bottom"] + 0.5 * profile["length"] + 0.05)
    info["hold_z"] = round(hold_z, 4)
    risen = move_ik((position[0], position[1], hold_z), [option("grasp", quat, 0)], arm, "rise", gripper=0.0, strategy=STRATEGY_FINAL, samples=2)
    info["rise_ok"] = risen is not None
    cap = profile["cap_dir"]
    axis = (cap[1], -cap[0], 0.0)  # cap x Z: rotating +90 deg about it takes the cap direction to +Z
    half = quat_mul(quat_axis_angle(axis, 0.25 * math.pi), quat)
    full = quat_mul(quat_axis_angle(axis, 0.5 * math.pi), quat)
    fan = []
    for degrees in (0, 30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180):
        # Every yaw about world z keeps the bottle vertical: let joint distance decide.
        fan.append(option("yaw%+d" % degrees, quat_mul(quat_axis_angle((0.0, 0.0, 1.0), math.radians(degrees)), full), 0))
    offsets = ((0.0, 0.0, 0.0), (-0.06, 0.0, 0.0), (0.0, 0.06, 0.0), (0.0, -0.06, 0.0), (-0.06, 0.0, 0.05), (0.06, 0.0, 0.0), (0.0, 0.0, 0.08))
    pivot = (position[0], position[1], hold_z)
    rotated = None
    for dx, dy, dz in offsets:
        spot = (pivot[0] + dx, pivot[1] + dy, pivot[2] + dz)
        # The 45-degree step is only a waypoint; the full rotation is tried either way (the yaw
        # fan lets the gripper end up pointing in any horizontal direction, e.g. forward).
        half_ok = move_ik(spot, [option("half", half, 0)], arm, "rotate_half", gripper=0.0, strategy=STRATEGY_FINAL, samples=2) is not None
        if report["terminated"]:
            return False
        rotated = move_ik(spot, fan, arm, "rotate_full", gripper=0.0, strategy=STRATEGY_FINAL, keep=6)
        if rotated is not None:
            pivot = spot
            info["half_ok"] = half_ok
            break
        if report["terminated"]:
            return False
    info["rotate_ok"] = rotated is not None
    if rotated is None:
        return False
    info["rotation"] = rotated["name"]
    info["pos"] = r3(pivot)
    hold_still(arm, "hold_upright", 0.0)
    place_z = profile["bottom"] + 0.5 * profile["length"] + 0.012
    info["set_down_ok"] = move_ik((pivot[0], pivot[1], place_z), [rotated], arm, "set_down", gripper=0.0, strategy=STRATEGY_FINAL, samples=2) is not None
    released = open_gripper(arm=arm)
    count_rows(released)
    info["released_ok"] = released.ok
    info["retreat_ok"] = move_ik((pivot[0], pivot[1], place_z + 0.10), [rotated], arm, "retreat", gripper=1.0, strategy=STRATEGY_FINAL, samples=2) is not None
    return info["set_down_ok"] and released.ok


# ----------------------------------------------------------------------------- episode
def run_episode():
    parsed = parse_sentence(context.language)
    kind = context.task_name if context.task_name in TASK_KINDS else kind_from_sentence(context.language, parsed)
    report["kind"] = kind
    report["kind_from_sentence"] = kind_from_sentence(context.language, parsed)
    report["parsed"] = {k: parsed[k] for k in ("arm", "phrase", "size_rule", "digit", "shape", "color", "alternatives")}
    resolved = resolve(context.language)
    stage("choose", kind=kind)
    target = choose_target(kind, parsed, resolved)
    if target is None:
        report["error"] = "no plausible target for " + kind
        return
    summary = target["summary"]
    report["target"] = {"how": target["how"], "track_queries": target["track_queries"], "geometry": describe(summary)}
    for key in ("query", "score", "iou", "candidates", "tried"):
        if key in target:
            report["target"][key] = target[key]
    arm = parsed["arm"] or (resolved.get("arm") if resolved.get("arm") in ("left", "right") else None)
    if arm is None:
        arm = "left" if summary["center"][1] > 0.0 else "right"
    report["arm"] = arm
    stage("grasp", arm=arm)
    if kind == "straighten_object":
        profile = bottle_profile(summary)
        held = pick(target, arm, grasp_options(summary, across_only=True))
        report["held"] = held
        seen = target.get("hover_seen")
        if seen is not None and seen["length"] > 1.5 * seen["width"]:
            # The close-range wrist view at hover gives the finer bottle axis; keep the head's cap end.
            d = (math.cos(seen["yaw"]), math.sin(seen["yaw"]))
            sign = 1.0 if d[0] * profile["cap_dir"][0] + d[1] * profile["cap_dir"][1] >= 0.0 else -1.0
            profile["head_cap_dir"] = profile["cap_dir"]
            profile["cap_dir"] = (sign * d[0], sign * d[1], 0.0)
        if held and not report["terminated"]:
            stage("upright")
            report["upright_done"] = upright(arm, profile)
        else:
            report["upright"] = {"skipped": "object not held"}
            report["upright_done"] = False
    else:
        report["held"] = pick(target, arm, grasp_options(summary))
    stage("done")


try:
    run_episode()
except Exception as exc:
    report["error"] = "exception: " + str(exc)[:300]
for attempt in report["attempts"]:
    attempt.pop("target", None)
report["object_query"] = (report.get("target") or {}).get("query")
last_attempt = report["attempts"][-1] if report["attempts"] else {}
report["summary"] = {
    "kind": report.get("kind"),
    "arm": report.get("arm"),
    "chooser": (report.get("target") or {}).get("how"),
    "approach": last_attempt.get("approach"),
    "aim_source": last_attempt.get("aim_source"),
    "finger_bias_xy_m": (last_attempt.get("finger_check") or {}).get("bias_xy_m"),
    "tcp_to_object_tool_at_close": (last_attempt.get("retry_descend_measure") or last_attempt.get("descend_measure") or {}).get("tcp_to_object_tool"),
    "fail": last_attempt.get("fail"),
    "retry_on_air": "retry" in last_attempt,
    "lift_m": last_attempt.get("lift"),
    "gripper_after": last_attempt.get("gripper_after"),
    "held": report.get("held"),
    "upright_done": report.get("upright_done"),
    "rows": report["rows"],
    "terminated": report["terminated"],
    "error": report["error"],
}
result = report
