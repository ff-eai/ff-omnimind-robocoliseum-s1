# G2 / GenieSim instruction board, skill library v3.6 (2026-10-28, Opus 5.5).
#
# v3.6 = v3.5 + fixes for every failure cause v3.4 / v3.5 left on the lb-style runs and job 1675
# (exec/skill_library_v3_6_2026-10-28.md, evidence/2026-10-28_v36):
#   * Grasp yaw by wrist-camera side (all tasks but straighten). The two "equivalent" yaws of a closing
#     direction are not: with the wrist camera / 9 mm pad offset toward the arm's table edge, 20-degree
#     grasps held 64/90 over every logged run, toward the robot's midline 878/908; on objects under
#     2.2 cm 8/23 vs 75/77. The outward yaw of each pair now comes after the other one and its vertical;
#     a square footprint (cube, hexagon, disc, cylinder) closes alike along either axis, so its four
#     yaws are ordered by camera side (the 2 cm cube: midline 5/5, fore/aft 1/4).
#   * Far-edge approach: the cause was the harness, not the program. The wrist cameras render through
#     OpenCV lens distortion (USD k1 ~ -0.04) that the planning scene ignored; each wrist camera sees the
#     OTHER start-pose gripper at its image corner, ~5 cm off, outside cuRobo's robot carve, so every
#     hover under a start-pose gripper was "goal state infeasible (scene_collision)". Fixed in the
#     harness (branch g2-v36: planning scene back-projected through each camera's lens). The program
#     keeps v3.5's restage as the fallback and enables it for every task (RESTAGE_KINDS).
#   * Object type: an empty / unparseable resolver answer is asked again with the category spelled out
#     (test s42: no answer, then SAM 3 "toys" found the orange); the chosen instance is checked against
#     the category with SAM 3's own category queries (test s39: "pink toy" on the pink pen).
#   * Block colour: when SAM 3's two best instances of the colour query score within 0.05, the one under
#     the resolver's box is taken (s33 "green": yellow cube 0.492 vs green 0.48, Qwen's box on the green).
#   * A descend with no IK solution approaches again with the remaining options (job 1675: a far-forward
#     cube reached its hover with a vertical grasp that had no IK at the grasp height).
#   * Block number (after the stage-1 run): votes and fitting blocks more than SIDE_MARGIN_M across the
#     midline from the instructed arm lose to any on its side (every one of the 657 upstream instructions
#     that names an arm has its target on that side; stage-1 s31 "2": the head voted for a far-side block).
#   * report["v36_flags"] lists every decision that differs from what v3.5 would do.
# Everything else is v3.5.
#
# v3.5 = v3.4 + block shapes by geometry over three views, the object-type category / side checks and
# the far-edge restage (exec/skill_library_v3_5_2026-10-27.md; its notes sit at the v3.5 constants).
#
# v3.4 = v3.3 + the pick_block_number fallbacks (decision logic only; exec/skill_library_v3_4_2026-10-26.md).
# Both v3.3 held-out number losses (seeds 31-50) came from fallback paths development never reached:
#   * s34 ("6"): the head reply was unparseable, the right wrist named "the purple block shaped like
#     number 8" (2 holes, ruled out), the third view was skipped by the 60 s budget, and the last
#     fallback took the ruled-out block although two blocks fit the digit. Now, when no answer
#     survives the shape filter, the remaining view is asked while the resolver time is under
#     VLM_RESCUE_BUDGET_S (70 s), and a vote the shape filter contradicts is never taken while a
#     block that fits exists (several fit and every view is spent: a guess among them).
#   * s35 ("0"): three unanimous answers named a block SAM 3 had not segmented in either wrist view,
#     and "the only census block that fits" overrode them. An answer whose block is not in the census
#     now counts as a vote on a block of unknown shape (votes within CUBE_MERGE_M are one block), so
#     it beats a guess from shape alone, as votes on census blocks of unknown shape already did.
# Offline: the 28 logged v3.3 number runs replayed through this choose_number(): 26/28 -> 27/28 right
# (s35; s34 asks the left wrist, which the same scene's dev probe answered right); dev probes with
# real SAM 3 and injected faults, v3.3 -> v3.4: real answers 12/12 -> 12/12 (same decisions),
# s34-like 7 -> 11, s35-like 7 -> 12, every view wrong 7 -> 9 of 12, never worse.
# Everything else is v3.3.
#
# v3.3 = v3.2 + fixes for the three remaining failure classes of the v3.2 formal run (177/200),
# developed on seeds 51-70 only (exec/skill_library_v3_3_2026-10-25.md, evidence/2026-10-25_v33_dev):
#   * straighten_object: 5 of the 6 held-but-not-upright failures were cans set UPSIDE DOWN
#     (Upright wants the asset's local +y up, allow_flipped=False; v3.2's "narrower end = cap"
#     is a coin flip on cans, whose two ends are equally wide). The cap guess is now the end
#     whose top silhouette sits lower near the end (neck taper; head-camera dev probes 12/12 on
#     Coca-Cola, Fanta, oolong), reversed for the Sprite (its top end sits raised: the cue took
#     the bottom on all 5 dev instances), and upright() then also tries the OTHER end up:
#     Upright latches and GenieSim ends the episode on success, so the second orientation only
#     runs after a wrong guess. Paid for by dropping v3.2's 4 cm rise and 45-degree waypoint
#     (PickUpOnGripper is not scored in straighten); the post-lift hold stays, each rotation is
#     the least joint motion over a fan of yaws, holds end just past the next evaluation step,
#     and the hover starts 17 cm up (Follow opens Upright's 300-step window).
#   * pick_block_shape ("cube"): Qwen named a bigger block "cube" in the 2 cm-cube scenes, even on
#     the high-resolution wrist images. Every block seen by the head and both wrist cameras at the
#     start pose (the wrists see the whole table at ~1 mm/px, no motion) gets a top-face test:
#     square, side ~ height, all four corners occupied. Qwen's block is kept if it passes, else the
#     geometric cube is taken (dev replay: v3.2 chooser 8/14, v3.3 14/14).
#   * pick_block_number: Qwen reads the digit on the head and both wrist images; an answer whose
#     block contradicts the digit's hole count / narrowness (wrist depth, 93/95 right) is dropped,
#     then the most-voted block wins (dev replay: v3.2 11/12, v3.3 12/12). The v3.2 "planner
#     refusal" s45 was also a misread (left arm sent to a far-right "2" instead of the "0").
# Everything else is v3.2.
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

# v3.3: "cube" is decided by geometry, seen by the head AND both wrist cameras at the start pose
# (no motion: the wrists already see the whole table, ~1 mm/px against the head's ~2.3 mm/px).
# Dev probes 10-25 (seeds 51-70): Qwen named a larger block "cube" in the 2 cm-cube scenes (7 of
# 9 views wrong on s53-s55, the wrist views included); a cube's top face is a square whose side
# equals its height and whose four corners are all occupied: corner occupancy 0.79-1.47 on
# cubes, 0.00 on the L-shape (its notch), 0.19-0.54 on discs, cylinders and hexagonal prisms.
CUBE_VIEWS = ("head", "hand_left", "hand_right")
CUBE_QUERIES = ("block", "cube")
CUBE_MIN_SCORE = 0.1        # head "block" scores were 0.15-0.17 for every block of one dev scene (s54)
CUBE_ASPECT_MIN = 0.8       # top face short / long side
CUBE_HEIGHT_RATIO = (0.78, 1.3)  # height above the table / top-face side
CUBE_CORNER_MIN = 0.6       # least occupied corner (20 % squares) / its share of a full rectangle
CUBE_SIDE_M = (0.015, 0.065)
CUBE_MERGE_M = 0.025        # instances of one block in different views / queries lie this close

# v3.5: every shape of the block-shape board, by geometry (exec/skill_library_v3_5_2026-10-27.md). Each
# scene holds one block of each class: cube (5 or 2 cm), rectangular prism (6 x 3 x 3, 6 x 1.5 x 1.5 or
# 2.7 x 1.9 x 1.5 cm), triangular prism (lying on a 5.8 x 2.4 cm face, 45-degree roof 2.9 cm high),
# cylinder (standing: 3 x 6, 5 x 2.5, 2.4 x 1.4 cm, or a 4.7 x 2.9 x 4 cm elliptic one), hexagonal
# prism (2.5 x 2.7 x 1.6 cm, on its hexagon) and an L-shaped block (3.8 x 3.5 x 3.6 cm, L from above).
# Per view (dev probes, 14 scenes, 223 block views, wrist depth at the wire's 1 mm): a roof's top
# over the end fifths of its long axis is 0.32-0.42 of the top over the middle fifth (every flat top
# >= 0.98); the L's notch leaves one corner of its top-face rectangle empty (0.00 in 40/40 views)
# while the others are full (>= 1.95); cubes and rectangular prisms fill all four corners (>= 0.55
# but 2 partial views); cylinders and the hexagon leave all four partly empty (<= 1.14); a wrist
# view tells the hexagon from a cylinder by the sixfold harmonic of its top-face outline (0.039-0.060
# vs 0.001-0.012; the head's 2.3 mm/px cannot: 0.004-0.27). Qwen's block was wrong in all 18
# rectangular / triangular / hexagonal all-instruction episodes on seeds 31-33, so it only breaks ties.
SHAPE_QUERIES = ("block", "cube", "cylinder")  # "cylinder": the 3 x 6 cm cylinder of s53 is no "block" in any view
SHAPE_CLASSES = {"rectangular": "rectangular", "cuboid": "rectangular", "triangular": "roof", "cylindrical": "cylinder",
                 "cylinder": "cylinder", "hexagonal": "hexagon"}
ROOF_RIDGE_MAX = 0.7        # top over the end fifths / top over the middle fifth of the long axis
L_CORNER_EMPTY = 0.15       # the L: one corner empty ...
L_CORNER_FULL = 1.5         # ... and another full
RECT_CORNER_MIN = 0.55      # cuboids fill all four corners (dev: >= 0.59 bar 2 partial views)
ROUND_CORNER_MAX = 1.2      # cylinders and the hexagon leave every corner partly empty
HEX_HARMONIC_MIN = 0.025    # sixfold outline harmonic / mean radius, wrist views
HEX_HEAD_HARMONIC_MIN = 0.06  # the same from the head alone (dev: hexagon 0.038-0.27, cylinders 0.004-0.086)
OUTLINE_SPREAD_MAX = 2.5    # a round outline whose radii spread more than this has a foreign mask merged in
SHAPE_TOP_MAX_M = 0.07      # top-face rectangle longer than any block: merged masks
SHAPE_FOOT_MAX_M = 0.09

# v3.3: "block shaped like number N". Qwen reads the digit on the head AND both wrist images;
# an answer whose block contradicts the digit's shape is dropped, then answers are counted.
# Shape = holes in the top face (rasterised wrist-camera depth: 93/95 dev observations right,
# the head's 2.3 mm/px is too coarse: 10/57) and narrowness ("1": short/long 0.51-0.57, every
# other digit >= 0.61). Dev probes: per view Qwen was right 10/12 (head), 11/12 and 8/12
# (wrists); errors include unparseable (truncated) replies and an "8" taken for a "6" in two
# views of one scene (s59, the "6" seen upside down from the right wrist).
NUMBER_VIEWS = ("head", "hand_left", "hand_right")
DIGIT_HOLES = {"0": 1, "1": 0, "2": 0, "3": 0, "4": 1, "5": 0, "6": 1, "7": 0, "8": 2, "9": 1}
NARROW_ASPECT = 0.59
HOLE_CELL_M = 0.0015
HOLE_MIN_CELLS = 3
VLM_BUDGET_S = 100.0        # no further resolver call once this much resolver time is spent
VLM_THIRD_BUDGET_S = 60.0   # the third view only if the first two took less (all thinking before the
                            # first motion must stay under GenieSim's 180 s reply timeout; dev s59:
                            # head 31.6 s + left wrist 23.9 s under load)
VLM_RESCUE_BUDGET_S = 70.0  # v3.4: the third view when no answer survives the shape filter (held-out s34:
                            # 61.8 s). The local station runs without keepalive holds, so the worst case
                            # must still fit 180 s: 70 + one view (<= 32 s logged, 4096-token cap) + census
                            # (~15 s) + the slowest logged hover planning (~60 s) = ~177 s

# v3.3 straighten: which end is the cap (see bottle_profile), and the hold height for the
# rotation (lower end of the vertical bottle this far above the table; v3.2 also rose 4 cm).
CAP_DROP_WINDOW_M = (0.005, 0.011)
CAP_DROP_MIN_M = 0.0005
REVERSED_CAP_NAMES = ("sprite",)  # assets whose cap end reads as the bottom (see bottle_profile)
UPRIGHT_CLEARANCE_M = 0.05

# v3.5 pick_object_type: every scene holds one fruit, one pen, one toy (toy car, toy plane or Rubik's
# cube) and one drink, and the instruction names the category. Seeds 31-33 (both runs): "toys" with
# the toy car of instance 9 -> Qwen answered "the green apple" ("the only object that could be
# interpreted as a toy ... unlike the pen, car, or bottle ... vehicle"). An answer naming another
# category is asked again once with the category spelled out (OBJECT_TYPE_HINTS); if that answer is
# off too, the category's own SAM 3 queries go before the resolver's.
TOY_WORDS = ("toy", "toys", "car", "plane", "airplane", "aeroplane", "aircraft", "jet", "truck", "vehicle", "rubik", "rubik's", "rubiks",
             "puzzle", "doll", "teddy", "robot", "train", "helicopter")
OBJECT_TYPE_HINTS = {
    "toy": "(The target is a toy: a toy car, a toy airplane, a Rubik's cube or another plaything counts as a toy; fruit, drinks and pens are not toys.)",
    "fruit": "(The target is a fruit, such as an apple, orange, peach or lemon; drinks, toys and pens are not fruit.)",
    "drink": "(The target is a drink: a bottle or can of beverage; fruit, toys and pens are not drinks.)",
    "stationery": "(The target is stationery: a pen, pencil, marker or other writing tool; fruit, drinks and toys are not stationery.)",
}
OBJECT_TYPE_QUERIES = {"toy": ["toy", "toy car", "toy airplane", "rubik's cube"], "fruit": ["fruit", "apple", "orange"],
                       "drink": ["bottle", "can", "soda can"], "stationery": ["pen", "pencil", "marker"]}
# v3.5 dev4: every object-type target lies on the instructed arm's side of the table (100/100 upstream
# instructions, signed y >= 0; 172 of 300 other objects lie across). Dev s56: "Right arm picks up toys"
# -> Qwen "the yellow toy" with its box on the lemon at y = +0.17; asked again with the category spelled
# out it answers "the toy car" (offline, the same frame). A choice further than SIDE_MARGIN_M across is
# asked again once with the category and the side named.
SIDE_MARGIN_M = 0.03
SIDE_HINTS = {"left": "(It lies on the left arm's side of the table: the left half of the image.)",
              "right": "(It lies on the right arm's side of the table: the right half of the image.)"}
# v3.5: objects right under a start-pose gripper (|y| ~ 0.21-0.29 m, x ~ 0.69-0.75 m) were unreachable:
# 9 "approach" failures on seeds 31-33 (number, common sense, object type, billiards, shape), every
# hover plan refused with "cuRobo goal state infeasible (scene_collision)". The planning scene fuses the
# wrist cameras' depth; each sees its own open gripper at 4-13 cm (~244k pixels) and the cuRobo robot
# carve leaves part of it, so a hover goal under that gripper collides with the gripper's own image
# (offline, dev probes: every goal refused with the full scene, every one feasible without the points
# within 12 cm of the start TCPs or with the head's points alone). The image moves with the gripper,
# so when both hovers and the integrated planner fail, the gripper first moves up and toward the robot's
# midline (orientation kept) and the approach runs once more, before the steep escalation (offline: 4/4
# probes reach the staging pose, then 4-5 of 8 hover options plan, 0 of 8 before). Only for the kinds
# below (the tasks v3.5 is tested on).
EDGE_RESTAGE_KINDS = ("pick_block_shape", "pick_object_type")
RESTAGE_OFFSETS_M = ((0.0, 0.15, 0.10), (-0.12, 0.0, 0.08))  # (dx, dy toward the midline, dz) of the TCP

# v3.6: the far-edge refusals came from the harness (planning scene without the wrist lenses' distortion;
# fixed on branch g2-v36). The restage stays as the fallback for whatever else refuses every hover (job 1675
# also starts episodes with an arm left over the table) and now runs for every task.
RESTAGE_KINDS = TASK_KINDS
SCENE_SENSITIVE_M = 0.15    # a target this close (xy) to the arm's start TCP lay in the old phantom's reach (logged)
# v3.6 grasp yaw by wrist-camera side. For the v3.x approach quaternions (tool z down, tilted 20 degrees toward
# +x or vertical; tool x = closing axis at `yaw`) the wrist camera / pad side (tool +y right arm, -y left arm)
# points away from the robot's midline by cos(yaw), for both arms. Every logged 20-degree grasp: camera toward
# the midline (cos < -0.3) 878/908 held, fore/aft 217/235, toward the arm's table edge (cos > 0.3) 64/90; under
# the lb wire on objects lower than 2.2 cm 75/77, 15/20 and 8/23 (scripts/g2_v36_grasp_side.py).
CAMERA_SIDE_COS = 0.3
SQUARE_FOOTPRINT = 0.8      # width / length at or above: the four yaws close alike and are ordered by camera side
# v3.6 object type: the chosen instance is checked with SAM 3's queries for each category; an instance another
# category explains (>= CATEGORY_OTHER_SCORE within CATEGORY_MATCH_M) that none of the instruction's category
# queries finds (>= CATEGORY_MIN_SCORE) is replaced by the category's best instance on the arm's side (test
# s39: "pink toy" -> the pink pen; every scene holds one fruit, one pen, one toy and one drink).
CATEGORY_CHECK_QUERIES = {"toy": ["toy car", "toy airplane", "rubik's cube"], "stationery": ["pen", "marker"],
                          "drink": ["bottle", "can"], "fruit": ["apple", "orange", "lemon", "peach"]}
CATEGORY_MATCH_M = 0.03
CATEGORY_MIN_SCORE = 0.3
CATEGORY_OTHER_SCORE = 0.5
CATEGORY_SWITCH_SCORE = 0.6  # the replacement must be a confident instance: SAM 3 scores the toy car of instance 23
                             # 0.25 as "toy car" and 0.50 as "can" (dev s54 frame), so a weak "other" alone proves nothing
# v3.6 block colour: two best SAM 3 instances of the colour query within this score -> the one under the
# resolver's box (s33 "green": the yellow cube 0.492, the green one 0.48, Qwen's box on the green one).
COLOR_TIE_SCORE = 0.05
COLOR_TIE_IOU = 0.1

# ----------------------------------------------------------------------------- report
context = get_task_context()
frame = get_robot_state().base_frame
report = {
    "program": "instruction_board_v3_6",
    "task": context.task_name,
    "instruction": context.language,
    "stages": [],
    "moves": [],
    "attempts": [],
    "rows": 0,
    "terminated": False,
    "error": None,
    "v36_flags": [],
}
last_command = {}
last_ik = {}


def stage(name, **fields):
    entry = {"stage": name, "rows_before": report["rows"]}
    entry.update(fields)
    report["stages"].append(entry)
    return entry


def flag(name, **fields):
    """v3.6: a decision that differs from what v3.5 would do (report["v36_flags"])."""
    entry = {"flag": name, "rows": report["rows"]}
    entry.update(fields)
    report["v36_flags"].append(entry)
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


def option(name, quat, rank, rank35=None):
    """rank35: the rank v3.5 gave the same option (only grasp options carry it; see ik_ranked)."""
    item = {"name": name, "quat": quat_normalize(quat), "rank": rank}
    if rank35 is not None:
        item["rank35"] = rank35
    return item


def camera_side(yaw):
    """v3.6: 0 = the wrist camera / pad side faces the robot's midline, 1 = fore / aft, 2 = the arm's table
    edge (CAMERA_SIDE_COS; the same for both arms with the v3.x approach quaternions)."""
    c = math.cos(yaw)
    if c < -CAMERA_SIDE_COS:
        return 0
    return 2 if c > CAMERA_SIDE_COS else 1


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


def grasp_options(summary, across_only=False, by_side=True):
    """Orientation candidates. Rank = 10 * grasp class + tilt rank: class 0 closes across the
    footprint's short axis (yaw and yaw + 180 are the same grasp), class 1 along it; within a
    class the 20-degree forward tilt (what v2.1 actually executed) comes before vertical.
    v3.2: a closing direction wider than the jaw is not offered (pen s48 closed along its 13 cm).
    The 20-degree and vertical options keep v3.1's ranking and height: their fingertip touching
    the table does not predict failure (v3 + v3.1: 195/200 held, 100% even 10 mm "below").
    v3.6 (by_side, every task but straighten): the yaw whose wrist camera faces the arm's table edge
    goes after the other yaw of its pair and that yaw's vertical; on a square footprint the four yaws
    are one class ordered midline, fore/aft, outward (camera_side). Straighten keeps v3.5's order: its
    bottles held 12/12 outward and the upright rotation starts from the grasp."""
    base = summary["yaw"]
    yaws = [(base + 0.5 * math.pi, 0), (base - 0.5 * math.pi, 0)]
    if not across_only:
        yaws = yaws + [(base, 1), (base + math.pi, 1)]
    square = summary["width"] >= SQUARE_FOOTPRINT * summary["length"]
    options = []
    for yaw, grasp_class in yaws:
        if grasp_class and span_across(summary, yaw) > JAW_SPAN_M:
            continue
        label = "yaw%+.0f" % math.degrees(math.atan2(math.sin(yaw), math.cos(yaw)))
        side = camera_side(yaw)
        rank = 10 * grasp_class
        if by_side and square:
            rank = 4 * side
        elif by_side and side == 2:
            rank = 10 * grasp_class + 4
        options.append(option(label + "_tilt20", approach_quat(yaw, TILT_RAD, 0.0), rank, 10 * grasp_class))
        options.append(option(label + "_vertical", approach_quat(yaw, 0.0, 0.0), rank + 1, 10 * grasp_class + 1))
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
    # v3.6: the candidate v3.5's ranking would have tried first, for the yaw_reranked flag
    last_ik.clear()
    by35 = sorted([entry for entry in ranked if "rank35" in entry[2]], key=lambda entry: (entry[2]["rank35"], entry[1]))
    if by35:
        last_ik["v35_first"] = by35[0][2]["name"]
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
    if last_ik.get("v35_first") and ranked[0][2]["name"] != last_ik["v35_first"]:
        entry["v35_first"] = last_ik["v35_first"]
        flag("yaw_reranked", label=label, v36_first=ranked[0][2]["name"], v35_first=last_ik["v35_first"])
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
    return move_joints(arm, last_command.get(arm) or current_joints(arm), label, gripper)


def move_joints(arm, target, label, gripper):
    """Joint-interpolated move to a known joint configuration (no IK), executed and servoed."""
    entry = {"label": label}
    report["moves"].append(entry)
    try:
        plan = plan_motion([float(v) for v in target], arm=arm, strategy=STRATEGY_FINAL)
    except Exception as exc:
        entry["fail"] = "exception: " + str(exc)[:80]
        return False
    if not plan.ok or plan.trajectory is None:
        entry["fail"] = error_text(plan)
        return False
    return execute(plan.trajectory, arm, entry, gripper)


def hold_tick(arm, label):
    """Hold the current pose just past GenieSim's next evaluation step. Its checkers run when the
    step counter is a multiple of 30, and the counter equals report["rows"] (dev: s61 latched
    Upright at step 360 inside a hold that began at 342; s53 at 270 inside a rotation). A
    move_to_joints of N steps costs exactly N rows (no chunk padding), so this costs 3-32 rows
    instead of a padded 30-row chunk."""
    rows = report["rows"]
    steps = max(3, (rows // 30 + 1) * 30 + 2 - rows)
    entry = {"label": label, "steps": steps, "rows_before": rows}
    report["moves"].append(entry)
    target = last_command.get(arm) or current_joints(arm)
    done = move_to_joints(target, tolerance=0.0001, max_steps=steps, arm=arm)
    count_rows(done)
    entry.update({"exec_ok": done.ok, "rows": done.steps_executed})
    if done.terminated:
        entry["terminated"] = True
    return done.ok


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
    report["vlm_seconds"] = report.get("vlm_seconds", 0.0) + float(resolved.get("latency_s") or 0.0)
    return resolved


def resolve_view(text, camera):
    """v3.3: the resolver on another camera (a wrist sees the table at ~1 mm/px from the start
    pose). Every answer is logged in report["resolver_views"]; the time spent is summed in
    report["vlm_seconds"] because all thinking before the first motion must stay well under
    GenieSim's 180 s reply timeout."""
    try:
        resolved = g2.resolve_instruction(text, camera_name=camera)
    except Exception as exc:
        resolved = {"ok": False, "error": str(exc)[:120]}
    if not resolved:
        resolved = {"ok": False}
    report["vlm_seconds"] = report.get("vlm_seconds", 0.0) + float(resolved.get("latency_s") or 0.0)
    entry = {"camera": camera, "text": text[:160]}
    entry.update({k: resolved.get(k) for k in ("ok", "query", "box_xyxy", "latency_s", "error")})
    report.setdefault("resolver_views", []).append(entry)
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
    elif kind == "pick_object_type" and (report.get("object_type") or {}).get("demoted"):
        # v3.5: the resolver named another category twice: the category's own queries go first
        category = object_category(parsed["phrase"])
        queries = [parsed["phrase"]] + OBJECT_TYPE_QUERIES.get(category, []) + queries
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


def categories_of(text):
    """v3.5: the object-type categories a phrase names ("drink", "stationery", "toy", "fruit"). A
    fruit word counts only when no other category is named ("orange bottle" is a drink, "orange
    toy car" a toy: orange, peach and lemon are colours too)."""
    words = str(text).lower().replace("-", " ").replace(",", " ").split()
    found = []
    if any(word in BEVERAGE_WORDS for word in words):
        found.append("drink")
    if any(word in STATIONERY_WORDS for word in words):
        found.append("stationery")
    if any(word in TOY_WORDS for word in words):
        found.append("toy")
    if not found and any(word in FRUIT_WORDS for word in words):
        found.append("fruit")
    return found


def check_object_side(kind, parsed, target):
    """v3.5 dev4: an object-type target more than SIDE_MARGIN_M on the far side of the instructed arm is
    the wrong object (SIDE_HINTS): ask once more with the category and the side named; if that choice is
    across too, take the category's own SAM 3 instances on the arm's side; else keep the choice."""
    arm = parsed["arm"]
    check = report.get("object_type")
    if target is None or check is None or arm not in ("left", "right"):
        return target
    sign = 1.0 if arm == "left" else -1.0
    check["side_y"] = round(target["summary"]["grasp_xy"][1], 4)
    if sign * target["summary"]["grasp_xy"][1] >= -SIDE_MARGIN_M:
        return target
    category = object_category(parsed["phrase"])
    again = resolve_view(context.language + " " + OBJECT_TYPE_HINTS.get(category, "") + " " + SIDE_HINTS[arm], "head")
    check["side_reask_query"] = again.get("query") if again.get("ok") else None
    if again.get("ok") and again.get("query"):
        retry = choose_target(kind, parsed, again)
        if retry is not None and sign * retry["summary"]["grasp_xy"][1] >= -SIDE_MARGIN_M:
            check["side_result"] = "reask_on_side"
            check["side_y_after"] = round(retry["summary"]["grasp_xy"][1], 4)
            return retry
    for query in query_variants([parsed["phrase"]] + OBJECT_TYPE_QUERIES.get(category, [])):
        for score, summary, box in text_instances("head", query):
            if sign * summary["grasp_xy"][1] >= -SIDE_MARGIN_M:
                check["side_result"] = "category_query_on_side"
                check["side_y_after"] = round(summary["grasp_xy"][1], 4)
                return target_of(summary, "text_side", [query, parsed["phrase"]], query=query, score=round(score, 3))
    check["side_result"] = "kept"
    return target


def object_category(phrase):
    """The category an object-type instruction names: "toys" -> "toy", "drinks" -> "drink"."""
    for word in ("toy", "drink", "fruit", "stationery"):
        if word in str(phrase).lower():
            return word
    return None


def check_object_type(parsed, resolved):
    """v3.5: an object-type answer that names another category than the instruction is asked again
    once with the category spelled out (OBJECT_TYPE_HINTS); the in-category answer replaces it, else
    text_queries puts the category's own queries first. Answers in the category, or naming none,
    stay as they are (v3.4)."""
    category = object_category(parsed["phrase"])
    query = resolved.get("query") if resolved.get("ok") else None
    named = categories_of(query) if query else []
    check = {"category": category, "resolver_query": query, "resolver_categories": named, "in_category": category in named, "demoted": False}
    report["object_type"] = check
    if category is None or category in named or (query and not named):
        return resolved
    # v3.6: no usable answer at all (test s42: unparseable JSON, then SAM 3 "toys" found the orange) is asked
    # again with the category spelled out, like an off-category answer
    empty = not query
    if empty:
        flag("reask_empty", error=str(resolved.get("error"))[:80])
    again = resolve_view(context.language + " " + OBJECT_TYPE_HINTS[category], "head")
    second = again.get("query") if again.get("ok") else None
    check["reask_query"] = second
    check["reask_categories"] = categories_of(second) if second else []
    if second and category in check["reask_categories"]:
        check["result"] = "reask_in_category"
        return again
    if second and empty and not check["reask_categories"]:
        check["result"] = "reask_after_empty"
        return again
    check["demoted"] = True
    check["result"] = "category_queries_first"
    return resolved


def check_object_category(parsed, target):
    """v3.6, pick_object_type: is the chosen instance one of the instruction's category? SAM 3's queries for
    each category (CATEGORY_CHECK_QUERIES) on the head image: if one of the category's own queries finds an
    instance within CATEGORY_MATCH_M of the choice, it stays. Otherwise, when another category's query does
    (score >= CATEGORY_OTHER_SCORE), the choice is that other kind of object (test s39: Qwen "pink toy" with
    its box on the pink pen; "pen" finds it, no toy query does) and the category's best own instance on the
    instructed arm's side replaces it, if one scores at least CATEGORY_SWITCH_SCORE. Otherwise the choice
    is kept."""
    category = object_category(parsed["phrase"])
    check = report.get("object_type")
    if target is None or check is None or category not in CATEGORY_CHECK_QUERIES:
        return target
    xy = target["summary"]["grasp_xy"]
    best = {}
    own = []

    def scan(name):
        for query in CATEGORY_CHECK_QUERIES[name]:
            for score, summary, box in text_instances("head", query, min_score=CATEGORY_MIN_SCORE):
                if name == category:
                    own.append((score, summary, query))
                if xy_distance(summary["grasp_xy"], xy) <= CATEGORY_MATCH_M:
                    best[name] = max(best.get(name, 0.0), score)

    scan(category)
    if category in best:
        check["category_check"] = "own"
        check["category_own_score"] = round(best[category], 3)
        return target
    for name in CATEGORY_CHECK_QUERIES:
        if name != category:
            scan(name)
    others = {name: round(score, 3) for name, score in best.items() if name != category}
    check["category_other_scores"] = others
    if not others or max(others.values()) < CATEGORY_OTHER_SCORE:
        check["category_check"] = "unexplained_kept"
        return target
    sign = 1.0 if parsed["arm"] == "left" else (-1.0 if parsed["arm"] == "right" else 0.0)
    candidates = sorted([item for item in own if item[0] >= CATEGORY_SWITCH_SCORE and (sign == 0.0 or sign * item[1]["grasp_xy"][1] >= -SIDE_MARGIN_M)],
                        key=lambda item: -item[0])
    if not candidates:
        check["category_check"] = "other_kept_no_alternative"
        return target
    score, summary, query = candidates[0]
    check["category_check"] = "switched"
    check["category_switch_to"] = {"query": query, "score": round(score, 3), "grasp_xy": r3(summary["grasp_xy"])}
    flag("category_switch", was=r3(xy), explained_by=others, to_query=query)
    return target_of(summary, "category_check", [query, parsed["phrase"]], query=query, score=round(score, 3))


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


def choose_color(queries, box, limit=8):
    """v3.6, pick_block_color: choose_by_text, except that when the two best instances of the query score
    within COLOR_TIE_SCORE, the one under the resolver's box is taken (s33 "green": SAM 3 scored the yellow
    cube 0.492 and the teal-green one 0.48; Qwen's box was on the green one)."""
    tried = []
    for query in queries[:limit]:
        tried.append(query)
        found = text_instances("head", query)
        if not found:
            continue
        score, summary, ibox = found[0]
        target = target_of(summary, "text", [query] + [q for q in queries if q != query][:2], query=query, score=round(score, 3), tried=tried)
        if box and len(found) > 1 and found[0][0] - found[1][0] < COLOR_TIE_SCORE:
            ious = [box_iou(box, item[2]) if item[2] is not None else 0.0 for item in found[:2]]
            report["color_tie"] = {"query": query, "scores": [round(found[0][0], 3), round(found[1][0], 3)], "iou": [round(v, 3) for v in ious]}
            if ious[1] > ious[0] and ious[1] >= COLOR_TIE_IOU:
                score, summary, ibox = found[1]
                flag("color_tie_box", query=query, scores=report["color_tie"]["scores"], iou=report["color_tie"]["iou"])
                target = target_of(summary, "text_tie_box", target["track_queries"], query=query, score=round(score, 3), tried=tried)
        return target
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


def top_face(summary, table_z):
    """The top face of a block cloud: points within max(3 mm, 15 % of the height) of its top
    (98th percentile), their min-area rectangle, and the occupancy of its four corner squares
    (20 % of each side; 1.0 = as full as the rest of the rectangle)."""
    pts = sampled(summary["cloud"].points, 3000)
    top = quantile([float(p[2]) for p in pts], 0.98)
    height = top - table_z
    if height < 0.008:
        return None
    band = max(0.003, 0.15 * height)
    xs = []
    ys = []
    for p in pts:
        if float(p[2]) >= top - band:
            xs.append(float(p[0]))
            ys.append(float(p[1]))
    if len(xs) < 20:
        return None
    yaw, a, b, middle = footprint(xs, ys)
    if a < 0.005 or b < 0.002:
        return None
    c, s = math.cos(yaw), math.sin(yaw)
    counts = [0, 0, 0, 0]
    for x, y in zip(xs, ys):
        u = (x - middle[0]) * c + (y - middle[1]) * s
        v = (y - middle[1]) * c - (x - middle[0]) * s
        if abs(u) <= 0.3 * a or abs(v) <= 0.3 * b:
            continue
        counts[(2 if u > 0 else 0) + (1 if v > 0 else 0)] += 1
    side = 0.5 * (a + b)
    return {"xy": (middle[0], middle[1]), "a": a, "b": b, "side": side, "height": height, "aspect": b / a,
            "ratio": height / side, "corners": min(counts) / (0.04 * len(xs)), "n_top": len(xs),
            "corner_max": max(counts) / (0.04 * len(xs)), "top": top, "band_xs": xs, "band_ys": ys}


def is_cube(face):
    return (face is not None and face["aspect"] >= CUBE_ASPECT_MIN and CUBE_HEIGHT_RATIO[0] <= face["ratio"] <= CUBE_HEIGHT_RATIO[1]
            and face["corners"] >= CUBE_CORNER_MIN and CUBE_SIDE_M[0] <= face["side"] <= CUBE_SIDE_M[1])


def block_census(noun, extra=()):
    """Every plausible block seen by the head and both wrist cameras, merged across views and
    queries by top-face position: [{"xy", "faces": {view: face}, "summaries": {view: (score, summary)},
    "tables": {view: table height}}].
    Per view the table height is the median bottom of the instances within 3 cm of the lowest;
    an instance whose top is more than 12 cm above it (a finger in a wrist view) is not a block.
    v3.5: `extra` queries (the shape census adds "cylinder"); the cube census keeps v3.4's."""
    objects = []
    queries = [noun] + [q for q in CUBE_QUERIES if q != noun] + [q for q in extra if q != noun and q not in CUBE_QUERIES]
    for view in CUBE_VIEWS:
        found = []
        for query in queries:
            for score, summary, box in text_instances(view, query, min_score=CUBE_MIN_SCORE, max_extent=0.15):
                found.append((score, summary))
        if not found:
            continue
        bottoms = sorted([item[1]["bottom"] for item in found])
        low = [z for z in bottoms if z <= bottoms[0] + 0.03]
        table_z = low[len(low) // 2]
        mine = []
        for score, summary in sorted(found, key=lambda item: -item[0]):
            if summary["top"] - table_z > 0.12:
                continue
            if any(xy_distance(summary["grasp_xy"], other["grasp_xy"]) < CUBE_MERGE_M for other_score, other in mine):
                continue
            mine.append((score, summary))
        for score, summary in mine:
            face = top_face(summary, table_z)
            xy = face["xy"] if face is not None else summary["grasp_xy"]
            entry = None
            for obj in objects:
                if xy_distance(obj["xy"], xy) < CUBE_MERGE_M:
                    entry = obj
                    break
            if entry is None:
                entry = {"xy": xy, "faces": {}, "summaries": {}, "tables": {}}
                objects.append(entry)
            if view not in entry["faces"]:
                entry["faces"][view] = face
                entry["summaries"][view] = (score, summary)
                entry["tables"][view] = table_z
    return objects


def digit_shape(summary):
    """Holes in the top face of a lying digit block and its narrowness, from one cloud: the top
    3 mm rasterised at HOLE_CELL_M in the face's own frame; a hole is a 4-connected empty region
    of at least HOLE_MIN_CELLS cells that does not reach the border. All points are used (a
    thinned cloud leaves empty cells that look like holes)."""
    pts = summary["cloud"].points
    if len(pts) > 12000:
        pts = sampled(pts, 12000)
    top = quantile([float(p[2]) for p in sampled(pts, 3000)], 0.98)
    xs = []
    ys = []
    for p in pts:
        if float(p[2]) >= top - 0.003:
            xs.append(float(p[0]))
            ys.append(float(p[1]))
    if len(xs) < 30:
        return None
    yaw, a, b, middle = footprint(sampled(xs, 2000), sampled(ys, 2000))
    if a < 0.01:
        return None
    c, s = math.cos(yaw), math.sin(yaw)
    cell = HOLE_CELL_M
    nu = int(math.ceil((a + 4 * cell) / cell))
    nv = int(math.ceil((b + 4 * cell) / cell))
    occupied = set()
    for x, y in zip(xs, ys):
        u = (x - middle[0]) * c + (y - middle[1]) * s
        v = (y - middle[1]) * c - (x - middle[0]) * s
        i = min(nu - 1, max(0, int((u + 0.5 * a + 2 * cell) / cell)))
        j = min(nv - 1, max(0, int((v + 0.5 * b + 2 * cell) / cell)))
        occupied.add(i * nv + j)
    visited = set()
    holes = 0
    for start in range(nu * nv):
        if start in occupied or start in visited:
            continue
        stack = [start]
        visited.add(start)
        size = 0
        border = False
        while stack:
            cellid = stack.pop()
            size += 1
            i, j = cellid // nv, cellid % nv
            if i == 0 or j == 0 or i == nu - 1 or j == nv - 1:
                border = True
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ii, jj = i + di, j + dj
                if 0 <= ii < nu and 0 <= jj < nv:
                    other = ii * nv + jj
                    if other not in occupied and other not in visited:
                        visited.add(other)
                        stack.append(other)
        if not border and size >= HOLE_MIN_CELLS:
            holes += 1
    return {"holes": holes, "aspect": b / a, "a": a, "b": b, "n_top": len(xs)}


def digit_census():
    """Blocks in both wrist views (the head is too coarse for holes), merged by position:
    [{"xy", "shapes": {view: shape}, "summaries": {view: (score, summary)}}]."""
    objects = []
    for view in ("hand_left", "hand_right"):
        found = []
        for query in ("block", "number block"):
            for score, summary, box in text_instances(view, query, min_score=CUBE_MIN_SCORE, max_extent=0.12):
                found.append((score, summary))
        if not found:
            continue
        bottoms = sorted([item[1]["bottom"] for item in found])
        table_z = bottoms[len(bottoms) // 4]
        mine = []
        for score, summary in sorted(found, key=lambda item: -item[0]):
            if summary["top"] - table_z > 0.08 or summary["height"] < 0.006:
                continue
            if any(xy_distance(summary["grasp_xy"], other["grasp_xy"]) < CUBE_MERGE_M for other_score, other in mine):
                continue
            mine.append((score, summary))
        for score, summary in mine:
            entry = None
            for obj in objects:
                if xy_distance(obj["xy"], summary["grasp_xy"]) < CUBE_MERGE_M:
                    entry = obj
                    break
            if entry is None:
                entry = {"xy": summary["grasp_xy"], "shapes": {}, "summaries": {}}
                objects.append(entry)
            if view not in entry["shapes"]:
                entry["shapes"][view] = digit_shape(summary)
                entry["summaries"][view] = (score, summary)
    return objects


def digit_fit(obj, digit):
    """True / False if the block's shape agrees / disagrees with `digit`, None if unknown.
    Views that disagree on the hole count make it unknown."""
    shapes = [f for f in obj["shapes"].values() if f is not None]
    if not shapes or digit not in DIGIT_HOLES:
        return None
    counts = sorted(set([f["holes"] for f in shapes]))
    # narrow only if every view says so: a partial mask (occlusion) can only make a block look
    # narrower (dev s60: one view measured 0.31 for a wide digit)
    narrow = max([f["aspect"] for f in shapes]) < NARROW_ASPECT
    if len(counts) > 1:
        return None if digit != "1" else (False if not narrow else None)
    return counts[0] == DIGIT_HOLES[digit] and (narrow == (digit == "1"))


def vlm_vote(camera, resolved, objects):
    """The census block under the resolver's box (point prompt at its centre on that camera)."""
    box = resolved.get("box_xyxy") if resolved.get("ok") else None
    if not box:
        return None
    centre = ((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0)
    found = point_instances(camera, centre, 0.12)
    if not found:
        return None
    xy = found[0][1]["grasp_xy"]
    best = None
    for obj in objects:
        gap = xy_distance(obj["xy"], xy)
        if gap < CUBE_MERGE_M and (best is None or gap < best[0]):
            best = (gap, obj)
    return {"xy": xy, "summary": found[0][1], "object": None if best is None else best[1]}


def choose_number(resolved, digit, arm, noun):
    """Votes of Qwen on up to three views, filtered by the digit's shape (holes, narrowness).
    Falls back to v3.2 (head box) when there is neither a vote nor a census block."""
    objects = digit_census()
    order = ["head"]
    first_wrist = "hand_" + arm if arm in ("left", "right") else "hand_left"
    order += [first_wrist] + [v for v in ("hand_left", "hand_right") if v != first_wrist]
    votes = []
    for view in order:
        if view == "head":
            answer = resolved
        elif view != order[1] and settled(votes, digit):
            report["number_third_view"] = "skipped: the first two views agree on a fitting block"
            continue
        elif report.get("vlm_seconds", 0.0) < (VLM_BUDGET_S if view == order[1] else VLM_THIRD_BUDGET_S):
            answer = resolve_view(context.language, view)
        elif view != order[1] and not surviving(votes, digit) and report.get("vlm_seconds", 0.0) < VLM_RESCUE_BUDGET_S:
            # v3.4 (held-out s34): without another answer the choice would be a block the shape
            # filter ruled out, or a guess among the blocks that fit.
            report["number_third_view"] = "asked: no answer survives the shape filter (resolver time %.0f s)" % report.get("vlm_seconds", 0.0)
            answer = resolve_view(context.language, view)
        else:
            report["number_third_view"] = "skipped: resolver time %.0f s" % report.get("vlm_seconds", 0.0)
            continue
        vote = vlm_vote(view, answer, objects)
        if vote is not None:
            vote["view"] = view
            votes.append(vote)
    tally = []
    for obj in objects:
        fit = digit_fit(obj, digit)
        mine = [v for v in votes if v["object"] is obj]
        tally.append((obj, fit, mine))
    rows = []
    for obj, fit, mine in tally:
        rows.append({"xy": r3(obj["xy"]), "fit": fit, "votes": [v["view"] for v in mine],
                     "shapes": {view: None if f is None else {"holes": f["holes"], "aspect": round(f["aspect"], 2)} for view, f in obj["shapes"].items()}})
    decision = {"digit": digit, "votes": [{"view": v["view"], "xy": r3(v["xy"]), "census": v["object"] is not None} for v in votes]}
    report["number_census"] = rows
    report["number_choice"] = decision
    # v3.4 (held-out s35): answers on a block the wrist census does not hold are votes on a block of
    # unknown shape, like votes on a census block whose views disagree on the hole count.
    unseen = off_census(votes)
    if unseen:
        decision["off_census"] = [{"xy": r3(obj["xy"]), "votes": [v["view"] for v in obj["votes"]]} for obj in unseen]
    for obj in unseen:
        tally.append((obj, None, obj["votes"]))
    agreeing = [(len(mine), -order.index(mine[0]["view"]), obj, mine) for obj, fit, mine in tally if mine and fit is not False]
    # v3.6 stage 1 (s31 "2", right arm): the head voted for a "2"-shaped block 25 cm across the midline, the right
    # wrist for the true one; every one of the 657 upstream instructions that name an arm has its target on that
    # arm's side, so a block more than SIDE_MARGIN_M across loses to any block on the side
    agreeing = on_arm_side(agreeing, arm, lambda item: item[2]["xy"], "votes")
    if agreeing:
        agreeing.sort(key=lambda item: (-item[0], -item[1]))
        count, rank, obj, mine = agreeing[0]
        if obj.get("off_census"):
            decision["result"] = "votes_off_census"
            return number_target(obj, mine[0], "vote_unseen", noun)
        decision["result"] = "votes_fitting_shape"
        return number_target(obj, mine[0], "vote_shape", noun)
    fitting = on_arm_side([obj for obj, fit, mine in tally if fit is True], arm, lambda obj: obj["xy"], "fitting")
    if len(fitting) == 1:
        decision["result"] = "only_fitting_shape"
        return number_target(fitting[0], None, "shape_only", noun)
    if fitting:
        # v3.4 (held-out s34): every answer names a block the shape filter rules out, and all views
        # have been asked (or the budget is spent); one of the blocks that fit is the target, so
        # take the one the most wrist views measured (ties: census order).
        best = sorted(fitting, key=lambda obj: -len([f for f in obj["shapes"].values() if f is not None]))[0]
        decision["result"] = "fitting_shape_guess"
        decision["fitting"] = len(fitting)
        return number_target(best, None, "shape_guess", noun)
    if votes:
        decision["result"] = "first_vote"
        return target_of(votes[0]["summary"], "vote_" + votes[0]["view"], [noun])
    decision["result"] = "none"
    return None


def on_arm_side(items, arm, xy_of, what):
    """v3.6: the items within SIDE_MARGIN_M of the instructed arm's half of the table, if there are any (else all)."""
    if arm not in ("left", "right") or not items:
        return items
    sign = 1.0 if arm == "left" else -1.0
    kept = [item for item in items if sign * xy_of(item)[1] >= -SIDE_MARGIN_M]
    if kept and len(kept) < len(items):
        flag("number_side", what=what, dropped=[r3(xy_of(item)) for item in items if sign * xy_of(item)[1] < -SIDE_MARGIN_M])
        return kept
    return items


def surviving(votes, digit):
    """v3.4: an answer survives the shape filter if its block fits the digit or its shape is
    unknown (views disagree, or the block is not in the census)."""
    for vote in votes:
        if vote["object"] is None or digit_fit(vote["object"], digit) is not False:
            return True
    return False


def off_census(votes):
    """v3.4: answers whose block is not in the wrist census (SAM 3 did not segment it there; held-out
    s35: all three answers 7-8 cm from every census block), grouped into blocks by position:
    [{"xy", "shapes", "summaries", "votes", "off_census"}], shaped like census entries."""
    blocks = []
    for vote in votes:
        if vote["object"] is not None:
            continue
        home = None
        for obj in blocks:
            if xy_distance(obj["xy"], vote["xy"]) < CUBE_MERGE_M:
                home = obj
                break
        if home is None:
            home = {"xy": vote["xy"], "shapes": {}, "summaries": {}, "votes": [], "off_census": True}
            blocks.append(home)
        home["votes"].append(vote)
        if vote["view"] not in home["summaries"]:
            home["summaries"][vote["view"]] = (0.0, vote["summary"])
    return blocks


def settled(votes, digit):
    """Two answers on the same census block, and that block's shape does not contradict the digit."""
    if len(votes) < 2 or votes[0]["object"] is None or votes[0]["object"] is not votes[1]["object"]:
        return False
    return digit_fit(votes[0]["object"], digit) is not False


def number_target(obj, vote, how, noun):
    """Grasp geometry for the chosen block: the head view's instance if it is there (what v3.2
    aimed with), else the wrist instance with the most points."""
    for score, summary, box in text_instances("head", noun, min_score=CUBE_MIN_SCORE, max_extent=0.12):
        if xy_distance(summary["grasp_xy"], obj["xy"]) < CUBE_MERGE_M:
            return target_of(summary, how + "_head", [noun], score=round(score, 3))
    if vote is not None and vote["view"] == "head":
        return target_of(vote["summary"], how + "_headbox", [noun])
    view = sorted(obj["summaries"], key=lambda v: -obj["summaries"][v][1]["points"])[0]
    score, summary = obj["summaries"][view]
    return target_of(summary, how + "_" + view, [noun], score=round(score, 3))


def choose_cube(box, noun):
    """'cube block': the VLM's block if it is a geometric cube, else the block that is a cube in
    more views than it is not (at least one), else the VLM's block (v3.2 behaviour)."""
    objects = block_census(noun)
    rows = []
    for obj in objects:
        faces = [f for f in obj["faces"].values() if f is not None]
        obj["passes"] = sum(1 for f in faces if is_cube(f))
        obj["fails"] = len(faces) - obj["passes"]
        row = {"xy": r3(obj["xy"]), "passes": obj["passes"], "fails": obj["fails"], "views": {}}
        for view, f in obj["faces"].items():
            row["views"][view] = None if f is None else {"side": round(f["side"], 4), "aspect": round(f["aspect"], 2), "ratio": round(f["ratio"], 2),
                                                          "corners": round(f["corners"], 2), "n_top": f["n_top"], "xy": r3(f["xy"])}
        rows.append(row)
    report["cube_census"] = rows
    cubes = [obj for obj in objects if obj["passes"] >= 1 and obj["passes"] > obj["fails"]]
    vlm = choose_by_box(box, noun)
    decision = {"cubes": len(cubes), "vlm": None if vlm is None else r3(vlm["summary"]["grasp_xy"])}
    report["cube_choice"] = decision
    if vlm is not None:
        for obj in cubes:
            if xy_distance(obj["xy"], vlm["summary"]["grasp_xy"]) < CUBE_MERGE_M:
                decision["result"] = "vlm_is_cube"
                vlm["how"] = vlm["how"] + "+cube"
                return vlm
    if cubes:
        best = sorted(cubes, key=lambda obj: (obj["fails"] - obj["passes"], -obj["passes"]))[0]
        view = "head" if "head" in best["summaries"] else sorted(best["summaries"], key=lambda v: -best["summaries"][v][1]["points"])[0]
        score, summary = best["summaries"][view]
        decision["result"] = "geometry" if vlm is None else "geometry_over_vlm"
        decision["view"] = view
        return target_of(summary, "cube_geometry", [noun], score=round(score, 3))
    decision["result"] = "vlm_only"
    return vlm


def roof_ratio(summary, table_z):
    """v3.5: the top over the end fifths of the cloud's footprint long axis / the top over its middle
    fifth (90th percentiles above the table). Flat tops 0.98-1.04, the triangular prism's 45-degree
    roof 0.32-0.42 (dev probes, every view that holds the whole block)."""
    pts = sampled(summary["cloud"].points, 3000)
    c, s = math.cos(summary["yaw"]), math.sin(summary["yaw"])
    mx, my = summary["grasp_xy"]
    length = summary["length"]
    middle = []
    ends = []
    for p in pts:
        u = (float(p[0]) - mx) * c + (float(p[1]) - my) * s
        if abs(u) <= 0.1 * length:
            middle.append(float(p[2]) - table_z)
        elif abs(u) >= 0.3 * length:
            ends.append(float(p[2]) - table_z)
    if len(middle) < 5 or len(ends) < 5:
        return None
    top = quantile(middle, 0.9)
    if top <= 0.005:
        return None
    return quantile(ends, 0.9) / top


def outline_harmonic(xs, ys, middle):
    """v3.5: the sixfold harmonic of a top face's outline (95th-percentile radius about the face
    centre per 10-degree bin, at least 24 of 36 bins), relative to the mean radius, and the radius
    spread max / min. A hexagon's corners stand ~1.8 mm out of its inscribed circle: wrist views
    0.039-0.060 on the hexagonal prism, 0.001-0.012 on every cylinder (dev probes)."""
    bins = [[] for index in range(36)]
    for x, y in zip(xs, ys):
        dx, dy = x - middle[0], y - middle[1]
        bins[int((math.atan2(dy, dx) + math.pi) / math.radians(10.0)) % 36].append(math.sqrt(dx * dx + dy * dy))
    radii = []
    for index, values in enumerate(bins):
        if len(values) >= 2:
            radii.append(((index + 0.5) * math.radians(10.0) - math.pi, quantile(values, 0.95)))
    if len(radii) < 24:
        return None, None
    mean = sum([r for t, r in radii]) / len(radii)
    if mean <= 0.0:
        return None, None
    c = sum([r * math.cos(6.0 * t) for t, r in radii]) / len(radii)
    s = sum([r * math.sin(6.0 * t) for t, r in radii]) / len(radii)
    return 2.0 * math.sqrt(c * c + s * s) / mean, max([r for t, r in radii]) / max(1e-6, min([r for t, r in radii]))


def shape_view(face, summary, table_z):
    """v3.5: the class of one block in one view - "roof", "l_shape", "cube", "rectangular", "round"
    or None - and the features that decided it (see SHAPE_QUERIES)."""
    if face is None:
        return None, {}
    feats = {"a": face["a"], "b": face["b"], "height": face["height"], "aspect": face["aspect"],
             "corner_min": face["corners"], "corner_max": face["corner_max"], "length": summary["length"]}
    if face["a"] > SHAPE_TOP_MAX_M or summary["length"] > SHAPE_FOOT_MAX_M:
        return None, feats
    ridge = roof_ratio(summary, table_z)
    feats["ridge"] = ridge
    if ridge is not None and ridge < ROOF_RIDGE_MAX:
        return "roof", feats
    if face["corners"] < L_CORNER_EMPTY and face["corner_max"] >= L_CORNER_FULL:
        return "l_shape", feats
    if is_cube(face):
        return "cube", feats
    if face["corners"] >= RECT_CORNER_MIN:
        return "rectangular", feats
    if face["corner_max"] < ROUND_CORNER_MAX:
        harmonic, spread = outline_harmonic(face["band_xs"], face["band_ys"], face["xy"])
        feats["harmonic6"] = harmonic
        feats["spread"] = spread
        return "round", feats
    return None, feats


def median(values):
    ordered = sorted(values)
    half = len(ordered) // 2
    return ordered[half] if len(ordered) % 2 else 0.5 * (ordered[half - 1] + ordered[half])


def vote_margin(obj, label):
    """Views that say `label` minus views that say anything else."""
    return 2 * obj["votes"].get(label, 0) - sum(obj["votes"].values())


def census_summary(obj):
    """The head view's instance of a census block if there is one (what the grasp aims with),
    else the wrist instance with the most points: (view, score, summary)."""
    view = "head" if "head" in obj["summaries"] else sorted(obj["summaries"], key=lambda v: -obj["summaries"][v][1]["points"])[0]
    score, summary = obj["summaries"][view]
    return view, score, summary


def choose_shape(box, noun, shape):
    """v3.5: 'rectangular / triangular prism / cylindrical / hexagonal prism block' by geometry over
    the head and both wrist views. Each census block gets one class per view (shape_view); a block
    is a candidate if no other class has more views. Rectangular and roof (triangular prism) are
    classes of their own; among the round blocks the hexagon is the one whose top-face outline has
    the strongest sixfold harmonic (median over wrist views, the head only when no wrist saw it) if
    that reaches HEX_HARMONIC_MIN, and every other round block is a cylinder. The candidate with the
    largest vote margin wins; Qwen's block (head box, v3.4) breaks a tie and is the fallback when no
    block has the shape."""
    wanted = SHAPE_CLASSES[shape]
    label = wanted if wanted in ("rectangular", "roof") else "round"
    objects = block_census(noun, SHAPE_QUERIES)
    rows = []
    for obj in objects:
        obj["votes"] = {}
        wrist = []
        head = []
        row = {"xy": r3(obj["xy"]), "views": {}}
        for view, face in obj["faces"].items():
            label_view, feats = shape_view(face, obj["summaries"][view][1], obj["tables"][view])
            row["views"][view] = {"class": label_view}
            for key, value in feats.items():
                row["views"][view][key] = None if value is None else round(value, 4)
            if label_view is None:
                continue
            obj["votes"][label_view] = obj["votes"].get(label_view, 0) + 1
            if label_view == "round" and feats.get("harmonic6") is not None and feats["spread"] <= OUTLINE_SPREAD_MAX:
                if view == "head":
                    head.append(feats["harmonic6"])
                else:
                    wrist.append(feats["harmonic6"])
        values = wrist or head
        obj["hexness"] = median(values) if values else None
        obj["hex_source"] = "wrist" if wrist else ("head" if head else None)
        row["votes"] = obj["votes"]
        row["hexness"] = None if obj["hexness"] is None else round(obj["hexness"], 4)
        row["hex_source"] = obj["hex_source"]
        rows.append(row)
    report["shape_census"] = rows
    if label == "round":
        rounds = [obj for obj in objects if obj["votes"].get("round", 0) >= 1 and vote_margin(obj, "round") >= 0]
        hexagons = []
        for source, threshold in (("wrist", HEX_HARMONIC_MIN), ("head", HEX_HEAD_HARMONIC_MIN)):
            # a wrist measurement outranks the head's (its scale differs: the head's 2.3 mm/px
            # roughens every outline), so the head decides only when no wrist saw a hexagon
            measured = sorted([obj for obj in rounds if obj["hex_source"] == source and obj["hexness"] >= threshold], key=lambda obj: -obj["hexness"])
            if measured:
                hexagons = measured[:1]
                break
        if wanted == "hexagon":
            candidates = hexagons
        else:
            plain = [obj for obj in rounds if obj["hexness"] is None
                     or obj["hexness"] < (HEX_HARMONIC_MIN if obj["hex_source"] == "wrist" else HEX_HEAD_HARMONIC_MIN)]
            candidates = plain or [obj for obj in rounds if not any(obj is other for other in hexagons)]
    else:
        candidates = [obj for obj in objects if obj["votes"].get(label, 0) >= 1 and vote_margin(obj, label) >= 0]
    vlm = choose_by_box(box, noun)
    decision = {"shape": shape, "wanted": wanted, "candidates": len(candidates), "vlm": None if vlm is None else r3(vlm["summary"]["grasp_xy"])}
    report["shape_choice"] = decision
    if candidates:
        best_margin = max([vote_margin(obj, label) for obj in candidates])
        tied = [obj for obj in candidates if vote_margin(obj, label) == best_margin]
        decision["tied"] = len(tied)
        if len(tied) > 1 and vlm is not None:
            for obj in tied:
                if xy_distance(obj["xy"], vlm["summary"]["grasp_xy"]) < CUBE_MERGE_M:
                    decision["result"] = "vlm_breaks_tie"
                    vlm["how"] = vlm["how"] + "+" + wanted
                    return vlm
        best = sorted(tied, key=lambda obj: -len(obj["faces"]))[0]
        view, score, summary = census_summary(best)
        decision["result"] = "geometry" if len(tied) == 1 else "geometry_most_views"
        decision["view"] = view
        decision["xy"] = r3(best["xy"])
        decision["vlm_agrees"] = vlm is not None and xy_distance(best["xy"], vlm["summary"]["grasp_xy"]) < CUBE_MERGE_M
        track = ["cylinder", noun] if wanted == "cylinder" else [noun]
        return target_of(summary, "shape_geometry", track, score=round(score, 3), floor_from_reference=True)
    decision["result"] = "vlm_only"
    return vlm


def choose_target(kind, parsed, resolved):
    box = resolved.get("box_xyxy") if resolved.get("ok") else None
    noun = "billiard ball" if kind == "pick_billiards_color" else "block"
    queries = text_queries(kind, parsed, resolved)
    target = None
    if kind == "pick_block_size":
        target = choose_by_size(parsed["size_rule"] or str(resolved.get("selection_rule")), noun)
        if target is None:
            target = choose_by_box(box, noun)
    elif kind == "pick_block_shape" and parsed["shape"] in ("cube", "cubic"):
        target = choose_cube(box, noun)
        if target is None:
            target = choose_by_text(queries)
    elif kind == "pick_block_shape" and parsed["shape"] in SHAPE_CLASSES:
        target = choose_shape(box, noun, parsed["shape"])
        if target is None:
            target = choose_by_text(queries)
    elif kind == "pick_block_number" and parsed["digit"] is not None:
        target = choose_number(resolved, parsed["digit"], parsed["arm"], noun)
        if target is None:
            target = choose_by_box(box, noun)
        if target is None:
            target = choose_by_text(queries)
    elif kind in ("pick_block_number", "pick_block_shape"):
        if kind == "pick_block_shape":
            shape_diagnostics(noun)
        target = choose_by_box(box, noun)
        if target is None:
            target = choose_by_text(queries)
    elif kind == "pick_block_color":
        target = choose_color(queries, box)
        if target is None:
            target = choose_by_box(box, noun, max_extent=0.3)
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
def approach(center, grasp_z, options, arm, summary, attempt, high_first=False):
    """Hover ABOVE_M over the grasp point: v2.1's ranked IK at two heights, then the
    integrated (scene-aware) planner, then steeper reach-direction tilts. v3.3: `high_first`
    tries the higher hover first (straighten: see run_episode)."""
    hover = (center[0], center[1], grasp_z + ABOVE_M)
    high = (center[0], center[1], grasp_z + ABOVE_M + ABOVE_HIGH_M)
    heights = (("above_high", high), ("above", hover)) if high_first else (("above", hover), ("above_high", high))
    for label, position in heights:
        chosen = move_ik(position, options, arm, label, gripper=1.0)
        if chosen is not None:
            attempt["approach"] = label
            return chosen
        if report["terminated"]:
            return None
        if high_first and label == "above_high":
            # v3.3: the integrated planner at the high hover before dropping to the low one
            # (planning only, no rows): dev s61 got the low hover from the integrated planner,
            # Follow passed there, and the second orientation ran out of the Upright window.
            chosen = move_pose(high, sorted(options, key=lambda item: item["rank"]), arm, "above_high_integrated", gripper=1.0)
            if chosen is not None:
                attempt["approach"] = "above_high_integrated"
                return chosen
            if report["terminated"]:
                return None
    chosen = move_pose(hover, sorted(options, key=lambda item: item["rank"]), arm, "above_integrated", gripper=1.0)
    if chosen is None and not report["terminated"] and report.get("kind") in RESTAGE_KINDS and "restage" not in attempt:
        # v3.5: every plain hover refused - move the gripper (and the image of its own fingers in the
        # planning scene) out of the way and approach once more, before the steep escalation: dev s60
        # reached a far-edge pen only at 35 degrees, where the fingertip clearance raised the TCP above
        # the 1.9 cm pen's top and both closes caught air (EDGE_RESTAGE_KINDS). v3.6: every task.
        if report.get("kind") not in EDGE_RESTAGE_KINDS:
            flag("restage_new_kind", kind=report.get("kind"))
        attempt["restage"] = restage(arm)
        if attempt["restage"] is not None and not report["terminated"]:
            return approach(center, grasp_z, options, arm, summary, attempt, high_first)
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


def shape_floor(target, seen, summary, grasp_z):
    """v3.5, block-shape geometry targets only: a wrist view from above holds little of a low
    block's side, so its bottom sits high; the grasp height never comes from a bottom above the head
    camera's. Dev s67: the 1.5 cm bar seen 0.9 cm high from the wrist put the TCP 3.5 mm under its
    top and both closes caught air; the low blocks held had it 6-10.5 mm under the top."""
    if not target.get("floor_from_reference"):
        return grasp_z
    return min(grasp_z, max(seen["top"] - GRASP_BELOW_TOP_M, summary["bottom"] + FLOOR_CLEARANCE_M))


def restage(arm):
    """v3.5: move the TCP by the first reachable RESTAGE_OFFSETS_M (up and toward the robot's midline,
    then back and up), orientation kept, with the scene-aware planner. Returns the offset used or None."""
    position, quat = tool_pose(arm)
    inward = -1.0 if arm == "left" else 1.0
    for dx, dy, dz in RESTAGE_OFFSETS_M:
        target = (position[0] + dx, position[1] + inward * dy, position[2] + dz)
        if move_ik(target, [option("restage", quat, 0)], arm, "restage", gripper=1.0, samples=2) is not None:
            return [dx, inward * dy, dz]
        if report["terminated"]:
            return None
    return None


def pick(target, arm, options, hold_after=True, high_first=False):
    """Approach with the TCP above the object's footprint centre (head camera), check at hover
    (grasp height and gross moves from the wrist view, finger differential logged), descend,
    close, lift, verify; one retry when the jaws closed on air. v3.3: `hold_after` False skips
    the post-lift evaluation hold (straighten does not score PickUpOnGripper; it needs the rows)."""
    summary = target["summary"]
    aim = (summary["grasp_xy"][0], summary["grasp_xy"][1], summary["center"][2])
    grasp_z = grasp_height(summary)
    cloud = summary["cloud"]
    attempt = {"arm": arm, "target": target, "center": r3(summary["center"]), "aim": r3(aim[:2]), "top": round(summary["top"], 4),
               "bottom": round(summary["bottom"], 4), "grasp_z": round(grasp_z, 4), "options": [item["name"] for item in options],
               "rows_start": report["rows"], "aim_source": "head"}
    report["attempts"].append(attempt)
    chosen = approach(aim, grasp_z, options, arm, summary, attempt, high_first)
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
        grasp_z = shape_floor(target, seen, summary, grasp_height(seen, summary))
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
    if not held and not report["terminated"] and attempt.get("fail") == "descend" and report["moves"][-1].get("fail") == "no IK solution":
        # v3.6: the option the hover used has no IK at the grasp height (job 1675: a far-forward cube reached
        # with a vertical grasp after both 20-degree hovers were refused): turn to the other options at the
        # hover (only those with an IK solution at the grasp height), then descend once more
        grasp_pos = (aim[0], aim[1], grasp_z)
        rest = [item for item in options if item["name"] != chosen["name"] and ik_ranked(grasp_pos, [item], arm, samples=2, keep=1)]
        here = tool_pose(arm)[0]
        again = move_ik(here, rest, arm, "above_reapproach", gripper=1.0) if rest else None
        flag("descend_reapproach", failed=chosen["name"], option=None if again is None else again["name"])
        if again is not None and not report["terminated"]:
            chosen = again
            attempt["option"] = chosen["name"]
            attempt["descend_reapproach"] = chosen["name"]
            attempt.pop("fail", None)
            steep = approach_tilt(chosen["quat"]) > STEEP_TILT_RAD
            table_floor = summary["bottom"] + (TIP_CLEARANCE_M + tip_drop(chosen["quat"]) if steep else FLOOR_CLEARANCE_M)
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
            grasp_z = shape_floor(target, seen, summary, grasp_height(seen, summary))
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
    if held and hold_after and not report["terminated"]:
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
    source = "width"
    # v3.3: the top of a lying can / bottle sits lower near its cap end. Coca-Cola and Fanta
    # necks taper over ~18 mm, their bottoms over ~8 mm (mesh profiles); the oolong bottle's
    # neck is 17 mm lower. Head-camera dev probes (17 scenes): this cue 12/12 right on Coca-Cola,
    # Fanta and oolong, 0/5 on the slim Sprite can, whose ends do not differ; the 30 % width rule
    # above (v3.2) was 11/17. upright() tries the other end second, so a wrong guess costs rows.
    drops = end_drops(along, [float(p[2]) for p in pts], quantile(along, 0.005), quantile(along, 0.995))
    if drops is not None and abs(drops[0] - drops[1]) >= CAP_DROP_MIN_M:
        cap_sign = 1.0 if drops[0] > drops[1] else -1.0
        source = "drop"
    if any(name in context.language.lower() for name in REVERSED_CAP_NAMES):
        # The Sprite reads the other way round: on all 5 dev instances (13 runs) its top end sat
        # 0.5-1.0 mm ABOVE the body level and its bottom end 0.4-1.4 mm below (margins 1.0-2.3 mm),
        # whether the top pointed away from the robot or towards it, so the cue took the bottom
        # every time (v3.2's width rule: 4 of 5 wrong); 20/20 right on the other assets. Its mesh
        # ends differ by ~1 mm in radius only; it probably rests slightly top-up (not verified).
        cap_sign = -cap_sign
        source += "_reversed"
    return {"yaw": yaw, "length": length, "cap_dir": (cap_sign * d[0], cap_sign * d[1], 0.0), "cap_source": source,
            "drop_pos": None if drops is None else round(drops[0], 4), "drop_neg": None if drops is None else round(drops[1], 4),
            "w_pos": round(w_pos, 4), "w_neg": round(w_neg, 4), "bottom": summary["bottom"]}


def end_drops(along, zs, lo, hi, step=0.002):
    """How far the top silhouette (90th percentile height per 2 mm bin along the axis) sits
    below the body level (median over the middle half) 5-11 mm from the + and - ends."""
    if hi - lo < 0.04:
        return None
    count = int((hi - lo) / step) + 1
    bins = [[] for index in range(count)]
    for s, z in zip(along, zs):
        if lo <= s <= hi:
            bins[min(count - 1, int((s - lo) / step))].append(z)
    tops = [quantile(b, 0.9) if len(b) >= 3 else None for b in bins]
    middle = [t for t in tops[count // 4: (3 * count) // 4] if t is not None]
    if not middle:
        return None
    body = quantile(middle, 0.5)
    near = {True: [], False: []}
    for index, top in enumerate(tops):
        if top is None:
            continue
        centre = lo + (index + 0.5) * step
        for at_hi in (True, False):
            gap = (hi - centre) if at_hi else (centre - lo)
            if CAP_DROP_WINDOW_M[0] <= gap <= CAP_DROP_WINDOW_M[1]:
                near[at_hi].append(top)
    if not near[True] or not near[False]:
        return None
    return (body - max(near[True]), body - max(near[False]))


def rotate_min_motion(arm, fan, pivot, offsets, label):
    """IK for every fan orientation at every offset of the pivot, then execute the solution with
    the least joint motion (joint interpolation). Dev batch 1: the holds that were not upright
    followed a 6.5 rad reconfiguration or a 3.4 rad move right after the grasp; v3.2's upright
    rotations were mostly 2.2-3.3 rad, the 60-row ones 4.1-4.4 rad tilted 1.9-4.1 deg."""
    candidates = []
    for dx, dy, dz in offsets:
        spot = (pivot[0] + dx, pivot[1] + dy, pivot[2] + dz)
        for rank, distance, item, joints in ik_ranked(spot, fan, arm, samples=1, keep=4):
            candidates.append((distance, spot, item, joints))
    candidates.sort(key=lambda c: c[0])
    entry = {"label": label, "ik_solutions": len(candidates)}
    report["moves"].append(entry)
    for index, candidate in enumerate(candidates[:6]):
        distance, spot, item, joints = candidate
        try:
            plan = plan_motion(joints, arm=arm, strategy=STRATEGY_FINAL)
        except Exception as exc:
            entry.setdefault("plan_errors", []).append("exception: " + str(exc)[:80])
            continue
        if not plan.ok or plan.trajectory is None:
            entry.setdefault("plan_errors", []).append(error_text(plan))
            continue
        entry.update({"option": item["name"], "joint_distance": round(distance, 3), "target": r3(spot), "plan_attempt": index + 1,
                      "waypoints": len(plan.trajectory.joint_positions)})
        if execute(plan.trajectory, arm, entry, 0.0):
            return item, spot
        if report["terminated"]:
            return None, spot
    if "option" not in entry:
        entry["fail"] = "no IK solution" if not candidates else "planning refused every IK candidate"
    return None, pivot


def upright(arm, profile):
    """Rotate the held bottle 90 degrees about the horizontal axis perpendicular to it so the
    guessed cap end points up (minimal rotation from the MEASURED tool orientation; a fan of
    yaws about world z, all keeping the bottle vertical, at small position offsets; the IK
    solution with the least joint motion is executed), hold one evaluation tick; then the same
    with the OTHER end up, hold one tick; set down, release.
    Upright() checks only the tilt (<= 5 deg, local +y up) and latches; GenieSim ends the
    episode as soon as it passes, so the second orientation only runs when the first was wrong.
    v3.2 lost 5 of 6 held-and-rotated failures (seeds 31-50) to an upside-down can: the narrower
    end segment it took for the cap is a coin flip on cans. To pay for the second try v3.3 drops
    v3.2's 4 cm rise and 45-degree waypoint (30-120 rows) and keeps its post-lift hold (the
    grasp settles: dev holds without it failed 2 of 7 with the cap right, v3.2 with it 0 of 13)."""
    info = {"cap_dir": r3(profile["cap_dir"]), "head_cap_dir": r3(profile.get("head_cap_dir") or profile["cap_dir"]),
            "cap_source": profile.get("cap_source"), "drop_pos": profile.get("drop_pos"), "drop_neg": profile.get("drop_neg"),
            "w_pos": profile["w_pos"], "w_neg": profile["w_neg"], "length": round(profile["length"], 4)}
    report["upright"] = info
    position, quat = tool_pose(arm)
    hold_z = max(position[2], profile["bottom"] + 0.5 * profile["length"] + UPRIGHT_CLEARANCE_M)
    info["hold_z"] = round(hold_z, 4)
    if hold_z > position[2] + 0.015:
        info["rise_ok"] = move_ik((position[0], position[1], hold_z), [option("grasp", quat, 0)], arm, "rise", gripper=0.0, strategy=STRATEGY_FINAL, samples=2) is not None
        if report["terminated"]:
            return False
    cap = profile["cap_dir"]
    axis = (cap[1], -cap[0], 0.0)  # cap x Z: rotating +90 deg about it takes the cap direction to +Z
    offsets = ((0.0, 0.0, 0.0), (-0.06, 0.0, 0.0), (0.0, 0.06, 0.0), (0.0, -0.06, 0.0), (-0.06, 0.0, 0.05), (0.06, 0.0, 0.0), (0.0, 0.0, 0.08))
    pivot = (position[0], position[1], hold_z)
    info["tries"] = []
    last = None
    lying = last_command.get(arm) or current_joints(arm)  # the lifted, still lying grasp
    for name, angle in (("guess_up", 0.5 * math.pi), ("other_up", -0.5 * math.pi)):
        if last is not None:
            # Back to the lying grasp first (the reverse of the rotation just made), then rotate
            # the other way (dev s56: one direct 5.4 rad reconfiguration, the can was not upright;
            # dev s57/s60/s61 with the return: upright at 2.1-3.1 deg).
            move_joints(arm, lying, "return_lying", 0.0)
            if report["terminated"]:
                return True
        full = quat_mul(quat_axis_angle(axis, angle), quat)
        fan = []
        for degrees in (0, 30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180):
            # Every yaw about world z keeps the bottle vertical: let joint motion decide.
            fan.append(option("%s_yaw%+d" % (name, degrees), quat_mul(quat_axis_angle((0.0, 0.0, 1.0), math.radians(degrees)), full), 0))
        rotated, spot = rotate_min_motion(arm, fan, pivot, offsets, "rotate_" + name)
        entry = {"try": name, "rotation": None if rotated is None else rotated["name"], "pos": r3(spot), "rows_before_hold": report["rows"]}
        info["tries"].append(entry)
        if report["terminated"]:
            entry["terminated_during_rotation"] = True
            return True
        if rotated is None:
            continue
        hold_tick(arm, "hold_" + name)
        last = (rotated, spot)
        if report["terminated"]:
            entry["terminated_during_hold"] = True
            return True
    info["rotate_ok"] = last is not None
    if last is None:
        return False
    rotated, spot = last
    place_z = profile["bottom"] + 0.5 * profile["length"] + 0.012
    info["set_down_ok"] = move_ik((spot[0], spot[1], place_z), [rotated], arm, "set_down", gripper=0.0, strategy=STRATEGY_FINAL, samples=2) is not None
    released = open_gripper(arm=arm)
    count_rows(released)
    info["released_ok"] = released.ok
    info["retreat_ok"] = move_ik((spot[0], spot[1], place_z + 0.10), [rotated], arm, "retreat", gripper=1.0, strategy=STRATEGY_FINAL, samples=2) is not None
    return info["set_down_ok"] and released.ok


# ----------------------------------------------------------------------------- episode
def run_episode():
    parsed = parse_sentence(context.language)
    kind = context.task_name if context.task_name in TASK_KINDS else kind_from_sentence(context.language, parsed)
    report["kind"] = kind
    report["kind_from_sentence"] = kind_from_sentence(context.language, parsed)
    report["parsed"] = {k: parsed[k] for k in ("arm", "phrase", "size_rule", "digit", "shape", "color", "alternatives")}
    resolved = resolve(context.language)
    if kind == "pick_object_type":
        resolved = check_object_type(parsed, resolved)
    stage("choose", kind=kind)
    target = choose_target(kind, parsed, resolved)
    if kind == "pick_object_type":
        target = check_object_side(kind, parsed, target)
        target = check_object_category(parsed, target)
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
    # v3.6: where the arm starts (job 1675 also began episodes with an arm left over the table) and whether
    # the target lies where v3.5's planning scene held the phantom of this arm's own start-pose gripper
    start = tool_pose(arm)[0]
    report["start_tcp"] = r3(start)
    gap = xy_distance(summary["grasp_xy"], start)
    if gap < SCENE_SENSITIVE_M:
        flag("scene_sensitive", tcp_dist=round(gap, 3))
    stage("grasp", arm=arm)
    if kind == "straighten_object":
        profile = bottle_profile(summary)
        report["upright_plan"] = {"cap_dir": r3(profile["cap_dir"]), "source": profile["cap_source"], "drop_pos": profile["drop_pos"],
                                  "drop_neg": profile["drop_neg"], "w_pos": profile["w_pos"], "w_neg": profile["w_neg"]}
        # Hover 17 cm above the grasp (v3.2's second hover height) first: Upright's 300-step
        # window opens when Follow passes (TCP inside a 0.2 m box around the object's prim
        # origin), and for the Coca-Cola and Sprite assets that origin sits ~one radius off
        # the can's axis, so the 12 cm hover already passed Follow (dev s56: the window opened
        # at row 60; Upright can pass up to Follow + 330, kb/06 v3.3). From 17 cm,
        # Follow passes during the descend, ~30 rows later.
        held = pick(target, arm, grasp_options(summary, across_only=True, by_side=False), high_first=True)
        report["held"] = held
        seen = target.get("hover_seen")
        if seen is not None and seen["length"] > 1.5 * seen["width"]:
            # v3.3: logged only. The bottle axis stays the head camera's: against the true axis
            # (scene truth, v3.2 formal + v3.3 dev, 29 runs) the head is a median 1.0 deg off
            # (> 3 deg in 2), the hover-wrist refinement v3 introduced 1.9 deg (> 3 deg in 5,
            # three of them from the 17 cm hover); Upright allows 5 deg in all (dev s66: +4.3 deg).
            d = (math.cos(seen["yaw"]), math.sin(seen["yaw"]))
            sign = 1.0 if d[0] * profile["cap_dir"][0] + d[1] * profile["cap_dir"][1] >= 0.0 else -1.0
            report["upright_plan"]["wrist_cap_dir"] = r3((sign * d[0], sign * d[1], 0.0))
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
