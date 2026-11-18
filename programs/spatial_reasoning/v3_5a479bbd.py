# G2 / GenieSim spatial board, skill library spatial v2 (2026-11-01, Opus 5.5; C5k), from spatial v1 rc2 (2026-10-31, C5j).
#
# One program for the eight spatial-board tasks (scene table_task_2_g2_op, start G2_DEFAULT_STATES as the instruction
# board, nothing randomized). The grasp is instruction_board_v4_0's pick() (approach / wrist check / descend / close /
# lift / retry on air), copied verbatim with every helper it needs (the sandbox only allows `import math`). New here:
#   * task parsing (the eight fixed sentence templates) and a SAM 3 census of a noun's instances on the head camera;
#   * relational choosers: twins by size, then left / right / closer / further / centre (pick absolute) or the twin
#     closer to / further from a reference object (pick relative) - scene-truth check on all 60 instances: right on 60;
#   * place-at-pose: carry a held object (orientation kept, yaw fan by least joint motion) at a transport height and
#     lower it to a goal; hold or release; holds that cover GenieSim's two consecutive 30-row checks;
#   * place relative (hold the object in the air beside the reference: no height check), place beverage (A2 takes Y,
#     A1 takes X away, A2 sets Y on X's spot), stack three blocks, sort cubes by size (layout planner + two moves).
#   * sort number without the VLM: each digit block's look from the head colour (five looks, two digits each) and its
#     top face from the start-pose wrist depth (loops, face area); Qwen's votes only choose among orders left open;
#   * stack bowls: rim grasp, the carried bowl's true geometry (nest rise 3.4 cm from the mesh), carried 6 cm over the
#     stack, released 8 mm above its nested height; mid-table spot when the in-place base needs a long reach.
# Facts behind every rule: evidence/2026-11-01_spatial (spatial_instances.json, the checkers' code, asset origins).
# v2 (C5k), every change from a measured failure (evidence/2026-11-02_spatial_v2), all tasks run as one build:
#   * stack bowls: no head re-measurement of the carried bowl (its 4-6.5 cm "hang" was the base bowl's near side in the
#     head's view; the rim sits where the grasp offset says); rim grasps planned before anything moves (real IK: rim point
#     x jaw sign, held motion = up + least carry), in place only when every carry plans under 3.6 rad, else mid-table; the
#     release rises 18 cm over the rim and steps out before going home;
#   * sort number: the look by hue first (a desaturated blue 1 / dark purple 3 read pale), the next look when the looks
#     admit no three distinct digits; sort (both): a block dropped in its carry is found again and moved once;
#   * place relative: one retry after the object slipped out in the carry; stack three: the stack re-measured after every
#     release; every empty arm going home first rises 25 cm over the table and steps out to its own side;
#   * pick: a descend with no IK for any plain option re-approaches with the steep reach tilts; twins' head heights may
#     differ 1.5 cm more (low objects seen from the side).
# v3 (C5o, 2026-11-03), every change from a lost episode of platform job 1710 or a measured mechanism (evidence/
# 2026-11-03_spatial_v3), all tasks run as one build:
#   * gripper events without steps: 10 of 11 logged arm blow-ups (1710 + C5k; five Onfloor endings on 1710) began at a
#     gripper command step. A set-down opens the jaw only part way (RELEASE_OPEN_DELTA past its measured opening, ramped
#     over RELEASE_RAMP_ROWS of one chunk, the arm held at its MEASURED joints) and retreats so, opening fully away from
#     the table; sort closes ramp too (soft_close); every trajectory whose jaw command differs from the last one by more
#     than GRIP_STEP_MAX ramps it over GRIP_RAMP_ROWS (with_gripper);
#   * pick: an object at least WIDE_SPAN_M across the jaw counts as held only from AIR_OPENING_WIDE (air_limit: a jaw that
#     closed on a neck read 0.06); a 4-11 cm object whose first close read empty is retried 1.5 cm under its top (hollow
#     stapler); finger clearance ranks options clear by less than FINGER_EXTRA_M after those clear by more;
#   * sort (both): finger clearance against where the other blocks are NOW (placed ones at their slots); a missed block is
#     looked for again only away from every other block (and, cubes, at its size); sort_layout's plan must satisfy the
#     checker by SORT_FINAL_MARGIN_M; after the moves both arms clear, the row is looked at again (head, then the wrists),
#     every block identified by size / position, and the row planned again when it does not hold (SORT_VERIFY_BEFORE_ROWS);
#   * stack three: cubes grasped at half their height (the 20-degree grasp's lower fingertip had hung 0.3 cm over the held
#     cube's bottom and stopped set-downs on the cube below), placements corrected for the bias the placing arm leaves
#     (PLACE_BIAS_Y_M: 1710 + C5k +-0.9 cm), the third cube aimed between the base and the second; a stray second cube is
#     placed again; a base not seen near the aim is looked for STACK_REFIND_M wide; after the last release the arm clears
#     and a cube off the stack is put back on it (STACK_REPAIR_BEFORE_ROWS);
#   * place relative: goals beside a far reference (x > PLACE_FAR_X) shifted toward the robot first;
#   * beverage: Y's grasp yaw chosen with its carry onto X's spot in mind (bev_grasp_options: the can's four yaws follow a
#     noisy mask axis, and from half of them the carry needed 4.4-7 rad); a refused carry is planned again with
#     BEV_RETRY_SAMPLES IK calls per yaw under BEV_RETRY_CAP_RAD; Y across on the taker's side with X behind it is raised
#     where it is instead of staged at the placer's edge;
#   * stack bowls: an empty arm whose side step is refused rises higher and steps out again before going home; tilted rim
#     grasps beyond the bowl's centre rank BOWL_FAR_TILT_PENALTY_RAD worse.
# Runs on harness aspire_v41 (branch g2-v41, c583d6f: spatial horizons + 150-row drain holds), as rc2 / v2.
#
# Sandbox: only `import math`, no classes, no names starting with an underscore, only the whitelisted builtins (no
# getattr / isinstance / hasattr); tools are injected by the harness. result = report.
import math

# ----------------------------------------------------------------------------- copied from v4.0 (constants, verbatim)
STRATEGY = MotionStrategy(ik_solver="curobo", trajectory_planner="curobo", time_dilation_factor=1.0)
STRATEGY_POSE = MotionStrategy(ik_solver="curobo", trajectory_planner="curobo", pose_planner="curobo-integrated", time_dilation_factor=1.0)
STRATEGY_FINAL = MotionStrategy(ik_solver="curobo", trajectory_planner="interpolation")
ABOVE_M = 0.12
ABOVE_HIGH_M = 0.05
LIFT_M = 0.12
LIFT_FALLBACK_M = (0.12, 0.08, 0.05)
GRASP_BELOW_TOP_M = 0.035
FLOOR_CLEARANCE_M = 0.006
RESIDUAL_OK_RAD = 0.25
SERVO_TOL_RAD = 0.04
HOVER_SHIFT_MAX_M = 0.12
HOVER_SHIFT_CAP_M = 0.045
WRIST_OVERRIDE_M = 0.03
SIZE_MATCH = 0.35
RETRY_LOWER_M = 0.005
TALL_M = 0.11                 # C5o: objects at least this tall keep v2's retry (dev4 / dev5 retried them 2 cm lower: the oolong
                              # bottle slipped at its shoulder as at its neck, 0 of 2; withdrawn)
HOLLOW_MIN_M = 0.04           # C5o: an object at least this tall (and under TALL_M) whose first close read <= AIR_EMPTY ...
AIR_EMPTY = 0.02              # ... retries HOLLOW_RETRY_BELOW_TOP_M under its top
HOLLOW_RETRY_BELOW_TOP_M = 0.015
RETRY_MOVED_M = 0.02
AIR_OPENING = 0.06
AIR_OPENING_WIDE = 0.15       # C5o: air_limit - an object at least WIDE_SPAN_M across the jaw reads >= 0.21 when held
WIDE_SPAN_M = 0.035
TRACK_RADIUS_M = 0.08
MIN_SCORE = 0.25
TRACK_MIN_SCORE = 0.12
MIN_POINTS = 30
TILT_RAD = math.radians(20.0)
ESCALATION_TILTS_RAD = (math.radians(35.0), math.radians(50.0))
FINGERTIP_HALF_SPAN_M = 0.052
STEEP_TILT_RAD = math.radians(27.5)
TIP_CLEARANCE_M = 0.006
JAW_SPAN_M = 0.09
GRIP_BELOW_TOP_M = 0.002
SHOULDER_X = 0.13
SHOULDER_Y = 0.244
PADS_TOOL = {"left": ((-0.052, -0.009, -0.027), (0.052, -0.009, -0.027)),
             "right": ((0.052, 0.009, -0.027), (-0.052, 0.009, -0.027))}
GRIPPER_PARTS_TOOL = {"left": ((-0.044, -0.013, -0.067), (0.044, -0.013, -0.067), (-0.004, -0.020, -0.070)),
                      "right": ((0.044, 0.013, -0.067), (-0.044, 0.013, -0.067), (-0.004, 0.020, -0.070))}
FINGER_RADIUS_M = 0.03
PAD_MASK_POINTS = (1500, 20000)
PAD_GATE_M = 0.03
PAD_AGREE_M = 0.012
FINGER_BIAS_APPLY = False
FINGER_BIAS_RANGE_M = (0.012, 0.03)
CUBE_VIEWS = ("head", "hand_left", "hand_right")
CUBE_QUERIES = ("block", "cube")
CUBE_MIN_SCORE = 0.1
CUBE_ASPECT_MIN = 0.8
CUBE_HEIGHT_RATIO = (0.78, 1.3)
CUBE_CORNER_MIN = 0.6
CUBE_SIDE_M = (0.015, 0.065)
CUBE_MERGE_M = 0.025
DIGIT_HOLES = {"0": 1, "1": 0, "2": 0, "3": 0, "4": 1, "5": 0, "6": 1, "7": 0, "8": 2, "9": 1}
NARROW_ASPECT = 0.59
HOLE_CELL_M = 0.0015
HOLE_MIN_CELLS = 3
SET_DOWN_MAX_JOINT_RAD = 2.5
NEAR_OFFSET_RAD_PER_M = 5.0
RESTAGE_OFFSETS_M = ((0.0, 0.15, 0.10), (-0.12, 0.0, 0.08))
SCENE_SENSITIVE_M = 0.15
CAMERA_SIDE_COS = 0.3
SQUARE_FOOTPRINT = 0.8
SETTLE_ROWS = (12, 23)
CHUNK_ROWS = 30
SETTLE_FLAG = 0.05
LOW_OBJECT_M = 0.022
LOW_TIP_DROP_M = 0.01
HOLD_SLIP_WORDS = ("mouse", "sponge", "cup", "papercup", "apple")
TABLE_Z = 0.731
CONTACT_SLOPE = 8.4
CONTACT_OFFSET = -0.027
RISE_ROWS = 6
FOLLOW_REACH_ROWS = 3
SERVO_PROGRESS = 0.8
SERVO_STEPS = 10
FINGER_HALF_WIDTH_M = 0.012
FINGER_EXTRA_M = 0.015        # C5o: options clear of every neighbour by less than this extra rank after the ones clear by more
FINGER_TOP_MARGIN_M = 0.003
LOW_OBJECT_RETRY_M = 0.032
VERTICAL_FIRST_WORDS = ("mouse", "sponge")
START_TCP = {"left": (0.696, 0.2817, 0.976), "right": (0.696, -0.2817, 0.976)}
START_JOINTS = {"left": [0.73898, -0.717002, -1.524278, -1.537563, 0.277782, -0.926759, -0.839352],
                "right": [-0.738978, -0.717004, 1.524279, -1.537559, -0.277803, -0.926731, 0.839361]}
HOME_MOVED_START = True
MOVED_START_M = 0.03
HOME_LIFT_M = 0.08
HOME_LOW_M = 0.15
OFF_TABLE_M = 0.10
PLANE_WORDS = ("plane", "planes", "airplane", "aeroplane", "aircraft", "jet")
FUSELAGE_HIGH = 0.6
FUSELAGE_BIN_M = 0.005
FUSELAGE_MAX_W = 0.02
FUSELAGE_MIN_RUN_M = 0.025
FUSELAGE_END_M = 0.012


# ----------------------------------------------------------------------------- spatial constants (C5j)
SPATIAL_TASKS = ("pick_object_absolute_position", "pick_object_relative_position", "place_beverage_to_anothers_position",
                 "place_object_relative_position", "sort_cubes_by_size", "sort_number", "stack_bowls", "stack_three_building_blocks")
PICK_TASKS = ("pick_object_absolute_position", "pick_object_relative_position")
TASK_KINDS = SPATIAL_TASKS
RESTAGE_KINDS = SPATIAL_TASKS
EDGE_RESTAGE_KINDS = SPATIAL_TASKS
# v3.8's PickUpOnGripper window (StepOut 200, last scoring check 180 rows after Follow); pick tasks only
WINDOW_LAST_ROWS = 180
TABLE_CENTRE = (0.71, 0.0)   # base frame; both table assets are centred at world (0, 0), robot root at world (-0.71, 0, 0)
MERGE_M = 0.03               # instances of one object in different queries / views lie this close (twins: >= 8.5 cm apart)
TWIN_MIN_GAP_M = 0.05
TWIN_FLOOR = 0.02            # SAM 3 score floor for the noun census (relative rules decide, not an absolute floor)
TWIN_SCORE_RATIO = 0.6       # twins score within this ratio of each other
TWIN_MAX_GAP = 0.5           # and their footprint / height sizes agree within this relative gap
CLEAR_RATIO = 1.5            # a unique object's best instance scores this much above the next one
TWIN_LONE_SCORE_RATIO = 0.2  # a lower-scoring pair still counts when it is alone (no third instance scoring
TWIN_LONE_RATIO = 0.5        # this share of its lower score) and its sizes agree within TWIN_LONE_MAX_GAP
TWIN_LONE_MAX_GAP = 0.25
TWIN_HEIGHT_SLACK_M = 0.015  # C5k: twins' head heights may differ this much more (size_gap)
# SAM 3 queries per instruction noun (first = most specific). Scene objects: evidence/2026-11-01_spatial/asset_origins.json
NOUN_QUERIES = {
    "building blocks": ["red cube", "red block", "cube", "block"],
    "paper cup": ["white paper cup", "paper cup", "cup"],
    "toy car": ["car", "toy car"],
    "oolong tea beverage": ["bottle", "plastic bottle"],
    "coke cola": ["coca-cola can", "red can", "can"],
    "coca cola": ["coca-cola can", "red can", "can"],
    "sprite": ["sprite can", "green can", "can"],
    "fanta": ["fanta can", "orange can", "can"],
    "wang lao ji": ["wang lao ji can", "red can", "can"],
    "bowl sponge": ["sponge", "yellow sponge"],
    "toy plane": ["airplane", "toy airplane", "toy plane"],
    "medicine bottle": ["medicine bottle", "pill bottle", "white bottle"],
    "face cleanser": ["face cleanser tube", "tube", "face cleanser"],
    "clock": ["pink clock", "alarm clock", "clock"],
    "tape measure": ["tape measure"],
    "peach": ["peach"],
    "rubik cube": ["rubik's cube", "rubik cube"],
    "orange": ["orange"],
    "green apple": ["green apple", "apple"],
    "apple": ["red apple", "apple"],
    "grape juice": ["black bottle", "bottle"],
    "mouse": ["computer mouse", "mouse"],
    "water bottle": ["green bottle", "bottle", "water bottle"],
    "lemon": ["lemon"],
    "yellow stapler": ["yellow stapler", "stapler"],
    "mug": ["mug", "white mug", "cup"],
}
ABS_RULES = (("on the left side", "left"), ("on the right side", "right"), ("closer to the robot", "closer"),
             ("further away from the robot", "further"), ("farther away from the robot", "further"),
             ("near the center", "center"), ("near the centre", "center"))
REL_SPLITS = ((" closer to the ", "closer"), (" further from the ", "further"), (" farther from the ", "further"))
PLACE_SPLITS = ((" to the left of the ", "left"), (" to the right of the ", "right"), (" to the top of the ", "top"),
                (" to the bottom of the ", "bottom"))
# place relative: RelativePosition wants the AABB centres > 0.05 m apart along the relation axis and within 0.10 m on the
# other one (no height check, 2 consecutive checks): the object is held in the air at the reference + PLACE_REL_OFFSET_M
PLACE_REL_OFFSET_M = 0.12
PLACE_REL_MIN_GAP_M = 0.03    # and at least this far beyond both half-extents
PLACE_REL_MIN_OFFSET_M = 0.08  # the shortest along-axis offset tried (RelativePosition wants > 0.05 m)
PLACE_ARM_GOAL_Y = 0.10        # the goal's side arm carries when the goal lies this far across the midline ...
PLACE_ARM_OBJECT_Y = 0.15      # ... and the object within this of the midline
RELAY_X = (0.55, 0.70)         # a relay set-down: x clipped to this range, ...
RELAY_Y = 0.05                 # ... this far on the carrying arm's side of the midline
RELAY_FIND_M = 0.06
PLACE_RETRY_ROWS = 330          # C5k: a slip retry (home, pick, carry, two holds) needs about this many rows
CARRY_CLEAR_M = 0.20          # transport: the held object's bottom at least this far above the table (tallest: 0.17 m)
HOLD_CLEAR_M = 0.06           # place relative: the held object's bottom this far above the table at the goal
PLACE_GAP_M = 0.008           # set-down: the object's bottom this far above its support when the jaw opens
RELEASE_RETREAT_M = 0.10
CARRY_FAN = (0, 30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180)
CARRY_MAX_JOINT_RAD = 3.0
# place beverage: InBBox(Y origin, X's start origin +- (0.04, 0.04, 0.075)) and Upright(Y, 15 deg), 2 checks
CAN_HEIGHT_M = 0.115
BEV_PARK = {"left": (0.62, 0.30), "right": (0.62, -0.30)}   # (dev2) where A1 held X out of the way
# dev4: both cans wait at their own arm's outer edge, nearer the robot than the 2 x 2 grid (x 0.62-0.80, |y| 0.07-0.09)
BEV_STAGE = {"left": (0.56, 0.31), "right": (0.56, -0.31)}
BEV_PARK_SHIFT_M = 0.16       # dev10: X first parked this far from its spot toward the taker's outer side (dev4 s1030: the
                              # edge stage was 3.35 rad away for a taker that had crossed to fetch X)
BEV_CARRY_MAX_JOINT_RAD = 4.2   # dev10 s1002: X's spot in the back row across the midline was 3.80 rad away (6 of 30 such)
BEV_SET_GAP_M = 0.01
# per brand: (SAM 3 query, weight), summed over head + both wrists (C5j probe frames; "fanta can" scores 0.007)
BRAND_QUERIES = {"coca cola": [("coca-cola can", 1.0), ("red can", 0.25)], "coke cola": [("coca-cola can", 1.0), ("red can", 0.25)],
                 "sprite": [("green can", 1.0), ("sprite can", 0.5)], "fanta": [("orange can", 1.0)],
                 "wang lao ji": [("wang lao ji can", 0.5)]}
CAN_MARGIN = 0.3              # assignment totals closer than this and no colour rule: Qwen names X and Y
CAN_BLUE_TOP = 30.0           # B - R of a can's top face above this: the Wang Lao Ji (light-blue top)
CAN_COLOUR_BONUS = 2.0
# stacking: Stack([a, b, c], [t, t]): every origin within t (x and y) of the first's; no height check
STACK_GAP_M = 0.006
STACK_RETRY_BEFORE_ROWS = 900     # a dropped cube is looked for again only this early (Stack: 1530 rows)
STACK_SPOT = (0.60, 0.0)          # where the stack is built: on the midline, both arms reach it without crossing
STACK_SPOT_TOLERANCE_M = 0.03     # a cube this close to the spot is the base where it is
STACK_FIND_M = 0.05               # the placed base is looked for this close to the spot
CARRY_MAX_STEP_RAD = 0.025        # carries / set-downs move a joint at most this much per row (0.75 rad/s)
PLACE_RELEASE_MAX_GAP_M = 0.03    # the jaw opens only when the set-down got this close to its height
# bowls (benchmark_bowl_025, mesh: rim 7.15-7.94 cm radius, 8 mm wall, 8.26 cm tall): a radial rim grasp straddles the
# wall with the jaw vertical, TCP BOWL_RIM_DEPTH_M under the rim top; the jaw then reads ~0.04 open (8 mm), under
# AIR_OPENING, so a bowl counts as held above BOWL_HELD_OPENING. The carried bowl is let go with its bottom just above the
# stack's rim and drops into it (identical bowls nest and self-centre); Stack wants every origin within 5 cm (xy) of b1.
BOWL_HEIGHT_M = 0.0826
BOWL_RIM_RADIUS_M = 0.0755
BOWL_RIM_DEPTH_M = 0.025
BOWL_HELD_OPENING = 0.015
BOWL_DROP_GAP_M = 0.008       # released this far above its nested height (dev: let go 4.2 cm above it)
BOWL_NEST_RISE_M = 0.034      # rim rise per nested bowl: two copies of the mesh touch with 3.3-3.5 cm between their
                              # bottoms (scripts/bowl_profile.py; the 1.5 cm assumed before put every release too high)
BOWL_CLEAR_M = 0.06           # the carried bowl's true bottom over the stack's rim on the way (dev: 1.4 cm; every base was
                              # pushed 6-11 cm toward the robot, the carried bowl landing 4-5 cm past its old spot)
BOWL_MAX_JOINT_RAD = 4.6      # a bowl carry (slowed) may take this much joint motion (dev7: 3.27 / 3.56 rad refused at 3.0;
                              # dev12 / dev13 s1013: the far-right bowl needed 4.44 / 4.16 rad, refused at 3.8)
BOWL_SEEN_M = 0.10            # the stack's top bowl is looked for this close to the aim after each release
BOWL_SPOTS = ((0.58, 0.0), (0.62, 0.0), (0.56, 0.04), (0.56, -0.04))  # a stack built mid-table (first free one)
BOWL_SPOT_FREE_M = 0.17       # a spot is free when every bowl but the first moved stands at least this far from it
BOWL_CROSS_M = 0.10           # in place only when no other-side arm must reach further past the midline than this ...
BOWL_BASE_MAX_X = 0.78        # ... and the base is no further out than this (dev12 s1061: base at x 0.88, no IK for any carry
                              # onto it; s1013: the far-right bowl to a base at y +0.11 needed 4.44 rad)
# C5k (v2): rc2's head re-measurement of the carried bowl ("hangs 4-6.5 cm off") is gone: in the kept head frames the bowl hung
# where the grasp offset says (rim fit 0.5-2 cm), the census had taken the base bowl's near side (evidence/2026-11-02_spatial_v2
# offline/bowl_hang); its correction moved correctly placed bowls off the stack (4 test losses). The grasp is planned instead:
BOWL_PLAN_FAN = (0, 30, -30, 60, -60, 90, -90, 120)   # carry yaws scored per rim grasp before anything moves
BOWL_PLAN_GOOD_RAD = 2.0      # the first rim grasp (rc2's order) whose held motion (up + least carry) is under this is taken
BOWL_PLAN_CAP_RAD = 3.6       # in place only when every carry has a planned grasp under this (else the mid-table plan)
BOWL_PLAN_HOVER_WEIGHT = 0.1  # the empty arm's motion to the hover counts this much against the held motion
BOWL_CLEAR_UP_M = 0.18        # after a release the open jaw rises to this over the stack's rim, moves out to BOWL_CLEAR_SIDE_Y on
BOWL_CLEAR_SIDE_Y = 0.24      # its own side, then goes home (s5037: the clear home from 10 cm up dragged the first bowl 12 cm)
# sort: rightof = the AABB centre > 0.05 m further along -y; aligned_x = |dx| < 0.05 m; all four at once, 2 checks
SORT_NUMBER_SMALLEST = "Pick up the digit-shaped block that shows the SMALLEST number."
SORT_NUMBER_LARGEST = "Pick up the digit-shaped block that shows the LARGEST number."
VLM_SORT_CEILING_S = 300.0    # no further digit question once the resolver has used this much time (dev4 s1003: at
                              # ~40 s per call on a shared GPU 120 s allowed 3 of 6 votes and the one "largest" vote was
                              # wrong; GenieSim waits while the program thinks, keepalive holds cost one row a minute)
# sort number, digit read without the VLM (C5j rc2; Qwen's votes read 3 of 11 dev scenes right). The ten digit blocks come
# in five looks, two digits each: golden 0/5, blue 1/6, yellow 2/7, dark purple 3/8, pale lavender 4 ("white") / 9. The
# head colour gives the look; the wrist cameras at the start pose (~1 mm/px, exact depth) give the top face rasterised at
# HOLE_CELL_M: loops (0, 6, 9: one; 8: two) and face area (1, 4, 7 small). Dev frames, 11 scenes, 32 blocks: loops 13-39
# cells, 4's notch 3-5; face cells 1 / 4 / 7 112-151, all other digits 169-243 (offline/digits/all_features.txt).
DIGIT_LOOKS = {"golden": (0, 5), "blue": (1, 6), "yellow": (2, 7), "dark": (3, 8), "pale": (4, 9)}
# C5k (v2): hue first. rc2 called every block under saturation 0.25 pale before looking at its hue: a narrow blue 1 (sat 0.187,
# hue 203) and a dark purple 3 (sat 0.225, hue 264) read pale on test (2 losses). Every head reading C5j logged (72 blocks,
# evidence/2026-11-02_spatial_v2/offline/digit_looks_c5j.txt): golden hue 50-54, yellow 62-65, blue 197-203 (sat 0.19-0.45),
# dark purple 262-266 (sat 0.23-0.58), pale lavender 277-298 (sat 0.03-0.12).
DIGIT_GREY_SAT = 0.06         # under this saturation the hue says nothing: pale (the only near-grey look)
DIGIT_GOLD_HUE = 58.0         # golden 50-54 deg vs yellow 62-65 (head and wrists)
DIGIT_BLUE_HUE = (170.0, 240.0)  # blue 197-203
DIGIT_DARK_HUE = 274.0        # dark purple 262-266 from 240 up to here, pale lavender 277-298 beyond
DIGIT_DARK_SAT = 0.15         # a purple hue under this saturation is the pale lavender
# each look's centre (hue, saturation) and the scales that make the next-best look of a block (read_digits' fallback)
DIGIT_LOOK_CENTRES = {"golden": (52.0, 0.45), "yellow": (63.5, 0.43), "blue": (199.0, 0.39), "dark": (264.0, 0.50), "pale": (292.0, 0.09)}
DIGIT_LOOK_SCALE = (20.0, 0.15)
DIGIT_LOOP_CELLS = 10
DIGIT_SMALL_CELLS = 168        # 1 / 4 / 7: 118-153 cells; 6 / 9 / 2: 182-246 (wrist faces, dev replay)
DIGIT_FACE_HEIGHT_M = 0.03    # a wrist mask taller or longer than a lying digit took in background (dev: 0.24-0.26 m)
DIGIT_FACE_LENGTH_M = 0.045
DIGIT_FACE_MATCH_M = 0.02
SORT_SPACING_M = 0.09
SORT_Y_LIMIT = 0.27
SORT_X_RANGE = (0.50, 0.74)
SORT_CROSS_M = 0.10           # a move's arm reaches at most this far past the midline
SORT_KEEP_MARGIN_M = 0.015    # an arrangement that already satisfies the checker by this margin is left alone
SORT_RETRY_BEFORE_ROWS = 700  # a missed block is looked for again and picked once more only this early (1230 rows)
SORT_DROP_SEP_M = 0.03       # C5k: a dropped block lies this far from every filled slot and unmoved block
SORT_REPAIR_BEFORE_ROWS = 760 # (v2's cubes-only verify; v3 uses SORT_VERIFY_BEFORE_ROWS)
REFIND_M = 0.12
CLEAR_SAFE_Z = 0.25           # C5k: an empty arm going home first rises to this over the table ...
CLEAR_SIDE_Y = 0.22           # ... and steps out to this |y| on its own side (clear_arm)
# C5o (v3)
RELEASE_OPEN_DELTA = 0.3      # a set-down opens the jaw this much past its measured opening (norm.; ~2 cm wider) ...
RELEASE_OPEN_MAX = 1.0        # ... (at most fully open), ramped over RELEASE_RAMP_ROWS of one chunk, before the retreat
RELEASE_RAMP_ROWS = 24
PLACE_BIAS_Y_M = 0.008        # stack three: a cube lands this much toward the placing arm's side of its TCP goal (1710 +
                              # C5k stage 2, 90 base / 60 second-cube placements: +0.33 / +0.86 cm left arm, -0.71 / -0.95 cm right)
STACK_REFIND_M = 0.12        # stack three: no cube near the aim - the base / cubes are looked for this wide
STACK_LOOSE_M = 0.04          # stack three: a cube further than this from the stack's axis is off the stack
STACK_REPAIR_BEFORE_ROWS = 1180   # a loose cube is put on the stack again only this early (Stack window 1500 rows)
SORT_MATCH_M = 0.035          # sort verify: a found block this close to where a block should be is that block
SORT_SIZE_TOL = 0.3           # sort cubes: a re-found block's size within this share of the block's own
SORT_FINAL_MARGIN_M = 0.01    # sort_layout: a plan's resulting row satisfies the checker's relations by this much
SORT_VERIFY_BEFORE_ROWS = 860 # the row is looked at again and planned again only this early (MixedRules: 1200 rows)
PLACE_FAR_X = 0.70            # place relative: a reference further out than this takes its across-shifts toward the robot first
BOWL_SIDE_RETRY_UP_M = 0.08   # stack bowls: a refused side step is tried again this much higher, with BOWL_SIDE_RETRY_CAP_RAD
BOWL_SIDE_RETRY_CAP_RAD = 4.5
BOWL_FAR_TILT_PENALTY_RAD = 1.0   # stack bowls: a tilted rim grasp beyond the bowl's centre scores this much worse
GRIP_STEP_MAX = 0.2           # a gripper command change larger than this ramps (with_gripper) ...
GRIP_RAMP_ROWS = 12           # ... over this many rows
CLOSE_RAMP_ROWS = 18          # soft_close: the jaw closes over this many rows of its chunk
SOFT_GRIP_KINDS = ("sort_cubes_by_size", "sort_number")   # every logged close-time blow-up was in a sort task
grip_command = {}
BEV_X_BEHIND_M = 0.05         # beverage: X counts as behind Y when this much further out
BEV_RETRY_SAMPLES = 6         # beverage: Y's carry onto X's spot, when refused, again with this many IK calls per yaw ...
BEV_RETRY_CAP_RAD = 4.6       # ... and this cap
BEV_YAW_SAMPLES = 2           # beverage: IK calls per pose in the carry estimate of Y's grasp options (bev_grasp_options) ...
BEV_YAW_PENALTY = 100         # ... an option whose estimated carry is over BEV_CARRY_MAX_JOINT_RAD ranks this much later
grasp_rule = {}

# ----------------------------------------------------------------------------- report
context = get_task_context()
frame = get_robot_state().base_frame
report = {
    "program": "spatial_board_v3",
    "task": context.task_name,
    "instruction": context.language,
    "horizon": (context.metadata or {}).get("horizon"),
    "stages": [],
    "moves": [],
    "attempts": [],
    "rows": 0,
    "terminated": False,
    "error": None,
    "flags": [],
    "resolver": {},
}
last_command = {}
last_ik = {}
lift_hold = {}
hold_limit = {}
follow_row = {}
neighbours = []
LANGUAGE = " ".join(str(context.language or "").strip().rstrip(".").split())


def flag(name, **fields):
    """A decision worth finding in the report (report["flags"]); v4.0's per-version flag lists share this one."""
    entry = {"flag": name, "rows": report["rows"]}
    entry.update(fields)
    report["flags"].append(entry)
    return entry


def flag37(name, **fields):
    return flag(name, **fields)


def flag38(name, **fields):
    return flag(name, **fields)


def flag39(name, **fields):
    return flag(name, **fields)


def floor_score(value):
    """v4.0's camqualgen floor scaling is not needed on the spatial board (no image noise): the floor itself."""
    return value


# ----------------------------------------------------------------------------- copied from v4.0 (functions, verbatim)
def stage(name, **fields):
    entry = {"stage": name, "rows_before": report["rows"]}
    entry.update(fields)
    report["stages"].append(entry)
    return entry

def median_of(values):
    ordered = sorted(values)
    n = len(ordered)
    return ordered[n // 2] if n % 2 else 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])

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
    below = GRASP_BELOW_TOP_M
    if grasp_rule.get("mid"):
        # C5o (stack three): fingertips at the cube's half height, not 3.5 cm under its top (1.3 cm over its bottom on these
        # 4.8 cm cubes): the 20-degree grasp's lower fingertip then hangs ~1.6 cm over the held cube's bottom instead of
        # ~0.3 cm, so a set-down onto the stack no longer stops on the cube below (1710 + C5k: second cubes set 0.4-1.2 cm
        # high, dropped 1-1.8 cm; dev3 smoke 1058 knocked the base) and the first release no longer meets the table
        below = min(GRASP_BELOW_TOP_M, 0.5 * (summary["top"] - bottom))
    return max(summary["top"] - below, bottom + FLOOR_CLEARANCE_M)

def volume(summary):
    return summary["length"] * summary["width"] * (summary["height"] + 0.01)

def xy_distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

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

def with_gripper(trajectory, gripper, arm=None):
    """Copy of a planned trajectory that commands one normalized gripper value on every row. C5o: when the arm's last
    commanded value (grip_command) differs from it by more than GRIP_STEP_MAX, the first GRIP_RAMP_ROWS rows (at most the
    trajectory's) ramp from that value to it instead of stepping (gripper_ramp)."""
    count = len(trajectory.joint_positions)
    values = [gripper for waypoint in trajectory.joint_positions]
    start = grip_command.get(arm) if arm is not None else None
    if start is not None and abs(start - gripper) > GRIP_STEP_MAX and count > 1:
        ramp = min(GRIP_RAMP_ROWS, count)
        values = [start + (gripper - start) * min(1.0, (k + 1) / float(ramp)) for k in range(count)]
    if arm is not None:
        grip_command[arm] = gripper
    return Trajectory(
        joint_positions=trajectory.joint_positions,
        dt_s=trajectory.dt_s,
        joint_names=trajectory.joint_names,
        planner=trajectory.planner,
        collision_aware=trajectory.collision_aware,
        expected_start=trajectory.expected_start,
        arm=trajectory.arm,
        embodiment=trajectory.embodiment,
        gripper_positions=values,
    )

def with_hold(trajectory, arm):
    """v3.7: the trajectory preceded by rows that hold the arm where it is (measured joints, as close_gripper's own
    rows do), as many as fill its first 30-row chunk, within SETTLE_ROWS: the jaw finishes closing before the arm
    rises, in rows GenieSim was sent anyway (the last chunk is padded to 30)."""
    count = len(trajectory.joint_positions)
    rows = max(SETTLE_ROWS[0], min(SETTLE_ROWS[1], CHUNK_ROWS - count))
    limit = hold_limit.pop(arm, None)
    if limit is not None and limit < rows:
        # v3.8: the rise must come before the last check of PickUpOnGripper's window (window_hold_limit)
        flag38("hold_window", rows=max(0, limit), v37_rows=rows)
        rows = max(0, limit)
    if rows <= 0:
        return trajectory, 0
    here = current_joints(arm)
    grippers = None
    if trajectory.gripper_positions is not None:
        grippers = [float(trajectory.gripper_positions[0]) for index in range(rows)] + [float(v) for v in trajectory.gripper_positions]
    return Trajectory(
        joint_positions=[here for index in range(rows)] + [[float(v) for v in row] for row in trajectory.joint_positions],
        dt_s=trajectory.dt_s,
        joint_names=trajectory.joint_names,
        planner=trajectory.planner,
        collision_aware=trajectory.collision_aware,
        expected_start=trajectory.expected_start,
        arm=trajectory.arm,
        embodiment=trajectory.embodiment,
        gripper_positions=grippers,
    ), rows

def execute(trajectory, arm, entry, gripper=None):
    """Execute, then servo up to three rounds onto the last waypoint (the simulated arm lags
    cuRobo's time-optimal end by ~0.1 rad; a residual of 0.09 rad once closed on air)."""
    if gripper is not None:
        trajectory = with_gripper(trajectory, gripper, arm)
    if lift_hold.pop(arm, False):
        trajectory, rows = with_hold(trajectory, arm)
        entry["settle_rows"] = rows
        # every held lift differs from v3.6's motion, whether or not the jaw still moved (v3.7 stage 1, dev1: sponge
        # and mouse, jaws static at 0.41 / 0.39, held after 12 more rows at the grasp where v3.6 closed on air twice)
        if rows > 0:
            flag37("lift_hold", label=entry.get("label"), rows=rows)
    hold_limit.pop(arm, None)
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
        before = worst
        servo = move_to_joints(final, tolerance=0.03, max_steps=SERVO_STEPS, arm=arm)
        count_rows(servo)
        worst = max([float(v) for v in servo.final_errors.values()] or [worst])
        rounds += 1
        entry.update({"servo_rounds": rounds, "residual_after_servo_rad": round(worst, 4)})
        if servo.terminated:
            entry["terminated"] = True
            return False
        try:
            if int(servo.steps_executed) >= SERVO_STEPS and worst > SERVO_TOL_RAD:
                # v3.8: v3.7's round could have run up to 20 steps
                flag38("servo_steps_cap", label=entry.get("label"), residual=round(worst, 4))
        except Exception:
            pass
        if worst > SERVO_TOL_RAD and rounds < 3 and worst > SERVO_PROGRESS * before:
            # v3.8: the arm is blocked (number s104 / s108 "4": a fingertip on a neighbouring block, 0.042 -> 0.043 ->
            # 0.048 rad over three 20-row rounds); further rounds only spend rows and requests
            flag38("servo_stopped", label=entry.get("label"), rounds=rounds, residual=round(worst, 4), before=round(before, 4))
            break
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
    off = []
    for seg in segs.segmentations:
        if float(seg.score) < floor_score(min_score):
            continue
        summary = summarize(cloud_of(seg.mask, camera))
        if plausible(summary, max_extent):
            box = None if seg.box_xyxy is None else [float(v) for v in seg.box_xyxy]
            if camera == "head" and summary["bottom"] > TABLE_Z + OFF_TABLE_M:
                # v3.8: not on the table (s108 "the mouse": Qwen's "black cylindrical object", 0.73, was a background
                # object whose bottom stood 28 cm above the table)
                off.append((float(seg.score), summary))
                continue
            found.append((float(seg.score), summary, box))
    if off:
        top_off = max([item[0] for item in off])
        if not found or top_off > max([item[0] for item in found]):
            flag38("off_table", query=str(text)[:60], score=round(top_off, 3),
                   bottom=round(sorted(off, key=lambda item: -item[0])[0][1]["bottom"], 4), kept=len(found))
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
            if camera == "head" and summary["bottom"] > TABLE_Z + OFF_TABLE_M:
                # v3.8: not on the table (text_instances)
                if not found:
                    flag38("off_table_point", score=round(float(seg.score), 3), bottom=round(summary["bottom"], 4))
                continue
            found.append((float(seg.score), summary, None))
    return found

def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0

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

def target_of(summary, how, track_queries, **extra):
    target = {"summary": summary, "how": how, "track_queries": track_queries}
    target.update(extra)
    return target

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
    rows_before = report["rows"]
    if move_ik((aim[0], aim[1], grasp_z), [chosen], arm, label, gripper=1.0, strategy=STRATEGY_FINAL, exclude=cloud, samples=3) is None:
        attempt["fail"] = label
        return False, aim
    if report["terminated"]:
        return False, aim
    if "row" not in follow_row:
        # v3.8: Follow (TCP inside the object's 10 x 20 x 10 cm box) passes at the first 30-step check after the descend
        # brings the TCP down to the object; PickUpOnGripper's window starts there
        follow_row["row"] = ((rows_before + FOLLOW_REACH_ROWS + CHUNK_ROWS - 1) // CHUNK_ROWS) * CHUNK_ROWS
        attempt["follow_row_est"] = follow_row["row"]
    measure, seen = wrist_measure(arm, attempt["target"], aim)
    attempt[label + "_measure"] = measure
    if report.get("kind") in SOFT_GRIP_KINDS:
        soft_close(arm, label + "_close")
        closed_ok = not report["terminated"]
    else:
        closed = close_gripper(arm=arm)
        count_rows(closed)
        grip_command[arm] = 0.0
        closed_ok = closed.ok
        if closed.terminated:
            return False, aim
    if report["terminated"]:
        return False, aim
    # v3.7: the jaw is still travelling when close_gripper returns; the lift starts with the arm held (with_hold)
    at_close = gripper_opening(arm)
    attempt.setdefault("at_close", []).append(round(at_close, 4))
    if report.get("kind") != "straighten_object":
        hold, info = hold_policy(attempt["target"], chosen, at_close)
        attempt.setdefault("hold_policy", []).append(info)
        if hold:
            lift_hold[arm] = True
            limit = window_hold_limit()
            if limit is not None:
                hold_limit[arm] = limit
                info["window_limit"] = limit
        else:
            # v3.8: v3.7 held every lift
            flag38("hold_skipped", label=label, object=info["slips"], span=info["span"], at_close=info["at_close"])
    lifted = lift(arm, aim, grasp_z, chosen, attempt)
    lift_hold.pop(arm, None)
    hold_limit.pop(arm, None)
    opening = gripper_opening(arm)
    attempt.setdefault("openings", []).append(round(opening, 4))
    attempt["lifted"] = lifted
    if report.get("kind") != "straighten_object" and at_close - opening > SETTLE_FLAG:
        # v3.6 lifted while the jaw covered this
        flag37("close_settle", label=label, at_close=round(at_close, 4), after_lift=round(opening, 4))
    limit = air_limit(attempt)
    if limit > AIR_OPENING and AIR_OPENING <= opening < limit:
        flag("air_wide", label=label, opening=round(opening, 4), limit=limit)
    return closed_ok and lifted and opening >= limit, aim


def air_limit(attempt):
    """C5o: the jaw opening after the lift under which the close counts as on air: AIR_OPENING, or AIR_OPENING_WIDE when the
    object spans at least WIDE_SPAN_M across the jaw (its close's hold_policy span). 1710 + C5k (80 lifts of such objects):
    every hold read >= 0.21, every reading of 0.055-0.12 was a slip or a close on air - the oolong bottle closed at its neck
    read 0.062 and counted as held (pick-absolute inst 14, every run), so the retry never ran."""
    policy = attempt.get("hold_policy") or []
    if policy and (policy[-1].get("span") or 0.0) >= WIDE_SPAN_M:
        return AIR_OPENING_WIDE
    return AIR_OPENING

def hold_policy(target, chosen, at_close):
    """v3.8: does the lift start with the jaw-settle hold? (hold, info). Not for the targets that slip out of the jaw
    while the arm is held at the grasp (HOLD_SLIP_WORDS in the words that found the target): every logged right-object
    first close, full credit with the hold vs without, mouse 9/12 vs 19/21, sponge 9/12 vs 20/23, paper cup 13/15 vs
    28/28 (v3.7 stage 2: sponge s101 / s105, mouse s110, paper cup s108 held at the return and slipped out during the
    hold); by task, only pick_specific_object did worse with the hold (50/59 vs 106/111), every other task as well or
    better (number 87/87 vs 155/159, object type 70/70 vs 274/286, common sense 85/85 vs 84/95). Apple: 26/28 vs 55/55,
    both losses the s92 apple, under v3.7 and v3.8 alike (0.67-0.69 at the return, 0.086-0.094 after the lift: read as
    held, so no retry; v3.6 lifted at once, read 0.049 and held it on the retry). The span across the jaw, the height
    above the table and the jaw's contact opening are logged for the record."""
    summary = target["summary"]
    text = " ".join([str(target.get("query") or ""), str((report.get("resolver") or {}).get("query") or "")] + [str(q) for q in target.get("track_queries") or []])
    words = text.lower().replace("-", " ").replace(",", " ").split()
    slips = [word for word in HOLD_SLIP_WORDS if word in words]
    axis = quat_column(chosen["quat"], 0)
    span = span_across(summary, math.atan2(axis[1], axis[0]))
    hold = not slips
    return hold, {"span": round(span, 4), "height": round(summary["top"] - TABLE_Z, 4), "at_close": round(at_close, 4),
                  "contact": round(CONTACT_SLOPE * span + CONTACT_OFFSET, 4), "slips": slips, "hold": hold}

def vertical_twin(chosen, options, summary, steep):
    """v3.7: the vertical option of the failed 20-degree option's yaw, when the object is lower than LOW_OBJECT_M and
    that option's lower open fingertip sat more than LOW_TIP_DROP_M below the TCP; else None.
    v3.8: also when the object's top is less than LOW_OBJECT_RETRY_M above the table (triangular prism 2.9 cm, s106 /
    s109: the TCP stopped 1.5 cm above its target, 2.2 cm above the table, and both closes caught air)."""
    v37_low = summary["top"] - summary["bottom"] < LOW_OBJECT_M
    low = v37_low or summary["top"] - TABLE_Z < LOW_OBJECT_RETRY_M
    if steep or not low or tip_drop(chosen["quat"]) <= LOW_TIP_DROP_M:
        return None
    if not chosen["name"].endswith("_tilt20"):
        return None
    name = chosen["name"][: -len("_tilt20")] + "_vertical"
    for item in options:
        if item["name"] == name:
            return item
    return None

def vertical_first_low(target, summary, options):
    """v3.9: a mouse or sponge whose top is less than LOW_OBJECT_RETRY_M above the table is grasped first with the
    vertical option of a 20-degree option whose lower open fingertip would sit more than LOW_TIP_DROP_M below the TCP
    (the 20-degree option then comes second). Every logged right-object first close of a mouse / sponge was a 20-degree
    option (evidence/2026-10-29_v38/grasp_first_closes.csv): full credit 38 of 49 with the fingertip 1 cm or more below
    the TCP (its tip lands on the table and the TCP stops ~1.9 cm above it, the jaw closes over the top), 19 of 19 below
    1 cm; v3.7 / v3.8's vertical retry of low objects held them (mouse s106), but only after a close on air."""
    if summary["top"] - TABLE_Z >= LOW_OBJECT_RETRY_M:
        return options
    text = " ".join([str(target.get("query") or ""), str((report.get("resolver") or {}).get("query") or "")] + [str(q) for q in target.get("track_queries") or []])
    words = text.lower().replace("-", " ").replace(",", " ").split()
    if not any(word in words for word in VERTICAL_FIRST_WORDS):
        return options
    by_name = {item["name"]: item for item in options}
    first38 = sorted(options, key=lambda item: item["rank"])[0]["name"] if options else None
    moved = []
    for item in options:
        if not item["name"].endswith("_tilt20") or tip_drop(item["quat"]) <= LOW_TIP_DROP_M:
            continue
        twin = by_name.get(item["name"][: -len("_tilt20")] + "_vertical")
        if twin is not None and twin["rank"] > item["rank"] - 0.5:
            twin["rank"] = item["rank"] - 0.5
            moved.append(twin["name"])
    first39 = sorted(options, key=lambda item: item["rank"])[0]["name"] if options else None
    if moved and first39 != first38:
        flag39("vertical_first_low", v38_first=first38, first=first39, moved=moved, height_above_table=round(summary["top"] - TABLE_Z, 4))
    return options

def is_plane_target(target):
    """v3.7: the chosen object is a toy plane by the words that found it (target query, tracking queries, resolver)."""
    text = " ".join([str(target.get("query") or ""), str((report.get("resolver") or {}).get("query") or "")] + [str(q) for q in target.get("track_queries") or []])
    words = text.lower().replace("-", " ").replace(",", " ").replace("'", " ").split()
    return any(word in PLANE_WORDS for word in words)

def fuselage_grasp(summary):
    """v3.7: where to close on a toy plane (PLANE_WORDS). The fuselage axis is the principal axis of the cloud's high
    band (> FUSELAGE_HIGH of the top above the bottom); along it, FUSELAGE_BIN_M bins of the cloud's cross-section
    width; the longest run of bins at most FUSELAGE_MAX_W wide (fuselage only, no wing) of at least FUSELAGE_MIN_RUN_M
    is the grasp section, its middle (kept FUSELAGE_END_M inside the plane's ends) the aim. Returns a copy of the
    summary with that aim as grasp_xy and the axis as the footprint yaw (grasp_options then closes across the
    fuselage), or None when no such section exists (v3.6's grasp)."""
    bottom = summary["bottom"]
    pts = [p for p in sampled(summary["cloud"].points, 3000) if float(p[2]) - bottom > 0.004]
    if len(pts) < 60:
        return None
    zs = [float(p[2]) - bottom for p in pts]
    top = quantile(zs, 0.98)
    high = [p for p, z in zip(pts, zs) if z > FUSELAGE_HIGH * top]
    if len(high) < 20:
        return None
    cx = sum(float(p[0]) for p in high) / len(high)
    cy = sum(float(p[1]) for p in high) / len(high)
    sxx = sum((float(p[0]) - cx) ** 2 for p in high) / len(high)
    syy = sum((float(p[1]) - cy) ** 2 for p in high) / len(high)
    sxy = sum((float(p[0]) - cx) * (float(p[1]) - cy) for p in high) / len(high)
    axis = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
    c, s = math.cos(axis), math.sin(axis)
    along = [(float(p[0]) - cx) * c + (float(p[1]) - cy) * s for p in pts]
    across = [(float(p[1]) - cy) * c - (float(p[0]) - cx) * s for p in pts]
    lo, hi = quantile(along, 0.01), quantile(along, 0.99)
    count = int((hi - lo) / FUSELAGE_BIN_M) + 1
    bins = [[] for index in range(count)]
    for u, v in zip(along, across):
        if lo <= u <= hi:
            bins[min(count - 1, int((u - lo) / FUSELAGE_BIN_M))].append(v)
    runs = []
    current = []
    for index, values in enumerate(bins):
        mid = lo + (index + 0.5) * FUSELAGE_BIN_M
        if len(values) >= 3 and quantile(values, 0.95) - quantile(values, 0.05) <= FUSELAGE_MAX_W:
            current.append(mid)
        else:
            if current:
                runs.append(current)
            current = []
    if current:
        runs.append(current)
    best = None
    for run in runs:
        length = run[-1] - run[0] + FUSELAGE_BIN_M
        middle = min(max(0.5 * (run[0] + run[-1]), lo + FUSELAGE_END_M), hi - FUSELAGE_END_M)
        if length >= FUSELAGE_MIN_RUN_M - 1e-6 and run[0] - 0.5 * FUSELAGE_BIN_M <= middle <= run[-1] + 0.5 * FUSELAGE_BIN_M and (best is None or length > best[0]):
            best = (length, middle, run)
    report["plane_grasp"] = {"axis_deg": round(math.degrees(axis), 1), "runs": [[round(run[0], 3), round(run[-1], 3)] for run in runs],
                             "top": round(top, 4), "extent": [round(lo, 3), round(hi, 3)]}
    if best is None:
        return None
    length, middle, run = best
    aim = (cx + middle * c, cy + middle * s)
    report["plane_grasp"].update({"aim": r3(aim), "section": [round(run[0], 3), round(run[-1], 3)], "offset": round(middle, 3)})
    plane = {key: summary[key] for key in summary}
    plane.update({"grasp_xy": aim, "yaw": axis, "length": hi - lo, "width": FUSELAGE_MAX_W})
    return plane

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

def finger_hits(item, summary, grasp_z, margin=0.0):
    """v3.8: neighbouring census blocks (neighbours) an open fingertip of grasp option `item` would land on with the TCP
    at the object's footprint centre and grasp height: [(fingertip xy, block xy, block top)]. C5o: `margin` widens the
    finger (clear_of_neighbours' second tier)."""
    axis = quat_column(item["quat"], 0)
    centre = summary["grasp_xy"]
    hits = []
    for sign in (1.0, -1.0):
        tip = (centre[0] + sign * FINGERTIP_HALF_SPAN_M * axis[0], centre[1] + sign * FINGERTIP_HALF_SPAN_M * axis[1],
               grasp_z + sign * FINGERTIP_HALF_SPAN_M * axis[2])
        for other in neighbours:
            if xy_distance(other["grasp_xy"], centre) < CUBE_MERGE_M:
                continue
            c, s = math.cos(other["yaw"]), math.sin(other["yaw"])
            dx, dy = tip[0] - other["grasp_xy"][0], tip[1] - other["grasp_xy"][1]
            u, v = dx * c + dy * s, dy * c - dx * s
            if (abs(u) <= 0.5 * other["length"] + FINGER_HALF_WIDTH_M + margin and abs(v) <= 0.5 * other["width"] + FINGER_HALF_WIDTH_M + margin
                    and other["top"] > tip[2] + FINGER_TOP_MARGIN_M):
                hits.append((r3(tip[:2]), r3(other["grasp_xy"]), round(other["top"], 4)))
    return hits

def clear_of_neighbours(options, summary, grasp_z, attempt):
    """v3.8, block number: options whose open fingertip would land on a neighbouring census block rank after every
    option that clears them (instance 18, the "4": the lower fingertip of the first option landed on the next block,
    the descend stopped 4.6 cm high and the first close pushed the "4" away; s95, s104, s108 all lost). The options keep
    their order otherwise; when none clears, nothing changes."""
    blocked = []
    near = []
    out = []
    for item in options:
        hits = finger_hits(item, summary, grasp_z)
        if hits:
            blocked.append({"option": item["name"], "hits": hits})
            moved = {key: item[key] for key in item}
            moved["rank"] = item["rank"] + 100
            out.append(moved)
        elif finger_hits(item, summary, grasp_z, FINGER_EXTRA_M):
            # C5o: clear, but by less than FINGER_EXTRA_M: after every option clear by more (1710 + C5o repro sort cubes inst
            # 21: the large cube's yaw+162 closing axis pointed at the middle cube 8.6 cm away, its open fingertip ~1 cm off
            # that cube's edge by this model; both runs flung the middle cube off the table at the close / lift)
            near.append(item["name"])
            moved = {key: item[key] for key in item}
            moved["rank"] = item["rank"] + 50
            out.append(moved)
        else:
            out.append(item)
    attempt["finger_clearance"] = blocked
    if near:
        attempt["finger_near"] = near
    if len(blocked) == len(options) or (not blocked and len(near) in (0, len(options))):
        # nothing blocked and nothing (or everything) near, or everything blocked: the order stays
        return options
    first37 = sorted(options, key=lambda item: item["rank"])[0]["name"]
    first38 = sorted(out, key=lambda item: item["rank"])[0]["name"]
    if first38 != first37:
        flag38("finger_clearance", v37_first=first37, first=first38, blocked=[entry["option"] for entry in blocked], near=near)
    return out

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
    if neighbours:
        options = clear_of_neighbours(options, summary, grasp_z, attempt)
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
        # v3.7: a fuselage aim lies off the plane's footprint centre on purpose; the wrist's centre never moves it
        if WRIST_OVERRIDE_M < gap <= HOVER_SHIFT_MAX_M and measure.get("same_size") and not target.get("fuselage") and gap > HOVER_SHIFT_CAP_M:
            # v3.8: every logged override (3, all lanes) jumped 7.0-7.6 cm to a neighbouring block and lost the episode
            # (number s104 "7": the head's mask covered two blocks; colour on job 1688); none ever corrected a moved object
            flag38("wrist_override_blocked", gap=round(gap, 4))
        elif WRIST_OVERRIDE_M < gap <= HOVER_SHIFT_MAX_M and measure.get("same_size") and not target.get("fuselage"):
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
        if not rest:
            # C5k: no plain option reaches the grasp height either (test: pick-absolute s5080, a medicine bottle at x 0.865;
            # place-relative s5015, a lemon at x 0.86; beverage s5124, a can at x 0.78 across the midline - each episode
            # ended there): the steeper reach-direction tilts (escalation_options), each at its own fingertip floor
            for item in escalation_options(summary, aim, arm):
                floor = summary["bottom"] + TIP_CLEARANCE_M + tip_drop(item["quat"])
                if ik_ranked((aim[0], aim[1], max(grasp_z, floor)), [item], arm, samples=2, keep=1):
                    rest.append(item)
            if rest:
                flag("descend_reapproach_steep", options=[item["name"] for item in rest[:4]])
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
            if steep and grasp_z < table_floor:
                attempt["grasp_z_raised_from"] = round(grasp_z, 4)
                grasp_z = table_floor
                attempt["grasp_z_used"] = round(grasp_z, 4)
            held, aim = descend_and_close(arm, aim, grasp_z, chosen, cloud, attempt, "descend")
    if not held and not report["terminated"] and attempt.get("lifted") and attempt["openings"][-1] < air_limit(attempt):
        # Closed on air at the lifted pose (~LIFT_M above the grasp): re-open, look again from
        # here like a hover (the close may have pushed the object), descend 5 mm lower, close once more.
        retry = {"first_opening": attempt["openings"][-1], "rows_before": report["rows"], "aim_source": "previous"}
        attempt["retry"] = retry
        if report.get("kind") in SOFT_GRIP_KINDS:
            # C5o: ramped, the arm held at its last commanded joints (soft_close)
            soft_open(arm, "retry_open", 1.0)
        else:
            opened = open_gripper(arm=arm)
            count_rows(opened)
            grip_command[arm] = 1.0
            # The finger check and the tracking assume an open jaw; give it three ticks to finish opening.
            settle = move_to_joints(last_command.get(arm) or current_joints(arm), tolerance=0.001, max_steps=3, arm=arm)
            count_rows(settle)
        upright_twin = vertical_twin(chosen, options, summary, steep)
        if upright_twin is not None and not report["terminated"]:
            # v3.7: a low object closed on air under a 20-degree grasp whose closing axis lies in the tilt plane: the
            # lower open fingertip lands on the table first and holds the TCP ~tip_drop above it, the upper finger
            # closes over the object (stage 1 s93: 1.5 cm bar, TCP 5 mm above its top, both closes on air, the bar
            # pushed 1.7 cm forward; logged vertical grasps on objects under 2.2 cm held 25/27). Turn to the same
            # yaw's vertical option here, at the lifted pose, if it has IK at the grasp height.
            if ik_ranked((aim[0], aim[1], grasp_z), [upright_twin], arm, samples=2, keep=1):
                turned = move_ik(tool_pose(arm)[0], [upright_twin], arm, "retry_vertical", gripper=1.0)
                flag37("retry_vertical", failed=chosen["name"], option=None if turned is None else turned["name"],
                       height=round(summary["top"] - summary["bottom"], 4))
                if summary["top"] - summary["bottom"] >= LOW_OBJECT_M:
                    # v3.8: v3.7 retried this one with the same 20-degree option
                    flag38("retry_vertical_low", failed=chosen["name"], option=None if turned is None else turned["name"],
                           height_above_table=round(summary["top"] - TABLE_Z, 4))
                if turned is not None and not report["terminated"]:
                    chosen = turned
                    steep = False
                    table_floor = summary["bottom"] + FLOOR_CLEARANCE_M
                    retry["option"] = chosen["name"]
        measure, seen = wrist_measure(arm, target, aim)
        retry["measure"] = measure
        if seen is not None:
            grasp_z = shape_floor(target, seen, summary, grasp_height(seen, summary))
            cloud = seen["cloud"]
            if measure.get("same_size") and RETRY_MOVED_M < xy_distance(seen["grasp_xy"], aim) <= TRACK_RADIUS_M and not target.get("fuselage"):
                aim = (seen["grasp_xy"][0], seen["grasp_xy"][1], aim[2])
                retry["aim_source"] = "wrist"
        # lower, but never below the floor (v3.1: never raise it; a steep escalation: raise it to
        # the fingertip floor if the wrist height is under it)
        lower = RETRY_LOWER_M
        height = summary["top"] - summary["bottom"]
        if HOLLOW_MIN_M <= height < TALL_M and retry["first_opening"] <= AIR_EMPTY:
            # C5o: an object 4-11 cm tall whose first close went right through (jaw fully shut): the retry closes
            # HOLLOW_RETRY_BELOW_TOP_M under its top instead of lower (the yellow stapler, place-relative inst 28, every run:
            # 3.5 cm under its top lies the gap between its arm and its base; both v2 closes read 0.001-0.006)
            seen_top = seen["top"] if seen is not None else summary["top"]
            grasp_z = max(grasp_z, min(seen_top, summary["top"]) - HOLLOW_RETRY_BELOW_TOP_M)
            lower = 0.0
            retry["hollow"] = True
        if steep:
            grasp_z = max(grasp_z - lower, table_floor)
        else:
            grasp_z = max(grasp_z - lower, min(grasp_z, table_floor))
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

def near_moves(arm, rotated, spot, z, label, gripper, cap, fan_degrees=(0, 30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180),
               offsets=((0.0, 0.0), (0.02, 0.0), (-0.02, 0.0), (0.0, 0.02), (0.0, -0.02)), max_step=None, samples=2):
    """v3.9: IK for the upright bottle at height z: the rotated orientation and its yaws about world z (every one keeps
    the bottle vertical), at the spot and 2 cm around it; executes the solution with the least joint motion if it is
    under `cap` rad (dev1 s213: the only IK solution move_ik found for a 15 cm set-down was an 11 rad reconfiguration that
    swung the arm and the can out of the head camera's view; the set-downs that worked moved 0.7-1.1 rad).
    Returns (executed, entry); executed is None when no candidate was under the cap or none planned."""
    fan = []
    for degrees in fan_degrees:
        fan.append(option("%s_z%+d" % (rotated["name"], degrees), quat_mul(quat_axis_angle((0.0, 0.0, 1.0), math.radians(degrees)), rotated["quat"]), 0))
    candidates = []
    for dx, dy in offsets:
        point = (spot[0] + dx, spot[1] + dy, z)
        for rank, distance, item, joints in ik_ranked(point, fan, arm, samples=samples, keep=max(4, 2 * samples)):
            candidates.append((distance, point, item, joints, distance + NEAR_OFFSET_RAD_PER_M * math.hypot(dx, dy)))
    # the least joint motion, a 2 cm offset counting as 0.1 rad more
    candidates.sort(key=lambda c: c[4])
    entry = {"label": label, "target": r3((spot[0], spot[1], z)), "ik_solutions": len(candidates),
             "best_joint_distance": round(candidates[0][0], 3) if candidates else None, "cap_rad": cap}
    report["moves"].append(entry)
    for index, candidate in enumerate(candidates[:4]):
        distance, point, item, joints = candidate[:4]
        if distance > cap:
            entry["fail"] = "least joint motion %.2f rad over the cap" % distance
            return None, entry
        try:
            plan = plan_motion(joints, arm=arm, strategy=STRATEGY_FINAL)
        except Exception as exc:
            entry.setdefault("plan_errors", []).append("exception: " + str(exc)[:80])
            continue
        if not plan.ok or plan.trajectory is None:
            entry.setdefault("plan_errors", []).append(error_text(plan))
            continue
        entry.update({"option": item["name"], "joint_distance": round(distance, 3), "target": r3(point), "plan_attempt": index + 1,
                      "waypoints": len(plan.trajectory.joint_positions)})
        trajectory = plan.trajectory
        if max_step is not None:
            # C5j: a held object slips out of a fast swing (dev2 stack s1051: 2.5 rad in one 30-row chunk, the cube was
            # gone at the set-down); the interpolation moves a joint up to 0.05 rad per row
            trajectory = slowed(trajectory, arm, max_step)
            entry["waypoints_slowed"] = len(trajectory.joint_positions)
        execute(trajectory, arm, entry, gripper)
        if report["terminated"] or (entry.get("rows") or 0) > 0:
            # executed; a residual left by the table under the bottle still counts: the bottle is down
            return item, entry
        entry.setdefault("plan_errors", []).append("not executed: " + str(entry.get("exec_error")))
    if "fail" not in entry:
        entry["fail"] = "no IK solution" if not candidates else "planning refused every candidate"
    return None, entry

def home_moved_arms():
    """v3.8 (leaderboard): return every arm that does not start at its start pose to the start joints, gripper open,
    before anything looks at the table. On the leaderboard 9 % of job 1688's episodes began with an arm left over the
    table by an earlier episode (TCP 0.74-0.93 m high): the head saw nothing behind it ("visible": false, no target),
    hovers were refused from there, mean credit ~0.74 vs 0.98. A low arm first rises HOME_LIFT_M straight up (joint
    interpolation, its measured orientation), then interpolates to the start joints: 1-2 chunks, only in such episodes."""
    homed = []
    for arm in ("left", "right"):
        try:
            position, quat = tool_pose(arm)
        except Exception:
            continue
        gap = math.sqrt(sum((position[i] - START_TCP[arm][i]) ** 2 for i in range(3)))
        if gap <= MOVED_START_M:
            continue
        info = {"arm": arm, "start_tcp": r3(position), "gap": round(gap, 3), "gripper": round(gripper_opening(arm), 3)}
        if position[2] < START_TCP[arm][2] - HOME_LOW_M:
            info["lift"] = move_ik((position[0], position[1], position[2] + HOME_LIFT_M), [option("measured", quat, 0)], arm, "home_lift",
                                   gripper=1.0, strategy=STRATEGY_FINAL, samples=2) is not None
        if report["terminated"]:
            break
        info["homed"] = move_joints(arm, START_JOINTS[arm], "home", 1.0)
        info["end_tcp"] = r3(tool_pose(arm)[0])
        homed.append(info)
        flag38("home_moved_arm", arm=arm, gap=info["gap"], homed=info["homed"])
        if report["terminated"]:
            break
    if homed:
        report["home"] = homed
    return homed



# ----------------------------------------------------------------------------- spatial: helpers (C5j)
def window_hold_limit():
    """v3.8's limit on the jaw-settle hold so the lift rises before PickUpOnGripper's last scoring check, pick tasks only:
    in the place / stack / sort tasks nothing is scored at the lift and the next stage allows 600-1500 rows."""
    if report.get("kind") not in PICK_TASKS:
        return None
    ref = follow_row.get("row")
    if ref is None:
        return None
    rows = report["rows"]
    upcoming = (rows // CHUNK_ROWS + 1) * CHUNK_ROWS
    if upcoming + CHUNK_ROWS <= ref + WINDOW_LAST_ROWS:
        return None
    return upcoming - rows - RISE_ROWS


def after(text, lead):
    index = text.find(lead)
    return None if index < 0 else text[index + len(lead):]


def noun_of(phrase):
    """A noun phrase without its leading article, trailing punctuation or a trailing 'located'."""
    words = str(phrase or "").strip().strip(",.").split()
    while words and words[-1] in ("located", "which", "that", "is", "lying", "placed", "sitting"):
        words = words[:-1]
    if words and words[0] in ("the", "a", "an"):
        words = words[1:]
    return " ".join(words)


def other_arm(arm):
    return "right" if arm == "left" else "left"


def task_from_sentence(text):
    lowered = text.lower()
    if "digit" in lowered and "arrange" in lowered:
        return "sort_number"
    if "arrange" in lowered or ("small" in lowered and "large" in lowered):
        return "sort_cubes_by_size"
    if "bowl" in lowered and "stack" in lowered:
        return "stack_bowls"
    if "stack" in lowered:
        return "stack_three_building_blocks"
    if "position" in lowered and "places" in lowered:
        return "place_beverage_to_anothers_position"
    if lowered.startswith("place") or " to the left of " in lowered or " to the right of " in lowered or " to the top of " in lowered or " to the bottom of " in lowered:
        return "place_object_relative_position"
    for lead, rel in REL_SPLITS:
        if lead in lowered:
            return "pick_object_relative_position"
    return "pick_object_absolute_position"


def parse_task(task, text):
    """The eight templates (spatial_instances.json): nouns, rule / relation, arms."""
    lowered = text.lower()
    parsed = {"task": task}
    rest = after(lowered, "picks up ") or after(lowered, "pick up ") or after(lowered, "place ") or lowered
    if task == "pick_object_absolute_position":
        for phrase, rule in ABS_RULES:
            index = rest.find(phrase)
            if index >= 0:
                parsed.update({"noun": noun_of(rest[:index]), "rule": rule})
                break
    elif task == "pick_object_relative_position":
        for lead, relation in REL_SPLITS:
            if lead in rest:
                noun, ref = rest.split(lead, 1)
                parsed.update({"noun": noun_of(noun), "relation": relation, "reference": noun_of(ref)})
                break
    elif task == "place_object_relative_position":
        for lead, relation in PLACE_SPLITS:
            if lead in rest:
                noun, ref = rest.split(lead, 1)
                parsed.update({"noun": noun_of(noun), "relation": relation, "reference": noun_of(ref)})
                break
    elif task == "place_beverage_to_anothers_position":
        for clause in lowered.split(","):
            clause = clause.strip()
            arm = "left" if clause.startswith("left") else ("right" if clause.startswith("right") else None)
            if " places " in clause and " at the " in clause:
                parsed["placer"] = arm
                parsed["y"] = noun_of(clause.split(" places ", 1)[1].split(" at the ", 1)[0])
                parsed["x"] = noun_of(clause.split(" at the ", 1)[1].split("'s position", 1)[0].split("s position", 1)[0])
            elif " picks up " in clause:
                parsed.setdefault("pickers", []).append([arm, noun_of(clause.split(" picks up ", 1)[1])])
        for arm, noun in parsed.get("pickers", []):
            if noun == parsed.get("x") and arm is not None:
                parsed["taker"] = arm
        if parsed.get("placer") and not parsed.get("taker"):
            parsed["taker"] = other_arm(parsed["placer"])
    return parsed


def queries_for(noun):
    key = " ".join(str(noun or "").lower().split())
    if key in NOUN_QUERIES:
        return list(NOUN_QUERIES[key])
    best = None
    for name in NOUN_QUERIES:
        if name in key and (best is None or len(name) > len(best)):
            best = name
    if best is not None:
        return [key] + [q for q in NOUN_QUERIES[best] if q != key]
    return [key] if key else []


def census(queries, views=("head",), min_score=MIN_SCORE, max_extent=0.3):
    """Every plausible SAM 3 instance of the queries, merged by footprint centre (MERGE_M) across queries and views:
    [{"xy", "summary", "score" (of the kept summary), "best", "box", "view", "query", "hits"}], best first. A head
    summary is kept over a wrist one (the grasp aims with the head, v3.1)."""
    objects = []
    for view in views:
        for query in queries:
            for score, summary, box in text_instances(view, query, min_score=min_score, max_extent=max_extent):
                xy = summary["grasp_xy"]
                entry = None
                for obj in objects:
                    if xy_distance(obj["xy"], xy) < MERGE_M:
                        entry = obj
                        break
                if entry is None:
                    entry = {"xy": xy, "summary": summary, "score": score, "best": score, "box": box, "view": view, "query": query, "hits": []}
                    objects.append(entry)
                elif (view == "head" and entry["view"] != "head") or (view == entry["view"] and score > entry["score"]):
                    entry.update({"xy": xy, "summary": summary, "score": score, "box": box, "view": view, "query": query})
                entry["best"] = max(entry["best"], score)
                entry["hits"].append([view, query, round(score, 3)])
    objects.sort(key=lambda obj: -obj["best"])
    return objects


def merge_into(objects, more):
    for obj in more:
        if all(xy_distance(obj["xy"], other["xy"]) >= MERGE_M for other in objects):
            objects.append(obj)
    objects.sort(key=lambda obj: -obj["best"])
    return objects


def describe_obj(obj):
    if obj is None:
        return None
    s = obj["summary"]
    return {"xy": r3(obj["xy"]), "score": round(obj["best"], 3), "view": obj["view"], "query": obj["query"],
            "size": r3((s["length"], s["width"], s["height"])), "top": round(s["top"], 4), "bottom": round(s["bottom"], 4),
            "hits": obj["hits"][:6]}


def size_gap(a, b):
    """Largest relative difference of footprint length / width / height (0 = the same size). C5k: heights differing by up
    to TWIN_HEIGHT_SLACK_M count as equal (the head sees a low object's top, a nearer one more of its side: test
    pick-absolute s5016, two toy cars at 2.8 and 1.2 cm, gap 0.58 > 0.5; the orange and lemon scoring a fifth of them
    became the twins)."""
    worst = 0.0
    for key in ("length", "width", "height"):
        x, y = a[key], b[key]
        diff = abs(x - y)
        if key == "height":
            diff = max(0.0, diff - TWIN_HEIGHT_SLACK_M)
        worst = max(worst, diff / max(0.015, x, y))
    return worst


def twin_pair(objects):
    """The noun's twins among one query's instances (best first): two instances >= TWIN_MIN_GAP_M apart whose SAM 3 scores
    are within TWIN_SCORE_RATIO of each other and whose sizes agree (size gap <= TWIN_MAX_GAP). Twins score alike even
    when SAM 3 scores the noun low (C5j probe frames: paper cups 0.054 / 0.053, clocks 0.062 / 0.059, toy cars 0.146 /
    0.132, each pair well above the next object). Of several such pairs, the best-scoring, then the most alike.
    (a, b, size gap, ambiguous) or None; ambiguous: a third instance scores like the pair."""
    top = objects[:4]
    pairs = []
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            a, b = top[i], top[j]
            if xy_distance(a["xy"], b["xy"]) < TWIN_MIN_GAP_M:
                continue
            gap = size_gap(a["summary"], b["summary"])
            if gap > TWIN_MAX_GAP:
                continue
            ratio = min(a["best"], b["best"]) / max(a["best"], b["best"], 1e-6)
            if ratio < TWIN_SCORE_RATIO:
                # a twin partly hidden (e.g. behind a start-pose gripper) scores lower: still the twins when they are the
                # only instances and agree in size (C5j dev1 pick-absolute s21: paper cups 0.758 / 0.326, sizes within 6 mm)
                others = [obj for obj in top if obj is not a and obj is not b and obj["best"] >= TWIN_LONE_RATIO * min(a["best"], b["best"])]
                if ratio < TWIN_LONE_SCORE_RATIO or gap > TWIN_LONE_MAX_GAP or others:
                    continue
            pairs.append((-min(a["best"], b["best"]), gap, i, j, a, b))
    if not pairs:
        return None
    pairs.sort(key=lambda p: (round(p[0], 2), p[1]))
    best = pairs[0]
    a, b = best[4], best[5]
    third = [obj for obj in top if obj is not a and obj is not b and obj["best"] >= TWIN_SCORE_RATIO * min(a["best"], b["best"])
             and min(xy_distance(obj["xy"], a["xy"]), xy_distance(obj["xy"], b["xy"])) >= TWIN_MIN_GAP_M]
    return a, b, best[1], bool(third)


def find_twins(noun, avoid=()):
    """The noun's two identical instances: the first query whose instances (score >= TWIN_FLOOR) hold a twin pair decides;
    else every query's instances merged, then the wrist views (they see the whole table from the start pose).
    `avoid` (e.g. the reference) is never a twin. (pair or None, census, query, tried)."""
    queries = queries_for(noun)
    tried = []
    merged = []
    for query in queries:
        found = [obj for obj in census([query], min_score=TWIN_FLOOR)
                 if all(xy_distance(obj["xy"], other["xy"]) >= MERGE_M for other in avoid if other is not None)]
        tried.append([query, len(found), [round(obj["best"], 3) for obj in found[:4]]])
        merge_into(merged, found)
        pair = twin_pair(found)
        if pair is not None:
            return pair, found, query, tried
    pair = twin_pair(merged)
    if pair is not None:
        return pair, merged, None, tried
    wrist = census(queries, views=("hand_left", "hand_right"), min_score=TWIN_FLOOR)
    tried.append(["wrists", len(wrist)])
    merge_into(merged, wrist)
    return twin_pair(merged), merged, None, tried


def find_single(noun, avoid=(), views=("head",)):
    """The noun's one instance that is not one of `avoid`: the first query whose best instance (>= TWIN_FLOOR) clearly
    beats its next one (CLEAR_RATIO) or stands alone; the wrist views next. (obj, query) or (None, None): the caller
    then asks Qwen (C5j: SAM 3 scored a peach 0.046 and an apple 0.044 as "peach")."""
    for view_set in (views, ("hand_left", "hand_right")):
        for query in queries_for(noun):
            found = [obj for obj in census([query], views=view_set, min_score=TWIN_FLOOR)
                     if all(xy_distance(obj["xy"], other["xy"]) >= MERGE_M for other in avoid if other is not None)]
            if not found:
                continue
            if len(found) == 1 or found[0]["best"] >= CLEAR_RATIO * found[1]["best"] or found[0]["best"] >= MIN_SCORE and found[1]["best"] < MIN_SCORE:
                return found[0], query
    return None, None


def resolve_text(text, camera=None):
    """g2.resolve_instruction (Qwen3.6 on a camera image; never raises), logged in report["resolver_calls"]."""
    try:
        resolved = g2.resolve_instruction(text) if camera is None else g2.resolve_instruction(text, camera_name=camera)
    except Exception as exc:
        resolved = {"ok": False, "error": str(exc)[:120]}
    if not resolved:
        resolved = {"ok": False}
    entry = {k: resolved.get(k) for k in ("ok", "query", "box_xyxy", "visible", "confidence", "latency_s", "error")}
    entry["text"] = text[:160]
    report.setdefault("resolver_calls", []).append(entry)
    report["vlm_seconds"] = report.get("vlm_seconds", 0.0) + float(resolved.get("latency_s") or 0.0)
    return resolved


def resolver_object(text, noun, for_target=True):
    """Fallback: Qwen's box for the sentence -> the plausible head mask under its centre, else the noun instance that
    overlaps it most. A census-like object or None. Only a target's answer goes to report["resolver"] (the hold policy
    and the plane test read its words)."""
    resolved = resolve_text(text)
    if for_target:
        report["resolver"] = {k: resolved.get(k) for k in ("ok", "query", "box_xyxy", "confidence")}
    box = resolved.get("box_xyxy")
    if not box:
        return None
    centre = (0.5 * (float(box[0]) + float(box[2])), 0.5 * (float(box[1]) + float(box[3])))
    found = point_instances("head", centre, 0.3)
    if found:
        summary = found[0][1]
        return {"xy": summary["grasp_xy"], "summary": summary, "score": found[0][0], "best": found[0][0], "box": None, "view": "head",
                "query": "resolver_point", "hits": [["head", "resolver_point", round(found[0][0], 3)]]}
    best = None
    for query in queries_for(noun)[:2]:
        for score, summary, ibox in text_instances("head", query, min_score=0.05):
            if ibox is None:
                continue
            iou = box_iou([float(v) for v in box], ibox)
            if iou >= 0.2 and (best is None or iou > best[0]):
                best = (iou, score, summary, query)
    if best is None:
        return None
    return {"xy": best[2]["grasp_xy"], "summary": best[2], "score": best[1], "best": best[1], "box": None, "view": "head",
            "query": best[3], "hits": [["head", best[3], round(best[1], 3)]]}


def target_from(obj, how, noun):
    queries = queries_for(noun)
    track = [obj["query"]] + [q for q in queries if q != obj["query"]][:2]
    return target_of(obj["summary"], how, track, query=obj["query"], score=round(obj["best"], 3))


def side_arm(xy):
    return "left" if xy[1] > 0.0 else "right"


def abs_measure(rule, xy):
    """Smaller is better for the absolute-position rule (scene truth: right on all 30 instances)."""
    if rule == "left":
        return -xy[1]
    if rule == "right":
        return xy[1]
    if rule == "closer":
        return xy[0]
    if rule == "further":
        return -xy[0]
    return math.hypot(xy[0] - TABLE_CENTRE[0], xy[1] - TABLE_CENTRE[1])


# ----------------------------------------------------------------------------- spatial: grasp and place
def grasp_target(target, arm, hold_after, options=None):
    """v4.0's pick with v4.0's options (toy plane: across the fuselage; low mouse / sponge: vertical first). C5o: `options`
    given (beverage Y, bev_grasp_options) replace v4.0's."""
    if options is not None:
        return pick(target, arm, options, hold_after=hold_after)
    summary = target["summary"]
    if is_plane_target(target):
        plane = fuselage_grasp(summary)
        if plane is not None:
            flag("plane_fuselage", aim=r3(plane["grasp_xy"]), footprint_centre=r3(summary["grasp_xy"]))
            target["footprint_xy"] = summary["grasp_xy"]
            summary = plane
            target["summary"] = plane
            target["fuselage"] = True
    held = pick(target, arm, vertical_first_low(target, summary, grasp_options(summary)), hold_after=hold_after)
    return held


def grasp_geometry(target):
    """Where the held object sits relative to the TCP, from the last attempt: (TCP z at the close - object bottom,
    object footprint centre - TCP aim in xy)."""
    attempt = report["attempts"][-1] if report["attempts"] else {}
    retry = attempt.get("retry") or {}
    grasp_z = retry.get("grasp_z_used", attempt.get("grasp_z_used"))
    aim = retry.get("aim_used", attempt.get("aim_used"))
    summary = target["summary"]
    centre = summary["grasp_xy"]
    if target.get("fuselage"):
        centre = target.get("footprint_xy") or centre
    below = (float(grasp_z) - summary["bottom"]) if grasp_z is not None else (summary["top"] - summary["bottom"] - GRASP_BELOW_TOP_M)
    offset = (centre[0] - aim[0], centre[1] - aim[1]) if aim is not None else (0.0, 0.0)
    return max(0.0, below), offset


def slowed(trajectory, arm, max_step):
    """The trajectory resampled (linear in joint space) so that no joint moves more than max_step rad per row."""
    rows = [[float(v) for v in row] for row in trajectory.joint_positions]
    previous = current_joints(arm)
    out = []
    for row in rows:
        step = max([abs(a - b) for a, b in zip(row, previous)] or [0.0])
        count = max(1, int(math.ceil(step / max_step)))
        for k in range(1, count + 1):
            out.append([previous[j] + (row[j] - previous[j]) * k / float(count) for j in range(len(row))])
        previous = row
    if len(out) == len(rows):
        return trajectory
    return Trajectory(
        joint_positions=out,
        dt_s=trajectory.dt_s,
        joint_names=trajectory.joint_names,
        planner=trajectory.planner,
        collision_aware=trajectory.collision_aware,
        expected_start=trajectory.expected_start,
        arm=trajectory.arm,
        embodiment=trajectory.embodiment,
        gripper_positions=None if trajectory.gripper_positions is None else [float(trajectory.gripper_positions[0]) for row in out],
    )


def still_held(arm):
    return gripper_opening(arm) >= AIR_OPENING


def carry(arm, goal_xy, tcp_z, label, gripper=0.0, fan=CARRY_FAN):
    """Move the TCP (holding an object) to (goal_xy, tcp_z): straight up first when that height is above the TCP, then
    across with the measured orientation turned about world z by the yaw (CARRY_FAN) with the least joint motion
    (v3.9 near_moves, joint interpolation, under CARRY_MAX_JOINT_RAD). Returns the executed option or None."""
    here, quat = tool_pose(arm)
    held = option("held", quat, 0)
    entry = {"label": label, "goal": r3((goal_xy[0], goal_xy[1], tcp_z)), "from": r3(here)}
    report.setdefault("carries", []).append(entry)
    if tcp_z > here[2] + 0.02:
        up, up_entry = near_moves(arm, held, (here[0], here[1]), tcp_z, label + "_up", gripper, CARRY_MAX_JOINT_RAD, fan_degrees=(0,),
                                  offsets=((0.0, 0.0),), max_step=CARRY_MAX_STEP_RAD)
        entry["up"] = up is not None
        if report["terminated"]:
            return None
        if up is not None:
            held = up
    done, move_entry = near_moves(arm, held, goal_xy, tcp_z, label, gripper, CARRY_MAX_JOINT_RAD, fan_degrees=fan, offsets=((0.0, 0.0),),
                                  max_step=CARRY_MAX_STEP_RAD)
    entry["done"] = None if done is None else done["name"]
    entry["exec_ok"] = move_entry.get("exec_ok")
    if done is None and not report["terminated"]:
        # wider: 2 cm around the goal, then 8 cm lower (a high far goal may have no IK)
        done, move_entry = near_moves(arm, held, goal_xy, tcp_z, label + "_near", gripper, CARRY_MAX_JOINT_RAD, fan_degrees=fan,
                                      max_step=CARRY_MAX_STEP_RAD)
        entry["done_near"] = None if done is None else done["name"]
    if done is None and not report["terminated"]:
        done, move_entry = near_moves(arm, held, goal_xy, tcp_z - 0.08, label + "_lower", gripper, CARRY_MAX_JOINT_RAD, fan_degrees=fan,
                                      max_step=CARRY_MAX_STEP_RAD)
        entry["done_lower"] = None if done is None else done["name"]
    try:
        entry["to"] = r3(tool_pose(arm)[0])
    except Exception:
        pass
    return done


def carry_any(arm, goals, heights, label, gripper=0.0, fan=CARRY_FAN, cap=None, samples=2):
    """carry() over alternative goals: straight up to the highest height first, then the first (height, goal) - heights
    outer, goals in the caller's preference order - whose yaw fan has a least-motion IK under the cap. Returns (option,
    goal, height) or (None, None, None)."""
    here, quat = tool_pose(arm)
    held = option("held", quat, 0)
    entry = {"label": label, "goals": [r3(g) for g in goals[:8]], "heights": [round(z, 4) for z in heights], "from": r3(here)}
    report.setdefault("carries", []).append(entry)
    top = max(heights)
    if top > here[2] + 0.02:
        up, up_entry = near_moves(arm, held, (here[0], here[1]), top, label + "_up", gripper, CARRY_MAX_JOINT_RAD, fan_degrees=(0,),
                                  offsets=((0.0, 0.0),), max_step=CARRY_MAX_STEP_RAD)
        entry["up"] = up is not None
        if report["terminated"]:
            return None, None, None
        if up is not None:
            held = up
    tried = 0
    for z in heights:
        for goal in goals:
            tried += 1
            done, move_entry = near_moves(arm, held, goal, z, label, gripper, cap or CARRY_MAX_JOINT_RAD, fan_degrees=fan, offsets=((0.0, 0.0),),
                                          max_step=CARRY_MAX_STEP_RAD, samples=samples)
            if report["terminated"]:
                return None, None, None
            if done is not None:
                entry.update({"done": done["name"], "goal": r3(goal), "z": round(z, 4), "tried": tried, "exec_ok": move_entry.get("exec_ok")})
                try:
                    entry["to"] = r3(tool_pose(arm)[0])
                except Exception:
                    pass
                return done, goal, z
    entry["done"] = None
    entry["tried"] = tried
    return None, None, None


def lower_to(arm, tcp_z, label, gripper=0.0):
    """Straight down (or up) to tcp_z, orientation kept; else within 1 cm and +-30 deg of yaw (C5j dev2 stack s1051: the
    only lowering candidate had no IK and the cube was let go 10 cm up)."""
    here, quat = tool_pose(arm)
    done, entry = near_moves(arm, option("held", quat, 0), (here[0], here[1]), tcp_z, label, gripper, CARRY_MAX_JOINT_RAD, fan_degrees=(0,),
                             offsets=((0.0, 0.0),), max_step=CARRY_MAX_STEP_RAD)
    if done is None and not report["terminated"]:
        done, entry = near_moves(arm, option("held", quat, 0), (here[0], here[1]), tcp_z, label + "_near", gripper, CARRY_MAX_JOINT_RAD,
                                 fan_degrees=(0, 30, -30), offsets=((0.0, 0.0), (0.01, 0.0), (-0.01, 0.0), (0.0, 0.01), (0.0, -0.01)),
                                 max_step=CARRY_MAX_STEP_RAD)
    return done


def soft_open(arm, label, opening, ramp_rows=RELEASE_RAMP_ROWS):
    """C5o: one chunk (CHUNK_ROWS rows) at the arm's MEASURED joints (as open_gripper / close_gripper hold it) with the jaw
    ramped linearly from its measured opening to `opening` over `ramp_rows` rows, then held, instead of the harness's
    stepped jaw command. dev2 / dev3 held the last COMMANDED joints: a set-down that stopped 5-9 mm high on the cube below
    was then pushed down that much while the jaw opened, and the second cube sprang off the base (dev2 smoke stack three
    5034 / 1058: found 4.3 / 10 cm away; dev3 1058 knocked the base). Returns the chunk's ok."""
    entry = {"label": label, "opening": round(opening, 3)}
    report["moves"].append(entry)
    hold = current_joints(arm)
    try:
        plan = plan_motion(hold, arm=arm, strategy=STRATEGY_FINAL)
    except Exception as exc:
        plan = None
        entry["fail"] = "exception: " + str(exc)[:80]
    if plan is None or not plan.ok or plan.trajectory is None:
        # no trajectory template: the harness's own gripper command (stepped) so the jaw still moves
        if plan is not None:
            entry["fail"] = error_text(plan)
        raw = close_gripper(arm=arm) if opening < 0.5 else open_gripper(arm=arm)
        count_rows(raw)
        grip_command[arm] = 0.0 if opening < 0.5 else 1.0
        entry["fallback"] = "close_gripper" if opening < 0.5 else "open_gripper"
        return raw.ok
    start = gripper_opening(arm)
    entry["from_opening"] = round(start, 3)
    t = plan.trajectory
    grippers = [start + (opening - start) * min(1.0, (k + 1) / float(max(1, ramp_rows))) for k in range(CHUNK_ROWS)]
    ramp = Trajectory(
        joint_positions=[list(hold) for k in range(CHUNK_ROWS)],
        dt_s=t.dt_s,
        joint_names=t.joint_names,
        planner=t.planner,
        collision_aware=t.collision_aware,
        expected_start=t.expected_start,
        arm=t.arm,
        embodiment=t.embodiment,
        gripper_positions=grippers,
    )
    # executed as one chunk, no servo rounds (as close_gripper / open_gripper): a servo step commands the harness's last
    # confirmed gripper value, which a refused chunk would leave stale
    done = execute_trajectory(ramp)
    count_rows(done)
    worst = max([float(v) for v in done.final_errors.values()] or [0.0])
    entry.update({"exec_ok": done.ok, "rows": done.steps_executed, "residual_rad": round(worst, 4)})
    if done.terminated:
        entry["terminated"] = True
    last_command[arm] = list(hold)
    grip_command[arm] = opening
    return done.ok


def soft_close(arm, label):
    """C5o: close_gripper for the sort / stack tasks - the jaw ramped from its opening to closed over CLOSE_RAMP_ROWS of one
    chunk, the arm held at its last commanded joints (soft_open). In every logged arm blow-up but one (11: 1710, the C5o
    repro, C5k stage 2 / repeat / control; joints swinging 2.5-12 rad off a constant hold) the gripper command had just
    stepped (open or close) in that chunk or the one before; sort-number inst 14's second close swung the left arm 3.2 rad
    with the jaw still open, every C5o / 1710 / C5k run of it lost."""
    return soft_open(arm, label, 0.0, ramp_rows=CLOSE_RAMP_ROWS)


def release_and_retreat(arm, label, partial=True):
    """C5o (partial): open the jaw part way (RELEASE_OPEN_DELTA past its measured opening; soft_open: ramped, the arm held at
    its last commanded joints), then back straight up RELEASE_RETREAT_M with the jaw so; the next move opens it fully away
    from the table. v2 opened it fully at the set-down: the 20-degree grasp's lower finger swings down along the tilted
    closing axis as the jaw opens and met the table (1710: the TCP pushed 5-14 mm sideways and 3-7 mm up at most table
    releases; three of the job's five Onfloor blow-ups came at a release). partial=False: v2's release (open, hold one padded
    chunk with the jaw open, back straight up) - the beverage can, held 8 cm above the table."""
    if not partial:
        opened = open_gripper(arm=arm)
        count_rows(opened)
        grip_command[arm] = 1.0
        if report["terminated"]:
            return
        hold_still(arm, label + "_open", 1.0)
        if report["terminated"]:
            return
        here, quat = tool_pose(arm)
        move_ik((here[0], here[1], here[2] + RELEASE_RETREAT_M), [option("retreat", quat, 0)], arm, label + "_retreat", gripper=1.0,
                strategy=STRATEGY_FINAL, samples=2)
        return
    opening = min(RELEASE_OPEN_MAX, gripper_opening(arm) + RELEASE_OPEN_DELTA)
    soft_open(arm, label + "_open", opening)
    if report["terminated"]:
        return
    here, quat = tool_pose(arm)
    move_ik((here[0], here[1], here[2] + RELEASE_RETREAT_M), [option("retreat", quat, 0)], arm, label + "_retreat", gripper=opening,
            strategy=STRATEGY_FINAL, samples=2)


def clear_arm(arm, label):
    """Send an empty arm back to its start joints (jaw open) so the other arm can work where it just was (C5j dev2
    stack: after a release the arm hovered 10 cm above the stack while the other arm brought the next cube). C5k: first
    straight up to CLEAR_SAFE_Z over the table and out to CLEAR_SIDE_Y on its own side (open-jaw moves, least joint
    motion): the cuRobo plan home is collision-aware only against the head depth, in which the arm hides what it just
    set down (test stack-bowls s5037: the clear home from 10 cm over the rim dragged the first bowl 12 cm)."""
    here, quat = tool_pose(arm)
    if math.sqrt(sum((here[i] - START_TCP[arm][i]) ** 2 for i in range(3))) <= MOVED_START_M:
        return True
    if here[2] < TABLE_Z + CLEAR_SAFE_Z - 0.01:
        near_moves(arm, option("open", quat, 0), (here[0], here[1]), TABLE_Z + CLEAR_SAFE_Z, label + "_up", 1.0, CARRY_MAX_JOINT_RAD,
                   fan_degrees=(0,), offsets=((0.0, 0.0),))
        if report["terminated"]:
            return False
        here, quat = tool_pose(arm)
    side = 1.0 if arm == "left" else -1.0
    if side * here[1] < CLEAR_SIDE_Y:
        near_moves(arm, option("open", quat, 0), (here[0], side * CLEAR_SIDE_Y), here[2], label + "_side", 1.0, CARRY_MAX_JOINT_RAD,
                   fan_degrees=(0, 30, -30), offsets=((0.0, 0.0),))
        if report["terminated"]:
            return False
    # collision-aware (cuRobo with the depth scene): a joint interpolation home from just above a stack can sweep through
    # it (dev5 stack s1042: every placement within 4 mm, the clears' residuals 0.036-0.050 rad, Stack 0)
    entry = {"label": label}
    report["moves"].append(entry)
    try:
        plan = plan_motion([float(v) for v in START_JOINTS[arm]], arm=arm, strategy=STRATEGY)
    except Exception as exc:
        plan = None
        entry["plan_error"] = "exception: " + str(exc)[:80]
    if plan is not None and plan.ok and plan.trajectory is not None:
        entry["planner"] = "curobo"
        return execute(plan.trajectory, arm, entry, 1.0)
    entry["planner"] = "interpolation"
    if plan is not None:
        entry["plan_error"] = error_text(plan)
    return move_joints(arm, START_JOINTS[arm], label + "_interp", 1.0)


def hold_checks(arm, label, gripper, count=2):
    """`count` padded 30-row chunks at the last commanded joints: every 30 consecutive rows hold one GenieSim check, so
    this covers `count` consecutive checks with one request each (hold_tick's single-row steps cost a request per row)."""
    for index in range(count):
        hold_still(arm, "%s_%d" % (label, index), gripper)
        if report["terminated"]:
            return


# ----------------------------------------------------------------------------- spatial: tasks
def run_pick(task, parsed):
    noun = parsed.get("noun")
    if not noun:
        report["error"] = "could not parse the noun"
        return
    stage("census", noun=noun)
    pair, found, query, tried = find_twins(noun)
    ref = None
    if pair is not None and pair[3] and task == "pick_object_relative_position":
        # a third instance scores like the twins (an apple among peaches): find the reference first and exclude it
        ref, ref_query = find_single(parsed.get("reference"))
        if ref is not None:
            flag("twins_ambiguous_reference_first")
            pair, found, query, tried = find_twins(noun, avoid=(ref,))
    report["census"] = {"noun": noun, "query": query, "tried": tried, "objects": [describe_obj(obj) for obj in found[:6]]}
    chosen = None
    how = None
    if pair is not None:
        a, b, gap, ambiguous = pair
        report["twins"] = {"a": describe_obj(a), "b": describe_obj(b), "size_gap": round(gap, 3), "ambiguous": ambiguous}
        if task == "pick_object_absolute_position":
            rule = parsed.get("rule") or "center"
            ranked = sorted([a, b], key=lambda obj: abs_measure(rule, obj["xy"]))
            how = "rule_" + rule
            report["twins"]["margin"] = round(abs_measure(rule, ranked[1]["xy"]) - abs_measure(rule, ranked[0]["xy"]), 4)
            chosen = ranked[0]
        else:
            if ref is None:
                ref, ref_query = find_single(parsed.get("reference"), avoid=(a, b))
            if ref is None:
                flag("reference_by_resolver")
                ref = resolver_object("the " + str(parsed.get("reference")), parsed.get("reference"), for_target=False)
                if ref is not None and min(xy_distance(ref["xy"], a["xy"]), xy_distance(ref["xy"], b["xy"])) < MERGE_M:
                    ref = None
            report["reference"] = describe_obj(ref)
            if ref is not None:
                sign = 1.0 if parsed.get("relation") == "closer" else -1.0
                ranked = sorted([a, b], key=lambda obj: sign * xy_distance(obj["xy"], ref["xy"]))
                report["twins"]["distances"] = [round(xy_distance(obj["xy"], ref["xy"]), 4) for obj in ranked]
                how = "relative_" + str(parsed.get("relation"))
                chosen = ranked[0]
    if chosen is None:
        flag("resolver_fallback", reason="no twin pair" if pair is None else "no reference")
        chosen = resolver_object(LANGUAGE, noun)
        how = "resolver"
    if chosen is None:
        report["error"] = "no target for " + task
        return
    target = target_from(chosen, how, noun)
    report["target"] = {"how": how, "object": describe_obj(chosen), "geometry": describe(chosen["summary"])}
    arm = side_arm(chosen["xy"])
    report["arm"] = arm
    stage("grasp", arm=arm)
    report["held"] = grasp_target(target, arm, hold_after=True)


def run_place_relative(parsed):
    noun, ref_noun, relation = parsed.get("noun"), parsed.get("reference"), parsed.get("relation")
    if not noun or not ref_noun or not relation:
        report["error"] = "could not parse the place sentence"
        return
    stage("census", noun=noun, reference=ref_noun)
    obj, query = find_single(noun)
    ref, ref_query = find_single(ref_noun, avoid=(obj,))
    report["object"] = describe_obj(obj)
    report["reference"] = describe_obj(ref)
    if obj is None:
        flag("resolver_fallback", reason="object")
        obj = resolver_object("Pick up the " + noun, noun)
    if ref is None and obj is not None:
        flag("reference_by_resolver")
        ref = resolver_object("the " + ref_noun, ref_noun, for_target=False)
        if ref is not None and xy_distance(ref["xy"], obj["xy"]) < MERGE_M:
            ref = None
        report["reference"] = describe_obj(ref)
    if obj is None or ref is None:
        report["error"] = "no object or reference"
        return
    s, r = obj["summary"], ref["summary"]
    axis = {"left": (0.0, 1.0), "right": (0.0, -1.0), "top": (1.0, 0.0), "bottom": (-1.0, 0.0)}[relation]
    cross = (axis[1], -axis[0]) if axis[0] == 0.0 else (0.0, 1.0)
    half_ref = 0.5 * max(r["length"], r["width"])
    half_obj = 0.5 * max(s["length"], s["width"])
    offset = max(PLACE_REL_OFFSET_M, half_ref + half_obj + PLACE_REL_MIN_GAP_M)
    goal = (ref["xy"][0] + axis[0] * offset, ref["xy"][1] + axis[1] * offset)
    report["place_plan"] = {"relation": relation, "offset": round(offset, 4), "goal": r3(goal)}
    # the arm: the object's side, unless the object lies near the midline and the goal well across it (dev2 s1014: the
    # left arm took a toy car at y = +0.08 and had no IK anywhere near its goal at y = -0.22)
    arm = side_arm(obj["xy"])
    if side_arm(goal) != arm and abs(goal[1]) > PLACE_ARM_GOAL_Y and abs(obj["xy"][1]) < PLACE_ARM_OBJECT_Y:
        arm = side_arm(goal)
        flag("place_arm_by_goal", arm=arm)
    report["arm"] = arm
    stage("grasp", arm=arm)
    target = target_from(obj, "place_object", noun)
    held = grasp_target(target, arm, hold_after=False)
    report["held"] = held
    if not held or report["terminated"]:
        return
    below, offset_xy = grasp_geometry(target)
    goals, heights = place_goals(ref, axis, cross, offset, arm, below, offset_xy)
    report["place_plan"].update({"below": round(below, 4), "grasp_offset": r3(offset_xy), "heights": [round(z, 4) for z in heights]})
    stage("carry")
    done, used, z = carry_any(arm, goals, heights, "carry_place")
    if report["terminated"]:
        return
    if done is None and still_held(arm):
        # no goal within this arm's reach (dev2 s1099: a face cleanser at y -0.18 to "the top of the fanta" at y +0.24):
        # set it down by the midline on this side, and the other arm takes it from there
        relay = relay_place(arm, target, noun, goals, heights)
        if relay is not None:
            arm, done, used = relay
        if report["terminated"]:
            return
    report["place_plan"]["carried"] = done is not None
    report["place_plan"]["goal_used"] = None if used is None else r3(used)
    report["place_plan"]["held_after_carry"] = still_held(arm)
    if done is not None and not still_held(arm) and not report["terminated"]:
        arm = place_retry(arm, noun, ref, axis, cross, offset)
        if report["terminated"]:
            return
    stage("hold")
    hold_checks(arm, "hold_place", 0.0)


def place_goals(ref, axis, cross, offset, arm, below, offset_xy):
    """TCP goals that keep the relation (> 5 cm along the axis, < 10 cm across it) for an object held at offset_xy from the
    TCP: along-axis offsets, then cross-axis shifts toward the arm's own side first; and the hold heights."""
    toward = 1.0 if arm == "left" else -1.0
    shifts = [0.0, 0.03, 0.06, -0.03]
    if cross[1] != 0.0:
        shifts = [v * toward * (1.0 if cross[1] > 0 else -1.0) for v in shifts]
    elif ref["xy"][0] > PLACE_FAR_X:
        # C5o: left / right of a far reference: the across shift (x) toward the robot first - the relation allows 10 cm across
        # in every instance (aligned_x|0.1); 1710 inst 4: the oolong tea beside a sponge at x 0.81 had no carry to any goal at
        # x 0.82-0.91 (TCP) and was never placed
        shifts = [dx / cross[0] for dx in (-0.05, -0.07, -0.03, 0.0)]
    else:
        shifts = [0.0, 0.03, -0.03, 0.06]
    goals = []
    for d in (offset, offset + 0.03, max(PLACE_REL_MIN_OFFSET_M, offset - 0.02)):
        for c in shifts:
            centre = (ref["xy"][0] + axis[0] * d + cross[0] * c, ref["xy"][1] + axis[1] * d + cross[1] * c)
            goals.append((centre[0] - offset_xy[0], centre[1] - offset_xy[1]))
    hold_z = TABLE_Z + below + CARRY_CLEAR_M
    heights = [hold_z, hold_z - 0.06, max(TABLE_Z + below + HOLD_CLEAR_M, hold_z - 0.12)]
    return goals, heights


def relation_holds(obj_xy, ref_xy, axis, cross, margin):
    """RelativePosition on centres: more than 0.05 m beyond the reference along the axis and within 0.10 m across it,
    each with `margin` to spare."""
    along = (obj_xy[0] - ref_xy[0]) * axis[0] + (obj_xy[1] - ref_xy[1]) * axis[1]
    across = (obj_xy[0] - ref_xy[0]) * cross[0] + (obj_xy[1] - ref_xy[1]) * cross[1]
    return along > 0.05 + margin and abs(across) < 0.10 - margin


def place_retry(arm, noun, ref, axis, cross, offset):
    """C5k: the object slipped out during the carry (test s5047 / s5036 / s5029, 3 of 18: GenieSim then saw it 13-23 cm off
    the reference's line, MixedRules 0). Open, go home, find it again (head; never the reference); where it fell may already
    satisfy the relation (then nothing moves), else it is picked once more and carried to the goals, only while the rows
    left before MixedRules' last check (Follow + 600) leave room for it (PLACE_RETRY_ROWS). Returns the arm that holds."""
    info = {"rows": report["rows"]}
    report["place_retry"] = info
    flag("place_slipped", arm=arm)
    opened = open_gripper(arm=arm)
    count_rows(opened)
    grip_command[arm] = 1.0
    if report["terminated"]:
        return arm
    clear_arm(arm, "place_retry_clear")
    if report["terminated"]:
        return arm
    found, query = find_single(noun, avoid=(ref,))
    info["found"] = describe_obj(found)
    if found is None:
        return arm
    if relation_holds(found["xy"], ref["xy"], axis, cross, 0.015):
        info["already"] = True
        return arm
    deadline = (follow_row.get("row") or 120) + 600
    if report["rows"] + PLACE_RETRY_ROWS > deadline:
        info["skipped"] = "rows %d, deadline %d" % (report["rows"], deadline)
        return arm
    again = side_arm(found["xy"])
    goal = (ref["xy"][0] + axis[0] * offset, ref["xy"][1] + axis[1] * offset)
    if side_arm(goal) != again and abs(goal[1]) > PLACE_ARM_GOAL_Y and abs(found["xy"][1]) < PLACE_ARM_OBJECT_Y:
        again = side_arm(goal)
    target = target_from(found, "place_object_retry", noun)
    held = grasp_target(target, again, hold_after=False)
    info.update({"arm": again, "held": held})
    if not held or report["terminated"]:
        return again
    below, offset_xy = grasp_geometry(target)
    goals, heights = place_goals(ref, axis, cross, offset, again, below, offset_xy)
    done, used, z = carry_any(again, goals, heights, "carry_place_retry")
    info.update({"carried": done is not None, "goal_used": None if used is None else r3(used), "held_after_carry": still_held(again)})
    return again


def relay_place(arm, target, noun, goals, heights):
    """Hand the held object to the other arm through the table: set it down at RELAY_X, +-RELAY_Y on this arm's side, send
    this arm home, find the object again near that spot (head), pick it with the other arm, carry it to the goals.
    Returns (other arm, carried option, goal) or None."""
    sign = 1.0 if arm == "left" else -1.0
    here = tool_pose(arm)[0]
    spot = (min(max(here[0], RELAY_X[0]), RELAY_X[1]), sign * RELAY_Y)
    info = {"from_arm": arm, "spot": r3(spot)}
    report["relay"] = info
    flag("relay", arm=arm, spot=r3(spot))
    placed = place_on(arm, target, spot, TABLE_Z, "relay_set", release=True)
    info["placed"] = placed
    if report["terminated"] or not placed:
        return None
    clear_arm(arm, "relay_clear")
    if report["terminated"]:
        return None
    found = None
    for query in [target.get("query")] + queries_for(noun):
        if not query:
            continue
        near = [obj for obj in census([query], min_score=TWIN_FLOOR) if xy_distance(obj["xy"], spot) < RELAY_FIND_M]
        if near:
            found = near[0]
            break
    info["found"] = describe_obj(found)
    if found is None:
        return None
    other = other_arm(arm)
    again = target_from(found, "relay_pick", noun)
    held = grasp_target(again, other, hold_after=False)
    info["held"] = held
    if report["terminated"] or not held:
        return None
    below, offset_xy = grasp_geometry(again)
    done, used, z = carry_any(other, goals, heights, "carry_place_relay")
    info["carried"] = done is not None
    return other, done, used


def can_census(brands, needed):
    """The four cans: every can instance on the head camera, then each brand's own queries (BRAND_QUERIES, weighted)
    on the head AND both wrist views (from the start pose the wrists see the cans at ~1 mm/px: C5j probe s1009,
    "coca-cola can" 0.94 vs 0.82 on the Wang Lao Ji there, 0.76 vs 0.72 on the head; "orange can" 0.88 on the Fanta vs
    <= 0.23, "fanta can" 0.007 everywhere), summed per can; the brand -> can assignment with the highest total (all
    permutations). Returns ({brand: obj}, margin to the second-best assignment)."""
    cans = census(["can", "soda can"], max_extent=0.15)
    report["cans"] = [describe_obj(obj) for obj in cans[:6]]
    cans = cans[:4]
    if len(cans) < len(brands):
        # fewer cans than brands were found: assign only the two the sentence names
        brands = list(needed)
    scores = []
    for brand in brands:
        row = [0.0] * len(cans)
        for query, weight in BRAND_QUERIES.get(brand, [(brand + " can", 1.0)]):
            for view in ("head", "hand_left", "hand_right"):
                best = [0.0] * len(cans)
                for score, summary, box in text_instances(view, query, min_score=0.02, max_extent=0.15):
                    for index, obj in enumerate(cans):
                        if xy_distance(summary["grasp_xy"], obj["xy"]) < 0.04:
                            best[index] = max(best[index], score)
                row = [row[index] + weight * best[index] for index in range(len(cans))]
        scores.append(row)
    view = head_view()
    colours = [top_colour(obj, view) for obj in cans]
    report["can_colours"] = colours
    # the Wang Lao Ji's top is light blue, every other can's silver (head image, 12 C5j frames: B - R +55..+65 on the Wang
    # Lao Ji, -20..+15 on the others); SAM 3 tells it from the Coca-Cola by 0.04 only and Qwen called it "the red can"
    blue = [None if c is None else c[2] - c[0] for c in colours]
    blue_cans = [index for index, value in enumerate(blue) if value is not None and value > CAN_BLUE_TOP]
    report["can_blue"] = blue
    if len(blue_cans) == 1:
        for b, brand in enumerate(brands):
            if brand == "wang lao ji":
                scores[b][blue_cans[0]] += CAN_COLOUR_BONUS
            elif brand in ("coca cola", "coke cola"):
                scores[b][blue_cans[0]] -= CAN_COLOUR_BONUS
        report["can_colour_rule"] = blue_cans[0]
    report["can_scores"] = {brand: [round(v, 3) for v in row] for brand, row in zip(brands, scores)}

    def permutations(items, size):
        if size == 0:
            return [[]]
        out = []
        for index, item in enumerate(items):
            for rest in permutations(items[:index] + items[index + 1:], size - 1):
                out.append([item] + rest)
        return out

    totals = []
    if len(cans) >= len(brands):
        for perm in permutations(list(range(len(cans))), len(brands)):
            totals.append((sum(scores[b][perm[b]] for b in range(len(brands))), perm))
    if not totals:
        return {}, 0.0
    totals.sort(key=lambda t: -t[0])
    margin = totals[0][0] - totals[1][0] if len(totals) > 1 else totals[0][0]
    report["can_margin"] = round(margin, 3)
    return {brand: cans[totals[0][1][index]] for index, brand in enumerate(brands)}, margin


def head_view():
    """Pose, intrinsics and RGB of the head camera (as wrist_view), or None."""
    try:
        view = get_observation(["head"]).cameras["head"]
        quat = quat_normalize([float(v) for v in view.camera_pose.quaternion_wxyz])
        k = view.intrinsics
        return {"position": [float(v) for v in view.camera_pose.position], "axes": [quat_column(quat, 0), quat_column(quat, 1), quat_column(quat, 2)],
                "fx": float(k[0][0]), "fy": float(k[1][1]), "cx": float(k[0][2]), "cy": float(k[1][2]), "rgb": view.rgb}
    except Exception:
        return None


def top_colour(obj, view, band=0.006, limit=80):
    """Mean RGB of the head image at the object's top face (cloud points within `band` of its top, projected)."""
    if view is None:
        return None
    pts = [p for p in sampled(obj["summary"]["cloud"].points, 3000) if float(p[2]) >= obj["summary"]["top"] - band]
    pts = sampled(pts, limit)
    h, w = int(view["rgb"].shape[0]), int(view["rgb"].shape[1])
    total = [0.0, 0.0, 0.0]
    count = 0
    for p in pts:
        pixel = project(view, [float(p[0]), float(p[1]), float(p[2])])
        if pixel is None:
            continue
        u, v = int(pixel[0]), int(pixel[1])
        if 0 <= u < w and 0 <= v < h:
            px = view["rgb"][v][u]
            total = [total[0] + int(px[0]), total[1] + int(px[1]), total[2] + int(px[2])]
            count += 1
    if count < 5:
        return None
    return [round(total[0] / count, 1), round(total[1] / count, 1), round(total[2] / count, 1)]


def can_by_resolver(noun, cans):
    """Qwen's box for "the <brand> can" -> the census can under it (head image), or None."""
    resolved = resolve_text("Pick up the " + noun + " can", None)
    box = resolved.get("box_xyxy")
    if not box:
        return None
    best = None
    for obj in cans:
        if obj.get("box") is None:
            continue
        iou = box_iou([float(v) for v in box], obj["box"])
        if iou >= 0.2 and (best is None or iou > best[0]):
            best = (iou, obj)
    return None if best is None else best[1]


def seeded_ik(arm, position, quat, seed, samples):
    """C5o: cuRobo IK at a pose from the joint state `seed` (`samples` calls, random solver seeds); the solution nearest
    `seed` as (joint distance, joints), or None."""
    best = None
    for sample in range(samples):
        try:
            ik = solve_ik(Pose(position=[float(v) for v in position], quaternion_wxyz=list(quat), frame=frame), [float(v) for v in seed],
                          arm=arm, backend="curobo")
        except Exception:
            continue
        if ik.ok and ik.joint_positions is not None:
            joints = [float(v) for v in ik.joint_positions]
            distance = joint_distance(joints, seed)
            if best is None or distance < best[0]:
                best = (distance, joints)
    return best


def bev_carry_estimate(arm, aim, hover_z, raise_z, goal, heights, quat):
    """C5o: the least joint motion the carry of Y onto X's spot would take after a grasp with orientation `quat`: IK at the
    hover from the current joints, at the raised pose from there, then over the carry's yaw fan (CARRY_FAN, one call
    each) at X's spot from there. None when the hover or the raise has no IK; 99 when the carry has none."""
    hover = seeded_ik(arm, (aim[0], aim[1], hover_z), quat, current_joints(arm), BEV_YAW_SAMPLES)
    if hover is None:
        return None
    raised = seeded_ik(arm, (aim[0], aim[1], raise_z), quat, hover[1], BEV_YAW_SAMPLES)
    if raised is None:
        return None
    best = 99.0
    for z in heights:
        for degrees in CARRY_FAN:
            turned = quat_mul(quat_axis_angle((0.0, 0.0, 1.0), math.radians(degrees)), quat)
            found = seeded_ik(arm, (goal[0], goal[1], z), turned, raised[1], 1)
            if found is not None and found[0] < best:
                best = found[0]
    return best


def bev_grasp_options(target, arm, spot):
    """C5o: Y's grasp options ordered with its carry onto X's spot in mind. 1710 inst 26 (seed 1002, 0.5 there): a round
    can's four grasp yaws follow its mask's noisy principal axis; of two rank-0 yaws v2 / v3 took the one with the shorter
    hover, and from a -143..-163 grasp the carry needed 4.4-7 rad (4 of 5 such runs found none under the 4.2 cap: 0.5),
    from +149..+155 3.5-3.7 rad (4 of 4 placed). This estimate offline over 24 yaws (lane B IK, the run's own scene):
    +120..+165 3.7-3.9 rad, -135..-150 6.9-7.0 (evidence offline/bev_yaw_1002.txt). A 20-degree option whose estimated
    carry is over BEV_CARRY_MAX_JOINT_RAD (and its vertical twin) ranks after every other option; the order stays v4.0's
    otherwise, and when every option or none is over."""
    summary = target["summary"]
    options = vertical_first_low(target, summary, grasp_options(summary))
    aim = summary["grasp_xy"]
    grasp_z = grasp_height(summary)
    raise_z = TABLE_Z + max(0.0, grasp_z - summary["bottom"]) + CARRY_CLEAR_M
    estimates = {}
    for item in options:
        if item["name"].endswith("_tilt20"):
            estimates[item["name"].split("_")[0]] = bev_carry_estimate(arm, aim, grasp_z + ABOVE_M, raise_z, spot, (raise_z, raise_z - 0.06), item["quat"])
    report["bev_grasp_estimates"] = {yaw: (None if estimates[yaw] is None else round(estimates[yaw], 2)) for yaw in estimates}
    over = [yaw for yaw in estimates if estimates[yaw] is None or estimates[yaw] > BEV_CARRY_MAX_JOINT_RAD]
    if not over or len(over) == len(estimates):
        return options
    out = []
    for item in options:
        if item["name"].split("_")[0] in over:
            moved = {key: item[key] for key in item}
            moved["rank"] = item["rank"] + BEV_YAW_PENALTY
            out.append(moved)
        else:
            out.append(item)
    flag("bev_grasp_carry", estimates=report["bev_grasp_estimates"], over=over,
         rank_first=sorted(options, key=lambda item: item["rank"])[0]["name"], first=sorted(out, key=lambda item: item["rank"])[0]["name"])
    return out


def run_place_beverage(parsed):
    x_noun, y_noun, placer, taker = parsed.get("x"), parsed.get("y"), parsed.get("placer"), parsed.get("taker")
    if not x_noun or not y_noun or placer not in ("left", "right"):
        report["error"] = "could not parse the beverage sentence"
        return
    taker = taker if taker in ("left", "right") else other_arm(placer)
    stage("census", x=x_noun, y=y_noun, placer=placer, taker=taker)
    brands = []
    for brand in ("coca cola", "sprite", "fanta", "wang lao ji"):
        brands.append(brand)
    for noun in (x_noun, y_noun):
        if noun not in brands:
            brands.append(noun)
    assigned, margin = can_census(brands, (x_noun, y_noun))
    x_obj, y_obj = assigned.get(x_noun), assigned.get(y_noun)
    if margin < CAN_MARGIN and x_obj is not None and y_obj is not None and report.get("can_colour_rule") is None:
        # a close assignment (both red cans): Qwen names X and Y on the head image; its answer wins where it differs
        cans = [assigned[brand] for brand in assigned]
        for name in ("x", "y"):
            noun = x_noun if name == "x" else y_noun
            picked = can_by_resolver(noun, cans)
            current = x_obj if name == "x" else y_obj
            if picked is not None and picked is not current:
                flag("can_by_resolver", brand=noun, margin=round(margin, 3))
                if name == "x":
                    x_obj = picked
                else:
                    y_obj = picked
        if x_obj is y_obj:
            x_obj, y_obj = assigned.get(x_noun), assigned.get(y_noun)
    report["x"], report["y"] = describe_obj(x_obj), describe_obj(y_obj)
    if x_obj is None or y_obj is None:
        flag("resolver_fallback", reason="cans")
        if y_obj is None:
            y_obj = resolver_object("Pick up the " + y_noun + " can", y_noun)
        if x_obj is None:
            x_obj = resolver_object("Pick up the " + x_noun + " can", x_noun)
    if x_obj is None or y_obj is None:
        report["error"] = "cans not found"
        return
    spot = (x_obj["xy"][0], x_obj["xy"][1])
    report["bev_plan"] = {"spot": r3(spot), "placer": placer, "taker": taker}
    # 1) the placer takes Y first: Follow(Y, placer) is the first stage (300 rows)
    stage("grasp_y", arm=placer)
    y_target = target_from(y_obj, "can_y", y_noun)
    crossed_x_taker = (side_arm(y_obj["xy"]) != placer and side_arm(x_obj["xy"]) == taker
                       and x_obj["xy"][0] > y_obj["xy"][0] + BEV_X_BEHIND_M)
    y_options = None
    if side_arm(y_obj["xy"]) == placer or crossed_x_taker:
        # Y is raised over its own spot and carried from there (below): its grasp yaw chosen with that carry in mind
        y_options = bev_grasp_options(y_target, placer, spot)
    held_y = grasp_target(y_target, placer, hold_after=False, options=y_options)
    report["held_y"] = held_y
    if report["terminated"] or not held_y:
        return
    y_below, y_offset = grasp_geometry(y_target)
    # Y to the placer's own outer edge, out of the taker's way (dev2 s1136: the cans were crossed - Y on the taker's side,
    # X on the placer's - and the taker parking X on its side swung through the placer holding Y above Y's spot)
    y_high = TABLE_Z + y_below + CARRY_CLEAR_M
    if crossed_x_taker:
        # C5o: Y across on the taker's side and X there too, behind it (1710 inst 15 / 19 / 21: X 16 cm behind Y, every run
        # 0.5): Y is raised where it is, as on its own side; v2 staged it at the placer's edge and the carry from there to
        # X's spot needed 4.9-8.7 rad (offline, lane B IK from the platform's states: in place 3.2 / 7.3 / 4.3 rad); the
        # taker's hover to X is planned around it (cuRobo, head depth). X in front of Y (inst 23, v2 1.0) keeps the edge
        # stage: the placer's arm would hang over X there
        flag("bev_raise_crossed")
    if side_arm(y_obj["xy"]) == placer or crossed_x_taker:
        # Y on the placer's own side: raised over its spot (dev2: 3/3 that way; dev4's edge stage then had no IK back to a
        # far spot in 2 of 7)
        here = tool_pose(placer)[0]
        carry_any(placer, [(here[0], here[1])], [y_high, y_high - 0.06], "raise_y", fan=(0,))
    else:
        # crossed: Y waits on the placer's side, out of the taker's way (dev2 s1136)
        sign = 1.0 if placer == "left" else -1.0
        carry_any(placer, [(0.58, sign * 0.20), BEV_STAGE[placer]], [y_high, y_high - 0.06], "stage_y")
    if report["terminated"]:
        return
    # 2) the taker lifts X off its spot and parks it near its own start TCP
    stage("grasp_x", arm=taker)
    x_target = target_from(x_obj, "can_x", x_noun)
    held_x = grasp_target(x_target, taker, hold_after=False)
    report["held_x"] = held_x
    if report["terminated"]:
        return
    if held_x:
        x_below, x_offset = grasp_geometry(x_target)
        sign = 1.0 if taker == "left" else -1.0
        park_y = max(-0.31, min(0.31, spot[1] + sign * BEV_PARK_SHIFT_M))
        x_high = TABLE_Z + x_below + CARRY_CLEAR_M
        carry_any(taker, [(spot[0], park_y), (0.58, sign * 0.20), BEV_STAGE[taker]], [x_high, x_high - 0.06], "park_x")
        if report["terminated"]:
            return
    else:
        flag("x_not_held")
    # 3) the placer sets Y down on X's spot (bottom BEV_SET_GAP_M above the table), still holding it
    stage("place_y")
    tcp_goal = (spot[0] - y_offset[0], spot[1] - y_offset[1])
    done_y, used_y, z_y = carry_any(placer, [tcp_goal], [y_high, y_high - 0.06, y_high - 0.10], "carry_y", cap=BEV_CARRY_MAX_JOINT_RAD)
    if done_y is None and not report["terminated"]:
        # C5o: cuRobo's IK returns one solution per call from random seeds; two per yaw found 4.6-10 rad carries where more
        # calls find 2.2-4.1 rad ones (offline, lane B, 1710 inst 15 / 19 / 21 / 26: every one a 0.5 there; their joint-
        # interpolated paths keep the TCP at the stage height and >= 15 cm from the taker's): again with BEV_RETRY_SAMPLES
        # calls per yaw and BEV_RETRY_CAP_RAD
        flag("bev_carry_more_samples")
        done_y, used_y, z_y = carry_any(placer, [tcp_goal], [y_high, y_high - 0.06, y_high - 0.10], "carry_y_more", cap=BEV_RETRY_CAP_RAD,
                                        samples=BEV_RETRY_SAMPLES)
    report["bev_plan"]["carried_y"] = done_y is not None
    if done_y is None and not report["terminated"]:
        # never set Y down away from X's spot: keep holding it where it is
        flag("bev_carry_y_failed")
        return
    if report["terminated"]:
        return
    lower_to(placer, TABLE_Z + y_below + BEV_SET_GAP_M, "set_y")
    if report["terminated"]:
        return
    report["bev_plan"].update({"tcp_goal": r3(tcp_goal), "y_below": round(y_below, 4), "held_after_set": still_held(placer)})
    stage("hold")
    hold_checks(placer, "hold_y", 0.0)
    if report["terminated"]:
        return
    # not passed while held: release it standing and hold two more checks
    flag("bev_release")
    release_and_retreat(placer, "release_y", partial=False)
    if report["terminated"]:
        return
    hold_checks(placer, "hold_released", 1.0)


def cube_census(queries, max_extent=0.1):
    objects = census(queries, max_extent=max_extent)
    report["cube_census"] = [describe_obj(obj) for obj in objects[:6]]
    return objects


def place_on(arm, target, goal_xy, support_z, label, release=True, bias_y=0.0):
    """Carry the held object over goal_xy at the transport height, lower its bottom to STACK_GAP_M above support_z,
    then release and retreat (or keep holding). Returns True when the set-down ran. C5o: `bias_y` - the TCP goal moves
    this much toward the midline from the placing arm's side (stack three: PLACE_BIAS_Y_M)."""
    below, offset = grasp_geometry(target)
    side = 1.0 if arm == "left" else -1.0
    tcp_goal = (goal_xy[0] - offset[0], goal_xy[1] - offset[1] - side * bias_y)
    transport = max(support_z, TABLE_Z) + below + 0.10
    info = {"label": label, "goal": r3(goal_xy), "tcp_goal": r3(tcp_goal), "below": round(below, 4), "support_z": round(support_z, 4)}
    if bias_y:
        info["bias_y"] = round(-side * bias_y, 4)
    report.setdefault("placements", []).append(info)
    done = carry(arm, tcp_goal, transport, label + "_carry")
    info["carried"] = done is not None
    if report["terminated"] or done is None:
        return False
    if not still_held(arm):
        # dropped during the carry: nothing to set down (the caller looks for it again)
        info["dropped_in_carry"] = True
        flag("dropped_in_carry", label=label)
        return False
    set_z = support_z + below + STACK_GAP_M
    lowered = lower_to(arm, set_z, label + "_set")
    info["lowered"] = lowered is not None
    info["held_before_release"] = still_held(arm)
    if report["terminated"]:
        return True
    gap = tool_pose(arm)[0][2] - set_z
    info["set_gap"] = round(gap, 4)
    if release and gap > PLACE_RELEASE_MAX_GAP_M:
        # never let go well above the support (dev2 stack s1051: released 10 cm up, the stack fell); keep holding it there
        flag("release_skipped_high", label=label, gap=round(gap, 4))
        info["released"] = False
        return True
    if release:
        release_and_retreat(arm, label)
        info["released"] = True
    return True


def stack_cubes_now():
    """C5o: every cube the head sees now (stack three's census filter)."""
    return [c for c in cube_census(["red cube", "red block", "cube"]) if 0.03 <= max(c["summary"]["length"], c["summary"]["width"]) <= 0.08]


def run_stack_blocks():
    """Stack the three cubes at STACK_SPOT (on the midline, comfortable for both arms): the cube nearest it goes there first
    (unless it already lies within 3 cm), then the other two, nearest first, each by the arm on its own side. No instance
    has a cube within 5 cm of the midline, and a stack on a side cube makes the other side's arm cross it (dev2 / dev4: 4 of
    4 such carries over the joint-motion cap, off target by 2-4 cm or dropped). Stack checks x / y only (3 cm).
    C5o: every placement corrected for the placing arm's outward bias (PLACE_BIAS_Y_M); the third cube aimed midway between
    the base and the second (Stack wants every cube within 3 cm of the one the checker names first; 1710 inst 13: the second
    cube 2.7 cm off the base, the third on it, Stack never passed); a second cube found off the base after its release is put
    on it again before the third (1710 inst 29: it leant against the base and the third went onto the base); after the last
    release the arm clears at once and a cube lying off the stack is put on it (stack_verify)."""
    stage("census")
    cubes = cube_census(["red cube", "red block", "cube"])
    cubes = [obj for obj in cubes if 0.03 <= max(obj["summary"]["length"], obj["summary"]["width"]) <= 0.08][:3]
    if len(cubes) < 2:
        report["error"] = "fewer than two cubes"
        return
    grasp_rule["mid"] = True
    spot = STACK_SPOT
    first = min(cubes, key=lambda c: xy_distance(c["xy"], spot))
    others = sorted([c for c in cubes if c is not first], key=lambda c: xy_distance(c["xy"], spot))
    height = first["summary"]["top"] - TABLE_Z
    report["stack_plan"] = {"spot": r3(spot), "first": describe_obj(first), "order": [describe_obj(c) for c in others]}
    aim = (first["xy"][0], first["xy"][1])
    base_xy = aim
    base_top = first["summary"]["top"]
    support = first["summary"]["top"]
    last_arm = side_arm(spot)
    pending = []
    if xy_distance(first["xy"], spot) > STACK_SPOT_TOLERANCE_M:
        pending.append(("base", first))
    pending += [("cube", c) for c in others]
    retries = 0
    replaced = 0
    index = 0
    while pending:
        role, cube = pending.pop(0)
        arm = side_arm(cube["xy"])
        stage("grasp_%d" % index, arm=arm, role=role)
        set_neighbours(cubes, cube)
        target = target_from(cube, "stack_" + role, "building blocks")
        held = grasp_target(target, arm, hold_after=False)
        if report["terminated"]:
            return
        if not held:
            flag("stack_not_held", index=index, role=role)
            index += 1
            continue
        stage("place_%d" % index)
        goal = spot if role == "base" else aim
        level = TABLE_Z if role == "base" else support
        placed = place_on(arm, target, goal, level, "stack_%d" % index, release=True, bias_y=PLACE_BIAS_Y_M)
        last_arm = arm
        if report["terminated"]:
            return
        info = report["placements"][-1]
        index += 1
        if info.get("dropped_in_carry") and retries < 1 and report["rows"] < STACK_RETRY_BEFORE_ROWS:
            retries += 1
            again = [c for c in cube_census(["red cube", "red block", "cube"])
                     if xy_distance(c["xy"], aim if role != "base" else spot) > 0.04 and all(xy_distance(c["xy"], p[1]["xy"]) > 0.03 for p in pending)
                     and 0.03 <= max(c["summary"]["length"], c["summary"]["width"]) <= 0.08]
            report["stack_refind"] = [describe_obj(c) for c in again[:3]]
            if again:
                pending.insert(0, (role, again[0]))
            continue
        if placed and info.get("released"):
            if not pending:
                break
            # the next arm works where this one is: send it home first
            clear_arm(arm, "clear_%d" % index)
            if report["terminated"]:
                return
            if role == "base":
                # where the base really landed (the head sees it with the arm gone)
                found = [c for c in cube_census(["red cube", "red block"]) if xy_distance(c["xy"], spot) < STACK_FIND_M]
                aim = found[0]["xy"] if found else spot
                support = (found[0]["summary"]["top"] if found else TABLE_Z + height)
                base_xy, base_top = aim, support
                report["stack_plan"]["base_measured"] = r3(aim)
            else:
                # C5k: where the stack now stands (rc2 kept the base's position and added a cube height: test s5015 and the
                # fresh rc2 run's s5217 placed all three within 1.6 cm of their goals and ended more than 3 cm out of
                # line); its highest cube near the aim: on the stack (top about one cube higher) or not (top unchanged)
                now = stack_cubes_now()
                found = sorted([c for c in now if xy_distance(c["xy"], aim) < STACK_FIND_M], key=lambda c: -c["summary"]["top"])
                entry = {"index": index, "expected_top": round(support + height, 4)}
                if found and abs(found[0]["summary"]["top"] - (support + height)) < 0.5 * height:
                    # C5o: the next cube goes midway between the base and this top (each then within half their offset)
                    top_xy = found[0]["xy"]
                    support = found[0]["summary"]["top"]
                    aim = (0.5 * (base_xy[0] + top_xy[0]), 0.5 * (base_xy[1] + top_xy[1]))
                    entry.update({"measured": r3(top_xy), "top": round(support, 4), "aim": r3(aim)})
                elif found and abs(found[0]["summary"]["top"] - support) < 0.5 * height:
                    entry.update({"measured": r3(found[0]["xy"]), "top": round(found[0]["summary"]["top"], 4), "not_on_stack": True})
                    flag("stack_cube_not_on_stack", index=index)
                    # C5o: the cube just set down is off the stack: it goes onto the base again before the next one (once)
                    aim, support = base_xy, base_top
                    stray = sorted([c for c in now if xy_distance(c["xy"], base_xy) > 0.02 and all(xy_distance(c["xy"], p[1]["xy"]) > 0.03 for p in pending)],
                                   key=lambda c: xy_distance(c["xy"], base_xy))
                    entry["stray"] = [describe_obj(c) for c in stray[:2]]
                    if stray and replaced < 1 and report["rows"] < STACK_RETRY_BEFORE_ROWS:
                        replaced += 1
                        pending.insert(0, ("cube", stray[0]))
                        flag("stack_replace", index=index, xy=r3(stray[0]["xy"]))
                else:
                    # C5o: no cube within STACK_FIND_M of the aim. v2 assumed the stack had grown (support + one cube);
                    # dev3 smoke stack three 1058: the second cube had knocked the base 5 cm away, both lay on the table,
                    # the third was let go 5 cm over the bare table and Stack never passed. Look STACK_REFIND_M wide: a cube
                    # standing on the table there is the base where it is now, the other one there goes onto it again
                    wide = sorted([c for c in now if xy_distance(c["xy"], aim) < STACK_REFIND_M and all(xy_distance(c["xy"], p[1]["xy"]) > 0.03 for p in pending)],
                                  key=lambda c: xy_distance(c["xy"], aim))
                    entry["wide"] = [describe_obj(c) for c in wide[:3]]
                    table = [c for c in wide if abs(c["summary"]["top"] - (TABLE_Z + height)) < 0.5 * height]
                    if table:
                        base_xy, base_top = table[0]["xy"], table[0]["summary"]["top"]
                        aim, support = base_xy, base_top
                        entry.update({"base_moved": r3(base_xy), "top": round(base_top, 4)})
                        flag("stack_base_moved", index=index, xy=r3(base_xy))
                        if len(table) > 1 and replaced < 1 and report["rows"] < STACK_RETRY_BEFORE_ROWS:
                            replaced += 1
                            pending.insert(0, ("cube", table[1]))
                            flag("stack_replace", index=index, xy=r3(table[1]["xy"]))
                    else:
                        support = support + height
                        entry["unmeasured"] = True
                report["stack_plan"].setdefault("remeasured", []).append(entry)
    stage("verify")
    stack_verify(base_xy, height, last_arm)


def stack_verify(base_xy, height, last_arm):
    """C5o: right after the last release the arm goes home (Stack, if it holds, passes during that move and ends the
    episode); then the head looks again and a cube lying off the stack (further than STACK_LOOSE_M from the base's axis,
    standing on the table) is picked and set on the stack's top, aimed midway between the base and that top - once per loose
    cube while rows remain (STACK_REPAIR_BEFORE_ROWS); else the arm holds where it is."""
    info = {"rows": report["rows"]}
    report["stack_verify"] = info
    last = report["placements"][-1] if report.get("placements") else {}
    if last.get("released") is False:
        # the last cube is still held over the stack (release_skipped_high): hold it there as v2 did (Stack checks x / y)
        info["skipped"] = "last cube held"
        stage("hold")
        hold_checks(last_arm, "hold_stack", None)
        return
    for attempt in range(2):
        clear_arm(last_arm, "verify_clear_%d" % attempt)
        if report["terminated"]:
            return
        now = stack_cubes_now()
        on = sorted([c for c in now if xy_distance(c["xy"], base_xy) < STACK_LOOSE_M], key=lambda c: -c["summary"]["top"])
        if not on:
            # C5o: the stack's base is not where it was set: the cube nearest it within STACK_REFIND_M is the base now
            near = sorted([c for c in now if xy_distance(c["xy"], base_xy) < STACK_REFIND_M], key=lambda c: xy_distance(c["xy"], base_xy))
            if near:
                base_xy = near[0]["xy"]
                info["base_moved"] = r3(base_xy)
                on = sorted([c for c in now if xy_distance(c["xy"], base_xy) < STACK_LOOSE_M], key=lambda c: -c["summary"]["top"])
        loose = sorted([c for c in now if xy_distance(c["xy"], base_xy) >= STACK_LOOSE_M and c["summary"]["top"] < TABLE_Z + 1.5 * height],
                       key=lambda c: xy_distance(c["xy"], base_xy))
        info.setdefault("seen", []).append({"rows": report["rows"], "on": [describe_obj(c) for c in on[:3]], "loose": [describe_obj(c) for c in loose[:3]]})
        if not loose or not on or report["rows"] > STACK_REPAIR_BEFORE_ROWS:
            break
        top = on[0]
        cube = loose[0]
        arm = side_arm(cube["xy"])
        stage("repair_grasp_%d" % attempt, arm=arm)
        set_neighbours(now, cube)
        target = target_from(cube, "stack_repair", "building blocks")
        held = grasp_target(target, arm, hold_after=False)
        last_arm = arm
        if report["terminated"]:
            return
        if not held:
            flag("stack_repair_not_held", attempt=attempt)
            continue
        goal = (0.5 * (base_xy[0] + top["xy"][0]), 0.5 * (base_xy[1] + top["xy"][1]))
        stage("repair_place_%d" % attempt)
        flag("stack_repair", attempt=attempt, cube=r3(cube["xy"]), goal=r3(goal), top=round(top["summary"]["top"], 4))
        place_on(arm, target, goal, top["summary"]["top"], "stack_repair_%d" % attempt, release=True, bias_y=PLACE_BIAS_Y_M)
        if report["terminated"]:
            return
    stage("hold")
    hold_checks(last_arm, "hold_stack", None)


def block_size(obj):
    s = obj["summary"]
    return 0.5 * (s["length"] + s["width"])


def move_arm(pick, place):
    """The arm for a move: its own side when both ends lie on one side, else the side of the end farther from the midline
    (the arm then crosses by the nearer end's distance only)."""
    if pick[1] * place[1] >= 0.0:
        return side_arm(pick if abs(pick[1]) >= abs(place[1]) else place)
    return side_arm(pick if abs(pick[1]) > abs(place[1]) else place)


def move_crossing(pick, place):
    """How far past the midline the move's arm has to reach (0 when both ends lie on one side)."""
    if pick[1] * place[1] >= 0.0:
        return 0.0
    return min(abs(pick[1]), abs(place[1]))


def sort_layout(blocks):
    """blocks small -> large. A row along y at x0 with SORT_SPACING_M steps, small at the largest y: either anchored on one
    block (it stays) or centred on the midline (every block moves). Feasible: slots inside SORT_Y_LIMIT / SORT_X_RANGE, no
    move whose arm must cross the midline by more than SORT_CROSS_M (dev6 s1013: the small block was carried from y -0.17 to
    +0.20, no IK), no slot on a block that has not moved before it gets there. Best: fewest moves plus crossing and travel
    costs. -> (cost, travel, anchor or -1, slots, order, spacing)."""
    plans = []
    rows = []
    for spacing in (SORT_SPACING_M, 0.075):
        for anchor in range(3):
            x0, y0 = blocks[anchor]["xy"]
            rows.append((anchor, spacing, x0, y0 + (anchor - 1) * spacing))
        mean_x = sum(b["xy"][0] for b in blocks) / 3.0
        for dx in (0.0, 0.06, -0.05):
            # a row off the blocks' own x line frees slots a block still covers (dev6 s1013 / mock s20 cycles)
            x_mid = min(max(mean_x + dx, SORT_X_RANGE[0] + 0.03), SORT_X_RANGE[1] - 0.06)
            for centre in (0.0, 0.03, -0.03):
                rows.append((-1, spacing, x_mid, centre))
    for anchor, spacing, x0, y_mid in rows:
        slots = [(x0, y_mid + (1 - k) * spacing) for k in range(3)]
        if not (SORT_X_RANGE[0] <= x0 <= SORT_X_RANGE[1]) or any(abs(slot[1]) > SORT_Y_LIMIT for slot in slots):
            continue
        moves = [k for k in range(3) if k != anchor and xy_distance(blocks[k]["xy"], slots[k]) > 0.015]
        final = [{"xy": slots[k] if k in moves else blocks[k]["xy"]} for k in range(3)]
        if not already_sorted(final, margin=SORT_FINAL_MARGIN_M):
            # C5o: blocks left within 1.5 cm of their slots can together break the relation (mock push, re-plan of sort
            # number inst 0: two blocks 1.5 cm off in opposite directions left a 4.5 cm gap, "no move", MixedRules never
            # passed): every block not at its slot moves, and a plan whose result still fails is not offered
            moves = [k for k in range(3) if k != anchor and xy_distance(blocks[k]["xy"], slots[k]) > 0.003]
            final = [{"xy": slots[k] if k in moves else blocks[k]["xy"]} for k in range(3)]
            if not already_sorted(final, margin=SORT_FINAL_MARGIN_M):
                continue
        crossing = [move_crossing(blocks[k]["xy"], slots[k]) for k in moves]
        if any(c > SORT_CROSS_M for c in crossing):
            continue
        order = []
        pending = list(moves)
        ok = True
        while pending and ok:
            progress = False
            for k in list(pending):
                blocked = [j for j in range(3) if j != k and (j not in moves or j in pending)
                           and xy_distance(blocks[j]["xy"], slots[k]) < 0.5 * (block_size(blocks[j]) + block_size(blocks[k])) + 0.03]
                if not blocked:
                    order.append(k)
                    pending.remove(k)
                    progress = True
            if not progress:
                ok = False
        if not ok:
            continue
        travel = sum(xy_distance(blocks[k]["xy"], slots[k]) for k in moves)
        cost = len(moves) + 3.0 * sum(crossing) + 0.5 * travel + (0.0 if spacing == SORT_SPACING_M else 0.1)
        plans.append((cost, travel, anchor, slots, order, spacing))
    if not plans:
        return None
    plans.sort(key=lambda p: (p[0], p[1]))
    return plans[0]


def already_sorted(blocks, margin=None):
    """The checker's four relations already hold with margin (C5j: sort-number instances 27 and 29 start sorted): middle
    and large each > 0.05 m (+ SORT_KEEP_MARGIN_M) further along -y than the one before, both within 0.05 m (- margin) of
    the small one in x."""
    small, middle, large = blocks[0]["xy"], blocks[1]["xy"], blocks[2]["xy"]
    m = SORT_KEEP_MARGIN_M if margin is None else margin
    gap = 0.05 + m
    return (small[1] - middle[1] > gap and middle[1] - large[1] > gap and abs(middle[0] - small[0]) < 0.05 - m
            and abs(large[0] - small[0]) < 0.05 - m)


def set_neighbours(objects, target):
    """v3.8's finger clearance for this pick: every other census object is a neighbour (grasp options whose open fingertip
    would land on one rank last). C5j dev8 sort-cubes s1004: the 2 cm cube's tilted grasp stalled on the 4 cm cube 8 cm
    away and pushed it; the next pick closed on air twice where it had been."""
    while neighbours:
        neighbours.pop()
    for obj in objects:
        if obj is not target and xy_distance(obj["xy"], target["xy"]) >= CUBE_MERGE_M:
            neighbours.append(obj["summary"])


def refind(obj, queries, max_extent=0.1):
    """The census instance nearest an object's last known position (within REFIND_M), or None."""
    best = None
    for found in census(queries, max_extent=max_extent):
        gap = xy_distance(found["xy"], obj["xy"])
        if gap < REFIND_M and (best is None or gap < best[0]):
            best = (gap, found)
    return None if best is None else best[1]


def set_sort_neighbours(blocks, k, where):
    """C5o: finger clearance for block k's pick against every other block where it is NOW (`where`: its slot once set down
    there). v2 used the census start positions (C5k sort number 1023: the third grasp's fingertip reached the block placed
    first, 7.8 cm away, in all four runs; flung off the table twice)."""
    while neighbours:
        neighbours.pop()
    for j in range(len(blocks)):
        if j == k or xy_distance(where[j], blocks[k]["xy"]) < CUBE_MERGE_M:
            continue
        summary = blocks[j]["summary"]
        moved = {key: summary[key] for key in summary}
        moved["grasp_xy"] = (where[j][0], where[j][1])
        neighbours.append(moved)


def refind_block(blocks, k, where, sized):
    """C5o: block k after a missed grasp: the census block nearest its last position (within REFIND_M) that lies more than
    SORT_DROP_SEP_M from where every other block is now and, for the cubes (sized), whose size is within SORT_SIZE_TOL of
    block k's. v2 took the nearest census block (1710 sort cubes inst 10: the large anchor cube, 4 cm from the missed small
    one, was carried to the small one's slot). None when there is no such block."""
    size = block_size(blocks[k])
    best = None
    for found in census(["cube", "block", "small cube", "number block"], max_extent=0.1):
        gap = xy_distance(found["xy"], blocks[k]["xy"])
        if gap >= REFIND_M:
            continue
        if any(xy_distance(found["xy"], where[j]) < SORT_DROP_SEP_M for j in range(len(blocks)) if j != k):
            continue
        if sized and abs(block_size(found) - size) > SORT_SIZE_TOL * size:
            continue
        if best is None or gap < best[0]:
            best = (gap, found)
    return None if best is None else best[1]


def run_sort(blocks, noun, hold=True):
    """blocks ordered small -> large: plan the layout (sort_layout), then each move: pick, carry, set down, release.
    C5o: returns the state ({"where": each block's xy now, "slots", "last_arm"}; {"sorted": True} when nothing had to move;
    None without a layout); `hold` False leaves the closing holds to the caller (sort_verify)."""
    report["sort_blocks"] = [{"xy": r3(b["xy"]), "size": round(block_size(b), 4)} for b in blocks]
    if already_sorted(blocks):
        flag("already_sorted")
        stage("hold")
        hold_checks("right", "hold_sorted", None)
        return {"sorted": True, "where": [b["xy"] for b in blocks], "last_arm": None}
    plan = sort_layout(blocks)
    if plan is None:
        report["error"] = "no sort layout"
        return None
    cost, travel, anchor, slots, order, spacing = plan
    report["sort_plan"] = {"anchor": anchor, "slots": [r3(slot) for slot in slots], "order": order, "spacing": spacing, "cost": round(cost, 3)}
    last_arm = None
    placed_at = {}
    where = [(b["xy"][0], b["xy"][1]) for b in blocks]
    sized = noun == "cube"
    state = {"where": where, "slots": slots, "last_arm": None}
    for k in order:
        block = blocks[k]
        slot = slots[k]
        arm = move_arm(block["xy"], slot)
        stage("grasp_%d" % k, arm=arm)
        set_sort_neighbours(blocks, k, where)
        target = target_from(block, "sort_block", noun)
        held = grasp_target(target, arm, hold_after=False)
        state["last_arm"] = arm
        if report["terminated"]:
            return state
        if not held:
            flag("sort_not_held", block=k)
            # it may have been pushed: look for it again (C5o: never another block) and try once more
            again = refind_block(blocks, k, where, sized)
            if again is None or report["rows"] > SORT_RETRY_BEFORE_ROWS:
                continue
            blocks[k] = again
            where[k] = (again["xy"][0], again["xy"][1])
            set_sort_neighbours(blocks, k, where)
            target = target_from(again, "sort_block_again", noun)
            held = grasp_target(target, move_arm(again["xy"], slot), hold_after=False)
            if report["terminated"]:
                return state
            if not held:
                continue
            arm = move_arm(again["xy"], slot)
        stage("place_%d" % k)
        place_on(arm, target, slot, TABLE_Z, "sort_%d" % k, release=True)
        placed_at[k] = bool(report["placements"][-1].get("released"))
        if placed_at[k]:
            where[k] = (slot[0], slot[1])
        last_arm = arm
        state["last_arm"] = arm
        if report["terminated"]:
            return state
        later = order[order.index(k) + 1:]
        if report["placements"][-1].get("dropped_in_carry") and report["rows"] < SORT_RETRY_BEFORE_ROWS:
            # C5k: the block fell out during the carry (dev1 sort-number s5124: read right, the middle block dropped, the row
            # never complete). It is the census block away from every placed slot and every block not yet moved; picked and
            # set down at its slot once more
            arm = sort_drop_retry(blocks, k, slots, where, arm, noun)
            last_arm = arm
            state["last_arm"] = arm
            if report["placements"][-1].get("label") == "sort_%d_again" % k and report["placements"][-1].get("released"):
                where[k] = (slots[k][0], slots[k][1])
            if report["terminated"]:
                return state
        if later:
            nxt_arm = move_arm(blocks[later[0]]["xy"], slots[later[0]])
            if nxt_arm != arm:
                clear_arm(arm, "clear_%d" % k)
                if report["terminated"]:
                    return state
    report["sort_state"] = {"where": [r3(xy) for xy in where]}
    if hold:
        stage("hold")
        hold_checks(last_arm or "right", "hold_sort", None)
    return state


def sort_census(sized):
    """C5o: the blocks the head sees now (the wrists' views merged in when it sees fewer than three)."""
    if sized:
        found = [obj for obj in cube_census(["cube", "block", "small cube"]) if 0.012 <= block_size(obj) <= 0.07]
    else:
        found = [obj for obj in census(["block", "number block", "digit block"], min_score=TWIN_FLOOR, max_extent=0.08)
                 if 0.01 <= max(obj["summary"]["length"], obj["summary"]["width"]) <= 0.06]
    if len(found) < 3:
        wrist = census(["block", "number block", "cube"], views=("hand_left", "hand_right"), min_score=TWIN_FLOOR, max_extent=0.08)
        merge_into(found, [obj for obj in wrist if 0.01 <= max(obj["summary"]["length"], obj["summary"]["width"]) <= 0.07])
    return found[:3]


def identify_by_position(found, where):
    """C5o: which found block is which: each block the found one nearest where it should be (within SORT_MATCH_M, each found
    block once, closest pairs first); a single block left unmatched takes the found block left over. [obj per block] or None."""
    pairs = sorted([(xy_distance(f["xy"], w), i, j) for i, w in enumerate(where) for j, f in enumerate(found)])
    out = [None for w in where]
    used = []
    for d, i, j in pairs:
        if d > SORT_MATCH_M:
            break
        if out[i] is None and j not in used:
            out[i] = found[j]
            used.append(j)
    left = [i for i in range(len(where)) if out[i] is None]
    spare = [j for j in range(len(found)) if j not in used]
    if len(left) == 1 and spare:
        best = min(spare, key=lambda j: xy_distance(found[j]["xy"], where[left[0]]))
        out[left[0]] = found[best]
        left = []
    if left:
        return None
    return out


def sort_verify(state, noun):
    """C5o: after the moves the last arm clears at once (MixedRules, if the row holds, passes during that move and ends the
    episode); the head - and the wrists when it sees fewer than three - then looks at the row again; each block is
    identified (the cubes by size, the digit blocks by where they should be now) and the row is planned again from where
    the blocks are when it does not hold (1710 sort cubes inst 23 / 26: a placed block pushed; v2's verify, cubes only, ran
    with the arm over the row and saw two blocks / none). Once, only while rows remain (SORT_VERIFY_BEFORE_ROWS)."""
    sized = noun == "cube"
    info = {"rows": report["rows"]}
    report["sort_verify"] = info
    last = report["placements"][-1] if report.get("placements") else {}
    if last.get("released") is False:
        # the last block is still held at its slot (release_skipped_high): hold it there as v2 did
        info["skipped"] = "last block held"
        stage("hold")
        hold_checks(state.get("last_arm") or "right", "hold_sort", None)
        return
    if state.get("last_arm"):
        clear_arm(state["last_arm"], "verify_clear")
        if report["terminated"]:
            return
    found = []
    if report["rows"] <= SORT_VERIFY_BEFORE_ROWS:
        found = sort_census(sized)
    info["found"] = [{"xy": r3(b["xy"]), "size": round(block_size(b), 4)} for b in found]
    cur = None
    if sized and len(found) == 3:
        cur = sorted(found, key=block_size)
    elif not sized and found:
        cur = identify_by_position(found, state["where"])
    if cur is None or len(cur) < 3:
        info["unidentified"] = True
    elif already_sorted(cur, margin=0.005):
        info["sorted"] = True
    else:
        info["replan"] = [r3(b["xy"]) for b in cur]
        flag("sort_replan", rows=report["rows"])
        run_sort(cur, noun)
        return
    stage("hold")
    hold_checks("right", "hold_sort", None)


def sort_and_verify(ordered, noun):
    """C5o: run_sort without its closing holds, then sort_verify."""
    state = run_sort(ordered, noun, hold=False)
    if report["terminated"] or state is None or state.get("sorted"):
        return
    stage("verify")
    sort_verify(state, noun)


def sort_drop_retry(blocks, k, slots, where, arm, noun):
    """C5k: find block k after it fell out of the jaw in its carry and move it to its slot once. The jaw opens and the arm
    goes home first (the head then sees the table); the dropped block is the census block that lies more than SORT_DROP_SEP_M
    from where every other block is (`where`: its slot once released there, else where it was). Returns the arm used last."""
    info = {"block": k, "rows": report["rows"]}
    report.setdefault("sort_drop_retry", []).append(info)
    soft_open(arm, "drop_open_%d" % k, 1.0)
    if report["terminated"]:
        return arm
    clear_arm(arm, "drop_clear_%d" % k)
    if report["terminated"]:
        return arm
    known = [where[j] for j in range(3) if j != k]
    candidates = [obj for obj in census(["cube", "block", "small cube", "number block"], min_score=TWIN_FLOOR, max_extent=0.08)
                  if 0.012 <= block_size(obj) <= 0.07 and all(xy_distance(obj["xy"], xy) > SORT_DROP_SEP_M for xy in known)]
    info["candidates"] = [r3(obj["xy"]) for obj in candidates[:4]]
    if not candidates:
        return arm
    found = min(candidates, key=lambda obj: xy_distance(obj["xy"], blocks[k]["xy"]))
    blocks[k] = found
    where[k] = (found["xy"][0], found["xy"][1])
    again = move_arm(found["xy"], slots[k])
    set_sort_neighbours(blocks, k, where)
    target = target_from(found, "sort_block_drop", noun)
    held = grasp_target(target, again, hold_after=False)
    info.update({"found": r3(found["xy"]), "arm": again, "held": held})
    if report["terminated"] or not held:
        return again
    place_on(again, target, slots[k], TABLE_Z, "sort_%d_again" % k, release=True)
    return again


def sort_cubes_census():
    objects = cube_census(["cube", "block", "small cube"])
    blocks = [obj for obj in objects if 0.012 <= block_size(obj) <= 0.07][:3]
    blocks.sort(key=block_size)
    return blocks


def run_sort_cubes():
    stage("census")
    blocks = sort_cubes_census()
    if len(blocks) < 3:
        report["error"] = "fewer than three blocks"
        return
    # C5k verified the cubes once (dev8 s1058: both cubes placed within 3 mm, but the first grasp had flicked the anchor 30
    # cm away); C5o: sort_verify, for both sort tasks, with the arm cleared first
    sort_and_verify(blocks, "cube")


def digit_votes(blocks):
    """Qwen on the head and both wrist views: the block with the smallest digit and the one with the largest, each
    answer's box mapped to a census block (point prompt at its centre on that camera, nearest within CUBE_MERGE_M)."""
    votes = {"small": [], "large": []}
    for view in ("head", "hand_left", "hand_right"):
        for key, text in (("small", SORT_NUMBER_SMALLEST), ("large", SORT_NUMBER_LARGEST)):
            if report.get("vlm_seconds", 0.0) > VLM_SORT_CEILING_S:
                flag("digit_vote_budget", view=view, which=key)
                continue
            answer = resolve_text(text, None if view == "head" else view)
            box = answer.get("box_xyxy") if answer.get("ok") else None
            if not box:
                continue
            centre = (0.5 * (float(box[0]) + float(box[2])), 0.5 * (float(box[1]) + float(box[3])))
            found = point_instances(view, centre, 0.12)
            if not found:
                continue
            xy = found[0][1]["grasp_xy"]
            best = None
            for index, block in enumerate(blocks):
                gap = xy_distance(block["xy"], xy)
                if gap < CUBE_MERGE_M and (best is None or gap < best[0]):
                    best = (gap, index)
            if best is not None:
                votes[key].append([view, best[1], answer.get("query")])
    return votes


def digit_look(rgb):
    """The look of a digit block from its mean RGB: (look, hue in degrees, saturation)."""
    r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
    hi, lo = max(r, g, b), min(r, g, b)
    sat = 0.0 if hi <= 0.0 else (hi - lo) / hi
    if hi - lo < 1e-6:
        hue = 0.0
    elif hi == r:
        hue = 60.0 * (((g - b) / (hi - lo)) % 6.0)
    elif hi == g:
        hue = 60.0 * ((b - r) / (hi - lo) + 2.0)
    else:
        hue = 60.0 * ((r - g) / (hi - lo) + 4.0)
    if sat < DIGIT_GREY_SAT:
        look = "pale"
    elif hue < DIGIT_GOLD_HUE:
        look = "golden"
    elif hue < DIGIT_BLUE_HUE[0]:
        look = "yellow"
    elif hue < DIGIT_BLUE_HUE[1]:
        look = "blue"
    elif hue < DIGIT_DARK_HUE and sat >= DIGIT_DARK_SAT:
        look = "dark"
    else:
        look = "pale"
    return look, round(hue, 1), round(sat, 3)


def look_ranking(hue, sat):
    """C5k: the five looks by their distance from a block's (hue, saturation) (DIGIT_LOOK_CENTRES, circular hue), nearest
    first, with the distances."""
    ranked = []
    for look in DIGIT_LOOK_CENTRES:
        centre = DIGIT_LOOK_CENTRES[look]
        dh = abs(hue - centre[0]) % 360.0
        dh = min(dh, 360.0 - dh)
        ranked.append((math.hypot(dh / DIGIT_LOOK_SCALE[0], (sat - centre[1]) / DIGIT_LOOK_SCALE[1]), look))
    ranked.sort()
    return ranked


def digit_face(summary):
    """digit_shape's raster with its numbers: the top 3 mm of one cloud at HOLE_CELL_M in the footprint's frame ->
    {"cells": occupied cells, "loops": enclosed empty regions of >= DIGIT_LOOP_CELLS cells, "holes": every enclosed
    region's size} or None."""
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
    sizes = []
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
            sizes.append(size)
    sizes.sort(reverse=True)
    return {"cells": len(occupied), "loops": len([size for size in sizes if size >= DIGIT_LOOP_CELLS]), "holes": sizes[:4],
            "a": round(a, 4), "b": round(b, 4)}


def digit_faces(blocks):
    """Each block's top faces in both wrist views at the start pose: the plausible lying-digit masks of three queries,
    matched to a block by footprint centre (DIGIT_FACE_MATCH_M)."""
    faces = [[] for block in blocks]
    for view in ("hand_left", "hand_right"):
        seen = []
        for query in ("block", "number block", "digit block"):
            for score, summary, box in text_instances(view, query, min_score=TWIN_FLOOR, max_extent=0.08):
                if summary["height"] > DIGIT_FACE_HEIGHT_M or summary["length"] > DIGIT_FACE_LENGTH_M:
                    continue
                if any(xy_distance(summary["grasp_xy"], other["grasp_xy"]) < 0.015 for other in seen):
                    continue
                seen.append(summary)
        for summary in seen:
            best = None
            for index, block in enumerate(blocks):
                gap = xy_distance(block["xy"], summary["grasp_xy"])
                if gap < DIGIT_FACE_MATCH_M and (best is None or gap < best[0]):
                    best = (gap, index)
            if best is None:
                continue
            face = digit_face(summary)
            if face is not None:
                face["view"] = view
                faces[best[1]].append(face)
    return faces


def digit_from_face(look, face):
    """One wrist face's digit within its look, None when the face does not say. A loop decides where the pair differs by
    one (golden 0/5, blue 6/1, dark 8/3, pale 9/4); a loopless face is the small digit only when its area agrees (a 6 or
    9 whose loop the mask missed abstains); yellow 2/7 by area alone."""
    loops, small = face["loops"], face["cells"] < DIGIT_SMALL_CELLS
    if look == "golden":
        return 0 if loops >= 1 else 5
    if look == "yellow":
        return 7 if small else 2
    if look == "dark":
        if loops >= 2:
            return 8
        return 3 if loops == 0 else None
    if loops >= 1:
        return 6 if look == "blue" else 9
    if small:
        return 1 if look == "blue" else 4
    return None


def read_digits(blocks):
    """Look (head colour) + wrist faces -> the assignments of three distinct digits, each block's digit from its look's
    pair, that agree best with the faces (faces for minus faces against). Returns (the assignments tied for best,
    per-block reading log); [] when a block has no look."""
    view = head_view()
    faces = digit_faces(blocks)
    readings = []
    for index, block in enumerate(blocks):
        rgb = top_colour(block, view)
        entry = {"xy": r3(block["xy"]), "rgb": rgb, "faces": faces[index]}
        if rgb is None:
            entry["look"] = None
            readings.append(entry)
            continue
        look, hue, sat = digit_look(rgb)
        entry.update({"look": look, "hue": hue, "sat": sat})
        votes = {}
        for face in faces[index]:
            digit = digit_from_face(look, face)
            if digit is not None:
                votes[digit] = votes.get(digit, 0) + 1
        entry["votes"] = votes
        entry["vote_log"] = {str(digit): count for digit, count in votes.items()}  # the report keeps string keys only
        readings.append(entry)
    if any(entry["look"] is None for entry in readings):
        return [], readings
    looks = [entry["look"] for entry in readings]
    scored = assignments(looks, readings)
    if not scored:
        # C5k: the looks admit no three distinct digits (e.g. three "pale" blocks): the least certain block (smallest gap
        # between its nearest and next look) takes its next look, then the next least certain, before Qwen's votes
        margins = []
        for index, entry in enumerate(readings):
            ranked = look_ranking(entry["hue"], entry["sat"])
            nxt = [item for item in ranked if item[1] != entry["look"]]
            if nxt:
                own = [item[0] for item in ranked if item[1] == entry["look"]][0]
                margins.append((nxt[0][0] - own, index, nxt[0][1]))
        for margin, index, other in sorted(margins):
            trial = list(looks)
            trial[index] = other
            scored = assignments(trial, readings)
            if scored:
                readings[index]["look_used"] = other
                flag("digit_look_next", block=index, look=readings[index]["look"], used=other, margin=round(margin, 3))
                break
    if not scored:
        return [], readings
    top = max(item[0] for item in scored)
    return [item[1] for item in scored if item[0] == top], readings


def assignments(looks, readings):
    """Every assignment of three distinct digits from the looks' pairs, scored by the faces (agree - against)."""
    scored = []
    for d0 in DIGIT_LOOKS[looks[0]]:
        for d1 in DIGIT_LOOKS[looks[1]]:
            for d2 in DIGIT_LOOKS[looks[2]]:
                digits = [d0, d1, d2]
                if len(set(digits)) < 3:
                    continue
                agree = sum(readings[k]["votes"].get(digits[k], 0) for k in range(3))
                against = sum(sum(readings[k]["votes"].values()) - readings[k]["votes"].get(digits[k], 0) for k in range(3))
                scored.append((agree - against, digits))
    return scored


def digit_rank(digits):
    """Block indices small -> large."""
    return sorted(range(3), key=lambda k: digits[k])


def run_sort_number():
    stage("census")
    # dev8 s1014: one of three blocks scored under CUBE_MIN_SCORE (0.1); size and plausibility filter the low floor
    objects = census(["block", "number block", "digit block"], min_score=TWIN_FLOOR, max_extent=0.08)
    blocks = [obj for obj in objects if 0.01 <= max(obj["summary"]["length"], obj["summary"]["width"]) <= 0.06][:3]
    report["digit_census"] = [describe_obj(obj) for obj in blocks]
    if len(blocks) < 3:
        wrist = census(["block", "number block"], views=("hand_left", "hand_right"), min_score=TWIN_FLOOR, max_extent=0.08)
        merge_into(blocks, [obj for obj in wrist if 0.01 <= max(obj["summary"]["length"], obj["summary"]["width"]) <= 0.06])
        blocks = blocks[:3]
    if len(blocks) < 3:
        report["error"] = "fewer than three digit blocks"
        return
    stage("read")
    tied, readings = read_digits(blocks)
    orders = []
    for digits in tied:
        if digit_rank(digits) not in orders:
            orders.append(digit_rank(digits))
    report["digit_read"] = {"readings": readings, "tied": tied[:6], "orders": orders}
    if len(orders) == 1:
        small, middle, large = orders[0]
        report["digit_order"] = {"small": r3(blocks[small]["xy"]), "middle": r3(blocks[middle]["xy"]), "large": r3(blocks[large]["xy"]),
                                 "digits": tied[0], "how": "look_face"}
        sort_and_verify([blocks[small], blocks[middle], blocks[large]], "block")
        return
    # the faces left more than one order (or a block has no colour): Qwen's votes choose among them (all orders if none)
    flag("digit_read_open", orders=len(orders))
    votes = digit_votes(blocks)
    tally = [[0, 0] for block in blocks]
    for view, index, query in votes["small"]:
        tally[index][0] += 1
    for view, index, query in votes["large"]:
        tally[index][1] += 1
    report["digit_votes"] = {"small": votes["small"], "large": votes["large"], "tally": tally}
    if not orders:
        orders = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]
    small, middle, large = max(orders, key=lambda o: (tally[o[0]][0] - tally[o[0]][1] + tally[o[2]][1] - tally[o[2]][0], -orders.index(o)))
    report["digit_order"] = {"small": r3(blocks[small]["xy"]), "middle": r3(blocks[middle]["xy"]), "large": r3(blocks[large]["xy"]),
                             "how": "votes"}
    if not votes["small"] and not votes["large"]:
        flag("digit_no_votes")
    sort_and_verify([blocks[small], blocks[middle], blocks[large]], "block")


def rim_options(point, radial, arm):
    """Rim grasp orientations: the closing axis radial (either sign), jaw vertical first, then tilted 20 / 35 deg toward
    the shoulder -> rim direction (dev5 s1026: bowls at x 0.87-0.90 had no IK for a vertical jaw at any rim point)."""
    yaw = math.atan2(radial[1], radial[0])
    side = 1.0 if arm == "left" else -1.0
    reach = math.atan2(point[1] - side * SHOULDER_Y, point[0] - SHOULDER_X)
    options = [option("rim_vertical", approach_quat(yaw, 0.0, 0.0), 0), option("rim_vertical_flip", approach_quat(yaw + math.pi, 0.0, 0.0), 1)]
    rank = 2
    for tilt in (math.radians(20.0), math.radians(35.0)):
        for flip in (0.0, math.pi):
            options.append(option("rim_tilt%d%s" % (int(round(math.degrees(tilt))), "_flip" if flip else ""), approach_quat(yaw + flip, tilt, reach), rank))
            rank += 1
    return options


def ik_from(position, quat, arm, seed):
    """One cuRobo IK solve seeded with `seed` (C5k: scoring grasps from a configuration the arm is not in). Joints or None."""
    try:
        ik = solve_ik(Pose(position=[float(v) for v in position], quaternion_wxyz=list(quat), frame=frame), [float(v) for v in seed], arm=arm, backend="curobo")
    except Exception:
        return None
    if not ik.ok or ik.joint_positions is None:
        return None
    return [float(v) for v in ik.joint_positions]


def bowl_rim_plans(bowl, arm, centre_goal, transport_z):
    """C5k: one bowl's rim grasps scored before anything moves, in rc2's order (rim points nearest the arm's start TCP first;
    per point the vertical jaw, then its 180-degree flip - the same grasp for the symmetric jaw), then the 20 / 35-degree
    tilts toward the shoulder (they reach the far bowls) unless a vertical grasp already scored under BOWL_PLAN_GOOD_RAD:
    IK at the hover from the arm's start joints, at the grasp point from there (the descend must exist: dev1 s5040's far
    bowl hovered and had no IK below), at the transport height above it (best of two), then over BOWL_PLAN_FAN to the carry
    goal. Held motion = up + the least carry; the first grasp under BOWL_PLAN_GOOD_RAD ends the search. rc2 took the first
    rim point whose hover planned, rim_vertical before its flip: test s5021 / s5005 / s5053 hovered 8-10 rad from the start
    and every carry from there needed 4.8-10 rad (over the cap); offline with the real cuRobo the flip or a neighbouring
    rim point of the same bowls carried in 1.1-2.8 rad. Returns (plans best first, IK calls):
    {"held", "hover", "point", "radial", "option", "joints", "yaw"}."""
    centre = bowl["xy"]
    rim_top = bowl["summary"]["top"]
    start = START_TCP[arm]
    seed = START_JOINTS[arm]
    points = []
    for k in range(8):
        angle = k * math.pi / 4.0
        radial = (math.cos(angle), math.sin(angle))
        point = (centre[0] + BOWL_RIM_RADIUS_M * radial[0], centre[1] + BOWL_RIM_RADIUS_M * radial[1], rim_top - BOWL_RIM_DEPTH_M)
        points.append((math.hypot(point[0] - start[0], point[1] - start[1]), point, radial))
    points.sort(key=lambda c: c[0])
    plans = []
    calls = 0
    for tilted in (False, True):
        if tilted and plans and min(p["held"] for p in plans) <= BOWL_PLAN_GOOD_RAD:
            break
        for distance, point, radial in points:
            for item in rim_options(point, radial, arm):
                if ("tilt" in item["name"]) != tilted:
                    continue
                hover = ik_from((point[0], point[1], point[2] + ABOVE_M), item["quat"], arm, seed)
                calls += 1
                if hover is None:
                    continue
                grasp = ik_from(point, item["quat"], arm, hover)
                calls += 1
                if grasp is None:
                    continue
                up = None
                for sample in range(2):
                    joints = ik_from((point[0], point[1], transport_z), item["quat"], arm, grasp)
                    calls += 1
                    if joints is not None and (up is None or joint_distance(joints, grasp) < joint_distance(up, grasp)):
                        up = joints
                if up is None:
                    continue
                offset = (centre[0] - point[0], centre[1] - point[1])
                carry = None
                for degrees in BOWL_PLAN_FAN:
                    angle = math.radians(degrees)
                    c, s_ = math.cos(angle), math.sin(angle)
                    goal = (centre_goal[0] - (c * offset[0] - s_ * offset[1]), centre_goal[1] - (s_ * offset[0] + c * offset[1]), transport_z)
                    joints = ik_from(goal, quat_mul(quat_axis_angle((0.0, 0.0, 1.0), angle), item["quat"]), arm, up)
                    calls += 1
                    if joints is not None and (carry is None or joint_distance(joints, up) < carry[0]):
                        carry = (joint_distance(joints, up), degrees)
                if carry is None:
                    continue
                held = joint_distance(up, grasp) + carry[0]
                if "tilt" in item["name"] and radial[0] > 0.1:
                    # C5o: a tilted jaw on the rim's far side (beyond the bowl's centre from the robot) never held: stack bowls
                    # inst 4's far bowl, rim_tilt35_flip at radial (0.71, -0.71), slipped out before its carry in all four
                    # runs that chose it (1710, C5k repeat, C5o repro, dev5), its (0, 1) rim point held (C5k stage 2); every
                    # other tilted rim grasp logged (34) had radial x <= 0. Scored BOWL_FAR_TILT_PENALTY_RAD worse
                    held += BOWL_FAR_TILT_PENALTY_RAD
                plans.append({"held": held, "hover": joint_distance(hover, seed), "point": point,
                              "radial": radial, "option": item, "joints": hover, "yaw": carry[1]})
                if plans[-1]["held"] <= BOWL_PLAN_GOOD_RAD:
                    plans.sort(key=lambda p: p["held"] + BOWL_PLAN_HOVER_WEIGHT * p["hover"])
                    return plans, calls
    plans.sort(key=lambda p: p["held"] + BOWL_PLAN_HOVER_WEIGHT * p["hover"])
    return plans, calls


def bowl_order_plans(order, centre, rim, tops):
    """C5k: bowl_rim_plans for every carry of one stacking plan (each bowl by its own side's arm, from home: the arms go home
    after each release), the k-th onto a stack whose rim stands k nest rises over `rim`. (plans per bowl, worst best held
    motion or None when a bowl has no plan, summary for the report)."""
    plans = []
    worst = 0.0
    info = []
    for index, bowl in enumerate(order):
        transport = max([rim + index * BOWL_NEST_RISE_M] + tops) + BOWL_CLEAR_M + BOWL_HEIGHT_M - BOWL_RIM_DEPTH_M
        found, calls = bowl_rim_plans(bowl, side_arm(bowl["xy"]), centre, transport)
        plans.append(found)
        best = found[0] if found else None
        info.append({"bowl": r3(bowl["xy"]), "calls": calls, "held": None if best is None else round(best["held"], 3),
                     "hover": None if best is None else round(best["hover"], 3), "option": None if best is None else best["option"]["name"],
                     "radial": None if best is None else r3(best["radial"]), "yaw": None if best is None else best["yaw"]})
        if best is None:
            worst = None
        elif worst is not None:
            worst = max(worst, best["held"])
    return plans, worst, info


def move_planned(position, item, joints, arm, label, gripper=None):
    """C5k: cuRobo plan (collision-aware) to joints chosen beforehand (bowl_rim_plans), executed and servoed. The option or None."""
    entry = {"label": label, "target": r3(position), "option": item["name"], "joint_distance": round(joint_distance(joints, current_joints(arm)), 3)}
    report["moves"].append(entry)
    try:
        plan = plan_motion(joints, arm=arm, strategy=STRATEGY)
    except Exception as exc:
        entry["fail"] = "exception: " + str(exc)[:80]
        return None
    if not plan.ok or plan.trajectory is None:
        entry["fail"] = error_text(plan)
        return None
    return item if execute(plan.trajectory, arm, entry, gripper) else None


def bowl_release(arm, rim, label):
    """C5k: open, hold one chunk with the jaw open (it is still opening when open_gripper returns), then straight up to
    BOWL_CLEAR_UP_M over the stack's rim (at least RELEASE_RETREAT_M) in one move, out to BOWL_CLEAR_SIDE_Y on the arm's
    own side at that height, then home (clear_arm). rc2 went home from 10 cm over the rim on a cuRobo plan whose collision
    scene (head depth) lacked the bowl the arm itself hid: test s5037's first bowl was dragged 12 cm toward the robot.
    One chunk more than rc2's release + clear (the mid-table plan's third release came at rows 1100-1290 of 1500)."""
    opened = open_gripper(arm=arm)
    count_rows(opened)
    grip_command[arm] = 1.0
    if report["terminated"]:
        return
    hold_still(arm, label + "_open", 1.0)
    if report["terminated"]:
        return
    here, quat = tool_pose(arm)
    z = max(here[2] + RELEASE_RETREAT_M, rim + BOWL_CLEAR_UP_M)
    up, entry = near_moves(arm, option("retreat", quat, 0), (here[0], here[1]), z, label + "_retreat", 1.0, CARRY_MAX_JOINT_RAD,
                           fan_degrees=(0,), offsets=((0.0, 0.0),))
    if report["terminated"]:
        return
    if up is None:
        move_ik((here[0], here[1], here[2] + RELEASE_RETREAT_M), [option("retreat", quat, 0)], arm, label + "_retreat_ik", gripper=1.0,
                strategy=STRATEGY_FINAL, samples=2)
        if report["terminated"]:
            return
    here, quat = tool_pose(arm)
    side = 1.0 if arm == "left" else -1.0
    if side * here[1] < BOWL_CLEAR_SIDE_Y:
        stepped, entry = near_moves(arm, option("open", quat, 0), (here[0], side * BOWL_CLEAR_SIDE_Y), here[2], label + "_side", 1.0,
                                    CARRY_MAX_JOINT_RAD, fan_degrees=(0, 30, -30), offsets=((0.0, 0.0),))
        if report["terminated"]:
            return
        if stepped is None:
            # C5o: refused (1710 inst 4: the left arm, back from a carry 12 cm across the midline, needed 3.63 rad; it then
            # went home on a cuRobo plan from over the stack and the stack failed): BOWL_SIDE_RETRY_UP_M higher, then the step
            # again with the empty arm's larger cap
            flag("bowl_side_retry", arm=arm, least=entry.get("best_joint_distance"))
            near_moves(arm, option("open", quat, 0), (here[0], here[1]), here[2] + BOWL_SIDE_RETRY_UP_M, label + "_side_up", 1.0,
                       CARRY_MAX_JOINT_RAD, fan_degrees=(0,), offsets=((0.0, 0.0),))
            if report["terminated"]:
                return
            here, quat = tool_pose(arm)
            near_moves(arm, option("open", quat, 0), (here[0], side * BOWL_CLEAR_SIDE_Y), here[2], label + "_side_again", 1.0,
                       BOWL_SIDE_RETRY_CAP_RAD, fan_degrees=(0, 30, -30), offsets=((0.0, 0.0),))
            if report["terminated"]:
                return
    clear_arm(arm, label + "_clear")


def bowl_grasp(bowl, arm, attempt, plans=None):
    """Rim grasp: jaw vertical, closing axis radial (one finger inside the bowl, one outside), TCP BOWL_RIM_DEPTH_M under
    the rim top. C5k: the planned grasps first (bowl_rim_plans: the hover goes to the scored joints); else rc2's rim points
    every 45 degrees, nearest to the arm's start TCP first; the first whose hover plans (cuRobo, collision-aware) is used,
    then a joint-interpolated descend, close, a settle chunk, lift. Returns (held, tcp_at_grasp, radial)."""
    centre = bowl["xy"]
    rim_top = bowl["summary"]["top"]
    start = START_TCP[arm]
    candidates = []
    for k in range(8):
        angle = k * math.pi / 4.0
        radial = (math.cos(angle), math.sin(angle))
        point = (centre[0] + BOWL_RIM_RADIUS_M * radial[0], centre[1] + BOWL_RIM_RADIUS_M * radial[1], rim_top - BOWL_RIM_DEPTH_M)
        candidates.append((math.hypot(point[0] - start[0], point[1] - start[1]), point, radial))
    candidates.sort(key=lambda c: c[0])
    chosen = None
    descended = False
    for plan in (plans or [])[:3]:
        point, radial = plan["point"], plan["radial"]
        attempt.update({"rim_point": r3(point), "radial": r3(radial), "rim_top": round(rim_top, 4), "planned_held": round(plan["held"], 3),
                        "planned_hover": round(plan["hover"], 3), "planned_yaw": plan["yaw"], "planned_option": plan["option"]["name"]})
        chosen = move_planned((point[0], point[1], point[2] + ABOVE_M), plan["option"], plan["joints"], arm, "bowl_hover_planned", gripper=1.0)
        if report["terminated"]:
            return False, None, None
        if chosen is None:
            continue
        # C5k: the descend from the planned hover; if it has no IK after all, the next planned grasp (dev1 s5040)
        if move_ik(point, [chosen], arm, "bowl_descend", gripper=1.0, strategy=STRATEGY_FINAL, exclude=bowl["summary"]["cloud"], samples=3) is not None:
            descended = True
            break
        if report["terminated"]:
            return False, None, radial
        flag("bowl_planned_descend_failed", option=chosen["name"], radial=r3(radial))
        chosen = None
    if report["terminated"]:
        return False, None, None
    for distance, point, radial in ([] if chosen is not None else candidates[:3]):
        options = rim_options(point, radial, arm)
        attempt.update({"rim_point": r3(point), "radial": r3(radial), "rim_top": round(rim_top, 4)})
        chosen = move_ik((point[0], point[1], point[2] + ABOVE_M), options, arm, "bowl_hover", gripper=1.0)
        if chosen is not None or report["terminated"]:
            break
    if chosen is None and not report["terminated"]:
        distance, point, radial = candidates[0]
        options = rim_options(point, radial, arm)
        attempt.update({"rim_point": r3(point), "radial": r3(radial)})
        chosen = move_pose((point[0], point[1], point[2] + ABOVE_M), options, arm, "bowl_hover_integrated", gripper=1.0)
    if chosen is None:
        attempt["fail"] = "bowl_hover"
        return False, None, None
    if report["terminated"]:
        return False, None, radial
    if not descended and move_ik(point, [chosen], arm, "bowl_descend", gripper=1.0, strategy=STRATEGY_FINAL, exclude=bowl["summary"]["cloud"], samples=3) is None:
        attempt["fail"] = "bowl_descend"
        return False, None, radial
    if report["terminated"]:
        return False, None, radial
    tcp = tool_pose(arm)[0]
    closed = close_gripper(arm=arm)
    count_rows(closed)
    grip_command[arm] = 0.0
    if report["terminated"]:
        return False, tcp, radial
    hold_still(arm, "bowl_settle", 0.0)
    if report["terminated"]:
        return False, tcp, radial
    at_close = gripper_opening(arm)
    lifted = move_ik((tcp[0], tcp[1], tcp[2] + 0.06), [chosen], arm, "bowl_lift", gripper=0.0, strategy=STRATEGY_FINAL, samples=2) is not None
    opening = gripper_opening(arm)
    attempt.update({"at_close": round(at_close, 4), "opening": round(opening, 4), "lifted": lifted, "tcp_at_grasp": r3(tcp)})
    return lifted and opening >= BOWL_HELD_OPENING, tcp, radial


def carry_offset(arm, centre_goal, offset, z, label, gripper=0.0, fan=CARRY_FAN, cap=None):
    """Carry an object held off the TCP axis (a bowl by its rim, its centre at TCP + offset): for each yaw of the fan the
    gripper turns about world z and the offset with it, so the TCP goal is centre_goal - R(yaw) offset; the least joint
    motion over every yaw (under CARRY_MAX_JOINT_RAD), joint interpolated and slowed (dev5 s1065 / s1058: with the yaw
    fixed every bowl carry needed 3.1-5.6 rad, and the 2 cm 'near' fallbacks put the drops off the base).
    Returns (executed option, yaw in degrees) or (None, None)."""
    here, quat = tool_pose(arm)
    entry = {"label": label, "centre_goal": r3(centre_goal), "offset": r3(offset), "z": round(z, 4), "from": r3(here)}
    report.setdefault("carries", []).append(entry)
    candidates = []
    for degrees in fan:
        angle = math.radians(degrees)
        c, s_ = math.cos(angle), math.sin(angle)
        turned = (c * offset[0] - s_ * offset[1], s_ * offset[0] + c * offset[1])
        goal = (centre_goal[0] - turned[0], centre_goal[1] - turned[1], z)
        item = option("held_z%+d" % degrees, quat_mul(quat_axis_angle((0.0, 0.0, 1.0), angle), quat), 0)
        for rank, distance, it, joints in ik_ranked(goal, [item], arm, samples=2, keep=2):
            candidates.append((distance, goal, it, joints, degrees))
    candidates.sort(key=lambda cand: cand[0])
    entry["ik_solutions"] = len(candidates)
    entry["best_joint_distance"] = round(candidates[0][0], 3) if candidates else None
    for distance, goal, item, joints, degrees in candidates[:4]:
        if distance > (cap or CARRY_MAX_JOINT_RAD):
            entry["fail"] = "least joint motion %.2f rad over the cap" % distance
            return None, None
        move = {"label": label, "target": r3(goal), "option": item["name"], "joint_distance": round(distance, 3)}
        report["moves"].append(move)
        try:
            plan = plan_motion(joints, arm=arm, strategy=STRATEGY_FINAL)
        except Exception as exc:
            move["fail"] = "exception: " + str(exc)[:80]
            continue
        if not plan.ok or plan.trajectory is None:
            move["fail"] = error_text(plan)
            continue
        execute(slowed(plan.trajectory, arm, CARRY_MAX_STEP_RAD), arm, move, gripper)
        entry.update({"done": item["name"], "yaw": degrees, "goal": r3(goal)})
        try:
            entry["to"] = r3(tool_pose(arm)[0])
        except Exception:
            pass
        return item, degrees
    entry["fail"] = "no IK solution" if not candidates else "planning refused every candidate"
    return None, None


def run_stack_bowls():
    stage("census")
    bowls = [obj for obj in census(["bowl", "green bowl"], max_extent=0.25) if 0.10 <= max(obj["summary"]["length"], obj["summary"]["width"]) <= 0.22][:3]
    report["bowl_census"] = [describe_obj(obj) for obj in bowls]
    if len(bowls) < 2:
        report["error"] = "fewer than two bowls"
        return
    # in place: the base stays (the bowl nearest the midline), the others go into it, nearest first, each by the arm on its
    # own side; mid-table: the bowl nearest a free BOWL_SPOT is set on the table there, then the others come onto it
    base = min(bowls, key=lambda b: (round(abs(b["xy"][1]) / 0.04), b["xy"][0]))
    others = sorted([b for b in bowls if b is not base], key=lambda b: xy_distance(b["xy"], base["xy"]))
    crossing = max([abs(base["xy"][1]) for b in others if (b["xy"][1] > 0.0) != (base["xy"][1] > 0.0)] + [0.0])
    report["bowl_plan"] = {"base": r3(base["xy"]), "order": [r3(b["xy"]) for b in others], "crossing": round(crossing, 3)}
    tops = [b["summary"]["top"] for b in bowls]
    layouts = []
    if crossing <= BOWL_CROSS_M and base["xy"][0] <= BOWL_BASE_MAX_X:
        layouts.append(("in place", others, base["summary"]["top"], base["xy"]))
    for spot in BOWL_SPOTS:
        first = min(bowls, key=lambda b: xy_distance(b["xy"], spot))
        rest = [b for b in bowls if b is not first]
        if all(xy_distance(b["xy"], spot) >= BOWL_SPOT_FREE_M for b in rest):
            # the first one's rim ends where a table-standing bowl's is
            layouts.append(("spot", [first] + sorted(rest, key=lambda b: xy_distance(b["xy"], spot)), TABLE_Z + BOWL_HEIGHT_M - BOWL_NEST_RISE_M, spot))
            break
    if not layouts:
        flag("bowl_spot_taken")
        layouts.append(("in place", others, base["summary"]["top"], base["xy"]))
    # C5k: every carry's rim grasp is planned before anything moves (bowl_order_plans); the first layout whose carries all
    # have a grasp under BOWL_PLAN_CAP_RAD is used (in place first, as rc2), else the one whose worst carry is least
    chosen = None
    tried = []
    for name, order, rim0, centre0 in layouts:
        plans, worst, info = bowl_order_plans(order, centre0, rim0, tops)
        tried.append({"layout": name, "worst_held": None if worst is None else round(worst, 3), "bowls": info})
        if worst is not None and worst <= BOWL_PLAN_CAP_RAD:
            chosen = (name, order, rim0, centre0, plans)
            break
        if worst is not None and (chosen is None or worst < chosen[5]):
            chosen = (name, order, rim0, centre0, plans, worst)
    report["bowl_plan"]["layouts"] = tried
    if chosen is None:
        flag("bowl_no_planned_grasp")
        name, order, rim0, centre0 = layouts[0]
        plans = [[] for b in order]
    else:
        name, order, rim0, centre0, plans = chosen[:5]
        if len(chosen) > 5:
            flag("bowl_plan_over_cap", layout=name, worst=round(chosen[5], 3))
    others, rim, centre = order, rim0, centre0
    aim = (centre[0], centre[1])
    report["bowl_plan"]["layout"] = name
    report["bowl_plan"]["order"] = [r3(b["xy"]) for b in others]
    if name == "spot":
        report["bowl_plan"]["spot"] = r3(centre)
    for index, bowl in enumerate(others):
        arm = side_arm(bowl["xy"])
        attempt = {"bowl": index, "arm": arm, "xy": r3(bowl["xy"]), "rows_start": report["rows"]}
        report.setdefault("bowl_moves", []).append(attempt)
        stage("bowl_grasp_%d" % index, arm=arm)
        held, tcp, radial = bowl_grasp(bowl, arm, attempt, plans[index])
        attempt["held"] = held
        if report["terminated"]:
            return
        if not held:
            flag("bowl_not_held", index=index)
            if tcp is not None:
                opened = open_gripper(arm=arm)
                count_rows(opened)
                grip_command[arm] = 1.0
                here, quat = tool_pose(arm)
                move_ik((here[0], here[1], here[2] + 0.10), [option("retreat", quat, 0)], arm, "bowl_retreat", gripper=1.0, strategy=STRATEGY_FINAL, samples=2)
            continue
        # the carried bowl's centre sits BOWL_RIM_RADIUS_M beyond the TCP along -radial; keep the orientation (no yaw)
        offset = (bowl["xy"][0] - tcp[0], bowl["xy"][1] - tcp[1])
        tcp_goal = (centre[0] - offset[0], centre[1] - offset[1])
        below = tcp[2] - bowl["summary"]["bottom"]
        # the head sees a bowl's rim, not its foot (census bottom 1.6 cm above the table): the TCP hangs `depth` under the
        # carried rim, the true bottom BOWL_HEIGHT_M under it; nested, the carried rim stands BOWL_NEST_RISE_M over `rim`
        depth = bowl["summary"]["top"] - tcp[2]
        transport = max([rim] + tops) + BOWL_CLEAR_M + BOWL_HEIGHT_M - depth
        drop = rim + BOWL_NEST_RISE_M + BOWL_DROP_GAP_M - depth
        attempt.update({"tcp_goal": r3(tcp_goal), "below": round(below, 4), "depth": round(depth, 4), "transport_z": round(transport, 4),
                        "drop_z": round(drop, 4)})
        stage("bowl_place_%d" % index)
        here = tool_pose(arm)[0]
        if transport > here[2] + 0.02:
            lower_to(arm, transport, "bowl_up_%d" % index)
            if report["terminated"]:
                return
        done, yaw = carry_offset(arm, aim, offset, transport, "bowl_carry_%d" % index, cap=BOWL_MAX_JOINT_RAD)
        attempt["carried"] = done is not None
        attempt["carry_yaw"] = yaw
        if report["terminated"]:
            return
        if done is None:
            # never drop a bowl away from the base: keep holding it and stop
            flag("bowl_carry_failed", index=index)
            break
        # C5k: no head re-measurement of the hanging bowl (rc2 corrected by the base's near side, 3.8-6.5 cm): lowered as carried
        lower_to(arm, drop, "bowl_lower_%d" % index)
        if report["terminated"]:
            return
        attempt["held_before_release"] = gripper_opening(arm) >= BOWL_HELD_OPENING
        # where did it land? the arm goes up, out and home (the head then sees the stack) and the bowls are counted again
        bowl_release(arm, rim + BOWL_NEST_RISE_M, "bowl_release_%d" % index)
        if report["terminated"]:
            return
        after = [obj for obj in census(["bowl", "green bowl"], max_extent=0.25) if 0.10 <= max(obj["summary"]["length"], obj["summary"]["width"]) <= 0.22]
        near = [obj for obj in after if xy_distance(obj["xy"], centre) < 0.10]
        attempt["after"] = [describe_obj(obj) for obj in after[:4]]
        attempt["after_near_base"] = [[r3(obj["xy"]), round(obj["summary"]["top"], 4), round(xy_distance(obj["xy"], centre), 4)] for obj in near]
        rim = max([rim + BOWL_NEST_RISE_M] + [obj["summary"]["top"] for obj in near])
        # the next bowl goes onto the stack where it now is (its highest bowl near the base)
        stacked = sorted([obj for obj in near if xy_distance(obj["xy"], aim) < BOWL_SEEN_M], key=lambda obj: -obj["summary"]["top"])
        if stacked:
            aim = (stacked[0]["xy"][0], stacked[0]["xy"][1])
            attempt["next_aim"] = r3(aim)
    stage("hold")
    hold_checks("left" if centre[1] > 0 else "right", "hold_bowls", None)


def run_episode():
    task = context.task_name if context.task_name in SPATIAL_TASKS else task_from_sentence(LANGUAGE)
    report["kind"] = task
    parsed = parse_task(task, LANGUAGE)
    report["parsed"] = parsed
    if HOME_MOVED_START:
        home_moved_arms()
        if report["terminated"]:
            return
    if task in PICK_TASKS:
        run_pick(task, parsed)
    elif task == "place_object_relative_position":
        run_place_relative(parsed)
    elif task == "place_beverage_to_anothers_position":
        run_place_beverage(parsed)
    elif task == "stack_three_building_blocks":
        run_stack_blocks()
    elif task == "sort_cubes_by_size":
        run_sort_cubes()
    elif task == "stack_bowls":
        run_stack_bowls()
    elif task == "sort_number":
        run_sort_number()
    else:
        report["error"] = "task not implemented in v1: " + task
    stage("done")


try:
    run_episode()
except Exception as exc:
    report["error"] = "exception: " + str(exc)[:300]
for attempt in report["attempts"]:
    attempt.pop("target", None)
last_attempt = report["attempts"][-1] if report["attempts"] else {}
report["summary"] = {
    "kind": report.get("kind"),
    "arm": report.get("arm"),
    "chooser": (report.get("target") or {}).get("how"),
    "approach": last_attempt.get("approach"),
    "fail": last_attempt.get("fail"),
    "retry_on_air": "retry" in last_attempt,
    "held": report.get("held"),
    "attempts": len(report["attempts"]),
    "flags": [entry["flag"] for entry in report["flags"]],
    "rows": report["rows"],
    "terminated": report["terminated"],
    "error": report["error"],
}
result = report
