# Development cost and run time: further detail

Report: Section 4.4 and Appendix B (Table 8).

## Sessions

The 13 logged coding-agent sessions (2026-10-26 to 2026-11-04, four tracks) took 138.6 agent hours and used 3.72
billion input and 8.48 million output tokens (Appendix B).

Each session ran unattended from a written briefing, and its log keeps running counts of its model tokens. We count
each session once, at the last counts in its log. One session stopped and was resumed in a second log file, and its
last counts cover both parts.

Of the 3.72 billion input tokens, 3.70 billion were cache reads, 18.03 million cache writes and 42 thousand uncached
input. The 8.48 million output tokens include thinking.

## The online agent of the comparison

The online-agent figures are quoted from the PyRUA-Lean paper: per RoboDojo episode, about 750,000 input and 9,000
output tokens, with cached input counted, at about $2.4. The same paper reports 25.7 model calls per episode on
average, 19.3 in solved episodes and 32.2 in unsolved ones.

## The pick episode (Section 5.1)

The episode of Section 5.1 took 61.8 s of wall-clock time (Figure 6d). Of this, 11.3 s went to the resolver and 8.7 s
to five SAM 3 calls, including saving the overlays of Figure 6a-c. Fourteen cuRobo calls took 1.7 s: one motion plan and
thirteen inverse-kinematics solves, eight of them for the grasp candidates. The simulator took 31.0 s of wall-clock time
to execute six motions. The program executed 163 action steps, which is 5.4 s of motion at the nominal 30 Hz command
rate.
