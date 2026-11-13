# Planspace evaluation: 4 arms, exploratory play data, fine-tuned checkpoint adaln_s0_k4_explore_lora (no joint stage)

Arms: **A1** = action search, IK of the predicted hand; **A2** = action search, action executed directly; **L** = latent search, IK of the predicted hand; **L2A** = latent search, L2A action executed directly.
Errors in mm: reach = mean distance of the 5 hand bodies to the target; grasp = wrist to object. Min = lowest error seen at any replan.
Videos: `<task>/<arm>/<demo>_<task>.mp4` (both cameras, every control step); `compare/<task>/<demo>.mp4` stacks all arms + the demo.

## reach (20 episodes)

| Arm | Done | Success | Mean final (mm) | Mean min (mm) | Mean replans | Mean wall (min) |
|---|---|---|---|---|---|---|
| A1 | 20/20 | 20/20 | 107 | 107 | 7.8 | 3.6 |
| A2 | 20/20 | 20/20 | 87 | 87 | 6.5 | 3.8 |
| L | 20/20 | 20/20 | 114 | 114 | 9.2 | 3.5 |
| L2A | 20/20 | 20/20 | 117 | 117 | 9.9 | 4.5 |

| Episode | A1 | A2 | L | L2A | Compare |
|---|---|---|---|---|---|
| demo_133 | [✓ 105](reach/A1/demo_133_reach.mp4) | [✓ 80](reach/A2/demo_133_reach.mp4) | [✓ 107](reach/L/demo_133_reach.mp4) | [✓ 137](reach/L2A/demo_133_reach.mp4) | [video](compare/reach/demo_133.mp4) |
| demo_606 | [✓ 108](reach/A1/demo_606_reach.mp4) | [✓ 91](reach/A2/demo_606_reach.mp4) | [✓ 110](reach/L/demo_606_reach.mp4) | [✓ 99](reach/L2A/demo_606_reach.mp4) | [video](compare/reach/demo_606.mp4) |
| demo_1364 | [✓ 117](reach/A1/demo_1364_reach.mp4) | [✓ 62](reach/A2/demo_1364_reach.mp4) | [✓ 131](reach/L/demo_1364_reach.mp4) | [✓ 115](reach/L2A/demo_1364_reach.mp4) | [video](compare/reach/demo_1364.mp4) |
| demo_1208 | [✓ 109](reach/A1/demo_1208_reach.mp4) | [✓ 78](reach/A2/demo_1208_reach.mp4) | [✓ 117](reach/L/demo_1208_reach.mp4) | [✓ 111](reach/L2A/demo_1208_reach.mp4) | [video](compare/reach/demo_1208.mp4) |
| demo_1341 | [✓ 107](reach/A1/demo_1341_reach.mp4) | [✓ 99](reach/A2/demo_1341_reach.mp4) | [✓ 108](reach/L/demo_1341_reach.mp4) | [✓ 99](reach/L2A/demo_1341_reach.mp4) | [video](compare/reach/demo_1341.mp4) |
| demo_1311 | [✓ 116](reach/A1/demo_1311_reach.mp4) | [✓ 87](reach/A2/demo_1311_reach.mp4) | [✓ 124](reach/L/demo_1311_reach.mp4) | [✓ 103](reach/L2A/demo_1311_reach.mp4) | [video](compare/reach/demo_1311.mp4) |
| demo_686 | [✓ 123](reach/A1/demo_686_reach.mp4) | [✓ 112](reach/A2/demo_686_reach.mp4) | [✓ 126](reach/L/demo_686_reach.mp4) | [✓ 130](reach/L2A/demo_686_reach.mp4) | [video](compare/reach/demo_686.mp4) |
| demo_1076 | [✓ 123](reach/A1/demo_1076_reach.mp4) | [✓ 99](reach/A2/demo_1076_reach.mp4) | [✓ 125](reach/L/demo_1076_reach.mp4) | [✓ 114](reach/L2A/demo_1076_reach.mp4) | [video](compare/reach/demo_1076.mp4) |
| demo_665 | [✓ 101](reach/A1/demo_665_reach.mp4) | [✓ 105](reach/A2/demo_665_reach.mp4) | [✓ 111](reach/L/demo_665_reach.mp4) | [✓ 136](reach/L2A/demo_665_reach.mp4) | [video](compare/reach/demo_665.mp4) |
| demo_1142 | [✓ 90](reach/A1/demo_1142_reach.mp4) | [✓ 86](reach/A2/demo_1142_reach.mp4) | [✓ 104](reach/L/demo_1142_reach.mp4) | [✓ 134](reach/L2A/demo_1142_reach.mp4) | [video](compare/reach/demo_1142.mp4) |
| demo_793 | [✓ 109](reach/A1/demo_793_reach.mp4) | [✓ 92](reach/A2/demo_793_reach.mp4) | [✓ 111](reach/L/demo_793_reach.mp4) | [✓ 114](reach/L2A/demo_793_reach.mp4) | [video](compare/reach/demo_793.mp4) |
| demo_131 | [✓ 96](reach/A1/demo_131_reach.mp4) | [✓ 65](reach/A2/demo_131_reach.mp4) | [✓ 123](reach/L/demo_131_reach.mp4) | [✓ 129](reach/L2A/demo_131_reach.mp4) | [video](compare/reach/demo_131.mp4) |
| demo_952 | [✓ 96](reach/A1/demo_952_reach.mp4) | [✓ 79](reach/A2/demo_952_reach.mp4) | [✓ 125](reach/L/demo_952_reach.mp4) | [✓ 127](reach/L2A/demo_952_reach.mp4) | [video](compare/reach/demo_952.mp4) |
| demo_907 | [✓ 106](reach/A1/demo_907_reach.mp4) | [✓ 95](reach/A2/demo_907_reach.mp4) | [✓ 125](reach/L/demo_907_reach.mp4) | [✓ 111](reach/L2A/demo_907_reach.mp4) | [video](compare/reach/demo_907.mp4) |
| demo_831 | [✓ 123](reach/A1/demo_831_reach.mp4) | [✓ 84](reach/A2/demo_831_reach.mp4) | [✓ 108](reach/L/demo_831_reach.mp4) | [✓ 127](reach/L2A/demo_831_reach.mp4) | [video](compare/reach/demo_831.mp4) |
| demo_170 | [✓ 107](reach/A1/demo_170_reach.mp4) | [✓ 111](reach/A2/demo_170_reach.mp4) | [✓ 108](reach/L/demo_170_reach.mp4) | [✓ 112](reach/L2A/demo_170_reach.mp4) | [video](compare/reach/demo_170.mp4) |
| demo_146 | [✓ 115](reach/A1/demo_146_reach.mp4) | [✓ 101](reach/A2/demo_146_reach.mp4) | [✓ 132](reach/L/demo_146_reach.mp4) | [✓ 114](reach/L2A/demo_146_reach.mp4) | [video](compare/reach/demo_146.mp4) |
| demo_739 | [✓ 108](reach/A1/demo_739_reach.mp4) | [✓ 82](reach/A2/demo_739_reach.mp4) | [✓ 129](reach/L/demo_739_reach.mp4) | [✓ 126](reach/L2A/demo_739_reach.mp4) | [video](compare/reach/demo_739.mp4) |
| demo_322 | [✓ 88](reach/A1/demo_322_reach.mp4) | [✓ 54](reach/A2/demo_322_reach.mp4) | [✓ 84](reach/L/demo_322_reach.mp4) | [✓ 106](reach/L2A/demo_322_reach.mp4) | [video](compare/reach/demo_322.mp4) |
| demo_814 | [✓ 101](reach/A1/demo_814_reach.mp4) | [✓ 70](reach/A2/demo_814_reach.mp4) | [✓ 81](reach/L/demo_814_reach.mp4) | [✓ 97](reach/L2A/demo_814_reach.mp4) | [video](compare/reach/demo_814.mp4) |

## grasp (20 episodes)

| Arm | Done | Success | Mean final (mm) | Mean min (mm) | Mean replans | Mean wall (min) |
|---|---|---|---|---|---|---|
| A1 | 20/20 | 17/20 | 159 | 145 | 46.5 | 20.1 |
| A2 | 20/20 | 17/20 | 152 | 132 | 53.0 | 22.1 |
| L | 20/20 | 18/20 | 159 | 148 | 46.5 | 20.8 |
| L2A | 20/20 | 17/20 | 175 | 162 | 52.5 | 19.9 |

| Episode | A1 | A2 | L | L2A | Compare |
|---|---|---|---|---|---|
| demo_133 | [✓ 163](grasp/A1/demo_133_grasp.mp4) | [✓ 168](grasp/A2/demo_133_grasp.mp4) | [✓ 165](grasp/L/demo_133_grasp.mp4) | [✓ 170](grasp/L2A/demo_133_grasp.mp4) | [video](compare/grasp/demo_133.mp4) |
| demo_606 | [✗ 366](grasp/A1/demo_606_grasp.mp4) | [✗ 366](grasp/A2/demo_606_grasp.mp4) | [✗ 354](grasp/L/demo_606_grasp.mp4) | [✗ 343](grasp/L2A/demo_606_grasp.mp4) | [video](compare/grasp/demo_606.mp4) |
| demo_1364 | [✗ 194](grasp/A1/demo_1364_grasp.mp4) | [✗ 203](grasp/A2/demo_1364_grasp.mp4) | [✗ 190](grasp/L/demo_1364_grasp.mp4) | [✗ 334](grasp/L2A/demo_1364_grasp.mp4) | [video](compare/grasp/demo_1364.mp4) |
| demo_1208 | [✓ 131](grasp/A1/demo_1208_grasp.mp4) | [✓ 123](grasp/A2/demo_1208_grasp.mp4) | [✓ 142](grasp/L/demo_1208_grasp.mp4) | [✓ 170](grasp/L2A/demo_1208_grasp.mp4) | [video](compare/grasp/demo_1208.mp4) |
| demo_1341 | [✓ 128](grasp/A1/demo_1341_grasp.mp4) | [✓ 124](grasp/A2/demo_1341_grasp.mp4) | [✓ 131](grasp/L/demo_1341_grasp.mp4) | [✓ 163](grasp/L2A/demo_1341_grasp.mp4) | [video](compare/grasp/demo_1341.mp4) |
| demo_1311 | [✓ 126](grasp/A1/demo_1311_grasp.mp4) | [✓ 100](grasp/A2/demo_1311_grasp.mp4) | [✓ 127](grasp/L/demo_1311_grasp.mp4) | [✓ 138](grasp/L2A/demo_1311_grasp.mp4) | [video](compare/grasp/demo_1311.mp4) |
| demo_1008 | [✓ 122](grasp/A1/demo_1008_grasp.mp4) | [✓ 111](grasp/A2/demo_1008_grasp.mp4) | [✓ 134](grasp/L/demo_1008_grasp.mp4) | [✓ 126](grasp/L2A/demo_1008_grasp.mp4) | [video](compare/grasp/demo_1008.mp4) |
| demo_870 | [✓ 140](grasp/A1/demo_870_grasp.mp4) | [✓ 140](grasp/A2/demo_870_grasp.mp4) | [✓ 144](grasp/L/demo_870_grasp.mp4) | [✓ 188](grasp/L2A/demo_870_grasp.mp4) | [video](compare/grasp/demo_870.mp4) |
| demo_686 | [✗ 156](grasp/A1/demo_686_grasp.mp4) | [✓ 168](grasp/A2/demo_686_grasp.mp4) | [✓ 162](grasp/L/demo_686_grasp.mp4) | [✓ 173](grasp/L2A/demo_686_grasp.mp4) | [video](compare/grasp/demo_686.mp4) |
| demo_1076 | [✓ 179](grasp/A1/demo_1076_grasp.mp4) | [✓ 133](grasp/A2/demo_1076_grasp.mp4) | [✓ 127](grasp/L/demo_1076_grasp.mp4) | [✓ 137](grasp/L2A/demo_1076_grasp.mp4) | [video](compare/grasp/demo_1076.mp4) |
| demo_665 | [✓ 169](grasp/A1/demo_665_grasp.mp4) | [✓ 158](grasp/A2/demo_665_grasp.mp4) | [✓ 177](grasp/L/demo_665_grasp.mp4) | [✓ 169](grasp/L2A/demo_665_grasp.mp4) | [video](compare/grasp/demo_665.mp4) |
| demo_1142 | [✓ 145](grasp/A1/demo_1142_grasp.mp4) | [✓ 166](grasp/A2/demo_1142_grasp.mp4) | [✓ 171](grasp/L/demo_1142_grasp.mp4) | [✓ 157](grasp/L2A/demo_1142_grasp.mp4) | [video](compare/grasp/demo_1142.mp4) |
| demo_793 | [✓ 163](grasp/A1/demo_793_grasp.mp4) | [✓ 139](grasp/A2/demo_793_grasp.mp4) | [✓ 145](grasp/L/demo_793_grasp.mp4) | [✓ 195](grasp/L2A/demo_793_grasp.mp4) | [video](compare/grasp/demo_793.mp4) |
| demo_131 | [✓ 142](grasp/A1/demo_131_grasp.mp4) | [✓ 119](grasp/A2/demo_131_grasp.mp4) | [✓ 136](grasp/L/demo_131_grasp.mp4) | [✓ 173](grasp/L2A/demo_131_grasp.mp4) | [video](compare/grasp/demo_131.mp4) |
| demo_952 | [✓ 134](grasp/A1/demo_952_grasp.mp4) | [✓ 110](grasp/A2/demo_952_grasp.mp4) | [✓ 131](grasp/L/demo_952_grasp.mp4) | [✓ 119](grasp/L2A/demo_952_grasp.mp4) | [video](compare/grasp/demo_952.mp4) |
| demo_907 | [✓ 116](grasp/A1/demo_907_grasp.mp4) | [✓ 118](grasp/A2/demo_907_grasp.mp4) | [✓ 117](grasp/L/demo_907_grasp.mp4) | [✓ 137](grasp/L2A/demo_907_grasp.mp4) | [video](compare/grasp/demo_907.mp4) |
| demo_1207 | [✓ 138](grasp/A1/demo_1207_grasp.mp4) | [✗ 126](grasp/A2/demo_1207_grasp.mp4) | [✓ 138](grasp/L/demo_1207_grasp.mp4) | [✗ 119](grasp/L2A/demo_1207_grasp.mp4) | [video](compare/grasp/demo_1207.mp4) |
| demo_769 | [✓ 160](grasp/A1/demo_769_grasp.mp4) | [✓ 142](grasp/A2/demo_769_grasp.mp4) | [✓ 161](grasp/L/demo_769_grasp.mp4) | [✓ 121](grasp/L2A/demo_769_grasp.mp4) | [video](compare/grasp/demo_769.mp4) |
| demo_731 | [✓ 158](grasp/A1/demo_731_grasp.mp4) | [✓ 164](grasp/A2/demo_731_grasp.mp4) | [✓ 160](grasp/L/demo_731_grasp.mp4) | [✓ 175](grasp/L2A/demo_731_grasp.mp4) | [video](compare/grasp/demo_731.mp4) |
| demo_118 | [✓ 157](grasp/A1/demo_118_grasp.mp4) | [✓ 164](grasp/A2/demo_118_grasp.mp4) | [✓ 156](grasp/L/demo_118_grasp.mp4) | [✓ 185](grasp/L2A/demo_118_grasp.mp4) | [video](compare/grasp/demo_118.mp4) |
