# 数据切分与画像分配统计

- train: 18014 条；条件 {'no_profile': 9004, 'irrelevant': 4505, 'aligned': 4505}；难度 {'easy': 3325, 'medium': 10601, 'hard': 4088}
- eval: 1092 条；条件 {'no_profile': 548, 'irrelevant': 272, 'aligned': 272}；难度 {'medium': 585, 'hard': 342, 'easy': 165}
- aligned 画像来源: {'own': 723, 'pool': 4054}
- irrelevant 未配到: 0
- dev: 1050 条
- 画像池: 4666 份（泄漏标记 773，clean 3893）
