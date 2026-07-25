"""Build the AW-Extend evaluation plan: 19 templates x 2 instances.

# note (luojiaxuan): AW-Extend 的 19 个模板来自 AgentProg 的 tasks_long.txt。
# 每个模板取 task_index 0 与 1 两个实例,与主表 "先对模板内实例求平均、再对模板做
# macro average" 的统计口径一致 —— 分母是 19 个模板,不是 38 条 episode。
"""

import json
import sys

TASKS = [
    "ContactsAddContactAndSms",
    "ContactsAddMultipleContactsAndSms",
    "MarkorFetchMultipleNotesAndSms",
    "MarkorFetchNoteAndSms",
    "MarkorTodoList",
    "SimpleCalendarDeleteEventsLong",
    "SimpleCalendarDeleteEventsOnRelativeDayLong",
    "SimpleCalendarDeleteEventsSuperLong",
    "SimpleCalendarDeleteEventsOnRelativeDaySuperLong",
    "ExpenseAddMultipleLong",
    "ExpenseAddMultipleSuperLong",
    "ExpenseDeleteMultipleSuperLong",
    "ExpenseDeleteMultipleLong2",
    "ExpenseDeleteMultipleSuperLong2",
    "RecipeAddMultipleRecipesLong",
    "RecipeAddMultipleRecipesSuperLong",
    "RecipeDeleteMultipleRecipesLong",
    "RecipeDeleteMultipleRecipesSuperLong",
    "MarkorMergeNotesLong",
]

INSTANCES_PER_TEMPLATE = 2

if len(TASKS) != 19:
    raise SystemExit(f"expected 19 AW-Extend templates, got {len(TASKS)}")

plan = [
    {"task_type": task, "task_index": index}
    for task in TASKS
    for index in range(INSTANCES_PER_TEMPLATE)
]

out = sys.argv[1] if len(sys.argv) > 1 else "awextend_plan.json"
with open(out, "w", encoding="utf-8") as handle:
    json.dump(plan, handle, ensure_ascii=False, indent=1)

print(f"templates: {len(TASKS)} | instances each: {INSTANCES_PER_TEMPLATE}")
print(f"episodes: {len(plan)} -> {out}")
