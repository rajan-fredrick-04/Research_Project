from crewai import Task
from tools import tool
from agents import course_plan,assessment,manager


course_plan_task=Task(
    description=(""),
    expected_output="",
    agent=assessment,
    tools=[tool]
)

assessment_task=Task(
    description=(""),
    expected_output="",
    agent=assessment,
    tools=[tool]
)
manager_task=Task(
    description=(""),
    expected_output="",
    agent=manager,
    tools=[tool]
)

