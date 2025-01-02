from crewai import Task
from tools import tool
from agents import course_plan,assessment,manager


course_plan_task = Task(
    description=(
        "Generate a detailed course plan by breaking down the provided unit contents into 1-hour sessions. "
        "Ensure even distribution of teaching hours, prioritize core concepts, and maintain a logical flow for student learning."
    ),
    expected_output=(
        "A structured course plan with individual 1-hour sessions, including session numbers, topics, content coverage, and alignment with unit objectives."
    ),
    agent=course_plan,
    tools=[tool]
)

assessment_task = Task(
    description=(
        "Design comprehensive and relevant assessments based on the provided unit contents and learning outcomes. "
        "Categorize assessments into types such as quizzes, written assignments, hands-on tasks, and visual assessments, and rank them by suitability."
    ),
    expected_output=(
        "A detailed assessment plan with categorized and ranked assessments for each unit, aligned with the course content and learning goals."
    ),
    agent=assessment,
    tools=[tool]
)

manager_task = Task(
    description=(
        "Combine the course plan and assessment outputs to generate a comprehensive final report. "
        "Ensure alignment between teaching sessions and assessments, creating a cohesive educational framework."
    ),
    expected_output=(
        "A unified final report integrating the course plan with corresponding assessments. "
        "The report should provide structured sessions alongside relevant and prioritized evaluation methods for a complete educational framework."
    ),
    agent=manager,
    tools=[tool]
)

