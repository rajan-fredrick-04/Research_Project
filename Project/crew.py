from crewai import Crew,Process
from agents import course_plan,assessment,manager
from tasks import assessment_task,course_plan_task,manager_task

crew=Crew(
    agents=[course_plan,assessment],
    tasks=[assessment_task,course_plan_task],
    manager_agent=manager,
    process=Process.hierarchical
)
result = crew.kickoff()
print(result)