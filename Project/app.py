import streamlit as st
from assessment_gen import assessment_generator
from course_plan_gen import course_plan_generator

# Welcome Page
def Home_page():
    st.title("Content Aware prompt engineering for course plan generation")
    st.divider()
    st.write("""
    Welcome to the Course and Assessment Plan Generator!
In today's rapidly evolving educational landscape, aligning course plans with well-defined learning outcomes is critical for effective teaching. This AI-powered tool simplifies the process by leveraging Bloom’s Taxonomy to structure learning objectives and assessments.
Educators often face challenges in manually designing detailed course plans that integrate session-wise content, assessments, and pedagogy. This tool automates the generation of course plans and assessments, ensuring they match specific Course Outcomes (COs).
By streamlining this traditionally time-intensive task, our solution allows educators to focus on delivering impactful learning experiences while adhering to pedagogical standards.
Simply input your course outcomes, and let AI handle the rest—effortlessly aligning content, objectives, and assessments for optimal learning.

    """)

# Main function to manage navigation
def main():
    # Initialize session state for page navigation
    if "page" not in st.session_state:
        st.session_state.page = "Home"

    # Sidebar for navigation buttons
    st.sidebar.title("Navigation")
    if st.sidebar.button("Home",use_container_width=True):
        st.session_state.page = "Home"
    if st.sidebar.button("Course Plan Generator",use_container_width=True):
        st.session_state.page = "Course Plan Generator"
    if st.sidebar.button("Assessment Generator",use_container_width=True):
        st.session_state.page = "Assessment Generator"


    # Render the appropriate page
    if st.session_state.page == "Home":
        Home_page()
    elif st.session_state.page == "Course Plan Generator":
        course_plan_generator()
    elif st.session_state.page == "Assessment Generator":
        assessment_generator()

if __name__ == "__main__":
    main()
