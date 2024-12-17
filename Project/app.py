import streamlit as st
from assessment_gen import assessment_generator
from course_plan_gen import course_plan_generator

# Welcome Page
def Home_page():
    st.title("Welcome to the Project!")
    st.write("""
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. 
    Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. 
    Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. 
    Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. 
    Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
    Curabitur pretium tincidunt lacus. Nulla gravida orci a odio. Nullam varius, turpis et commodo pharetra, est eros bibendum elit.
    """)

# Main function to manage navigation
def main():
    # Initialize session state for page navigation
    if "page" not in st.session_state:
        st.session_state.page = "Home"

    # Sidebar for navigation buttons
    st.sidebar.title("Navigation")
    if st.sidebar.button("Home"):
        st.session_state.page = "Home"
    if st.sidebar.button("Course Plan Generator"):
        st.session_state.page = "Course Plan Generator"
    if st.sidebar.button("Assessment Generator"):
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
