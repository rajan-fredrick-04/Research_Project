import streamlit as st
from assessment_gen import assessment_generator
from course_plan_gen import course_plan_generator

# Injecting custom CSS for styling
st.markdown("""
    <style>
    
  

    /* Header and Title for Main Content */
    .stTitle {
        font-size: 36px;
        color: #4CAF50;
        font-weight: bold;
        text-align: center;
        padding: 30px;
        background-color: #ffffff;
        border-radius: 8px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
    }

    .stDivider {
        border-top: 2px solid #4CAF50;
        margin: 20px 0;
    }

    .stText {
        font-size: 18px;
        line-height: 1.8;
        color: #555;
        text-align: justify;
    }

    /* Sidebar Styling */
    .sidebar .sidebar-content {
        background-color: #4CAF50;
        color: white;
        padding: 20px;
    }

    .sidebar .sidebar-header {
        font-size: 24px;
        color: white;
        font-weight: 600;
        padding-bottom: 15px;
        border-bottom: 2px solid #fff;
        margin-bottom: 20px;
    }

    .sidebar .sidebar-button {
        background-color: #45a049;
        color: white;
        border-radius: 5px;
        padding: 12px 20px;
        margin-bottom: 15px;
        width: 100%;
    }

    .sidebar .sidebar-button:hover {
        background-color: #387038;
    }

    .sidebar .sidebar-footer {
        color: white;
        font-size: 12px;
        margin-top: 30px;
        text-align: center;
    }

    /* Main Content - Image Background */
    .main-content {
        background-image: url('https://cdn.pixabay.com/photo/2014/06/28/12/35/books-378903_1280.jpg');
        background-size: cover;
        background-position: center;
        padding: 100px 20px;
        color: white;
        text-align: center;
        min-height: 100vh;
    }

    .main-content h1 {
        font-size: 48px;
        font-weight: bold;
        margin-bottom: 20px;
    }

    .main-content p {
        font-size: 20px;
        margin-bottom: 20px;
        line-height: 1.8;
    }

    .stButton {
        background-color: transparent;
        color: white;
        border-radius: 5px;
        font-size: 16px;
        margin-top: 20px;
    }

    .stButton:hover {
        background-color: transparent;
    }
    </style>
""", unsafe_allow_html=True)

# Welcome Page
def Home_page():
    st.markdown("""
        <style>
        stApp {
            margin: 0;
            padding: 0;
            height: 100vh;
        }

        .main-content {
            background-image: url('https://cdn.pixabay.com/photo/2014/06/28/12/35/books-378903_1280.jpg'); 
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center center;
            width:100vh;
            min-height: 100vh; 
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center; /* Horizontally center the content */
            padding: 20px;
            color: white;
            text-align: center;
        }

        .main-content h1 {
            font-size: 48px;
            font-weight: bold;
            margin-bottom: 20px;
        }

        .main-content p {
            font-size: 20px;
            margin-bottom: 20px;
            line-height: 1.8;
        }
        </style>

        <div class="main-content">
            <h1>Content Aware Prompt Engineering for Course Plan Generation</h1>
            <p>Welcome to the Course and Assessment Plan Generator! In today's rapidly evolving educational landscape, aligning course plans with well-defined learning outcomes is critical for effective teaching. This AI-powered tool simplifies the process by leveraging Bloom’s Taxonomy to structure learning objectives and assessments.
            Educators often face challenges in manually designing detailed course plans that integrate session-wise content, assessments, and pedagogy. This tool automates the generation of course plans and assessments, ensuring they match specific Course Outcomes (COs).
            By streamlining this traditionally time-intensive task, our solution allows educators to focus on delivering impactful learning experiences while adhering to pedagogical standards. Simply input your course outcomes, and let AI handle the rest—effortlessly aligning content, objectives, and assessments for optimal learning.</p>
        </div>
    """, unsafe_allow_html=True)


# Main function to manage navigation
def main():
    # Initialize session state for page navigation
    if "page" not in st.session_state:
        st.session_state.page = "Home"

    # Sidebar for navigation buttons
    st.sidebar.title("Navigation")
    st.sidebar.header("Welcome to the Tool")
    if st.sidebar.button("Home", use_container_width=True):
        st.session_state.page = "Home"
    if st.sidebar.button("Course Plan Generator", use_container_width=True):
        st.session_state.page = "Course Plan Generator"
    if st.sidebar.button("Assessment Generator", use_container_width=True):
        st.session_state.page = "Assessment Generator"

    # Sidebar Footer
    st.sidebar.markdown("""
        <div class="sidebar-footer">
            <p>&copy; 2025 Course Plan Generator. All rights reserved.</p>
        </div>
    """, unsafe_allow_html=True)

    # Render the appropriate page
    if st.session_state.page == "Home":
        Home_page()
    elif st.session_state.page == "Course Plan Generator":
        course_plan_generator()
    elif st.session_state.page == "Assessment Generator":
        assessment_generator()

if __name__ == "__main__":
    main()
