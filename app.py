import json
from PIL import Image
import io
import os
import numpy as np
import streamlit as st
from streamlit import session_state
from tensorflow.keras.models import load_model
from keras.preprocessing import image as img_preprocessing
from tensorflow.keras.applications.efficientnet import preprocess_input
import base64


session_state = st.session_state
if "user_index" not in st.session_state:
    st.session_state["user_index"] = 0


def signup(json_file_path="data.json"):
    st.title("Signup Page")
    with st.form("signup_form"):
        st.write("Fill in the details below to create an account:")
        name = st.text_input("Name:")
        email = st.text_input("Email:")
        age = st.number_input("Age:", min_value=0, max_value=120)
        sex = st.radio("Sex:", ("Male", "Female", "Other"))
        password = st.text_input("Password:", type="password")
        confirm_password = st.text_input("Confirm Password:", type="password")

        if st.form_submit_button("Signup"):
            if password == confirm_password:
                user = create_account(name, email, age, sex, password, json_file_path)
                session_state["logged_in"] = True
                session_state["user_info"] = user
            else:
                st.error("Passwords do not match. Please try again.")


def check_login(username, password, json_file_path="data.json"):
    try:
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)

        for user in data["users"]:
            if user["email"] == username and user["password"] == password:
                session_state["logged_in"] = True
                session_state["user_info"] = user
                st.success("Login successful!")
                render_dashboard(user)
                return user

        st.error("Invalid credentials. Please try again.")
        return None
    except Exception as e:
        st.error(f"Error checking login: {e}")
        return None


def predict_retina(image_path, model):
    img = img_preprocessing.load_img(image_path, target_size=(224, 224))
    img_array = img_preprocessing.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make predictions using the model
    predictions = model.predict(img_array)

    # Get the predicted label
    predicted_label_index = np.argmax(predictions)

    # Define class labels
    class_labels = {
        0: "Healthy",
        1: "Mild DR",
        2: "Moderate DR",
        3: "Proliferate DR",
        4: "Severe DR"
    }

    predicted_label = class_labels[predicted_label_index]

    return predicted_label


def generate_medical_report(predicted_label):
    # Define class labels and corresponding medical information
    medical_info = {
        "Healthy": {
            "report": "Great news! It seems like the patient's eyes are healthy and free from diabetic retinopathy. Regular check-ups are recommended to keep an eye on their eye health.",
            "preventative_measures": [
                "Keep up the good work with a healthy lifestyle",
                "Keep blood sugar levels in check",
                "Regular exercise can further boost eye health",
            ],
            "precautionary_measures": [
                "Stay on top of regular eye exams",
                "Consider scheduling annual comprehensive eye check-ups to monitor any changes",
            ],
        },
        "Mild DR": {
            "report": "It looks like there are some early signs of diabetic retinopathy. Nothing to panic about, but it's important to keep a close eye on things and make some lifestyle adjustments.",
            "preventative_measures": [
                "Maintain strict control over blood sugar levels",
                "A healthy diet can make a big difference",
                "Regular exercise is key to managing diabetes",
            ],
            "precautionary_measures": [
                "Plan for more frequent eye check-ups",
                "Consider consulting with an eye specialist to discuss any concerns",
            ],
        },
        "Moderate DR": {
            "report": "The patient appears to have moderate diabetic retinopathy. This calls for immediate attention and some lifestyle changes to manage the condition effectively.",
            "preventative_measures": [
                "Keep blood sugar levels closely monitored and controlled",
                "A balanced diet is crucial for managing diabetes",
                "Regular exercise can improve circulation and eye health",
            ],
            "precautionary_measures": [
                "Don't delay regular eye exams",
                "Seek advice from an eye specialist for personalized guidance and treatment options",
            ],
        },
        "Proliferate DR": {
            "report": "It seems like the patient is dealing with proliferative diabetic retinopathy. Urgent action is needed to prevent vision loss.",
            "preventative_measures": [
                "Maintain strict control over blood sugar levels",
                "Follow a healthy diet to support overall eye health",
                "Regular exercise can help manage diabetes and improve blood flow",
            ],
            "precautionary_measures": [
                "Seek immediate consultation with an eye specialist",
                "Explore treatment options such as laser therapy or surgery to prevent further complications",
            ],
        },
        "Severe DR": {
            "report": "The patient's condition appears to be severe diabetic retinopathy. Immediate medical intervention is critical to prevent blindness.",
            "preventative_measures": [
                "Maintain strict control over blood sugar levels",
                "Follow a healthy diet and lifestyle to support overall health",
                "Prompt treatment is essential to preserve vision",
            ],
            "precautionary_measures": [
                "Seek emergency consultation with an eye specialist",
                "Consider aggressive treatment options such as laser therapy or surgery to halt disease progression",
            ],
        },
    }

    # Retrieve medical information based on predicted label
    medical_report = medical_info[predicted_label]["report"]
    preventative_measures = medical_info[predicted_label]["preventative_measures"]
    precautionary_measures = medical_info[predicted_label]["precautionary_measures"]

    # Generate conversational medical report with each section in a paragraphic fashion
    report = (
        f"Disease: {predicted_label}\n\n"  # Prepend the disease (predicted label)
        "Medical Report:\n\n"
        + medical_report
        + "\n\nPreventative Measures:\n\n- "
        + ",\n- ".join(preventative_measures)
        + "\n\nPrecautionary Measures:\n\n- "
        + ",\n- ".join(precautionary_measures)
    )

    # Return both the comprehensive report and the list of precautionary measures
    return report, precautionary_measures


def initialize_database(json_file_path="data.json"):
    try:
        # Check if JSON file exists
        if not os.path.exists(json_file_path):
            # Create an empty JSON structure
            data = {"users": []}
            with open(json_file_path, "w") as json_file:
                json.dump(data, json_file)
    except Exception as e:
        print(f"Error initializing database: {e}")



def save_retina_image(image_file, json_file_path="data.json"):
    try:
        if image_file is None:
            st.warning("No file uploaded.")
            return

        if not session_state["logged_in"] or not session_state["user_info"]:
            st.warning("Please log in before uploading images.")
            return

        # Load user data from JSON file
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)

        # Find the user's information
        for user_info in data["users"]:
            if user_info["email"] == session_state["user_info"]["email"]:
                image = Image.open(image_file)

                if image.mode == "RGBA":
                    image = image.convert("RGB")

                # Convert image bytes to Base64-encoded string
                image_bytes = io.BytesIO()
                image.save(image_bytes, format="JPEG")
                image_base64 = base64.b64encode(image_bytes.getvalue()).decode("utf-8")

                # Update the user's information with the Base64-encoded image string
                user_info["retina"] = image_base64

                # Save the updated data to JSON
                with open(json_file_path, "w") as json_file:
                    json.dump(data, json_file, indent=4)

                session_state["user_info"]["retina"] = image_base64
                return

        st.error("User not found.")
    except Exception as e:
        st.error(f"Error saving pupil image to JSON: {e}")

def create_account(name, email, age, sex, password, json_file_path="data.json"):
    try:
        # Check if the JSON file exists or is empty
        if not os.path.exists(json_file_path) or os.stat(json_file_path).st_size == 0:
            data = {"users": []}
        else:
            with open(json_file_path, "r") as json_file:
                data = json.load(json_file)

        # Append new user data to the JSON structure
        user_info = {
            "name": name,
            "email": email,
            "age": age,
            "sex": sex,
            "password": password,
            "report": None,
            "precautions": None,
            "retina":None

        }
        data["users"].append(user_info)

        # Save the updated data to JSON
        with open(json_file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)

        st.success("Account created successfully! You can now login.")
        return user_info
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON: {e}")
        return None
    except Exception as e:
        st.error(f"Error creating account: {e}")
        return None



def login(json_file_path="data.json"):
    st.title("Login Page")
    username = st.text_input("Email:")
    password = st.text_input("Password:", type="password")

    login_button = st.button("Login")

    if login_button:
        user = check_login(username, password, json_file_path)
        if user is not None:
            session_state["logged_in"] = True
            session_state["user_info"] = user
        else:
            st.error("Invalid credentials. Please try again.")

def get_user_info(email, json_file_path="data.json"):
    try:
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)
            for user in data["users"]:
                if user["email"] == email:
                    return user
        return None
    except Exception as e:
        st.error(f"Error getting user information: {e}")
        return None


def render_dashboard(user_info, json_file_path="data.json"):
    try:
        st.title(f"Welcome to the Dashboard, {user_info['name']}!")
        st.subheader("User Information:")
        st.write(f"Name: {user_info['name']}")
        st.write(f"Sex: {user_info['sex']}")
        st.write(f"Age: {user_info['age']}")

        # Open the JSON file and check for the 'retina' key
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)
            for user in data["users"]:
                if user["email"] == user_info["email"]:
                    if "retina" in user and user["retina"] is not None:
                        image_data = base64.b64decode(user["retina"])
                        st.image(Image.open(io.BytesIO(image_data)), caption="Uploaded Pupil Image", use_container_width=True)

                    if isinstance(user_info["precautions"], list):
                        st.subheader("Precautions:")
                        for precautopn in user_info["precautions"]:
                            st.write(precautopn)                    
                    else:
                        st.warning("Reminder: Please upload Pupil images and generate a report.")
    except Exception as e:
        st.error(f"Error rendering dashboard: {e}")
def fetch_precautions(user_info):
    return (
        user_info["precautions"]
        if user_info["precautions"] is not None
        else "Please upload Pupil images and generate a report."
    )


def main(json_file_path="data.json"):
    st.sidebar.title("Diabetic Retinopathy prediction system")
    page = st.sidebar.radio(
        "Go to",
        ("Signup/Login", "Dashboard", "Upload Pupil Image", "View Reports"),
        key="Diabetic Retinopathy prediction system",
    )

    if page == "Signup/Login":
        st.title("Signup/Login Page")
        login_or_signup = st.radio(
            "Select an option", ("Login", "Signup"), key="login_signup"
        )
        if login_or_signup == "Login":
            login(json_file_path)
        else:
            signup(json_file_path)

    elif page == "Dashboard":
        if session_state.get("logged_in"):
            render_dashboard(session_state["user_info"])
        else:
            st.warning("Please login/signup to view the dashboard.")

    elif page == "Upload Pupil Image":
        if session_state.get("logged_in"):
            st.title("Upload Pupil Image")
            uploaded_image = st.file_uploader(
                "Choose a Pupil image (JPEG/PNG)", type=["jpg", "jpeg", "png"]
            )
            if st.button("Upload") and uploaded_image is not None:
                st.image(uploaded_image, use_container_width=True)
                st.success("Pupil image uploaded successfully!")
                save_retina_image(uploaded_image, json_file_path)

                model = load_model("models/efficientnet_model.h5")
                condition = predict_retina(uploaded_image, model)
                report, precautions = generate_medical_report(condition)

                # Read the JSON file, update user info, and write back to the file
                with open(json_file_path, "r+") as json_file:
                    data = json.load(json_file)
                    user_index = next((i for i, user in enumerate(data["users"]) if user["email"] == session_state["user_info"]["email"]), None)
                    if user_index is not None:
                        user_info = data["users"][user_index]
                        user_info["report"] = report
                        user_info["precautions"] = precautions
                        session_state["user_info"] = user_info
                        json_file.seek(0)
                        json.dump(data, json_file, indent=4)
                        json_file.truncate()
                    else:
                        st.error("User not found.")
                st.write(report)
        else:
            st.warning("Please login/signup to upload a pupil image.")

    elif page == "View Reports":
        if session_state.get("logged_in"):
            st.title("View Reports")
            user_info = get_user_info(session_state["user_info"]["email"], json_file_path)
            if user_info is not None:
                if user_info["report"] is not None:
                    st.subheader("Retina Report:")
                    st.write(user_info["report"])
                else:
                    st.warning("No reports available.")
            else:
                st.warning("User information not found.")
        else:
            st.warning("Please login/signup to view reports.")



if __name__ == "__main__":
    initialize_database()
    main()
