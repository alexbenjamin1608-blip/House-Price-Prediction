
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import uuid

from pathlib import Path
from streamlit_cookies_controller import CookieController

from auth import (
    register_user,
    login_user,
    get_user_id,
    get_user_profile,
    save_user_profile,
    clear_profile_image,
    save_prediction,
    get_prediction_history,
    create_login_session,
    verify_login_token,
    revoke_login_session
)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==================================================
# PATH CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

COOKIE_NAME = "houseprice_login_token"
COOKIE_MAX_AGE = 30 * 24 * 60 * 60  # 30 days


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="House Price Analytics",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# COOKIE CONTROLLER
# ==================================================

cookie_controller = CookieController(key="houseprice_cookies")


# ==================================================
# WHITE AND PINK THEME
# ==================================================

st.markdown("""
<style>

.stApp {
    background-color: #FFFCFE;
    color: #352B32;
}

[data-testid="stAppViewContainer"] {
    background-color: #FFFCFE;
}

[data-testid="stMainBlockContainer"] {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

[data-testid="stSidebar"] {
    background-color: #FFF0F5;
    border-right: 1px solid #F4D5E1;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {
    color: #493440 !important;
}

h1, h2, h3 {
    color: #382632 !important;
}

p, label, li {
    color: #574650;
}

[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #F0D5E0;
    border-bottom: 3px solid #D95B91;
    border-radius: 12px;
    padding: 18px;
}

[data-testid="stMetricLabel"] {
    color: #725B68 !important;
}

[data-testid="stMetricValue"] {
    color: #C0447B !important;
    font-weight: 700;
}

.stButton button,
.stFormSubmitButton button,
.stDownloadButton button {
    background-color: #D95B91;
    color: white !important;
    border: 1px solid #D95B91;
    border-radius: 8px;
    font-weight: 600;
    padding: 9px 16px;
}

.stButton button:hover,
.stFormSubmitButton button:hover,
.stDownloadButton button:hover {
    background-color: #BD4278;
    border-color: #BD4278;
}

.stTextInput input,
.stNumberInput input,
.stTextArea textarea {
    background-color: white;
    color: #352B32 !important;
    border: 1px solid #E8C9D7;
    border-radius: 7px;
}

[data-testid="stForm"] {
    background-color: #FFFFFF;
    border: 1px solid #F0D5E0;
    border-radius: 12px;
    padding: 22px;
}

[data-baseweb="select"] > div {
    background-color: white;
    border-color: #E8C9D7;
}

.stTabs [aria-selected="true"] {
    color: #C0447B !important;
    border-bottom: 3px solid #D95B91 !important;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: #F0D5E0 !important;
    border-radius: 12px;
}

[data-testid="stHeader"] {
    background-color: #FFFCFE;
}

hr {
    border-color: #F0D5E0;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# SESSION STATE
# ==================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user_email" not in st.session_state:
    st.session_state.user_email = ""

if "user_id" not in st.session_state:
    st.session_state.user_id = None

if "login_token" not in st.session_state:
    st.session_state.login_token = None


# ==================================================
# REMEMBER ME - AUTO LOGIN
# ==================================================

if not st.session_state.logged_in:

    try:
        saved_token = cookie_controller.get(COOKIE_NAME)

        if saved_token:

            verified_user = verify_login_token(saved_token)

            if verified_user:

                restored_user_id, restored_email = verified_user

                st.session_state.logged_in = True
                st.session_state.user_id = restored_user_id
                st.session_state.user_email = restored_email
                st.session_state.login_token = saved_token

            else:

                cookie_controller.remove(COOKIE_NAME)

    except Exception:
        pass


# ==================================================
# HELPER: SAFE PROFILE IMAGE PATH
# ==================================================

def get_safe_profile_image_path(relative_path):

    if not relative_path:
        return None

    try:

        image_path = (BASE_DIR / relative_path).resolve()
        upload_path = UPLOAD_DIR.resolve()

        # Image must be inside uploads folder
        image_path.relative_to(upload_path)

        if image_path.is_file():
            return image_path

    except (ValueError, OSError):
        return None

    return None


# ==================================================
# HELPER: DELETE OLD PROFILE IMAGE
# ==================================================

def delete_old_profile_image(relative_path):

    image_path = get_safe_profile_image_path(relative_path)

    if image_path is not None:

        try:
            image_path.unlink()
        except OSError:
            pass


# ==================================================
# LOGIN AND SIGNUP
# ==================================================

if not st.session_state.logged_in:

    st.markdown("""
        <h1 style="
            text-align:center;
            color:#D95B91 !important;
            font-size:42px;
        ">
            🏠 HousePrice
        </h1>

        <p style="text-align:center;">
            Smart House Price Analytics and Prediction Platform
        </p>
    """, unsafe_allow_html=True)

    st.write("")

    left, center, right = st.columns([1, 1.5, 1])

    with center:

        with st.container(border=True):

            auth_option = st.radio(
                "Welcome",
                ["Login", "Sign Up"],
                horizontal=True
            )

            if auth_option == "Login":

                st.subheader("Welcome Back!")

                with st.form("login_form"):

                    email = st.text_input("Email Address")

                    password = st.text_input(
                        "Password",
                        type="password"
                    )

                    remember_me = st.checkbox(
                        "Remember Me (30 days)",
                        value=False
                    )

                    login_button = st.form_submit_button(
                        "Login",
                        use_container_width=True
                    )

                if login_button:

                    if not email.strip() or not password:

                        st.error(
                            "Please enter your email and password."
                        )

                    elif login_user(email, password):

                        normalized_email = email.strip().lower()
                        user_id = get_user_id(normalized_email)

                        if user_id is None:

                            st.error("User account could not be found.")

                        else:

                            st.session_state.logged_in = True
                            st.session_state.user_email = normalized_email
                            st.session_state.user_id = user_id

                            if remember_me:

                                try:

                                    token = create_login_session(
                                        user_id,
                                        days=30
                                    )

                                    st.session_state.login_token = token

                                    cookie_controller.set(
                                        COOKIE_NAME,
                                        token,
                                        max_age=COOKIE_MAX_AGE
                                    )

                                except Exception as e:

                                    st.warning(
                                        "Login successful, but Remember Me "
                                        f"could not be enabled: {e}"
                                    )

                            else:

                                # Remove any previous remembered session
                                try:

                                    old_token = cookie_controller.get(
                                        COOKIE_NAME
                                    )

                                    if old_token:
                                        revoke_login_session(old_token)

                                    cookie_controller.remove(COOKIE_NAME)

                                except Exception:
                                    pass

                            st.success("Login successful!")
                            st.rerun()

                    else:

                        st.error("Invalid email or password.")

            else:

                st.subheader("Create Your Account")

                with st.form("signup_form"):

                    email = st.text_input("Email Address")

                    password = st.text_input(
                        "Create Password",
                        type="password"
                    )

                    confirm_password = st.text_input(
                        "Confirm Password",
                        type="password"
                    )

                    signup_button = st.form_submit_button(
                        "Create Account",
                        use_container_width=True
                    )

                if signup_button:

                    if (
                        not email.strip()
                        or "@" not in email
                        or "." not in email.split("@")[-1]
                    ):

                        st.error("Please enter a valid email address.")

                    elif len(password) < 8:

                        st.error(
                            "Password must contain at least 8 characters."
                        )

                    elif password != confirm_password:

                        st.error("Passwords do not match.")

                    elif register_user(email, password):

                        st.success(
                            "Account created successfully! Please login."
                        )

                    else:

                        st.error(
                            "This email is already registered."
                        )

    st.stop()


# ==================================================
# CURRENT USER INFORMATION
# ==================================================

user_id = st.session_state.user_id
user_email = st.session_state.user_email

if user_id is None:

    st.error("User session expired. Please login again.")

    st.session_state.logged_in = False
    st.session_state.user_email = ""
    st.session_state.user_id = None

    st.rerun()


# ==================================================
# LOAD DATA AND MODEL
# ==================================================

@st.cache_data
def load_data():

    return pd.read_csv(BASE_DIR / "data" / "Housing.csv")


@st.cache_resource
def load_model():

    return joblib.load(
        BASE_DIR / "models" / "house_price_model.pkl"
    )


try:

    df = load_data()
    model = load_model()

except Exception as e:

    st.error(f"Error loading files: {e}")
    st.stop()


# ==================================================
# GET CURRENT USER HISTORY
# ==================================================

try:

    user_history = get_prediction_history(user_email)

except Exception as e:

    st.error(f"Unable to load your prediction history: {e}")
    st.stop()


# ==================================================
# MODEL EVALUATION
# ==================================================

@st.cache_data
def evaluate_models(data):

    X = data.drop("price", axis=1)
    y = data["price"]

    numeric_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    def create_preprocessor():

        return ColumnTransformer([
            (
                "numeric",
                StandardScaler(),
                numeric_features
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first"
                ),
                categorical_features
            )
        ])

    # Linear Regression

    linear_model = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("regressor", LinearRegression())
    ])

    linear_model.fit(X_train, y_train)

    linear_predictions = linear_model.predict(X_test)

    # SGD Regressor

    sgd_model = Pipeline([
        ("preprocessor", create_preprocessor()),
        (
            "regressor",
            TransformedTargetRegressor(
                regressor=SGDRegressor(
                    loss="squared_error",
                    penalty=None,
                    max_iter=5000,
                    learning_rate="constant",
                    eta0=0.01,
                    tol=1e-6,
                    random_state=42
                ),
                transformer=StandardScaler()
            )
        )
    ])

    sgd_model.fit(X_train, y_train)

    sgd_predictions = sgd_model.predict(X_test)

    results = []

    for name, predictions in [
        ("Linear Regression", linear_predictions),
        ("SGD Regressor", sgd_predictions)
    ]:

        mse = mean_squared_error(y_test, predictions)

        results.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, predictions),
            "MSE": mse,
            "RMSE": np.sqrt(mse),
            "R2 Score": r2_score(y_test, predictions)
        })

    return pd.DataFrame(results)


try:

    evaluation_df = evaluate_models(df)

except Exception as e:

    st.error(f"Model evaluation error: {e}")
    st.stop()


# ==================================================
# SIDEBAR NAVIGATION
# ==================================================

with st.sidebar:

    st.markdown(
        "<h2 style='color:#C0447B !important;'>🏠 HousePrice</h2>",
        unsafe_allow_html=True
    )

    st.caption("Property Analytics Platform")

    st.divider()

    st.write("Welcome!")

    st.write(user_email)

    st.divider()

    selected_page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Data Analysis",
            "Price Prediction",
            "Prediction History",
            "Model Performance",
            "My Profile"
        ]
    )

    st.divider()

    st.caption("PROJECT INFORMATION")
    st.caption("Housing Dataset")
    st.caption("Linear Regression")
    st.caption("SGD Regressor")
    st.caption("Python & Streamlit")

    st.divider()

    if st.button("Logout", use_container_width=True):

        # Revoke remembered login token
        try:

            logout_token = (
                st.session_state.login_token
                or cookie_controller.get(COOKIE_NAME)
            )

            if logout_token:
                revoke_login_session(logout_token)

            cookie_controller.remove(COOKIE_NAME)

        except Exception:
            pass

        st.session_state.logged_in = False
        st.session_state.user_email = ""
        st.session_state.user_id = None
        st.session_state.login_token = None

        st.rerun()


# ==================================================
# MAIN HEADER
# ==================================================

st.title("House Price Analytics")

st.caption(
    "Property Data Analysis and Machine Learning Prediction"
)

st.divider()


# ==================================================
# DASHBOARD
# ==================================================

if selected_page == "Dashboard":

    st.header("Dashboard")

    st.write(f"Welcome, {user_email}")

    st.subheader("Your Personal Statistics")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Your Total Predictions",
        len(user_history)
    )

    if user_history:

        prices = [row[3] for row in user_history]

        col2.metric(
            "Your Average Predicted Price",
            f"₹{np.mean(prices):,.0f}"
        )

        col3.metric(
            "Your Latest Prediction",
            f"₹{prices[0]:,.0f}"
        )

    else:

        col2.metric("Average Predicted Price", "₹0")
        col3.metric("Latest Prediction", "Not available")

    st.divider()

    st.subheader("Housing Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Properties", f"{len(df):,}")

    col2.metric("Total Features", df.shape[1] - 1)

    col3.metric(
        "Average Price",
        f"₹{df['price'].mean():,.0f}"
    )

    col4.metric(
        "Median Price",
        f"₹{df['price'].median():,.0f}"
    )

    st.divider()

    st.subheader("Dataset Information")

    col1, col2 = st.columns(2)

    with col1:

        with st.container(border=True):

            st.markdown("### Dataset Dimensions")

            st.write(f"Total Rows: **{df.shape[0]}**")
            st.write(f"Total Columns: **{df.shape[1]}**")

    with col2:

        with st.container(border=True):

            st.markdown("### Data Quality")

            st.write(
                f"Missing Values: **{df.isnull().sum().sum()}**"
            )

            st.write(
                f"Duplicate Rows: **{df.duplicated().sum()}**"
            )

    st.divider()

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download Housing Dataset",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="Housing_Data.csv",
        mime="text/csv"
    )


# ==================================================
# DATA ANALYSIS
# ==================================================

elif selected_page == "Data Analysis":

    st.header("Exploratory Data Analysis")

    st.write(
        "Explore the housing dataset using visual analytics."
    )

    st.subheader("House Price Distribution")

    price_bins = pd.cut(df["price"], bins=10)

    price_distribution = (
        df.groupby(price_bins, observed=False)
        .size()
        .reset_index(name="Number of Houses")
    )

    price_distribution["Price Range"] = (
        price_distribution.iloc[:, 0].astype(str)
    )

    st.bar_chart(
        price_distribution.set_index("Price Range")[
            "Number of Houses"
        ],
        color="#D95B91",
        height=350
    )

    st.divider()

    st.subheader("Area vs House Price")

    st.scatter_chart(
        df,
        x="area",
        y="price",
        color="#D95B91",
        height=350
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Bedroom Analysis")

        bedroom_data = (
            df.groupby("bedrooms")["price"]
            .mean()
            .reset_index()
        )

        st.bar_chart(
            bedroom_data.set_index("bedrooms"),
            color="#D95B91"
        )

    with col2:

        st.subheader("Furnishing Analysis")

        furnishing_data = (
            df.groupby("furnishingstatus")["price"]
            .mean()
            .reset_index()
        )

        st.bar_chart(
            furnishing_data.set_index("furnishingstatus"),
            color="#D95B91"
        )

    st.divider()

    st.subheader("Missing Values Analysis")

    missing_data = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": df.isnull().sum().values
    })

    st.dataframe(
        missing_data,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Dataset Data Types")

    data_types = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values
    })

    st.dataframe(
        data_types,
        use_container_width=True,
        hide_index=True
    )


# ==================================================
# PRICE PREDICTION
# ==================================================

elif selected_page == "Price Prediction":

    st.header("House Price Prediction")

    st.write(
        "Enter the property details to estimate its selling price."
    )

    with st.form("prediction_form"):

        st.subheader("Property Details")

        col1, col2 = st.columns(2)

        with col1:

            area = st.number_input(
                "Area (sq. ft.)",
                min_value=100,
                value=3000,
                step=100
            )

            bedrooms = st.number_input(
                "Bedrooms",
                min_value=1,
                max_value=10,
                value=3
            )

            bathrooms = st.number_input(
                "Bathrooms",
                min_value=1,
                max_value=5,
                value=2
            )

            stories = st.number_input(
                "Stories",
                min_value=1,
                max_value=4,
                value=2
            )

            parking = st.number_input(
                "Parking Spaces",
                min_value=0,
                max_value=5,
                value=1
            )

            furnishingstatus = st.selectbox(
                "Furnishing Status",
                [
                    "furnished",
                    "semi-furnished",
                    "unfurnished"
                ]
            )

        with col2:

            mainroad = st.selectbox(
                "Main Road",
                ["yes", "no"]
            )

            guestroom = st.selectbox(
                "Guestroom",
                ["yes", "no"]
            )

            basement = st.selectbox(
                "Basement",
                ["yes", "no"]
            )

            hotwaterheating = st.selectbox(
                "Hot Water Heating",
                ["yes", "no"]
            )

            airconditioning = st.selectbox(
                "Air Conditioning",
                ["yes", "no"]
            )

            prefarea = st.selectbox(
                "Preferred Area",
                ["yes", "no"]
            )

        predict_button = st.form_submit_button(
            "Predict House Price",
            use_container_width=True
        )

    if predict_button:

        input_data = pd.DataFrame([{
            "area": area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "stories": stories,
            "mainroad": mainroad,
            "guestroom": guestroom,
            "basement": basement,
            "hotwaterheating": hotwaterheating,
            "airconditioning": airconditioning,
            "parking": parking,
            "prefarea": prefarea,
            "furnishingstatus": furnishingstatus
        }])

        try:

            predicted_price = float(
                model.predict(input_data)[0]
            )

            save_prediction(
                user_email,
                area,
                bedrooms,
                bathrooms,
                predicted_price
            )

            st.success(
                "Prediction completed and saved successfully!"
            )

            st.divider()

            st.subheader("Estimated House Price")

            st.markdown(
                f"""
                <h1 style="
                    color:#C0447B !important;
                    font-size:36px;
                ">
                    ₹ {predicted_price:,.2f}
                </h1>
                """,
                unsafe_allow_html=True
            )

            st.caption(
                "Estimated price generated by the trained ML model."
            )

            result = input_data.copy()
            result["Predicted Price"] = predicted_price

            st.subheader("Property Details")

            st.dataframe(
                result,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "Download Prediction Result",
                data=result.to_csv(index=False).encode("utf-8"),
                file_name="House_Price_Prediction.csv",
                mime="text/csv"
            )

            st.info(
                "Your prediction has been saved in your personal history."
            )

        except Exception as e:

            st.error(f"Prediction Error: {e}")


# ==================================================
# PREDICTION HISTORY
# ==================================================

elif selected_page == "Prediction History":

    st.header("My Prediction History")

    st.write(
        "View and download your previously saved house price predictions."
    )

    try:

        history = get_prediction_history(user_email)

        if history:

            history_df = pd.DataFrame(
                history,
                columns=[
                    "Area (sq. ft.)",
                    "Bedrooms",
                    "Bathrooms",
                    "Predicted Price",
                    "Prediction Date"
                ]
            )

            st.success(
                f"Total Predictions: {len(history_df)}"
            )

            st.divider()

            st.subheader("Your Previous Predictions")

            st.dataframe(
                history_df,
                use_container_width=True,
                hide_index=True
            )

            st.divider()

            csv_data = history_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "Download Prediction History",
                data=csv_data,
                file_name="Prediction_History.csv",
                mime="text/csv",
                use_container_width=True
            )

        else:

            st.info(
                "You have not made any predictions yet."
            )

            st.write(
                "Go to Price Prediction to predict a house price."
            )

    except Exception as e:

        st.error(
            f"Unable to load prediction history: {e}"
        )


# ==================================================
# MODEL PERFORMANCE
# ==================================================

elif selected_page == "Model Performance":

    st.header("Machine Learning Model Performance")

    st.write(
        "Comparison of Linear Regression and SGD Regressor "
        "using the same training and testing dataset."
    )

    st.divider()

    st.dataframe(
        evaluation_df.style.format({
            "MAE": "{:,.2f}",
            "MSE": "{:,.2f}",
            "RMSE": "{:,.2f}",
            "R2 Score": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Model R² Score Comparison")

    r2_chart = evaluation_df.set_index("Model")[["R2 Score"]]

    st.bar_chart(
        r2_chart,
        color="#D95B91",
        height=350
    )

    st.caption(
        "A higher R² score indicates a better fit on this test dataset."
    )


# ==================================================
# MY PROFILE
# ==================================================

elif selected_page == "My Profile":

    st.header("My Profile")

    st.write(
        "View and update your personal account information."
    )

    profile = get_user_profile(user_id)

    if profile is None:

        full_name = ""
        phone = ""
        bio = ""
        profile_image = ""

    else:

        full_name, phone, bio, profile_image = profile

    # ----------------------------------------------
    # PROFILE PICTURE DISPLAY
    # ----------------------------------------------

    col1, col2 = st.columns([1, 2])

    with col1:

        st.subheader("Profile Picture")

        current_image_path = get_safe_profile_image_path(
            profile_image
        )

        if current_image_path:

            st.image(
                str(current_image_path),
                width=180
            )

        else:

            st.info("No profile picture uploaded yet.")

    with col2:

        st.subheader("Account Information")

        st.write(f"**Email:** {user_email}")
        st.write(f"**User ID:** {user_id}")

    # ----------------------------------------------
    # DELETE PROFILE PICTURE
    # ----------------------------------------------

    if profile_image:

        if st.button(
            "Delete Profile Picture",
            key="delete_profile_picture"
        ):

            try:

                old_image_path = clear_profile_image(user_id)

                if old_image_path:
                    delete_old_profile_image(old_image_path)

                st.success("Profile picture deleted successfully!")

                st.rerun()

            except Exception as e:

                st.error(
                    f"Unable to delete profile picture: {e}"
                )

    st.divider()

    st.subheader("Edit Your Profile")

    with st.form("profile_form"):

        new_name = st.text_input(
            "Full Name",
            value=full_name or ""
        )

        new_phone = st.text_input(
            "Phone Number",
            value=phone or ""
        )

        new_bio = st.text_area(
            "About Me",
            value=bio or "",
            height=120
        )

        uploaded_image = st.file_uploader(
            "Upload or Replace Profile Picture",
            type=["jpg", "jpeg", "png"]
        )

        save_button = st.form_submit_button(
            "Save Profile",
            use_container_width=True
        )

    if uploaded_image is not None:

        st.image(
            uploaded_image,
            caption="Selected Profile Picture",
            width=150
        )

    # ----------------------------------------------
    # SAVE / REPLACE PROFILE
    # ----------------------------------------------

    if save_button:

        new_image_path = None

        try:

            image_path_to_save = profile_image or ""

            if uploaded_image is not None:

                extension = Path(
                    uploaded_image.name
                ).suffix.lower()

                if extension not in [".jpg", ".jpeg", ".png"]:

                    st.error(
                        "Only JPG, JPEG and PNG images are allowed."
                    )

                    st.stop()

                # Generate unique image filename
                image_filename = (
                    f"profile_{user_id}_{uuid.uuid4().hex}{extension}"
                )

                full_image_path = UPLOAD_DIR / image_filename

                # Save new image first
                with open(full_image_path, "wb") as file:

                    file.write(uploaded_image.getbuffer())

                new_image_path = full_image_path

                image_path_to_save = (
                    f"uploads/{image_filename}"
                )

            # Save profile information in database
            save_user_profile(
                user_id,
                new_name.strip(),
                new_phone.strip(),
                new_bio.strip(),
                image_path_to_save
            )

            # Delete old image only after database save succeeds
            if uploaded_image is not None and profile_image:

                delete_old_profile_image(profile_image)

            st.success(
                "Your profile has been updated successfully!"
            )

            st.rerun()

        except Exception as e:

            # Remove newly uploaded file if database save fails
            if new_image_path is not None:

                try:

                    if new_image_path.exists():
                        new_image_path.unlink()

                except OSError:
                    pass

            st.error(
                f"Unable to update profile: {e}"
            )


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "House Price Analytics | Machine Learning & Data Analytics Project"
)
