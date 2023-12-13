import streamlit as st

# Define a function 'app()' which accepts 'census_df' as an input.
def app(diabetes_df):
  # Set the title to the home page contents.
  st.title('Early Diabetes Predictoin Web App')  
  # Provide a brief description for the web app.
  st.markdown("<p style='color:red;font-size:25px'>Diabetes is a chronic (long-lasting) health condition that affects how your body turns food into energy. There isn’t a cure yet for diabetes, but losing weight, eating healthy food, and being active can really help in reducing the impact of diabetes. This Web app will help you to predict whether a person has diabetes or is prone to get diabetes in future by analysing the values of several features using the Decision Tree Classifier. </p>", unsafe_allow_html = True)
  # Add the 'beta_expander' to view full dataset 
  st.header("View Data")
  with st.beta_expander("View Data"):
      st.table(diabetes_df)

  st.subheader("Columns Description:")
  beta_col1, beta_col2, beta_col3 = st.beta_columns(3)
  # Add a checkbox in the first column. Display the column names of 'diabetes_df' on the click of checkbox.
  with beta_col1:
    if st.checkbox("Show all column names"):
      st.table(list(diabetes_df.columns))

  # Add a checkbox in the second column. Display the column data-types of 'diabetes_df' on the click of checkbox.
  with beta_col2: 
    if st.checkbox("View column data-type"):
        st.table(diabetes_df.dtypes)

  # Add a checkbox in the third column followed by a selectbox which accepts the column name whose data needs to be displayed.
  with beta_col3:
    if st.checkbox("View column data"):
        column_data = st.selectbox('Select column', tuple(diabetes_df.columns))
        st.write(diabetes_df[column_data])