import streamlit as st  # Ifor building web applications
import pandas as pd  # for data manipulation and analysis
import matplotlib.pyplot as plt  # or plotting
import seaborn as sns  # for statistical data visualization
import numpy as np  # for numerical operations
import plotly.express as px
from joblib import load
from sklearn.preprocessing import LabelEncoder
##########################################


# Defining the title and description
TITLE = "Streamlit Demonstration"
DESCRIPTION = "Survivorship Analysis from the Titanic Dataset"

# Displaying the title and description in the main content area
st.title(TITLE)
st.markdown(DESCRIPTION)

# Displaying the title in the sidebar
st.sidebar.title(TITLE)
# Adding a markdown message in the sidebar
st.sidebar.markdown("Choose your Visualizations")
##########################################


@st.cache_data                         # Decorating the function with st.cache
def load_data():
	data = pd.read_csv("titanic.csv")
	data = data.drop(columns=["Name"])

	return data
data = load_data()                      # Calling the load_data() function
st.write(data)                          # Displaying the data
##########################################


@st.cache_data
def subset_by_sex(df):
	df=df.groupby(['Sex', 'Survived',]).count().iloc[:, :1].rename(columns={"Pclass": "Count"})
	df=df.reset_index()
	df['Survived']=df['Survived'].astype(str)
	return df

if st.sidebar.checkbox("Male and Female Comparative", True):
	sex_subset=subset_by_sex(data)
	st.header("Male andFemale Comparative")
	if st.checkbox("Show Table", False):
		st.write(sex_subset)
		pass
##########################################


plot_choice = st.radio("Choose Your Plotting Tool", ['pyplot', 'plotly'])
st.header("Survivors and Non-Survivors by Sex")
if plot_choice == 'pyplot':
	plt_figure = plt.figure()
	sns.countplot(data=data, x="Sex", palette=['royalblue','orangered'])
	sns.despine()
	st.pyplot(plt_figure)
else:
	px_figure = px.bar(sex_subset, y="Count", x="Sex", color="Survived", barmode="group")
	st.plotly_chart(px_figure)
##########################################



@st.cache_data
def get_fares_min_max(df):
	min_fares = df['Fare'].min()
	max_fares = df['Fare'].max()
	return min_fares, max_fares

def subset_by_fare(df, lower, upper):
	df = df[df['Fare'] >= lower]
	df = df[df['Fare'] <= upper]
	return df

if st.sidebar.checkbox("Analysis by Fares", False):
	st.header("Analysis by Fares")
	min_fares, max_fares = get_fares_min_max(data)
	lower, upper = st.slider("Fare range", min_fares, max_fares, (100.0, float(max_fares/2)))
	fare_subset =subset_by_fare(data, lower, upper)
	st.write(f"Number of Observations: {fare_subset.shape[0]}")
	if st.checkbox("Show Table", False, key =0):
		st.write(fare_subset)
##########################################

plt_figure = plt.figure()
sns.countplot(data=fare_subset, x="Survived", palette=["royalblue", "orangered"])
sns.despine()
plt.title("Survivors Vs Non-Survivors")
st.pyplot(plt_figure)

def subset_by_family(df, col, values): 
	if values:
		df = df[df[col].isin(values)]
	else:
		st.error("Please Select Values.")
	return df
##########################################

if st.sidebar.checkbox("Family Abord Analysis"):
	st.header("Family Abord Analysis")
	col_1, col_2 = st.columns(2)

	col_1.header("Siblings/Spouses abord")
	sibli_spou = col_1.multiselect("Quantities", np.sort(data["Siblings/Spouses Aboard"].unique()))
	sibling_spouse_df = subset_by_family(data, "Siblings/Spouses Aboard", sibli_spou)
	plt_figure = plt.figure()
	sns.countplot(data=sibling_spouse_df, x="Survived", palette=["royalblue", "orangered"])
	sns.despine()
	plt.title("Survivors Vs Non-Survivors")
	col_1.pyplot(plt_figure)
##########################################
	col_2.header("Parents/Children abord")
	par_chil = col_2.multiselect("Quantities", np.sort(data["Parents/Children Aboard"].unique()))
	parents_children_df = subset_by_family(data, "Parents/Children Aboard", par_chil)
	plt_figure = plt.figure()
	sns.countplot(data=parents_children_df, x="Survived", palette=["royalblue", "orangered"])
	sns.despine()
	plt.title("Survivors Vs Non-Survivors")
	col_2.pyplot(plt_figure)


################## Deploying and interactive ML Model on Streamlit ################
# label_encoder = LabelEncoder()

###########################################
clf = load('rf.joblib')  # Loaded the Model
###########################################
if st.sidebar.checkbox("ML: Would I Survive the Titanic Disaster", False):
	st.header("Would I Survive the Disaster?")



	myForm = st.form("Ml_form")

	P_class = myForm.selectbox("Class, (1 is the Highest and 3 Lowest)", [1, 2, 3]) # 1st
	sex = myForm.radio("Sex", ["Male", "Female"]) # 2nd

	if sex == "Male":                        # 3rd
		sex_male = 1
		sex_female = 0

	elif sex == "Female":
		sex_male = 0
		sex_female = 1

	else:
		sex_male = None
		sex_female = None
		myForm.error("Could not find Gender input")

###########################################
age = myForm.slider(label="Age", min_value=0, max_value=120)                                 # 4th
sibling_spouse = myForm.selectbox('Siblings and Spouses Aboard', [i for i in range(0, 10)])  # 5th
parents_children = myForm.selectbox('Parents and Children Aboard', [i for i in range(0, 8)]) # 6th
min_fares, max_fares = get_fares_min_max(data)
fare = myForm.slider("Fare", min_fares, max_fares, 150.0)                                    # 7th

####### Load the Model at this point#######
submit = myForm.form_submit_button(label="Submit!")

if submit:
	my_data = np.array([P_class, age, sibling_spouse, parents_children,
                            fare, sex_female, sex_male])
	try:
		predictions = clf.predict([my_data])

	except Exception as e:
		myForm.error(f"Could not prediction. {e}")

	if predictions[0] == 0:
		st.error("Sorry! You Would Not Survive")

	else:
		st.warning("Woot! You Would Survive")







# try:
#     prediction = clf.predict([my_data])
#     except Exception as e:

#         myForm.error(f"Could not make prediction. {e}")
#     if prediction[0] == 0:

#         st.error("Sorry! You would not survive :(")
#     else:


#         st.warning("Woot! You would survive!")
	
        
 
	# my_data = np.array([P_class, sex, age, sibling_spouse, parents_children,
	# 	                fare, Sex_male, Sex_female])

	# enoded_data = label_encoder.transform(my_data)
	# reshape_data = enoded_data.reshape(1, 8)


	# ######Make the Predictions using the ML Model######

	# try:
	# 	prediction = clf.predict(reshape_data)
	# except Exception as e:
	# 	myForm.error(f"Could not make predictions. {e}")
	# if prediction[0] == 0:
	# 	st.error("Sorry you would not Survive :(" )
	# else:
	# 	st.warning("Woot! You Would Survive")
	


pass
	




































