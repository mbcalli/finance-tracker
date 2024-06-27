# Import data science libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import streamlit as st

from finance import Income, Expense

st.set_page_config(layout="wide")
st.title('Finance Projection')

starting_amount = st.number_input('Current Amount', min_value=0, value=10_000, step=1_000)
start_year, end_year = st.slider('Year Range', min_value=2000, max_value=2100, step=1, value=(2024, 2065))

with st.container(border=True):
    if 'num_incomes' not in st.session_state:
        st.session_state.num_incomes = 0

    # Function to add more incomes
    def add_income():
        st.session_state.num_incomes += 1

    # Button to add more inputs
    if st.button('Add income'):
        add_income()

    # Display the inputs
    for i in range(st.session_state.num_incomes):
        with st.container(border=True):
            st.text_input("label", value="", placeholder=f"Income {i+1}", key=f'income_name_{i+1}', label_visibility=False)
            st.number_input(label='Amount', min_value=0, value=10_000, key=f'income_amount_{i+1}')
            st.number_input(label='Period', min_value=0, value=5, key=f'income_period_{i+1}')
            st.number_input(label='Starting Year', min_value=0, value=2024, key=f'income_starting_year_{i+1}')
            st.number_input(label='Appreciation (%)', min_value=0, max_value=100, value=3, key=f'income_appreciation_{i+1}')
            
with st.container(border=True):
    if 'num_expenses' not in st.session_state:
        st.session_state.num_expenses = 0

    # Function to add more expenses
    def add_expense():
        st.session_state.num_expenses += 1

    # Button to add more inputs
    if st.button('Add expense'):
        add_expense()

    # Display the inputs
    for i in range(st.session_state.num_expenses):
        with st.container(border=True):
            st.text_input("label", value="", placeholder=f"Expense {i+1}", key=f'expense_name_{i+1}', label_visibility=False)
            st.number_input(label='Amount', min_value=0, value=10_000, key=f'expense_amount_{i+1}')
            st.number_input(label='Period', min_value=0, value=5, key=f'expense_period_{i+1}')
            st.number_input(label='Starting Year', min_value=0, value=2024, key=f'expense_starting_year_{i+1}')
            st.number_input(label='Appreciation (%)', min_value=0, max_value=100, value=3, key=f'expense_appreciation_{i+1}')
         
def create_dataframe(start_year: int, end_year: int, starting_amount: int) -> pd.DataFrame:
    years = np.arange(start_year, end_year+1)
    net_worth = starting_amount * np.ones(len(years))

    df = pd.DataFrame(dict(year=years, net_worth=net_worth))
    
    return df

def apply_incomes_and_expenses(dataframe: pd.DataFrame) -> pd.DataFrame:
    
    for income_idx in range(st.session_state.num_incomes):
        
        income = Income(
            amount = st.session_state[f'income_amount_{income_idx+1}'],
            period = st.session_state[f'income_period_{income_idx+1}'],
            starting_year = st.session_state[f'income_starting_year_{income_idx+1}'],
            appreciation = st.session_state[f'income_appreciation_{income_idx+1}'] / 100
        )
        
        income_name = st.session_state[f'income_name_{income_idx+1}']
        
        dataframe = income.apply(dataframe)
        dataframe[income_name] = income.get_amount_vector_over_timeframe(dataframe)
    
    for expense_idx in range(st.session_state.num_expenses):
        
        expense = Expense(
            amount = st.session_state[f'expense_amount_{expense_idx+1}'],
            period = st.session_state[f'expense_period_{expense_idx+1}'],
            starting_year = st.session_state[f'expense_starting_year_{expense_idx+1}'],
            appreciation = st.session_state[f'expense_appreciation_{expense_idx+1}'] / 100
        )
        
        expense_name = st.session_state[f'expense_name_{expense_idx+1}']
        
        dataframe = expense.apply(dataframe)
        dataframe[expense_name] = expense.get_amount_vector_over_timeframe(dataframe)
        
    return dataframe

def generate(start_year: int, end_year: int, starting_amount: int, filter_start_year: int, filter_end_year: int) -> go.Figure:
    
    dataframe = create_dataframe(start_year, end_year, starting_amount)
    dataframe = apply_incomes_and_expenses(dataframe)
    
    dataframe = dataframe[(dataframe.year >= filter_start_year) & (dataframe.year <= filter_end_year)]
    
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=dataframe['year'],
            y=dataframe['net_worth'],
            mode='lines+markers',
            name='Net Worth'
        )
    )
    
    for income_idx in range(st.session_state.num_incomes):
        income_name = st.session_state[f'income_name_{income_idx+1}']

        fig.add_trace(
            go.Scatter(
                x=dataframe['year'],
                y=dataframe[income_name],
                mode='lines+markers',
                name=income_name
            )
        )
        
    for expense_idx in range(st.session_state.num_expenses):
        expense_name = st.session_state[f'expense_name_{expense_idx+1}']

        fig.add_trace(
            go.Scatter(
                x=dataframe['year'],
                y=dataframe[expense_name],
                mode='lines+markers',
                name=expense_name
            )
        )
    
    return fig

with st.container(border=True):
    filter_start_year, filter_end_year = st.slider('Filter Year Range', min_value=start_year, max_value=end_year, step=1, value=(start_year, end_year))
    st.plotly_chart(generate(start_year, end_year, starting_amount, filter_start_year, filter_end_year))