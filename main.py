# Import data science libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import streamlit as st

from finance import Income, Expense, Recession, UnexpectedLifeEvent

# st.set_page_config(layout="wide")
st.title('finance-tracker')

with st.container(border=True):
    st.subheader('About')
    st.markdown("Welcome to my version of a financial events tracker. I created this tool after I realized that my financial situation doesn't fit the cookie cutter mold that most advisors use. I wanted a tool that could handle future events, so I developed `financial-tracker`. I hope to add more complexity to the tool, but right now it supports variable incomes and expenses. Keep an eye out for investments in the future. [github](https://github.com/mbcalli)")

with st.container(border=True):
    
    st.subheader('Parameters')
    
    starting_amount = st.number_input('Current Net Worth', min_value=0, value=10_000, step=1_000)
    start_year, end_year = st.slider('Year Range', min_value=2000, max_value=2100, step=1, value=(2024, 2065))

with st.container(border=True):
    
    # st.subheader('Income(s)')
    st.markdown("### $ coming in\nIncome, investments, etc...")
    
    if 'num_incomes' not in st.session_state:
        st.session_state.num_incomes = 0

    # Function to add more incomes
    def add_income():
        st.session_state.num_incomes += 1

    # Button to add more inputs
    if st.button('➕', key='add_income'):
        add_income()

    # Display the inputs
    for i in range(st.session_state.num_incomes):
        with st.container(border=True):
            st.text_input("label", value=f"Income {i+1}", key=f'income_name_{i+1}', label_visibility='hidden')
            income_monthly = st.toggle("Monthly", value=False, key=f'income_monthly_{i+1}')
            income_frequency = 'month' if income_monthly else 'year'
            st.number_input(label=f'How much per {income_frequency}?', min_value=0, value=10_000, key=f'income_amount_{i+1}')
            st.number_input(label=f'How many years?', min_value=0, value=5, key=f'income_period_{i+1}')
            st.number_input(label=f'What year will it start?', min_value=0, value=2024, key=f'income_starting_year_{i+1}')
            st.number_input(label=f'How much will it appreciate per year? (%)', min_value=0, max_value=100, value=3, key=f'income_appreciation_{i+1}')
                
            
with st.container(border=True):
    
    # st.subheader('Expense(s)')
    st.markdown("### $ going out\nExpenses, mortgages, debts, etc...")
    
    if 'num_expenses' not in st.session_state:
        st.session_state.num_expenses = 0

    # Function to add more expenses
    def add_expense():
        st.session_state.num_expenses += 1

    # Button to add more inputs
    if st.button('➕', key='add_expense'):
        add_expense()

    # Display the inputs
    for i in range(st.session_state.num_expenses):
        with st.container(border=True):
            st.text_input("label", value=f"Expense {i+1}", key=f'expense_name_{i+1}', label_visibility='hidden')
            expense_monthly = st.toggle("Monthly", value=False, key=f'expense_monthly_{i+1}')
            expense_frequency = 'month' if expense_monthly else 'year'
            st.number_input(label=f'How much per {expense_frequency}?', min_value=0, value=10_000, key=f'expense_amount_{i+1}')
            st.number_input(label=f'How many years?', min_value=0, value=5, key=f'expense_period_{i+1}')
            st.number_input(label=f'What year will it start?', min_value=0, value=2024, key=f'expense_starting_year_{i+1}')
            st.number_input(label=f'How much will it appreciate per year? (%)', min_value=0, max_value=100, value=3, key=f'expense_appreciation_{i+1}')
            
with st.container(border=True):
    
    # st.subheader('Recession(s)')
    st.markdown("### Recessions\nSimulate recessions")
    
    if 'num_recessions' not in st.session_state:
        st.session_state.num_recessions = 0

    # Function to add more recessions
    def add_recession():
        st.session_state.num_recessions += 1

    # Button to add more inputs
    if st.button('➕', key='add_recession'):
        add_recession()

    # Display the inputs
    for i in range(st.session_state.num_recessions):
        with st.container(border=True):
            st.text_input("label", value=f"Recession {i+1}", key=f'recession_name_{i+1}', label_visibility='hidden')
            recession_monthly = st.toggle("Monthly", value=False, key=f'recession_monthly_{i+1}')
            recession_frequency = 'month' if recession_monthly else 'year'
            st.number_input(label=f'How much per {recession_frequency}?', min_value=0, value=10_000, key=f'recession_amount_{i+1}')
            st.number_input(label=f'How many years?', min_value=0, value=5, key=f'recession_period_{i+1}')
            st.number_input(label=f'What year will it start?', min_value=0, value=2024, key=f'recession_starting_year_{i+1}')

with st.container(border=True):
    
    # st.subheader('Life_event(s)')
    st.markdown("### Unexpected Life Events\nNatural disasters, medical events, etc.")
    
    if 'num_life_events' not in st.session_state:
        st.session_state.num_life_events = 0

    # Function to add more life_events
    def add_life_event():
        st.session_state.num_life_events += 1

    # Button to add more inputs
    if st.button('➕', key='add_life_event'):
        add_life_event()

    # Display the inputs
    for i in range(st.session_state.num_life_events):
        with st.container(border=True):
            st.text_input("label", value=f"Life Event {i+1}", key=f'life_event_name_{i+1}', label_visibility='hidden')
            st.number_input(label=f'How much per event?', min_value=0, value=10_000, key=f'life_event_amount_{i+1}')
            st.toggle(label=f'Randomize amount per event', value=False, key=f'life_event_randomize_amount_{i+1}')
            st.number_input(label=f'How many events?', min_value=0, value=1, key=f'life_event_number_{i+1}')
            
def create_dataframe(start_year: int, end_year: int, starting_amount: int) -> pd.DataFrame:
    years = np.arange(start_year, end_year+1)
    net_worth = starting_amount * np.ones(len(years))

    df = pd.DataFrame(dict(year=years, net_worth=net_worth))
    
    return df

def apply_incomes_and_expenses(dataframe: pd.DataFrame) -> pd.DataFrame:
    
    for income_idx in range(st.session_state.num_incomes):
        
        frequency_factor = 1 if st.session_state[f'income_monthly_{i+1}'] == False else 12
        
        print(frequency_factor, st.session_state[f'income_monthly_{i+1}'])
        
        income = Income(
            amount = frequency_factor * st.session_state[f'income_amount_{income_idx+1}'],
            period = st.session_state[f'income_period_{income_idx+1}'],
            starting_year = st.session_state[f'income_starting_year_{income_idx+1}'],
            appreciation = st.session_state[f'income_appreciation_{income_idx+1}'] / 100
        )
        
        income_name = st.session_state[f'income_name_{income_idx+1}']
        
        dataframe = income.apply(dataframe)
        dataframe[income_name] = income.get_amount_vector_over_timeframe(dataframe)
    
    for expense_idx in range(st.session_state.num_expenses):
        
        frequency_factor = 1 if st.session_state[f'expense_monthly_{i+1}'] == False else 12
        
        expense = Expense(
            amount = frequency_factor * st.session_state[f'expense_amount_{expense_idx+1}'],
            period = st.session_state[f'expense_period_{expense_idx+1}'],
            starting_year = st.session_state[f'expense_starting_year_{expense_idx+1}'],
            appreciation = st.session_state[f'expense_appreciation_{expense_idx+1}'] / 100
        )
        
        expense_name = st.session_state[f'expense_name_{expense_idx+1}']
        
        dataframe = expense.apply(dataframe)
        dataframe[expense_name] = expense.get_amount_vector_over_timeframe(dataframe)
    
    for recession_idx in range(st.session_state.num_recessions):
        
        frequency_factor = 1 if st.session_state[f'recession_monthly_{i+1}'] == False else 12
        
        recession = Recession(
            amount = frequency_factor * st.session_state[f'recession_amount_{recession_idx+1}'],
            period = st.session_state[f'recession_period_{recession_idx+1}'],
            starting_year = st.session_state[f'recession_starting_year_{recession_idx+1}']
        )
        
        recession_name = st.session_state[f'recession_name_{recession_idx+1}']
        
        dataframe = recession.apply(dataframe)
        dataframe[recession_name] = recession.get_amount_vector_over_timeframe(dataframe)
        
    for life_event_idx in range(st.session_state.num_life_events):
        
        frequency_factor = 1 if st.session_state[f'life_event_monthly_{i+1}'] == False else 12
        
        life_event = UnexpectedLifeEvent(
            amount = frequency_factor * st.session_state[f'life_event_amount_{life_event_idx+1}'],
            starting_year = start_year,
            ending_year = end_year,
            randomize_amount= st.session_state[f'life_event_randomize_amount_{life_event_idx+1}'],
            n_events = st.session_state[f'life_event_number_{i+1}']
        )
        
        life_event_name = st.session_state[f'life_event_name_{life_event_idx+1}']
        
        dataframe = life_event.apply(dataframe)
        dataframe[life_event_name] = life_event.get_amount_vector_over_timeframe(dataframe)
      
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
        
    for recession_idx in range(st.session_state.num_recessions):
        recession_name = st.session_state[f'recession_name_{recession_idx+1}']

        fig.add_trace(
            go.Scatter(
                x=dataframe['year'],
                y=dataframe[recession_name],
                mode='lines+markers',
                name=recession_name
            )
        )
    
    for life_event_idx in range(st.session_state.num_life_events):
        life_event_name = st.session_state[f'life_event_name_{life_event_idx+1}']

        fig.add_trace(
            go.Scatter(
                x=dataframe['year'],
                y=dataframe[life_event_name],
                mode='lines+markers',
                name=life_event_name
            )
        )
        
    fig.update_layout(
        xaxis=dict(
            title='Year'
        ),
        yaxis=dict(
            title='Net Worth ($)'
        )
    )
    
    return fig

with st.container(border=True):
    
    st.subheader('Forecast')
    
    filter_start_year, filter_end_year = st.slider('Filter Year Range', min_value=start_year, max_value=end_year, step=1, value=(start_year, end_year))
    st.plotly_chart(generate(start_year, end_year, starting_amount, filter_start_year, filter_end_year))