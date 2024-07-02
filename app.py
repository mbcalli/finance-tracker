# Import data science libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import streamlit as st

from finance import *

# st.set_page_config(layout="wide")
st.set_page_config(page_title="finance-tracker")
st.title('finance-tracker')

with st.container(border=True):
    st.subheader('About')
    st.markdown("Welcome to my version of a financial events tracker. I created this tool after I realized that my financial situation doesn't fit the cookie cutter mold that most advisors use. I wanted a tool that could handle future events, so I developed `financial-tracker`. I hope to add more complexity to the tool, but right now it supports variable incomes and expenses. Keep an eye out for investments in the future. [github](https://github.com/mbcalli)")

with st.container(border=True):
    
    st.subheader('Parameters')
    
    starting_amount = st.number_input('Current Net Worth', min_value=0, value=10_000, step=1_000)
    start_year, end_year = st.slider('Year Range', min_value=2000, max_value=2100, step=1, value=(2024, 2065))

with st.container(border=True):
    
    st.markdown("### $ coming in\nIncome, investments, etc...")
    
    def add_income():
        if st.session_state.income_idxs:
            st.session_state.income_idxs.append(max(st.session_state.income_idxs) + 1)
        else:
            st.session_state.income_idxs = [1]
            
    def remove_income():
        st.session_state.income_idxs = [x for x in st.session_state.income_idxs if x != i]
    
    if 'income_idxs' not in st.session_state:
        st.session_state.income_idxs = []
    
    col1, _, _, col2 = st.columns([1,1,1,1])
    
    with col1:
        if st.button('➕ Add', key='add_income'):
            add_income()
    
    with col2:
        if st.button('🔄 Refresh', key='refresh_income'):
            pass
        
    # Display the inputs
    for i in st.session_state.income_idxs:
        with st.container(border=True):
            if st.button('➖ Remove', key=f'remove_income_{i}'):
                remove_income()
            st.text_input("Name", value=f"Income {i}", key=f'income_name_{i}')
            income_monthly = st.toggle("Monthly", value=False, key=f'income_monthly_{i}')
            income_frequency = 'month' if income_monthly else 'year'
            st.number_input(label=f'How much per {income_frequency}?', min_value=0, value=10_000, key=f'income_amount_{i}')
            st.number_input(label=f'How many years?', min_value=0, value=5, key=f'income_period_{i}')
            st.number_input(label=f'What year will it start?', min_value=0, value=2024, key=f'income_starting_year_{i}')
            st.number_input(label=f'How much will it appreciate per year? (%)', min_value=0, max_value=100, value=3, key=f'income_appreciation_{i}')
                   
with st.container(border=True):
    
    st.markdown("### $ going out\nExpenses, mortgages, debts, etc...")
    
    def add_expense():
        if st.session_state.expense_idxs:
            st.session_state.expense_idxs.append(max(st.session_state.expense_idxs) + 1)
        else:
            st.session_state.expense_idxs = [1]
            
    def remove_expense():
        st.session_state.expense_idxs = [x for x in st.session_state.expense_idxs if x != i]
    
    if 'expense_idxs' not in st.session_state:
        st.session_state.expense_idxs = []
    
    col1, _, _, col2 = st.columns([1,1,1,1])
    
    with col1:
        if st.button('➕ Add', key='add_expense'):
            add_expense()
    
    with col2:
        if st.button('🔄 Refresh', key='refresh_expense'):
            pass
        
    # Display the inputs
    for i in st.session_state.expense_idxs:
        with st.container(border=True):
            if st.button('➖ Remove', key=f'remove_expense_{i}'):
                remove_expense()
            st.text_input("Name", value=f"Expense {i}", key=f'expense_name_{i}')
            expense_monthly = st.toggle("Monthly", value=False, key=f'expense_monthly_{i}')
            expense_frequency = 'month' if expense_monthly else 'year'
            st.number_input(label=f'How much per {expense_frequency}?', min_value=0, value=10_000, key=f'expense_amount_{i}')
            st.number_input(label=f'How many years?', min_value=0, value=5, key=f'expense_period_{i}')
            st.number_input(label=f'What year will it start?', min_value=0, value=2024, key=f'expense_starting_year_{i}')
            st.number_input(label=f'How much will it appreciate per year? (%)', min_value=0, max_value=100, value=3, key=f'expense_appreciation_{i}')

with st.container(border=True):
    
    st.markdown("### Market Simulations\nSimulate market investments, including recessions within the market")
    
    def add_investment():
        if st.session_state.investment_idxs:
            st.session_state.investment_idxs.append(max(st.session_state.investment_idxs) + 1)
        else:
            st.session_state.investment_idxs = [1]
            
    def remove_investment():
        st.session_state.investment_idxs = [x for x in st.session_state.investment_idxs if x != i]
    
    if 'investment_idxs' not in st.session_state:
        st.session_state.investment_idxs = []
    
    col1, _, _, col2 = st.columns([1,1,1,1])
    
    with col1:
        if st.button('➕ Add', key='add_investment'):
            add_investment()
    
    with col2:
        if st.button('🔄 Refresh', key='refresh_investment'):
            pass
        
    # Display the inputs
    for i in st.session_state.investment_idxs:
        with st.container(border=True):
            if st.button('➖ Remove', key=f'remove_investment_{i}'):
                remove_investment()
            st.text_input("Name", value=f"Investment {i}", key=f'investment_name_{i}')
            st.number_input(label=f'How much do you start with?', min_value=0, value=10000, key=f'investment_starting_amount_{i}')
            investment_monthly = st.toggle("Monthly", value=False, key=f'investment_monthly_{i}')
            investment_frequency = 'month' if investment_monthly else 'year'
            st.number_input(label=f'How much contributed per {investment_frequency}?', min_value=0, value=10_000, key=f'investment_amount_{i}')
            st.number_input(label=f'How many years?', min_value=0, value=5, key=f'investment_period_{i}')
            st.number_input(label=f'What year will it start?', min_value=0, value=2024, key=f'investment_starting_year_{i}')
            st.number_input(label=f'How many recessions?', min_value=0, value=3, key=f'investment_n_recessions_{i}')
            st.number_input(label=f'How long will recessions last?', min_value=0, value=2, key=f'investment_recession_length_{i}')

with st.container(border=True):
    
    st.markdown("### Recessions\nSimulate Recesssions that directly hit net worth, e.g., increased cost of living from financial hardships")
    
    def add_recession():
        if st.session_state.recession_idxs:
            st.session_state.recession_idxs.append(max(st.session_state.recession_idxs) + 1)
        else:
            st.session_state.recession_idxs = [1]
            
    def remove_recession():
        st.session_state.recession_idxs = [x for x in st.session_state.recession_idxs if x != i]
    
    if 'recession_idxs' not in st.session_state:
        st.session_state.recession_idxs = []
    
    col1, _, _, col2 = st.columns([1,1,1,1])
    
    with col1:
        if st.button('➕ Add', key='add_recession'):
            add_recession()
    
    with col2:
        if st.button('🔄 Refresh', key='refresh_recession'):
            pass
        
    # Display the inputs
    for i in st.session_state.recession_idxs:
        with st.container(border=True):
            if st.button('➖ Remove', key=f'remove_recession_{i}'):
                remove_recession()
            st.text_input("Name", value=f"Recession {i}", key=f'recession_name_{i}')
            recession_monthly = st.toggle("Monthly", value=False, key=f'recession_monthly_{i}')
            recession_frequency = 'month' if recession_monthly else 'year'
            st.number_input(label=f'How much per {recession_frequency}?', min_value=0, value=10_000, key=f'recession_amount_{i}')
            st.number_input(label=f'How many years?', min_value=0, value=5, key=f'recession_period_{i}')
            st.number_input(label=f'What year will it start?', min_value=0, value=2024, key=f'recession_starting_year_{i}')

with st.container(border=True):
    
    st.markdown("### Unexpected Life Events\nNatural disasters, medical events, etc.")
    
    def add_life_event():
        if st.session_state.life_event_idxs:
            st.session_state.life_event_idxs.append(max(st.session_state.life_event_idxs) + 1)
        else:
            st.session_state.life_event_idxs = [1]
            
    def remove_life_event():
        st.session_state.life_event_idxs = [x for x in st.session_state.life_event_idxs if x != i]
    
    if 'life_event_idxs' not in st.session_state:
        st.session_state.life_event_idxs = []
    
    col1, _, _, col2 = st.columns([1,1,1,1])
    
    with col1:
        if st.button('➕ Add', key='add_life_event'):
            add_life_event()
    
    with col2:
        if st.button('🔄 Refresh', key='refresh_life_event'):
            pass
        
    # Display the inputs
    for i in st.session_state.life_event_idxs:
        with st.container(border=True):
            if st.button('➖ Remove', key=f'remove_life_event_{i}'):
                remove_life_event()
            st.text_input("Name", value=f"Life Event {i}", key=f'life_event_name_{i}')
            st.number_input(label=f'How much per event?', min_value=0, value=10_000, key=f'life_event_amount_{i}')
            st.toggle(label=f'Randomize amount per event', value=False, key=f'life_event_randomize_amount_{i}')
            st.number_input(label=f'How many events?', min_value=0, value=1, key=f'life_event_number_{i}')

def create_dataframe(start_year: int, end_year: int, starting_amount: int) -> pd.DataFrame:
    years = np.arange(start_year, end_year)
    net_worth = starting_amount * np.ones(len(years))

    df = pd.DataFrame(dict(year=years, net_worth=net_worth))
    
    return df

def apply_incomes_and_expenses(dataframe: pd.DataFrame) -> pd.DataFrame:
    
    for income_idx in st.session_state.income_idxs:
        
        income = Income(
            amount = (1 if st.session_state[f'income_monthly_{income_idx}'] == False else 12) * st.session_state[f'income_amount_{income_idx}'],
            period = st.session_state[f'income_period_{income_idx}'],
            starting_year = st.session_state[f'income_starting_year_{income_idx}'],
            appreciation = st.session_state[f'income_appreciation_{income_idx}'] / 100
        )
        
        income_name = st.session_state[f'income_name_{income_idx}']
        
        dataframe = income.apply(dataframe)
        dataframe[income_name] = income.get_amount_vector_over_timeframe(dataframe)
    
    for expense_idx in st.session_state.expense_idxs:
        
        expense = Expense(
            amount = (1 if st.session_state[f'expense_monthly_{expense_idx}'] == False else 12) * st.session_state[f'expense_amount_{expense_idx}'],
            period = st.session_state[f'expense_period_{expense_idx}'],
            starting_year = st.session_state[f'expense_starting_year_{expense_idx}'],
            appreciation = st.session_state[f'expense_appreciation_{expense_idx}'] / 100
        )
        
        expense_name = st.session_state[f'expense_name_{expense_idx}']
        
        dataframe = expense.apply(dataframe)
        dataframe[expense_name] = expense.get_amount_vector_over_timeframe(dataframe)
    
    for investment_idx in st.session_state.investment_idxs:
        
        investment = Investment(
            amount = st.session_state[f'investment_starting_amount_{investment_idx}'],
            period = st.session_state[f'investment_period_{investment_idx}'],
            starting_year = st.session_state[f'investment_starting_year_{investment_idx}'],
            yearly_contribution = (1 if st.session_state[f'investment_monthly_{investment_idx}'] == 'year' else 12) * st.session_state[f'investment_amount_{investment_idx}'],
            n_recessions = st.session_state[f'investment_n_recessions_{investment_idx}'],
            recession_length = st.session_state[f'investment_recession_length_{investment_idx}']
        )
        
        investment_name = st.session_state[f'investment_name_{investment_idx}']
        
        dataframe = investment.apply(dataframe)
        dataframe[investment_name] = investment.get_amount_vector_over_timeframe(dataframe)
    
    for recession_idx in st.session_state.recession_idxs:
        
        recession = Recession(
            amount = (1 if st.session_state[f'recession_monthly_{recession_idx}'] == False else 12) * st.session_state[f'recession_amount_{recession_idx}'],
            period = st.session_state[f'recession_period_{recession_idx}'],
            starting_year = st.session_state[f'recession_starting_year_{recession_idx}']
        )
        
        recession_name = st.session_state[f'recession_name_{recession_idx}']
        
        dataframe = recession.apply(dataframe)
        dataframe[recession_name] = recession.get_amount_vector_over_timeframe(dataframe)
        
    for life_event_idx in st.session_state.life_event_idxs:
        
        life_event = UnexpectedLifeEvent(
            amount = st.session_state[f'life_event_amount_{life_event_idx}'],
            starting_year = start_year,
            ending_year = end_year,
            randomize_amount= st.session_state[f'life_event_randomize_amount_{life_event_idx}'],
            n_events = st.session_state[f'life_event_number_{i}']
        )
        
        life_event_name = st.session_state[f'life_event_name_{life_event_idx}']
        
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
    
    for income_idx in st.session_state.income_idxs:
        income_name = st.session_state[f'income_name_{income_idx}']

        fig.add_trace(
            go.Scatter(
                x=dataframe['year'],
                y=dataframe[income_name],
                mode='lines+markers',
                name=income_name
            )
        )
        
    for expense_idx in st.session_state.expense_idxs:
        expense_name = st.session_state[f'expense_name_{expense_idx}']

        fig.add_trace(
            go.Scatter(
                x=dataframe['year'],
                y=dataframe[expense_name],
                mode='lines+markers',
                name=expense_name
            )
        )
    
    for investment_idx in st.session_state.investment_idxs:
        investment_name = st.session_state[f'investment_name_{investment_idx}']

        fig.add_trace(
            go.Scatter(
                x=dataframe['year'],
                y=dataframe[investment_name],
                mode='lines+markers',
                name=investment_name
            )
        )
        
    for recession_idx in st.session_state.recession_idxs:
        recession_name = st.session_state[f'recession_name_{recession_idx}']

        fig.add_trace(
            go.Scatter(
                x=dataframe['year'],
                y=dataframe[recession_name],
                mode='lines+markers',
                name=recession_name
            )
        )
    
    for life_event_idx in st.session_state.life_event_idxs:
        life_event_name = st.session_state[f'life_event_name_{life_event_idx}']

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