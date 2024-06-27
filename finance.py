import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Income:
    
    def __init__(self, amount: float, period: int, starting_year: int, appreciation: float = 0, fraction: float = 1):
        """Initialize an income source of amount float that occurs over a period. 
           Example, an income of $100,000 per year for 5 years, starting in 2024.

        Args:
            amount (float): Income amount, per year
            period (int): Number of years during which the income is received
            starting_year (int): Year in which income starts
            appreciation (float): Increase period over period (3% -> 0.03)
            fraction (float): Fraction of amount received, e.g., after taxes
        """
        
        self.amount = amount * fraction if fraction < 1 else amount
        self.period = period
        self.starting_year = starting_year
        self.appreciation = appreciation
        
    def get_amount_vector(self) -> np.array:
        """Returns the amount vector over the years.

        Returns:
            np.array: Amounts year by year
        """
        
        amounts = [self.amount] * self.period
        
        for i in range(1, len(amounts)):
            amounts[i] = amounts[i-1] * (1 + self.appreciation)
        
        return np.array(amounts)
    
    def get_amount_vector_over_timeframe(self, dataframe: pd.DataFrame) -> np.array:
        """Returns the amount vector for a given timeframe, with 0s if no income.

        Args:
            dataframe (pd.DataFrame): Net worth dataframe

        Returns:
            np.array: Amounts over time frame
        """
        dataframe = dataframe.copy()
        
        years = dataframe['year'].to_numpy()
        
        starting_year_index = np.where(years == self.starting_year)[0][0]
        
        amounts = np.zeros(len(years))
        
        amounts[starting_year_index : starting_year_index + self.period] = self.get_amount_vector()
        
        return amounts
        
    def apply(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Takes an input net worth dataframe and applies an income.

        Args:
            dataframe (pd.DataFrame): Net worth dataframe, with columns year and net worth.

        Returns:
            pd.DataFrame: New net worth dataframe with net worth applied
        """
        dataframe = dataframe.copy()
        
        income = self.get_amount_vector_over_timeframe(dataframe)
        
        cumulative_income = np.cumsum(income)
        
        dataframe['net_worth'] = dataframe['net_worth'] + cumulative_income
        
        return dataframe
    
class Expense:
    
    def __init__(self, amount: float, period: int, starting_year: int, appreciation: float = 0, fraction: float = 1):
        """Initialize an expense source of amount float that occurs over a period. 
           Example, a rent of $10,000 per year for 5 years, starting in 2024.

        Args:
            amount (float): Expense amount, per year
            period (int): Number of years during which the expense is incurred
            starting_year (int): Year in which income starts
            appreciation (float): Increase period over period (3% -> 0.03)
            fraction (float): Fraction of amount received, e.g., after taxes
        """
        
        self.amount = amount * fraction if fraction < 1 else amount
        self.period = period
        self.starting_year = starting_year
        self.appreciation = appreciation
        
    def get_amount_vector(self) -> np.array:
        """Returns the amount vector over the years.

        Returns:
            np.array: Amounts year by year
        """
        
        amounts = [self.amount] * self.period
        
        for i in range(1, len(amounts)):
            amounts[i] = amounts[i-1] * (1 + self.appreciation)
        
        return np.array(amounts)
    
    def get_amount_vector_over_timeframe(self, dataframe: pd.DataFrame) -> np.array:
        """Returns the amount vector for a given timeframe, with 0s if no expense.

        Args:
            dataframe (pd.DataFrame): Net worth dataframe

        Returns:
            np.array: Amounts over time frame
        """
        dataframe = dataframe.copy()
        
        years = dataframe['year'].to_numpy()
        
        starting_year_index = np.where(years == self.starting_year)[0][0]
        
        amounts = np.zeros(len(years))
        
        amounts[starting_year_index : starting_year_index + self.period] = self.get_amount_vector()
        
        return amounts
        
    def apply(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Takes an input net worth dataframe and applies an expense.

        Args:
            dataframe (pd.DataFrame): Net worth dataframe, with columns year and net worth.

        Returns:
            pd.DataFrame: New net worth dataframe with net worth applied
        """
        dataframe = dataframe.copy()
        
        expense = self.get_amount_vector_over_timeframe(dataframe)
        
        cumulative_expense = np.cumsum(expense)
        
        dataframe['net_worth'] = dataframe['net_worth'] - cumulative_expense
        
        return dataframe