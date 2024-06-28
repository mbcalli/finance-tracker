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
    
class Recession:
    
    def __init__(self, starting_year, period, amount):
        self.starting_year = starting_year
        self.period = period
        self.amount = amount
        
    def get_amount_vector(self) -> np.array:
        """Returns the amount vector over the years.

        Returns:
            np.array: Amounts year by year
        """
        
        amounts = [self.amount] * self.period
        
        return np.array(amounts)
    
    def get_amount_vector_over_timeframe(self, dataframe: pd.DataFrame) -> np.array:
        """Returns the amount vector for a given timeframe, with 0s if no recession.

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
        """Takes an input net worth dataframe and applies a recession.

        Args:
            dataframe (pd.DataFrame): Net worth dataframe, with columns year and net worth.

        Returns:
            pd.DataFrame: New net worth dataframe with net worth applied
        """
        dataframe = dataframe.copy()
        
        recession = self.get_amount_vector_over_timeframe(dataframe)
        
        cumulative_recession = np.cumsum(recession)
        
        dataframe['net_worth'] = dataframe['net_worth'] - cumulative_recession
        
        return dataframe
        
class UnexpectedLifeEvent:
    
    def __init__(self, starting_year: int, ending_year: int, amount: int = None, randomize_amount=False, n_events=1):
        
        self.starting_year = np.random.randint(starting_year, ending_year, size=n_events)
        self.period = 1
        
        assert (amount is not None) or (randomize_amount)
        if randomize_amount:
            self.amount = -int(np.random.uniform(0, 100_000))
        else:
            self.amount = -amount
        
        
    def get_amount_vector(self) -> np.array:
        """Returns the amount vector over the years.

        Returns:
            np.array: Amounts year by year
        """
        
        amounts = [self.amount] * self.period
        
        return np.array(amounts)
    
    def get_amount_vector_over_timeframe(self, dataframe: pd.DataFrame) -> np.array:
        """Returns the amount vector for a given timeframe, with 0s if no recession.

        Args:
            dataframe (pd.DataFrame): Net worth dataframe

        Returns:
            np.array: Amounts over time frame
        """
        dataframe = dataframe.copy()
        
        years = dataframe['year'].to_numpy()
        
        amounts = np.zeros(len(years))
        
        for year in self.starting_year:
            starting_year_index = np.where(years == year)[0][0]
            
            amounts[starting_year_index : starting_year_index + self.period] = self.get_amount_vector()
        
        return amounts
        
        
    def apply(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Takes an input net worth dataframe and applies a recession.

        Args:
            dataframe (pd.DataFrame): Net worth dataframe, with columns year and net worth.

        Returns:
            pd.DataFrame: New net worth dataframe with net worth applied
        """
        dataframe = dataframe.copy()
        
        event = self.get_amount_vector_over_timeframe(dataframe)
        
        cumulative_event = np.cumsum(event)
        
        dataframe['net_worth'] = dataframe['net_worth'] + cumulative_event
        
        return dataframe
        
