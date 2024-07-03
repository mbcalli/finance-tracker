import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

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
        
class Laplace:
    
    def __init__(self, loc: float, scale: float):
        """This class contains a nicer version of scipy stats Laplace distribution.

        Args:
            loc (float): The mean of the distribution
            scale (float): The scale of the distribution
        """
        self.loc = loc
        self.scale = scale
        self.dist = stats.laplace(loc=self.loc, scale=self.scale)
        
    def get_rvs(self, n: int):
        """Returns an array of n random variables taken from the distribution.

        Args:
            n (int): number of samples to return

        Returns:
            np.array: samples taken from distribution
        """
        return self.dist.rvs(n)
    
class Investment:
    
    def __init__(self, amount: float, period: int, starting_year: int, yearly_contribution: float = 0, n_recessions: int = 0, recession_length: int = 1, n_simulations: int = 1):
        """This class mimics the fluctuations of the stock market by taking samples of Laplace distributions to be the day-to-day gains or losses.
        Handles recession simulations, which are taken from the 2007-2008 recession. Laplace parameters are hard-coded.

        Args:
            amount (float): starting amount of money
            period (int): number of years
            starting_year (int): starting year of investment
            yearly_contribution (float, optional): amount to contribute annualy. Defaults to 0.
            n_recessions (int, optional): number of recessions to simulate. Defaults to 0.
            recession_length (int, optional): length of recessions. Defaults to 1 year.
        """
        assert (recession_length > 0) & (n_recessions >= 0)
        
        self.amount = amount
        self.period = period
        self.starting_year = starting_year
        self.ending_year = self.starting_year + self.period
        self.yearly_contribution = yearly_contribution
        self.n_recessions = n_recessions
        self.recession_length = recession_length
        self.n_simulations = n_simulations
        
        if self.n_recessions > 0:
            self.recession_starting_years = np.random.choice(np.arange(self.starting_year, self.ending_year), self.n_recessions, replace=False)
            self.recession_ending_years = self.recession_starting_years + self.recession_length
            self.recession_ending_years[self.recession_ending_years > self.ending_year] = self.ending_year
        else:
            self.recession_starting_years = []
            self.recession_ending_years = []
        
        self.normal_dist = Laplace(
            loc = np.float64(0.8999607008738539), 
            scale = np.float64(12.997665073156822)
        )
        
        self.recession_dist = Laplace(
            loc = np.float64(-1.4800001575100807), 
            scale = np.float64(13.99833543746421)
        )
        
        self.dist_start = 5475.089844
        
    def get_rvs(self):
        """Returns the random variables taken from the underlying distributions; accounts for recessions

        Returns:
            np.array: random variables
        """
        
        n_days = self.period * 365
        
        rvs = self.normal_dist.get_rvs(n_days)
        
        for recession_starting_year, recession_ending_year in zip(self.recession_starting_years, self.recession_ending_years):
            recession_starting_index = (recession_starting_year - self.starting_year) * 365
            recession_ending_index = (recession_ending_year - self.starting_year) * 365
            recession_rvs = self.recession_dist.get_rvs(365 * (recession_ending_year - recession_starting_year))
            rvs[recession_starting_index:recession_ending_index] = recession_rvs
            
        return rvs
    
    def get_yearly_contribution(self):
        n_days = self.period * 365
        contribution = np.zeros(n_days)
        contribution[::365] = self.yearly_contribution
        return contribution
    
    def get_cumulative_yearly_contribution(self):
        return np.cumsum(self.get_yearly_contribution())
    
    def get_cumulative_rvs(self):
        """cumulatively sums rvs

        Returns:
            np.array: cumulative sum of rvs
        """
        return (self.dist_start + np.cumsum(self.get_rvs())) / self.dist_start
    
    def get_simulation(self):
        """returns a single simulation (day-by-day)

        Returns:
            np.array: day-by-day amount of money in investment
        """
        return self.amount * self.get_cumulative_rvs() + self.get_cumulative_yearly_contribution()
    
    
    def get_amount_vector(self):
        """returns a single simulation (yearly)

        Returns:
            np.array: yearly amount of money in investment
        """
        simulation = np.array([self.get_simulation()[::365]])
        for _ in range(self.n_simulations - 1):
            simulation = np.vstack((
                simulation,
                self.get_simulation()[::365]
            ))
        simulation = np.mean(simulation, axis=0)
        return np.concatenate((np.array([0]), np.diff(simulation)))
        
    def get_amount_vector_over_timeframe(self, dataframe: pd.DataFrame) -> np.array:
        """Returns the amount vector for a given timeframe, with 0s if no investment.

        Args:
            dataframe (pd.DataFrame): Net worth dataframe

        Returns:
            np.array: Amounts over time frame
        """
        dataframe = dataframe.copy()
        
        years = dataframe['year'].to_numpy()
        
        amounts = np.zeros(len(years))
                
        starting_year_index = np.where(years == self.starting_year)[0][0]
        
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