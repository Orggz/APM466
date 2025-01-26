import scipy.optimize as optimize
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt


# Function to calculate Yield to Maturity (YTM)
# `freq` is set to 2 for semi-annual calculation
def bond_ytm(curr_price, face_value, time, coupon_rate, freq=2, guess=0.05):
    freq = float(freq)
    periods = time * freq  # Total number of periods
    coupon_ = coupon_rate * face_value / freq  # Semi-annual coupon payment
    dt = [(i + 1) / freq for i in range(int(periods))]  # Time intervals for payments

    # Define the YTM calculation function
    ytm_func = lambda y: sum([coupon_ / (1 + y / freq) ** (freq * t) for t in dt]) + face_value / (1 + y / freq) ** \
                         (freq * time) - curr_price

    # Solve for YTM using Newton's method
    return optimize.newton(ytm_func, guess) * 100  # Return YTM as a percentage


matplotlib.use('TkAgg')
# Load bond data from the Excel file
bonds = pd.read_excel(r'D:\\pythonProject\\Formatted Bond.xlsx')

# Convert date columns to datetime format
bonds['issue date'] = pd.to_datetime(bonds['issue date'])
bonds['maturity date'] = pd.to_datetime(bonds['maturity date'])

# Extract columns corresponding to current prices (Jan 6 to Jan 17)
cur_time_list = list(bonds.columns)[4:14]  # Columns for each pricing date, Jan 20th's data not used
format_cur_time = [pd.to_datetime(date) for date in cur_time_list]  # Convert to datetime

# Calculate periods (T) left until maturity for each date
T_lst_total = []
for curr in format_cur_time:
    T_lst = []
    for item in bonds["maturity date"]:
        delta = item - curr  # Days remaining until maturity
        left = delta.days
        T_lst.append(int(left // 180 + 1))  # Approximate periods in half-years
    T_lst_total.append(T_lst)

# Prepare other parameters needed for YTM calculation
coupon = list(bonds["Coupon"])
face_val = 100  # Face value of each bond
num_bonds = len(bonds["maturity date"])  # Total number of bonds

# Apply the bond_ytm function to calculate YTM for each bond and date
ytm_total = []
for i in range(len(cur_time_list)):
    ytm_lst = []
    cur_price_lst = bonds[cur_time_list[i]]
    cur_T_list = T_lst_total[i]
    for j in range(num_bonds):
        cur_price = cur_price_lst[j]
        cur_T = cur_T_list[j]
        cur_coupon = coupon[j]
        cur_ytm = bond_ytm(cur_price, face_val, cur_T, cur_coupon)
        ytm_lst.append(cur_ytm)
    ytm_total.append(ytm_lst)

# Organize the YTM data into a dictionary
ytm_dict = {cur_time_list[idx]: ytm_total[idx] for idx in range(len(cur_time_list))}
ytm_frame = pd.DataFrame(data=ytm_dict)  # Convert to DataFrame for analysis
ytm_frame.to_csv(r'D:\\pythonProject\\Q4a.csv', index=False)

# Define maturities (x-axis) and map YTM data for plotting
FIVE_YEARS = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
df = pd.DataFrame({'x': FIVE_YEARS,
                   'Jan 6': ytm_total[0],
                   'Jan 7': ytm_total[1],
                   'Jan 8': ytm_total[2],
                   'Jan 9': ytm_total[3],
                   'Jan 10': ytm_total[4],
                   'Jan 13': ytm_total[5],
                   'Jan 14': ytm_total[6],
                   'Jan 15': ytm_total[7],
                   'Jan 16': ytm_total[8],
                   'Jan 17': ytm_total[9]})

# Initialize the plot
plt.figure(figsize=(15, 5), dpi=80)
plt.style.use('classic')  # Apply a classic Matplotlib style
palette = plt.get_cmap('Set1')  # Use a color palette

# Plot each date's YTM curve
for idx, column in enumerate(df.columns[1:]):  # Skip the 'Maturity' column
    plt.plot(df['x'], df[column], marker='', color=palette(idx),
             linewidth=2, alpha=0.9, label=column)

# Add a legend
plt.legend(loc='upper left', ncol=2, title='Dates')

# Add titles and labels
plt.title("Yield to Maturity", fontsize=20, fontweight=0, color='orange')
plt.xlabel("Years to Maturity")
plt.ylabel("Yield to Maturity (YTM)")
folder_path = r'D:\pythonProject'
file_name = 'Q4a.png'
plt.savefig(f'{folder_path}\\{file_name}')


# Q4b:
# Function to calculate the first spot rate
def calculate_initial_spot_rate(price, face_value, maturity):
    spot = -math.log(price / face_value) / maturity
    return spot * 100


# Calculate initial spot rates for zero-coupon bonds
initial_spot_rates = []

for i in range(10):
    time_to_maturity = (bonds['maturity date'][0] - cur_time_list[i]).days / 365
    bond_price = bonds[cur_time_list[i]][0]
    face_value = 100
    spot_rate = calculate_initial_spot_rate(bond_price, face_value, time_to_maturity)
    initial_spot_rates.append(spot_rate)


# Function to calculate dirty price
def calculate_dirty_price(current_date, last_coupon_date, coupon_rate, face_value, clean_price):
    accrued_interest = (current_date - last_coupon_date).days / 365 * coupon_rate * face_value
    dirty_price = accrued_interest + clean_price
    return dirty_price


# Function for bootstrapping spot rates
def bootstrap_spot_rate(dirty_price, coupon_rate, face_value, previous_spot_rate, t_previous, t_current):
    payment_1 = coupon_rate * face_value
    payment_2 = (1 + coupon_rate) * face_value
    numerator = dirty_price - payment_1 * math.exp(-previous_spot_rate * t_previous)
    spot_rate = -math.log(numerator / payment_2) / t_current
    return spot_rate


# Perform bootstrapping to calculate the full spot rate curve
total_spot_rates = []
maturity_dates = bonds['maturity date']
coupon_rates = bonds['Coupon']

for i in range(10):
    last_coupon_date = maturity_dates[0] - relativedelta(months=6)
    current_date = cur_time_list[i]
    current_spot = initial_spot_rates[i]
    spots = [initial_spot_rates[i]]
    face_value = 100

    for j in range(9):
        if j == 0:
            t_previous = (maturity_dates[0] - current_date).days / 365
            t_current = (maturity_dates[1] - current_date).days / 365
            initial_spot = current_spot
            current_coupon_rate = coupon_rates[1]
            current_maturity_date = maturity_dates[1]
            current_price = bonds[current_date][0]
            dirty_price = calculate_dirty_price(current_maturity_date, last_coupon_date, current_coupon_rate,
                                                face_value,
                                                current_price)
            next_spot_rate = bootstrap_spot_rate(dirty_price, current_coupon_rate, face_value, initial_spot, t_previous,
                                                 t_current)
            spots.append(next_spot_rate * 100)
            current_spot = next_spot_rate
            last_coupon_date = maturity_dates[1]
        else:
            t_previous = (maturity_dates[j] - maturity_dates[j - 1]).days / 365
            t_current = (maturity_dates[j + 1] - maturity_dates[j - 1]).days / 365
            current_coupon_rate = coupon_rates[j + 1]
            current_maturity_date = maturity_dates[j + 1]
            current_price = bonds[current_date][j + 1]
            initial_spot = current_spot
            dirty_price = calculate_dirty_price(current_maturity_date, last_coupon_date, current_coupon_rate,
                                                face_value,
                                                current_price)
            next_spot_rate = bootstrap_spot_rate(dirty_price, current_coupon_rate, face_value, initial_spot, t_previous,
                                                 t_current)
            spots.append(next_spot_rate * 100)
            current_spot = next_spot_rate
            last_coupon_date = maturity_dates[j + 1]

    total_spot_rates.append(spots)

# Organize spot rate data into a DataFrame
spot_rate_dict = {cur_time_list[idx]: total_spot_rates[idx] for idx in range(len(cur_time_list))}
spot_rate_frame = pd.DataFrame(data=spot_rate_dict)
ytm_frame.to_csv(r'D:\\pythonProject\\Q4b.csv', index=False)

# Plot the spot rate curve
maturities = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
spot_rate_df = pd.DataFrame({'Maturity': maturities,
                             'Jan 6': total_spot_rates[0],
                             'Jan 7': total_spot_rates[1],
                             'Jan 8': total_spot_rates[2],
                             'Jan 9': total_spot_rates[3],
                             'Jan 10': total_spot_rates[4],
                             'Jan 13': total_spot_rates[5],
                             'Jan 14': total_spot_rates[6],
                             'Jan 15': total_spot_rates[7],
                             'Jan 16': total_spot_rates[8],
                             'Jan 17': total_spot_rates[9]})

plt.figure(figsize=(15, 5), dpi=80)
plt.style.use('classic')
palette = plt.get_cmap('Set1')

# Plot multiple lines for spot rates over different dates
for idx, column in enumerate(spot_rate_df.drop('Maturity', axis=1)):
    plt.plot(spot_rate_df['Maturity'], spot_rate_df[column], marker='', color=palette(idx),
             linewidth=2, alpha=0.9, label=column)

# Add legend, titles, and labels
plt.legend(loc='lower center', ncol=2, title='Dates')
plt.title("Spot Rate Curve", loc='center', fontsize=20, fontweight='bold', color='orange')
plt.xlabel("Years to Maturity")
plt.ylabel("Spot Rate (%)")

folder_path = r'D:\pythonProject'
file_name = 'Q4b.png'
plt.savefig(f'{folder_path}\\{file_name}')


# 4c: Function to calculate one-year forward rates based on spot rates
def one_year_forward_rates(spot_rates):
    t_a = 2  # Initial time period for the second spot rate (1-year)
    t_b = 1  # Initial time period for the first spot rate (0.5-year)
    forward_rates = []  # List to store the calculated forward rates

    # Loop through the first 4 spot rates to calculate forward rates
    for i in range(4):
        # r_a: the spot rate at time (i + 1) * 2 (e.g., 2 years, 3 years, etc.)
        # r_b: the spot rate at time i * 2 (e.g., 1 year, 2 years, etc.)
        r_a = spot_rates[(i + 1) * 2]  # Spot rate at time (i + 1) * 2
        r_b = spot_rates[i * 2]  # Spot rate at time i * 2
        # Calculate the forward rate using the formula
        forward_rate = (1 + r_a / 100) ** t_a / (1 + r_b / 100) ** t_b - 1
        forward_rates.append(abs(forward_rate) * 100)  # Append forward rate as a percentage
        t_a += 1  # Increment the time for the next forward rate
        t_b += 1  # Increment the time for the next forward rate

    return forward_rates  # Return the list of calculated forward rates


# List to store forward rates for all dates
total_forward_rate = []

# Calculate forward rates for each date
for i in range(10):
    forward_rate = one_year_forward_rates(total_spot_rates[i])  # Call function to calculate forward rates
    total_forward_rate.append(forward_rate)  # Store the forward rates for each date

# Print the forward rates for verification
print(total_forward_rate)

# Create a dictionary to store the forward rates for each date
for_dict = {}
for idx in range(len(cur_time_list)):
    for_dict[cur_time_list[idx]] = total_forward_rate[idx]

# Create a DataFrame from the dictionary for easy plotting
for_frame = pd.DataFrame(data=for_dict)

# Plot the forward rate curves
FIVE_YEARS = [i for i in range(2, 6)]  # Years to maturity for plotting
df = pd.DataFrame({'x': FIVE_YEARS,
                   'Jan 6': total_forward_rate[0],
                   'Jan 7': total_forward_rate[1],
                   'Jan 8': total_forward_rate[2],
                   'Jan 9': total_forward_rate[3],
                   'Jan 10': total_forward_rate[4],
                   'Jan 13': total_forward_rate[5],
                   'Jan 14': total_forward_rate[6],
                   'Jan 15': total_forward_rate[7],
                   'Jan 16': total_forward_rate[8],
                   'Jan 17': total_forward_rate[9]
                   })

# Initialize the plot
plt.figure(figsize=(15, 5), dpi=80)  # Set figure size
plt.style.use('classic')  # Apply classic Matplotlib style
palette = plt.get_cmap('Set1')  # Use the 'Set1' color palette

# Plot multiple forward rate curves for each date
num = 0
for column in df.drop('x', axis=1):
    num += 1
    plt.plot(df['x'], df[column], marker='', color=palette(num), linewidth=2, alpha=0.9, label=column)

# Add legend and labels
plt.legend(loc=2, ncol=2)  # Place the legend in the top left
plt.legend(loc='upper right', ncol=2, title='Dates')  # Add title to legend
plt.title("Forward Rate", loc='center', fontsize=20, fontweight=0, color='orange')  # Title of the plot
plt.xlabel("Maturity")  # X-axis label
plt.ylabel("Forward Rate")  # Y-axis label
folder_path = r'D:\pythonProject'
file_name = 'Q4c.png'
plt.savefig(f'{folder_path}\\{file_name}')

# Q5: Covariance Matrix Calculation for YTM
cov_mat = np.zeros([5, 9])  # Initialize a matrix to store log returns for YTM
for i in range(5):
    for j in range(1, 10):
        # Calculate log return for YTM
        X_ij = math.log((ytm_frame.iloc[i * 2, j]) / (ytm_frame.iloc[i * 2, j - 1]))
        cov_mat[i, j - 1] = X_ij  # Store the log return to the matrix

# Calculate the covariance matrix for YTM
ytm_cov = np.cov(cov_mat)
eig_val_ytm, eig_vec_ytm = np.linalg.eig(ytm_cov)  # Compute eigenvalues and eigenvectors
#print(ytm_cov)  # Print the YTM covariance matrix
#print(eig_val_ytm, eig_vec_ytm)  # Print eigenvalues and eigenvectors
print(eig_val_ytm[0] / sum(eig_val_ytm))  # Print the proportion of variance explained by the first eigenvalue


# Covariance Matrix Calculation for Forward Rates
cov_mat2 = np.zeros([4, 9])  # Initialize a matrix to store log returns for forward rates
for i in range(4):
    for j in range(1, 10):
        # Calculate log return for forward rates
        X_ij = math.log((for_frame.iloc[i, j]) / (for_frame.iloc[i, j - 1]))
        cov_mat2[i, j - 1] = X_ij  # Store the log return to the matrix

# Calculate the covariance matrix for forward rates
forward_cov = np.cov(cov_mat2)
eig_val_for, eig_vec_for = np.linalg.eig(forward_cov)  # Compute eigenvalues and eigenvectors
#print(forward_cov)  # Print the forward rate covariance matrix
#print(eig_val_for, eig_vec_for)  # Print eigenvalues and eigenvectors
print(eig_val_for[0] / sum(eig_val_for))  # Print the proportion of variance explained by the first eigenvalue
