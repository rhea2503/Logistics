import pandas as pd
from sqlalchemy import create_engine,MetaData,select
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import statsmodels.api as sm
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def logistics():
    engine = create_engine('sqlite:////Users/riyatharu/Downloads/shipments.db')
    metadata = MetaData()
    metadata.reflect(bind=engine)
    conn = engine.connect()
    eval = metadata.tables['eval']
    query = select(eval)
    result = conn.execute(query)
    df = pd.DataFrame(result.fetchall(), columns=result.keys())
    df.dropna(inplace=True)
    df['total_weight'] = df['total_weight'].str.replace(',', '.').astype(float)
    # Define weight bins and labels
    weight_bins = [ 900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500,2600,2700  ]
    weight_labels = [ f'{weight_bins[ i - 1 ]}-{weight_bins[ i ]}' for i in range( 1, len( weight_bins ) ) ]

    # Bin the weights and calculate average cost per kg for each carrier and weight bin
    df[ 'weight_bin' ] = pd.cut( df[ 'total_weight' ], bins=weight_bins, labels=weight_labels )
    avg_cost = df.groupby( [ 'carrier_name', 'weight_bin' ] ).mean( )[ 'costperkg' ].unstack( )

    # Create stacked bar chart
    fig, ax = plt.subplots( figsize=(10, 6) )
    avg_cost.plot( kind='bar', stacked=True, ax=ax )

    # Set axis labels and title
    ax.set_xlabel( 'Freight weight (kg)' )
    ax.set_ylabel( 'Average cost per kg' )
    ax.set_title( 'Average cost per kg by carrier and freight weight' )

    # Add legend
    ax.legend( title='Carrier', bbox_to_anchor=(1.01, 1), loc='upper left' )

    plt.show( )

    # Linear
    grouped = df.groupby( [ 'carrier_name', 'total_weight' ] ).mean( ).reset_index( )
    # Plot scatter plot of total weight vs. cost per kg for each carrier
    for carrier in grouped[ 'carrier_name' ].unique( ):
     data = grouped[ grouped[ 'carrier_name' ] == carrier ]
     plt.scatter( data[ 'total_weight' ], data[ 'total_cost' ], label=carrier )
     # Fit linear regression model to data for each carrier
     model = LinearRegression( ).fit( data[ [ 'total_weight' ] ], data[ 'total_cost' ] )
     x_range = range( int( data[ 'total_weight' ].min( ) ), int( data[ 'total_weight' ].max( ) ) )
     plt.plot( x_range, model.predict( [ [ x ] for x in x_range ] ) )
     # Calculate slope, intercept, and R-squared value for each carrier
     slope = model.coef_[ 0 ]
     intercept = model.intercept_
     r_squared = r2_score( data[ 'total_cost' ], model.predict( data[ [ 'total_weight' ] ] ) )
     print( f"Carrier: {carrier}" )
     print( f"Slope: {slope}" )
     print( f"Intercept: {intercept}" )
     print( f"R-squared: {r_squared}\n" )
     plt.xlabel( 'Total weight (kg)' )
     plt.ylabel('Cost ')
     plt.title( 'Cost  vs. total weight for each carrier' )
     plt.legend( )
    plt.show( )







    conn.close()





if __name__ == '__main__':
     logistics()