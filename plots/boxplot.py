import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import argparse

# Function to read and parse the file
def read_and_parse_file(filepath):
    data = []
    with open(filepath, 'r') as file:
        for line in file:
            if line.startswith('./maxflow'):
                parts = line.split(' ')
                filename = parts[8]
                filename = filename.split('/')[-1]
                filename = filename.split('_')[0]

                algorithm = parts[10].rstrip(':')
                algorithm = algorithm.split(':')[0]
                if algorithm == '0':
                    algorithm = 'TC'
                else:
                    algorithm = 'VC'
            else:
                stats = line.strip().split(', ')
                stats_dict = {stat.split(': ')[0]: float(stat.split(': ')[1]) for stat in stats}
                stats_dict['Filename'] = filename
                stats_dict['Algorithm'] = algorithm
                data.append(stats_dict)
    return pd.DataFrame(data)

# Function to normalize statistics to the Avg value
def normalize_to_avg(df):
    for col in ['Min', 'Lower Quartile', 'Median', 'Upper Quartile', 'Max']:
        df[col] = df[col] / df['Avg']
    return df

# Adjusted plotting function
def plot_data(df, output_path):
    sns.set(style="whitegrid")
    
    # Normalize the data
    df_normalized = normalize_to_avg(df.copy())
    
    # Melt the dataframe for seaborn
    melted_df = pd.melt(df_normalized, id_vars=['Filename', 'Algorithm'], var_name='Variable', value_vars=['Min', 'Lower Quartile', 'Median', 'Upper Quartile', 'Max'])
    
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Filename', y='value', hue='Algorithm', data=melted_df, palette="Set3")  # Ensure correct column names are used
    plt.title('Normalized Statistics Box Plot by File and Algorithm')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)

def boxplot_test(df, output_path):

    # Define colors for each 'Algorithm' type
    colors = {'TC': '#ee9b00', 'VC': '#0081a7'}

    # Ordered the filenames
    filenames = [
        "corporate-leadership",
        "unicode",
        "ucforum",
        "movielens-u-i",
        "marvel",
        "movielens-u-t",
        "movielens-t-i",
        "youtube",
        "dbpedia-location",
        "bookcrossing",
        "stackoverflow",
        "IMDB",
        "dblp-author"
    ]


    # Change the subplot layout
    
    fig, ax = plt.subplots(figsize=(14, 6), dpi=200)
    plt.subplots_adjust(hspace=0.5)
    plt.ylabel('Workload distribution (normalized to average)', fontsize=15)
    plt.grid(axis = 'y', color = 'grey', linestyle = '--')
    

    # Position counter for plotting
    position = 0
    count = 0
    # ax.axvline(x=-0.5, color='black', linestyle='--')
    for filename in filenames:
        # Filter the dataframe for each filename
        df_filtered = df[df['Filename'] == filename]

        for index, row in df_filtered.iterrows():
            stats = [row['Min'], row['Lower Quartile'], row['Median'], row['Upper Quartile'], row['Max']]
            box_color = colors[row['Algorithm']]
            ax.bxp([{
                'whislo': row['Min'],  # Lower whisker
                'q1': row['Lower Quartile'],  # Q1
                'med': row['Median'],  # Median
                'q3': row['Upper Quartile'],  # Q3
                'whishi': row['Max'],  # Upper whisker
                'fliers': []  # Assuming no outliers
            }], positions=[position], widths=0.7, patch_artist=True, manage_ticks=False,
            boxprops=dict(facecolor=box_color, color=box_color),  # Box color
            medianprops=dict(color="yellow"),  # Median line color
            whiskerprops=dict(color=box_color),  # Whisker color
            capprops=dict(color=box_color))  # Cap color (ends of whiskers)
            
            # Add algorithm name below the box for clarity
            # ax.text(position, -0.03, row['Algorithm'], ha='center', va='top', transform=ax.get_xaxis_transform(), size=12)
            
            position += 1
            # ax.legend()
        ax.text(position - 1.5, -0.03, "B"+str(count) , ha='center', va='top', transform=ax.get_xaxis_transform(), size=14)
        
        # Create custom patches for the legend
        legend_patches = [patches.Patch(color=color, label=(algorithm+"+BCSR")) for algorithm, color in colors.items()]

        # Adding the legend to the plot
        ax.legend(handles=legend_patches, title="Algorithms", fontsize=13, title_fontsize=13, loc='upper right')

        # ax.axvline(x=position - 0.5, color='black', linestyle='--')
        count += 1
    # Add some space between different filenames
    position += 1

    # Set y-axis to log scale
    ax.set_yscale('log')
    ax.axhline(color='grey', linestyle='--')
    ax.set_xlabel('')
    # ax.set_ylabel('Load distribution across warps (normalized to average)')
    
    # Hide x ticks and labels
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    # plt.legend(['TC', 'VC'], loc='upper right', fontsize=12, labelcolor=colors.values(), handler_map={ line, line2})
    plt.savefig(output_path, bbox_inches='tight')




if __name__ == "__main__":
    # Replace 'your_file_path.txt' with the path to your file
    parser = argparse.ArgumentParser(description="Plot normalized statistics box plot by file and algorithm.")
    parser.add_argument("--file", help="Path to the file with statistics.", required=True)
    parser.add_argument("--output", help="Path to the output file.", required=True)


    file_path = parser.parse_args().file
    output_path = parser.parse_args().output


    df = read_and_parse_file(file_path)
    #print(df)
    df_normalized = normalize_to_avg(df)
    #print(df_normalized)
    #plot_data(df_normalized, output_path)
    boxplot_test(df_normalized, output_path)